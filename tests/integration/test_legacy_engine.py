"""Current-target consumers on the retained native streaming scheduler."""
from dataclasses import asdict
from types import SimpleNamespace
import time
import numpy as np
import pytest
import torch

from env.core.run_env import SlayTheSpireEnv
from env.search.anchor_wave import CpuFallback
from env.search.anchored_chains import paired_anchor_search, ChainRequest
from env.core.native_observation import NativeRootObservationBatch
from rl.algorithms.search_auxiliary import SearchTrainingConfig
from rl.policies.deck_latent import DeckLatentPolicy
from rl.search.deck_latent import DeckLatentScorer

pytestmark = [pytest.mark.native, pytest.mark.timeout(240)]


class ReferenceService:
    """CPU model in place of transport only; real scorer and native states."""
    def __init__(self, policy):
        self.policy, self.replies, self.sequence = policy, {}, 0
        self.requests = {}

    def begin_call(self, kind, payload):
        self.sequence += 1
        self.replies[self.sequence] = self.call(kind, payload)
        return self.sequence

    def take_reply(self, ident):
        return True, self.replies.pop(ident)

    def call(self, kind, payload):
        if kind == 'fallback_begin':
            _, obs, masks, slots, capacity = payload
            roots = [{k:v[j] for k,v in obs.items()} for j in range(len(masks))]
            self.scorer = DeckLatentScorer(self.policy, roots)
            self.caches = {}
            return dict(fields=self.scorer.observation_keys, static=self.scorer.context_keys,
                actor=dict(logprobs=np.zeros_like(masks,float), probabilities=np.zeros_like(masks,float),
                           values=np.zeros(len(masks))))
        if kind == 'fallback_end':
            return dict(context_refreshes=self.scorer.context_refreshes)
        _, rows = payload
        requests = [ChainRequest(r,dict(self.scorer.roots[r],**delta),mask,kind,
                       self.caches.setdefault(lane,{}),lane) for r,lane,kind,mask,delta in rows]
        for row in requests:
            self.requests.setdefault(row.lane,[]).append((row.kind,
                {k:v.copy() for k,v in row.observation.items()},row.action_mask.copy()))
        return self.scorer(requests)


@pytest.mark.parametrize('scenario', ['ordinary','win','loss','draw','multi','multi_prefix','potion_multi'])
def test_native_wave_preserves_paired_actions_observations_and_root(complete_combat_factory, scenario):
    env = complete_combat_factory(['STRIKE_RED']*4+['DEFEND_RED']*4+['BASH'],seed=42101,
        card_language=True,upper_public_context=True,event_option_observation=True,deck_latent_context=True)
    if scenario == 'win':
        env.thick.bc.set_player_strength(999)
    elif scenario == 'loss':
        env.thick.bc.apply_custom_state({'player':{'hp':1}})
    elif scenario in ('draw','multi','multi_prefix','potion_multi'):
        import slaythespire as sts
        env.thick.bc.apply_custom_state(dict(hand=['SECRET_TECHNIQUE' if scenario=='draw' else 'PURITY',
            'DEFEND_RED','STRIKE_RED','DEFEND_RED'], draw=['DEFEND_RED','SHRUG_IT_OFF','STRIKE_RED'],
            potions=['Gamblers Brew']))
        if scenario == 'potion_multi':
            env.thick.bc.execute(sts.Action(sts.ActionType.POTION,0))
        else:
            env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    env._invalidate_backend_cache()
    if scenario == 'multi_prefix':
        from env.core.native_actions import AUTOREGRESSIVE_MULTI
        _, reward, terminated, truncated, info = env.step(AUTOREGRESSIVE_MULTI+1)
        assert reward == 0 and not terminated and not truncated and not info['simulator_transition']
        assert env._multi == {1}
    obs=env._obs();mask=env.action_masks();before=env.export_training_state()
    torch.manual_seed(20261111)
    policy=DeckLatentPolicy(env.observation_space,env.action_space,lambda _:5e-5,
        d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
        combat_value_target='pre-reward-anchor',deck_combat_encoder='transformer64x2').eval()
    policy.set_training_mode(False)
    budget=int(mask.sum())*2;depth=12;seed=20261111
    scorer=DeckLatentScorer(policy,[obs]);reference_rows={}
    def score(requests):
        for row in requests:
            reference_rows.setdefault(row.lane-1,[]).append((row.kind,
                {k:v.copy() for k,v in row.observation.items()},row.action_mask.copy()))
        return scorer(requests)
    expected, counts=paired_anchor_search([env],[budget],score,seed=seed,max_depth=depth,
        max_active=16,timeout_seconds=60,observation_batch=NativeRootObservationBatch(env))
    worlds=np.random.default_rng(seed).integers(0,2**63-1,2,dtype=np.int64)
    key=(0,0);metadata={key:dict(slot=0,mask=mask,row=dict(observation=obs,worlds=worlds,
        budget=budget,root=dict(actions=np.flatnonzero(mask))))}
    service=ReferenceService(policy)
    wave=CpuFallback([key],metadata,[env],service,SearchTrainingConfig(max_depth=depth,
        normal_budget=budget,boss_budget=budget,max_active=16,timeout_seconds=60),threads=2)
    try:
        while not wave.complete:
            result=wave.step()
        got=result[key]
        np.testing.assert_array_equal(got['actions'],expected[0]['actions'])
        np.testing.assert_array_equal(got['visits'],expected[0]['visits'])
        np.testing.assert_allclose(got['values'],expected[0]['values'],rtol=2e-5,atol=2e-6)
        for field in ('chains','atomic_steps','policy_rows','forced_rows','anchor_rows','cutoff_rows','terminal_rows'):
            assert wave.stats[field]==counts[field],field
        assert service.requests.keys()==reference_rows.keys()
        fields=set(service.scorer.observation_keys)
        for job,rows in reference_rows.items():
            actual=service.requests[job];assert len(rows)==len(actual)
            for (kind,o,m),(akind,ao,am) in zip(rows,actual,strict=True):
                assert kind==akind
                np.testing.assert_array_equal(m,am)
                for name in fields:
                    np.testing.assert_array_equal(o[name],ao[name],err_msg=name)
        assert env.export_training_state()==before
    finally:
        wave.close()


@pytest.mark.cuda
@pytest.mark.parametrize('force_fallback,environment_count', [(False,4), (True,4), (False,40)])
def test_legacy_native_real_ppo_two_rounds_and_ordinary_restore(tmp_path,cuda_battle_artifacts,
                                                            monkeypatch,force_fallback,environment_count):
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
    from gymnasium.wrappers import TimeLimit
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.algorithms.packed_ppo import configure_packed_updates
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.training.environments import TrainingStateWrapper
    from rl.training.reward import StrategicRewardWrapper
    from rl.training.checkpoints import save_bundle,restore_learner
    from rl.search.resident_service import close_collection_service
    if force_fallback:
        from env.search.resident_transport import ResidentRootExporter
        original = ResidentRootExporter.export
        def native_first_slot(self,indices,generation=None):
            result = original(self,indices,generation)
            for i,row in zip(indices,result,strict=True):
                if i == 0:row['native_fallback'] = True
            return result
        monkeypatch.setattr(ResidentRootExporter,'export',native_first_slot)
    def factory():
        return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
            card_language=True,global_map_view=True,upper_public_context=True,event_option_observation=True,
            combat_action_contract='autoregressive-v2',combat_exit_anchor=True,reward_flow='sequential-v1',
            deck_latent_context=True)),6000))
    def vector():return VecMonitor(NativeSearchVecEnv([factory]*environment_count,threads=2))
    class Verify(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_end(self):
            b=self.model.rollout_buffer
            assert b.full and np.all(b.slot_positions==b.buffer_size)
            assert np.isfinite(b.values).all() and np.isfinite(b.returns).all()
            assert self.model._resident_search_service.parked
            assert b.search_actions
            for (t,e),(a,lp) in b.search_actions.items():
                assert b.actions[t,e,0]==a and b.action_masks[t,e,a]
                assert b.log_probs[t,e]==pytest.approx(lp,abs=1e-6)
    env=vector();target=None;model=restored=None
    try:
        model=AccumulatingMaskablePPO(DeckLatentPolicy,env,seed=20261111,n_steps=32,batch_size=32,
            n_epochs=1,device='cuda',gamma=1.,gae_lambda=1.,learning_rate=5e-5,
            reasoning_coef=0,reasoning_updates=0,nosl_is_az=False,
            policy_kwargs=dict(d_model=32,n_layers=1,n_heads=4,room_lambda=1.,reward_flow='sequential-v1',
                combat_value_target='pre-reward-anchor',deck_combat_encoder='transformer64x2'))
        configure_packed_updates(model,dict(token_budget=1024,max_rows=8,precision='fp32',context_grouping=True))
        model.deck_search_config=asdict(SearchTrainingConfig(fraction=1.,normal_budget=8,boss_budget=16,
            max_depth=4,max_active=16,timeout_seconds=60,backend='cuda-resident',
            cuda_artifacts=str(cuda_battle_artifacts[0]),inference_cache_bytes=64*1024**2,
            action_selection='search-filter-policy-rank-v1'))
        model.deck_search_snapshot_dir=str(tmp_path/'weights')
        samples=environment_count*32*2
        model.learn(samples,callback=Verify())
        assert (model.deck_search_last_stats['native_fallback_roots'] > 0) == force_fallback
        assert model._resident_search_service.status['generation']==2
        source=save_bundle(model,tmp_path/'first',rank=0,world_size=1,run_id='legacy-test',
                           wandb_id=None,config={'ascension':0})
        close_collection_service(model)
        target=vector()
        restored=AccumulatingMaskablePPO.load(source/'model.zip',env=target,device='cuda')
        restored._setup_learn(samples,reset_num_timesteps=False)
        restore_learner(restored,source,rank=0,world_size=1)
        assert restored._recovery_environment_restored
        assert restored.deck_search_rng==model.deck_search_rng
        for key, value in restored._last_obs.items():
            np.testing.assert_array_equal(target.venv.buf_obs[key], value, err_msg=key)
            np.testing.assert_array_equal(model._last_obs[key], value, err_msg=key)
        restored.learn(samples,reset_num_timesteps=False,callback=Verify())
        assert restored.num_timesteps==samples*2
        assert restored._resident_search_service.status['generation']==2
        save_bundle(restored,tmp_path/'resumed',rank=0,world_size=1,run_id='legacy-test',
                    wandb_id=None,config={'ascension':0})
    finally:
        if model is not None:close_collection_service(model)
        if restored is not None:close_collection_service(restored)
        env.close()
        if target is not None:target.close()

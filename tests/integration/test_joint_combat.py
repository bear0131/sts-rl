import numpy as np
import pytest
import torch
import slaythespire as sts
from sb3_contrib import MaskablePPO

from env.core import native_actions as C
from rl.policies.split_policy import SplitStSPointerPolicy

pytestmark = pytest.mark.native


def test_joint_model_predicts_atomic_targeted_actions(combat_factory):
    env = combat_factory(['BASH','STRIKE_RED','DEFEND_RED','STRIKE_RED','DEFEND_RED'],
                         'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    model = MaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cpu')
    model.policy.set_training_mode(False)
    obs = env._obs()
    assert env.action_space.n == C.JOINT_SIZE
    assert obs['combat_legal'].shape == (C.JOINT_WIDTH,)
    mask = env.action_masks()
    assert not mask[:C.JOINT_END].any()
    predicted, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert mask[int(predicted)]
    pair = next((aid,bits) for aid,bits in env.thick.bc.joint_policy_actions()
                if bits >> 29 == 0 and env.thick.bc.is_targeted_action(bits))
    aid, bits = pair
    before = env.thick.bc.clone()
    before.execute(sts.Action(bits))
    result = env.step(aid)
    assert env._pending is None and not result[4].get('illegal')
    assert env.thick.bc.to_dict() == before.to_dict()


@pytest.mark.parametrize('multi,full',[(False,False),(True,False),(True,True)])
def test_native_complete_policy_survives_search_and_execution(combat_factory,multi,full):
    env = combat_factory(['PURITY' if multi else 'STRIKE_RED']*5, 'Two Louse',
                         combat_action_contract=C.JOINT_CONTRACT)
    if multi:
        env.thick.bc.apply_custom_state({'hand':['PURITY']+['STRIKE_RED']*9})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
        env._invalidate_backend_cache()
    before = env.thick.bc.to_dict()
    logits = np.zeros(C.JOINT_SIZE)
    legal = np.flatnonzero(env.action_masks())
    assert len(legal) == (130 if multi else 11)
    favored = legal[-1] if multi else C.JOINT_PLAY+1
    logits[favored] = 12
    response = env.nosl_is_az_cpp_begin(logits,state_nonce=42,simulations=4,depth=1,
        leaf_batch_size=4,compact_observation=True,root_search_mode=2 if full else 0,
        gumbel_scale=0.)
    for _ in range(8):
        if response['done']:
            break
        obs = response['observation_batch']
        assert obs['combat_legal'].shape[1] == C.JOINT_WIDTH
        for row, leaf in enumerate(response['leaf_meta']):
            np.testing.assert_array_equal(leaf['action_mask'][C.JOINT_END:], obs['combat_legal'][row])
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            [(leaf['request_id'], .3, np.zeros(C.JOINT_SIZE)) for leaf in response['leaf_meta']],max_leaves=4)
    assert response['done']
    result = response['result']
    assert result['visit_policy'].shape == (C.JOINT_SIZE,)
    assert result['visit_policy'].sum() == pytest.approx(1)
    assert result['action'] == favored
    if full:
        assert np.count_nonzero(result['visit_policy']) == 130
        assert np.count_nonzero(result['root_visit_policy']) <= 4
    assert env.thick.bc.to_dict() == before
    env.step(result['action'])
    assert env.thick.bc.to_dict() != before


def test_complete_target_prior_reaches_internal_search_nodes(combat_factory):
    env = combat_factory(['DEFEND_RED']*5, 'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['DEFEND_RED','STRIKE_RED','STRIKE_RED']})
    env._invalidate_backend_cache()
    root_logits = np.zeros(C.JOINT_SIZE)
    root_logits[C.JOINT_PLAY+5] = 14  # First play the untargeted Defend.
    for target in (0,1):
        response = env.nosl_is_az_cpp_begin(root_logits,state_nonce=42,simulations=4,depth=2,
            leaf_batch_size=1,compact_observation=True,fixed_worldline=True)
        for _ in range(20):
            if response['done']:
                break
            logits = np.zeros(C.JOINT_SIZE)
            logits[C.JOINT_PLAY+target] = 14
            response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
                [(leaf['request_id'],.3,logits) for leaf in response['leaf_meta']],max_leaves=1)
        assert response['done']
        root = env.nosl_is_az_cpp_tree_snapshot()
        first = next(e for e in root['children'] if e['action_type']==0 and e['source_idx']==0)
        child = env.nosl_is_az_cpp_tree_snapshot([first['edge_index']])
        best = max(child['children'], key=lambda e:e['prior'])
        assert best['action_type']==0 and best['source_idx']==0 and best['target_idx']==target
        assert best['prior'] > .99


def test_cpp_and_python_complete_codec_match_public_entities(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE']*5, combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['SECRET_TECHNIQUE'],
        'draw':['STRIKE_RED']*12+['SHRUG_IT_OFF','DEFEND_RED']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
    desc=env.thick.bc.select_info()
    handles=[c['native_index'] for c in desc['candidates']]
    for aid,bits in env.thick.bc.joint_policy_actions():
        assert aid == C.joint_action_id(bits, targeted=env.thick.bc.is_targeted_action(bits),candidate_indices=handles)
        assert C.joint_native_bits(aid,candidate_indices=handles) == bits


def test_pending_target_views_share_the_source_conditional_distribution(combat_factory):
    from env.core import action_space as A
    from env.core import vocab as V
    env = combat_factory(['BASH','STRIKE_RED','DEFEND_RED','STRIKE_RED','DEFEND_RED'],
                         'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['BASH','STRIKE_RED','DEFEND_RED']})
    env._invalidate_backend_cache()
    model = MaskablePPO(SplitStSPointerPolicy, env, n_steps=2, batch_size=2,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cpu')
    model.policy.set_training_mode(False)
    with torch.no_grad():
        model.policy.core.combat.complete_head.source_query.weight.normal_(0, .5)
    obs = env._obs()
    variants = [obs]
    for slot in (0,1):
        pending = {k:v.copy() for k,v in obs.items()}
        pending['arg_step'][0] = V.ARG_STEP_VOCAB_INDEX[A.PICK_TARGET]
        pending['pending_card_slot'][0] = slot+1
        variants.append(pending)
    tensors = {k:torch.as_tensor(np.stack([view[k] for view in variants])) for k in obs}
    with torch.no_grad():
        logits, values = model.policy.core.combat_actor_critic(tensors)
    for row, slot in enumerate((0,1), 1):
        root = logits[0,C.JOINT_PLAY+6*slot:C.JOINT_PLAY+6*slot+2].softmax(-1)
        pending = logits[row,A._TARGET:A._TARGET+2].softmax(-1)
        torch.testing.assert_close(root,pending,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(values[0],values[row])
    assert not torch.allclose(logits[1,A._TARGET:A._TARGET+2].softmax(-1),
                              logits[2,A._TARGET:A._TARGET+2].softmax(-1),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('mixed', [False, True])
def test_forced_end_roots_skip_forward_and_search_but_keep_value_samples(short_combat_env_type, monkeypatch, mixed):
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    class OnlyEnd(short_combat_env_type):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.thick.bc.apply_custom_state({'hand':['WOUND']*5, 'draw':['WOUND']*20, 'discard':[]})
            self._invalidate_backend_cache()
            return self._obs(), self._info()
    env = OnlyEnd(combat_action_contract=C.JOINT_CONTRACT)
    vector = DummyVecEnv([lambda:env] + ([lambda:short_combat_env_type(combat_action_contract=C.JOINT_CONTRACT)] if mixed else []))
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, vector, n_steps=4, batch_size=4,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cpu',
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_simulations=4, nosl_is_az_updates=1, hierarchical_combat_hp=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("forced End Turn must not create a search")
    monkeypatch.setattr(env, "nosl_is_az_cpp_begin", forbidden)
    calls = []
    original = model.policy.core.actor_critic
    def count(obs, *args, **kwargs):
        calls.append(len(obs['phase']))
        return original(obs, *args, **kwargs)
    monkeypatch.setattr(model.policy.core, "actor_critic", count)
    model.learn(total_timesteps=4 * vector.num_envs)
    assert model.num_timesteps == 4 * vector.num_envs
    assert model._nosl_is_az_rollout_upper_counts['forced_end_turn_roots'] == 4
    # Ordinary and hierarchical return bootstraps remain; forced roots do not forward.
    if not mixed:
        assert calls == [1, 1]
    else:
        assert calls[:4] == [1] * 4 and calls[-2:] == [2, 2]
    roots = [s for s in model._nosl_is_az_pending_fights[0] if s['sample_kind']=='combat_root']
    assert len(roots) == 4
    assert all(s['policy_weight']==1 and s['policy_target'][C.JOINT_END]==1 for s in roots)
    assert all(np.isnan(s['estimated_value']) for s in roots)
    boundaries = [s for s in model._nosl_is_az_pending_fights[0] if s['sample_kind']=='end_turn_boundary']
    assert len(boundaries) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="joint AMP learner requires a GPU")
def test_joint_bf16_learner_updates_autoregressive_and_target_heads(combat_factory):
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from stable_baselines3.common.logger import configure
    env = combat_factory(['PURITY']*5, 'Two Louse', combat_action_contract=C.JOINT_CONTRACT)
    env.thick.bc.apply_custom_state({'hand':['STRIKE_RED']*5})
    env._invalidate_backend_cache()
    normal, normal_mask = env._obs(), env.action_masks()
    model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, n_steps=4, batch_size=4,
        policy_kwargs=dict(d_model=32,n_layers=2,n_heads=4), seed=42, device='cuda',
        nosl_is_az=True,nosl_is_az_cpp=True,nosl_is_az_inner_only=True,amp_mode='bf16',
        nosl_is_az_updates=2,nosl_is_az_batch_size=4,nosl_is_az_micro_batch_size=2,
        nosl_is_az_replay_capacity=8)
    model.set_logger(configure(format_strings=[]))
    env.thick.bc.apply_custom_state({'hand':['PURITY']+['STRIKE_RED']*9})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD,0))
    env._invalidate_backend_cache()
    multi, multi_mask = env._obs(), env.action_masks()
    samples=[]
    for obs,mask,aid in ((normal,normal_mask,C.JOINT_PLAY+1),
                         (multi,multi_mask,C.JOINT_MULTI+3)):
        target=np.zeros(C.JOINT_SIZE,np.float32);target[aid]=1
        samples.append(dict(observation=obs,action_mask=mask,policy_target=target,value_target=.25,policy_weight=1.))
        samples.append(dict(observation=obs,action_mask=np.zeros(C.JOINT_SIZE,bool),
                            policy_target=np.zeros(C.JOINT_SIZE,np.float32),value_target=.25,policy_weight=0.))
    model._nosl_is_az_samples=samples
    stats=model._train_nosl_is_az()
    assert stats['updates']==2 and stats['finite_fraction']==1
    head=model.policy.core.combat.complete_head
    assert torch.count_nonzero(head.source_query.weight)>0
    assert torch.count_nonzero(head.subset.choice_key.weight)>0
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())

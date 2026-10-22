import numpy as np
import pytest
import torch
import slaythespire as sts
from sb3_contrib import MaskablePPO

from env.core import native_actions as C, encode
from env.search.native_batch import NativeSearchBatch
from rl.policies.entity_combat import EntityCombatPolicy
from rl.search.native import NativeSearchConfig, search

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def _model(env):
    return MaskablePPO(EntityCombatPolicy, env, seed=42, n_steps=4, batch_size=4,
        device='cpu', policy_kwargs=dict(entity_config=dict(width=32, layers=1,
            heads=4, quantiles=8, boss_win_weight=.9)))


@pytest.mark.parametrize('boss', [False, True])
def test_boss_bit_canonical_packed_and_threaded_leaves(combat_factory, boss):
    env=combat_factory(['DEFEND_RED','STRIKE_RED']*3, 'Slime Boss',
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True)
    env.thick.gc.cur_room = sts.Room.BOSS if boss else sts.Room.MONSTER
    env._invalidate_backend_cache()
    assert env._obs()['combat_is_boss'][0] == int(boss)
    logits=np.zeros(env.action_space.n)
    kwargs=dict(state_nonce=42, simulations=4, depth=1, leaf_batch_size=1, compact_observation=True,
                root_search_mode=0, paper_full_gumbel=False, boss_win_weight=.9)
    batch=NativeSearchBatch([env], threads=1)
    try:
        batch.begin({0:((logits,),kwargs)})
        assert not batch.done and batch.rows > 0
        obs=batch.observations()
        assert np.all(obs['combat_is_boss']==int(boss))
        while not batch.done:
            batch.advance(np.zeros((batch.rows,env.action_space.n)),np.full(batch.rows,.4))
    finally:
        batch.close()


@pytest.mark.parametrize('boss,expected', [(False,.05),(True,.905)])
def test_exact_native_victory_utility_before_relic_healing(combat_factory, boss, expected):
    env=combat_factory(['STRIKE_RED']*5, 'Jaw Worm', relics=['Burning Blood'],
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, card_language=True)
    env.thick.gc.cur_room=sts.Room.BOSS if boss else sts.Room.MONSTER
    env.thick.bc.apply_custom_state({'hand':['STRIKE_RED'], 'player':{'hp':10,'energy':3}})
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    mask=env.action_masks(); ids=np.flatnonzero(mask)
    action=next(int(a) for a in ids if a != 0)
    logits=np.full(env.action_space.n,-1000.);logits[action]=1000
    response=env.nosl_is_az_cpp_begin(logits,state_nonce=42,simulations=1,depth=0,
        root_search_mode=0,paper_full_gumbel=False,boss_win_weight=.9)
    assert response['done'], response
    assert response['result']['root_value'] == pytest.approx(expected)


def test_model_save_reload_graph_forward_and_search_contract(combat_factory,tmp_path):
    env=combat_factory(['STRIKE_RED','DEFEND_RED']*3,
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,card_language=True)
    model=_model(env);model.policy.set_training_mode(False)
    obs=env._obs();tensors={k:torch.as_tensor(np.repeat(v[None],2,axis=0)) for k,v in obs.items()}
    tensors['combat_is_boss'][:,0]=torch.tensor([0.,1.])
    with torch.no_grad():
        logits,values,out=model.policy.core.combat(tensors)
        p=out['win_logits'].sigmoid();h=out['hp_quantiles'].mean(-1,keepdim=True)
        torch.testing.assert_close(values[0],(p*h)[0])
        torch.testing.assert_close(values[1],(p*(.9+.1*h))[1])
    path=tmp_path/'boss.zip';model.save(path)
    loaded=MaskablePPO.load(path,device='cpu');loaded.policy.set_training_mode(False)
    with torch.no_grad():
        _,again,_=loaded.policy.core.combat(tensors)
    torch.testing.assert_close(again,values,rtol=0,atol=0)
    assert loaded.policy.entity_config['boss_win_weight']==.9
    config=NativeSearchConfig.from_model(loaded,simulations=4,depth=1,gumbel_scale=0)
    result=search(loaded,env,obs,config,nonce=42)
    assert env.action_masks()[result['action']]
    with pytest.raises(ValueError,match='must match'):
        search(loaded,env,obs,NativeSearchConfig(simulations=4),nonce=42)


@pytest.mark.parametrize('boss,expected', [(False, .05), (True, .905)])
def test_context_control_keeps_exact_train47_terminal_and_observed_hp(boss, expected):
    from env.core.run_env import SlayTheSpireEnv
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract=C.AUTOREGRESSIVE_CONTRACT,
        outer_return_context=True, combat_value_objective='combat-outcome-quantiles-v1')
    try:
        env.reset(seed=42101)
        for _ in range(64):
            if env.thick.bc is not None:
                break
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert env.thick.bc is not None
        env.thick.bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
        env.thick.gc.cur_room = sts.Room.BOSS if boss else sts.Room.MONSTER
        env.thick.bc.apply_custom_state({'hand': ['STRIKE_RED'], 'player': {'hp': 10, 'energy': 3}})
        env.thick.bc.set_player_strength(999)
        env._invalidate_backend_cache()
        action = next(int(a) for a in np.flatnonzero(env.action_masks())
                      if C.AUTOREGRESSIVE_PLAY <= a < C.AUTOREGRESSIVE_POTION)
        logits = np.full(env.action_space.n, -1000.)
        logits[action] = 1000.
        pool = NativeSearchBatch([env], 1)
        try:
            pool.begin({0: ((logits,), dict(state_nonce=42, simulations=1, depth=0,
                root_search_mode=0, paper_full_gumbel=False, boss_win_weight=.9,
                fixed_worldline=True, compact_observation=True))})
            assert pool.done  # Exact terminal; no upper critic or neural bootstrap.
            assert pool.results()[0]['root_value'] == pytest.approx(expected)
        finally:
            pool.close()
        _, _, _, _, info = env.step(action)
        assert info['nosl_value_terminal_target'] == pytest.approx(.05)
        assert info['nosl_value_terminal_outcome'] == 'win'
        terminal = info['nosl_value_terminal_observation']
        assert terminal['run_hp'][0] == pytest.approx(10 / encode.HP_SCALE)
        assert {'outer_deck_misc', 'deck_ids', 'floor'} <= terminal.keys()
        assert 'outer_combat_exit' not in info
    finally:
        env.close()

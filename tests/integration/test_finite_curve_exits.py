import numpy as np
import pytest
import torch

from env.core.finite_curves import FiniteCurveBank, OBJECTIVE
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms import outer_return as outer

pytestmark = [pytest.mark.native, pytest.mark.timeout(120)]


@pytest.mark.parametrize('outlet', ['win', 'escape', 'loss'])
def test_real_search_settlement_and_executed_exit_resources_match(finite_curve_bank, outlet):
    import slaythespire as sts
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
                         outer_return_context=True, combat_value_objective=OBJECTIVE)
    env.reset(seed=42101)
    for _ in range(64):
        if env.thick.bc is not None: break
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.thick.bc is not None
    gc = env.thick.gc
    gc.set_potion_state(['Smoke Bomb', 'Fire Potion']); gc.cur_hp=30
    env.thick.bc=sts.create_battle(gc,env.thick._encounter_name())
    monsters=[dict(hp=1) for _ in env.thick.bc.to_dict()['monsters']]
    env.thick.bc.apply_custom_state(dict(hand=['CLEAVE'],player=dict(hp=30,energy=3,strength=999),monsters=monsters))
    if outlet=='loss':
        # Explicit one-HP lethal incoming hit, no Fairy revival.
        gc.set_potion_state([])
        env.thick.bc=sts.create_battle(gc,'Cultist')
        env.thick.bc.apply_custom_state(dict(player=dict(hp=1,block=0),monsters=[dict(move='CULTIST_DARK_STRIKE')]))
    env._invalidate_backend_cache()
    action = sts.Action(sts.ActionType.POTION,0) if outlet=='escape' else sts.Action(sts.ActionType.END_TURN) if outlet=='loss' else sts.Action(sts.ActionType.CARD,0)
    leaf=env.thick.bc.clone();leaf.execute(action);assert leaf.is_battle_over
    search_env,outcome=env.thick.settle_combat_leaf(leaf)
    # Same complete action through the real native execution/settlement path.
    # Use the same committed-native-action transport as resident training,
    # preserving the ordinary autoregressive thin action contract.
    bits = int(action.action_type) << 29
    thin = env._native_action_to_thin(bits)
    assert thin is not None and env.action_masks()[thin]
    env._nosl_cpp_pending = (thin, bits)
    _,reward,done,truncated,info=env.step(thin)
    assert not truncated and info['finite_curve_exit']==outcome.curve_state
    state=info['finite_curve_exit'];assert state['outlet']==outlet
    assert state['potions']==(1 if outlet=='escape' else 0 if outlet=='loss' else 2)
    bank=FiniteCurveBank(finite_curve_bank)
    assert bank.value(state)==bank.value(outcome.curve_state)
    assert reward==outcome.reward
    if outlet=='escape':
        loot=env.thick.gc.get_reward_info()
        assert loot['card_count']==0
        assert not loot['gold'] and not loot['relics'] and not loot['potions']
    if outlet=='loss':assert done and outcome.target()==0
    env.close()

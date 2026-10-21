"""COMBAT01: genuine Smoke Bomb/theft actions and complete branch settlement."""
import pytest
import slaythespire as sts

from env.core.outer_return import CombatExit, strategic_reward
from schemas.combat_state import normalize_sim_global


def _end_turn(battle):
    battle.set_player_block(999)
    battle.execute(sts.Action(sts.ActionType.END_TURN))


@pytest.mark.timeout(30)
@pytest.mark.parametrize('steal_first', [False, True])
def test_smoke_has_no_loot_and_retains_stolen_gold_loss(run_combat_factory, steal_first):
    root = run_combat_factory(potions=['Smoke Bomb'])
    before = root.bc.to_dict()
    leaf = root.bc.clone()
    if steal_first:
        _end_turn(leaf)
        assert leaf.to_dict()['player']['gold'] < 99
    leaf.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    assert leaf.is_battle_over
    actual = root.fork_run()
    actual.bc = leaf.clone()
    actual._settle()
    branch, outcome = root.settle_combat_leaf(leaf)
    assert outcome == CombatExit(0., 'upper')
    assert outcome.target(1.4) == pytest.approx(1.4)  # No HP / [0,1] substitute.
    assert branch.gc.get_reward_info() == actual.gc.get_reward_info()
    rewards = branch.gc.get_reward_info()
    assert rewards['gold'] == rewards['relics'] == rewards['potions'] == []
    assert rewards['card_count'] == 0
    assert branch.gc.gold == leaf.to_dict()['player']['gold']
    assert normalize_sim_global(branch.gc)['gold'] == branch.gc.gold
    assert normalize_sim_global(branch.gc)['potions'] == ['PAD'] * branch.gc.potion_capacity
    assert root.bc.to_dict() == before and root.gc.gold == 99


@pytest.mark.timeout(30)
@pytest.mark.parametrize('kill_thief', [False, True])
def test_stolen_money_is_not_credited_before_claim(run_combat_factory, kill_thief):
    root = run_combat_factory()
    before = root.bc.to_dict()
    leaf = root.bc.clone()
    _end_turn(leaf)
    stolen = 99 - leaf.to_dict()['player']['gold']
    assert stolen > 0
    if kill_thief:
        leaf.set_player_strength(999)
        leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    else:
        for _ in range(10):
            if leaf.is_battle_over:
                break
            _end_turn(leaf)
    assert leaf.is_battle_over
    branch, outcome = root.settle_combat_leaf(leaf)
    assert outcome == CombatExit(0., 'upper')
    wallet = leaf.to_dict()['player']['gold']
    assert branch.gc.gold == wallet < 99
    assert normalize_sim_global(branch.gc)['gold'] == wallet
    rewards = branch.gc.get_reward_info()['gold']
    if kill_thief:
        assert len(rewards) == 2 and rewards[0] == stolen
        branch.gc.claim_gold(0)
        assert branch.gc.gold == 99
    else:
        assert len(rewards) == 1  # Only the ordinary room's gold offer remains.
    assert root.bc.to_dict() == before and root.gc.gold == 99


def test_outer_return_terminal_and_censor_contract():
    assert strategic_reward(3, 50, {'floor': 51, 'act': 3,
        'outcome': 'PLAYER_VICTORY'}, done=True, censored=False) == 1.002
    assert strategic_reward(1, 4, {'floor': 4, 'act': 1,
        'outcome': 'PLAYER_LOSS'}, done=True, censored=False) == 0
    assert strategic_reward(1, 16, {'floor': 17, 'act': 2,
        'outcome': 'UNDECIDED'}, done=False, censored=False) == pytest.approx(.102)
    assert CombatExit(0., 'terminal').target() == 0
    assert CombatExit(0., 'combat').target(-.2) == -.2
    with pytest.raises(ValueError, match='bootstrap'):
        CombatExit(0., 'terminal').target(1)
    with pytest.raises(ValueError, match='censored'):
        CombatExit(0., 'censored').target()
    with pytest.raises(ValueError, match='finite'):
        CombatExit(0., 'upper').target(float('nan'))

"""Unbounded native values and explicit victory/escape/death handoff."""
import numpy as np
import pytest

from env.core import native_actions as C


@pytest.mark.timeout(30)
@pytest.mark.parametrize('kind', ['victory', 'escape', 'loss'])
@pytest.mark.parametrize('value', [-.4, 2.7])
def test_all_exits_request_complete_settlement(run_combat_factory, kind, value):
    root = run_combat_factory(potions=['Smoke Bomb'] if kind == 'escape' else [])
    battle = root.bc
    if kind == 'victory':
        battle.set_player_strength(999)
        action = C.AUTOREGRESSIVE_PLAY  # Strike slot 0, monster slot 0.
    elif kind == 'escape':
        action = C.AUTOREGRESSIVE_POTION + 5  # No target.
    else:
        battle.apply_custom_state({'player': {'hp': 1}})
        action = 0
    original = battle.to_dict()
    logits = np.full(198, -1000., dtype=np.float64)
    logits[action] = 1000.
    session = battle.nosl_neural_search_begin(logits, simulations=1, depth=1,
        nonce=42101, root_value=value, fixed_worldline=True, outer_return=True)
    request = session.next_batch_packed(1)
    assert request['outer_return_schema'] == 1
    assert list(request['combat_exits']) == [0]
    assert request['needs_policy'] == [False]
    assert request['legal_action_counts'].tolist() == [0]
    leaf = request['combat_exits'][0]
    branch, exit = root.settle_combat_leaf(leaf)
    assert exit.kind == ('terminal' if kind == 'loss' else 'upper')
    if kind == 'escape':
        assert branch.gc.get_reward_info()['card_count'] == 0
    rid = request['request_ids'][0]
    with pytest.raises(ValueError, match='finite'):
        session.backprop(rid, float('nan'), [])
    session.backprop(rid, value, [])
    assert session.result()['root_value'] == pytest.approx(value)
    assert battle.to_dict() == original


def test_native_rejects_mixed_objectives(run_combat_factory):
    battle = run_combat_factory().bc
    with pytest.raises(ValueError, match='fixed worldline'):
        battle.nosl_neural_search_begin(np.zeros(198), outer_return=True)
    with pytest.raises(ValueError, match='HP/Boss'):
        battle.nosl_neural_search_begin(np.zeros(198), outer_return=True,
            fixed_worldline=True, boss_win_weight=.9)


@pytest.mark.timeout(30)
@pytest.mark.parametrize('kind', ['victory', 'escape', 'loss'])
@pytest.mark.parametrize('value', [-.4, 2.7])
def test_fixed_terminal_is_evaluated_once_without_losing_visits(run_combat_factory, kind, value):
    battle = run_combat_factory(potions=['Smoke Bomb'] if kind == 'escape' else []).bc
    if kind == 'victory':
        battle.set_player_strength(999)
        action = C.AUTOREGRESSIVE_PLAY
    elif kind == 'escape':
        action = C.AUTOREGRESSIVE_POTION + 5
    else:
        battle.apply_custom_state({'player': {'hp': 1}})
        action = 0
    logits = np.full(198, -1000., dtype=np.float64)
    logits[action] = 1000.
    session = battle.nosl_neural_search_begin(logits, simulations=128, depth=0,
        c_puct=1000., nonce=42101, root_value=value, fixed_worldline=True, outer_return=True)
    request = session.next_batch_packed(1)
    assert list(request['combat_exits']) == [0]
    session.backprop(request['request_ids'][0], value, [])
    # Remaining simulations must still contribute the same unbounded value;
    # they must not request the same immutable terminal 127 more times.
    assert session.next_batch_packed(1)['done']
    result = session.result()
    assert result['root_value'] == pytest.approx(value)
    assert result['total_root_visits'] == 128
    assert sum(visits for _, visits in result['root_visits']) == 128

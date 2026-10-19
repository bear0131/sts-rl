"""F03: whole-run selection handles retain full native pile indices."""
import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A, game_actions as ga, vocab as V

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]

CASES = [
    ('SECRET_TECHNIQUE', 'draw', 'STRIKE_RED', ['DEFEND_RED', 'SHRUG_IT_OFF'], 'hand'),
    ('SECRET_WEAPON', 'draw', 'DEFEND_RED', ['STRIKE_RED', 'BASH'], 'hand'),
    ('HEADBUTT', 'discard', 'DEFEND_RED', ['STRIKE_RED', 'BASH'], 'draw'),
    ('EXHUME', 'exhaust', 'EXHUME', ['STRIKE_RED', 'BASH'], 'hand'),
]


@pytest.fixture
def selection_factory(combat_factory):
    def make(case, padding, *, contract='legacy'):
        source, zone, filler, choices, destination = case
        env = combat_factory([source] * 5, 'Cultist', combat_action_contract=contract)
        state = dict(hand=[source], draw=[], discard=[], exhaust=[], player={'energy': 3})
        state[zone] = [filler] * padding + choices
        env.thick.bc.apply_custom_state(state)
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        env._invalidate_backend_cache()
        return env
    return make


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('padding', [10, 94])
@pytest.mark.parametrize('path', ['step', 'search', 'fast_search'])
def test_high_selection_index_executes_same_native_choice(selection_factory, case, padding, path):
    env = selection_factory(case, padding)
    source = env
    if path != 'step':
        env = env.fork_combat()
    battle = env.thick.bc
    before = battle.to_dict()
    native = [int(a.source_idx) for a in battle.legal_actions()]
    expected_mask = np.zeros(A.N_ACTIONS, dtype=bool)
    expected_mask[[A._PICK + index for index in native]] = True
    np.testing.assert_array_equal(env.action_masks(), expected_mask)
    assert env._obs()['arg_step'][0] == V.ARG_STEP_VOCAB_INDEX[A.PICK_CARD]
    index = padding + 1
    control = battle.clone()
    control.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, index))
    if path == 'step':
        _, _, terminal, truncated, _ = env.step(A._PICK + index)
        assert not terminal and not truncated
    else:
        terminal, _ = env.search_step(A._PICK + index, validate=path != 'fast_search')
        assert not terminal
    assert battle.to_dict() == control.to_dict()
    assert battle.to_dict() != before
    assert env.thick.select_descriptor() is None
    assert any(c[0] == case[3][1] for c in battle.to_dict()[case[4]])
    if path != 'step':
        assert source.thick.bc.to_dict() == before
        env.close()


@pytest.mark.parametrize('case', CASES[:2])
def test_native_search_keeps_high_index_visit_policy_and_can_commit(selection_factory, case):
    env = selection_factory(case, 10)
    result = env.nosl_is_az_cpp_decide(state_nonce=42, simulations=8, depth=2)
    assert result['native']
    assert result['action'] in (A._PICK + 10, A._PICK + 11)
    assert set(np.flatnonzero(result['visit_policy'])) <= {A._PICK + 10, A._PICK + 11}
    assert np.isclose(result['visit_policy'].sum(), 1)
    _, _, terminal, truncated, _ = env.step(result['action'])
    assert not terminal and not truncated
    assert env.thick.select_descriptor() is None


def test_reasoning_trace_can_execute_full_index_selection(selection_factory):
    env = selection_factory(CASES[0], 10)
    handled, trace = env.thick._apply_with_reasoning_trace(ga.select(11))
    assert handled and trace['precision'] == 'atomic'
    assert trace['semantic_action'] == ga.select(11)
    assert env.thick.bc.to_dict()['hand'][0][0] == 'SHRUG_IT_OFF'


def test_over_capacity_policy_choice_is_explicit_and_native_actions_still_work(selection_factory):
    env = selection_factory(CASES[0], 96, contract='native-v1')
    bits = env.native_combat_actions()
    assert [b & 0xffff for b in bits] == [96, 97]
    assert not env.action_masks().any()  # No spill into the adjacent map segment.
    legacy = env.fork_combat()
    try:
        legacy.combat_action_contract = 'legacy'
        legacy._invalidate_backend_cache()
        with pytest.raises(ValueError, match='96-slot policy contract'):
            legacy.action_masks()
    finally:
        legacy.close()
    _, _, terminal, truncated, _ = env.step_native_combat(bits[1])
    assert not terminal and not truncated
    assert env.thick.bc.to_dict()['hand'][0][0] == 'SHRUG_IT_OFF'

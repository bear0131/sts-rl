"""Thin adapters may omit redundant backend legal lists, never transition labels."""
from functools import partial

import numpy as np
import pytest

from env.core import action_space as A
from env.core.game_env import SimGameEnv

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_backend_legal_omission_is_explicit_and_reset_contract_stays_complete():
    regular = SimGameEnv()
    lean = SimGameEnv(emit_step_legal=False)
    np.testing.assert_equal(regular.reset(seed=123), lean.reset(seed=123))
    for _ in range(64):
        legal = regular.legal_semantic_actions()
        assert legal
        expected = regular.step(legal[0])
        actual = lean.step(legal[0])
        assert 'legal' in expected[3] and 'legal' not in actual[3]
        np.testing.assert_equal(expected[:3], actual[:3])
        np.testing.assert_equal({k: v for k, v in expected[3].items() if k != 'legal'}, actual[3])
        if regular.bc is not None:
            break
    assert regular.bc is not None


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('branch', [False, True])
def test_thin_metadata_omission_keeps_boundary_terminal_and_parent_state(
        short_combat_env_type, native, branch):
    contract = 'native-v1' if native else 'legacy'
    expected_parent = short_combat_env_type(global_map_view=True,
        backend_factory=partial(SimGameEnv, global_map_view=True), combat_action_contract=contract)
    actual_parent = short_combat_env_type(global_map_view=True, combat_action_contract=contract)
    expected = actual = None
    try:
        np.testing.assert_equal(expected_parent.reset(seed=123), actual_parent.reset(seed=123))
        expected = expected_parent.fork_combat() if branch else expected_parent
        actual = actual_parent.fork_combat() if branch else actual_parent
        parent_state = actual_parent.thick.bc.to_dict()
        saw_boundary = saw_terminal = False
        for index in range(64):
            if native:
                legal = expected.native_combat_actions()
                choices = [int(a) for a in legal if int(a) != (4 << 29)]
                action = (4 << 29) if index == 0 else (choices or legal)[0]
                first, second = expected.step_native_combat(action), actual.step_native_combat(action)
            else:
                expected._nosl_value_capture_pending = actual._nosl_value_capture_pending = True
                legal = np.flatnonzero(expected.action_masks())
                choices = [int(a) for a in legal if int(a) != A.END_TURN]
                action = A.END_TURN if index == 0 else int((choices or legal)[0])
                first, second = expected.step(action), actual.step(action)
            np.testing.assert_equal(first, second)
            saw_boundary |= 'nosl_value_boundary_observation' in first[4]
            saw_terminal |= 'nosl_value_terminal_target' in first[4]
            if saw_terminal:
                assert 'nosl_value_terminal_outcome' in first[4]
                break
        assert saw_terminal
        if native:
            assert saw_boundary
        if branch:
            np.testing.assert_equal(actual_parent.thick.bc.to_dict(), parent_state)
    finally:
        if branch:
            if expected is not None:
                expected.close()
            if actual is not None:
                actual.close()
        expected_parent.close()
        actual_parent.close()

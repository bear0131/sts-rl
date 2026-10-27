"""Finishing a search preserves pending actions and the diagnostic tree."""
import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('contract', ['legacy', 'joint-v2', 'autoregressive-v2'])
@pytest.mark.parametrize('through_batch', [False, True])
def test_completed_search_needs_no_empty_leaf_packet(combat_factory, contract, through_batch):
    env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5,
                         combat_action_contract=contract)
    pool = NativeSearchBatch([env], 2, direct_observations=True)
    try:
        pool.begin({0: ((np.zeros(env.action_space.n),), dict(
            simulations=8, depth=0, state_nonce=42, fixed_worldline=True,
            end_turn_as_leaf=True, compact_observation=True, leaf_batch_size=1))})
        session = env._nosl_cpp_az_session
        with pytest.raises(RuntimeError, match='not complete'):
            env._finish_native_neural_session()
        assert env._nosl_cpp_az_session is session
        while not pool.done:
            pool.advance(np.zeros((pool.rows, env.action_space.n), np.float32),
                         np.full(pool.rows, .5, np.float32))

        class Completed:
            def __getattr__(self, name):
                if name.startswith('next_batch'):
                    raise AssertionError('a completed search must not request another leaf packet')
                return getattr(session, name)

        completed = Completed()
        env._nosl_cpp_az_session = completed
        result = (pool.results()[0] if through_batch else env.nosl_is_az_cpp_leaf_batch(1)['result'])
        assert result['simulations'] == 8
        assert env._nosl_cpp_az_session is None
        assert env._nosl_cpp_last_session is completed
        assert env._nosl_cpp_pending == (result['action'], result['native_action_bits'])
        assert env._nosl_value_capture_pending
        assert env.nosl_is_az_cpp_tree_snapshot()['valid']
        before = env.thick.bc.to_dict()
        env.step(result['action'])
        assert env._nosl_cpp_pending is None
        assert env.thick.bc.to_dict() != before
    finally:
        pool.close()


def test_result_mapping_preserves_duplicate_accumulation(combat_factory):
    from env.core import native_actions as C
    env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5,
                         combat_action_contract='autoregressive-v2')
    actions = env._joint_policy_actions()
    end = 4 << 29
    card = next(bits for _, bits in actions if bits >> 29 == 0)
    card_id = env._native_action_to_thin(card)
    result = env._adapt_native_neural_result(dict(
        action_bits=end, root_visits=[(end, 2), (card, 5), (end, 3)],
        root_policy=[(end, .25), (card, .5), (end, .25)], simulations=10))
    assert result['root_visits'][C.AUTOREGRESSIVE_END] == 5
    assert result['root_visits'][card_id] == 5
    assert result['visit_policy'][C.AUTOREGRESSIVE_END] == .5
    assert result['visit_policy'][card_id] == .5
    assert result['selected_target'] == -1
    assert not result['selected_action_targeted']

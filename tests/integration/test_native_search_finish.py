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


@pytest.mark.parametrize('threads', [1, 4])
def test_retired_tree_release_preserves_external_owners_and_pending_batch(combat_factory, threads):
    envs = [combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5,
                           seed=42+i, combat_action_contract='autoregressive-v2')
            for i in range(2)]
    pool = NativeSearchBatch(envs, threads, direct_observations=True)
    retained = []

    def calls(nonce):
        return {i: ((np.zeros(env.action_space.n),), dict(
            simulations=16, depth=0, state_nonce=nonce+i, fixed_worldline=True,
            end_turn_as_leaf=True, compact_observation=True, leaf_batch_size=1))
                for i, env in enumerate(envs)}

    def finish():
        while not pool.done:
            pool.advance(np.zeros((pool.rows, envs[0].action_space.n), np.float32),
                         np.full(pool.rows, .5, np.float32))
        return pool.results()

    try:
        # Explicitly require the rebuilt extension; missing native support is
        # not a passed lifecycle test.
        assert hasattr(pool.pool, 'release_retired')
        for generation in range(4):
            pool.begin(calls(100+generation*10))
            before = {k: v.copy() for k, v in pool.observations().items()}
            counts = pool.pool.work_counts()
            pool.pool.release_retired()
            pool.pool.release_retired()
            assert pool.pool.work_counts() == counts
            for key, value in pool.observations().items():
                assert value.tobytes() == before[key].tobytes()
            results = finish()
            assert all(result['simulations'] == 16 for result in results.values())
            for env in envs:
                session = env._nosl_cpp_last_session
                retained.append((session, session.result()))
                assert env.nosl_is_az_cpp_tree_snapshot()['valid']
            for session, expected in retained:
                assert session.result() == expected
        pool.begin(calls(200))
        pool.clear()
        pool.pool.release_retired()
        assert pool.pool.work_counts()['completed_simulations'] == 0
        pool.begin(calls(210))
        finish()
        for session, expected in retained:
            assert session.done() and session.result() == expected
    finally:
        pool.close()

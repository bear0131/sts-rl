"""Search work units count executed game actions separately from choice prefixes."""
import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch
from env.core import native_actions as C

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('threads', [1, 4])
def test_end_turn_work_counts_include_every_simulation_and_reset(combat_factory, threads):
    envs = [combat_factory(['DEFEND_RED'] * 5, 'Jaw Worm', seed=42+i,
                           combat_action_contract='autoregressive-v2') for i in range(2)]
    for env in envs:
        env.thick.bc.apply_custom_state({'hand': [], 'player': {'hp': 100, 'max_hp': 100}})
        env._invalidate_backend_cache()
    pool = NativeSearchBatch(envs, threads, direct_observations=True)
    empty = dict(simulator_transitions=0, selection_prefix_decisions=0, completed_simulations=0)
    calls = {i: ((np.zeros(C.AUTOREGRESSIVE_SIZE),), dict(simulations=16,
        depth=0, state_nonce=42+i, compact_observation=True, leaf_batch_size=1,
        fixed_worldline=True, end_turn_as_leaf=True)) for i in range(2)}
    try:
        assert pool.pool.work_counts() == empty
        for _ in range(2):
            pool.begin(calls)
            while not pool.done:
                pool.advance(np.zeros((pool.rows, C.AUTOREGRESSIVE_SIZE), np.float32),
                             np.full(pool.rows, .5, np.float32))
            assert pool.pool.work_counts() == dict(simulator_transitions=32,
                selection_prefix_decisions=0, completed_simulations=32)
            assert all(result['simulations'] == 16 for result in pool.results().values())
            assert pool.pool.work_counts()['simulator_transitions'] == 32
        pool.clear()
        assert pool.pool.work_counts() == empty
    finally:
        pool.close()


def test_selection_prefix_is_not_a_simulator_transition(combat_factory):
    import slaythespire as sts
    env = combat_factory(['PURITY'] * 5, combat_action_contract='autoregressive-v2')
    env.thick.bc.apply_custom_state({'hand': ['PURITY'] + ['STRIKE_RED'] * 5})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    original = env.thick.bc.to_dict()
    logits = np.full(C.AUTOREGRESSIVE_SIZE, -30., np.float32)
    logits[C.AUTOREGRESSIVE_MULTI] = 30.
    pool = NativeSearchBatch([env], 1, direct_observations=True)
    try:
        pool.begin({0: ((logits,), dict(simulations=1, state_nonce=42, depth=1,
            compact_observation=True, leaf_batch_size=1, root_search_mode=0,
            fixed_worldline=True))})
        assert pool.observations()['select_hand_mask'].sum() == 1
        assert pool.pool.work_counts() == dict(simulator_transitions=0,
            selection_prefix_decisions=1, completed_simulations=0)
        pool.advance(logits[None], np.array([.5], np.float32))
        assert pool.done
        assert pool.pool.work_counts() == dict(simulator_transitions=0,
            selection_prefix_decisions=1, completed_simulations=1)
        assert env.thick.bc.to_dict() == original
    finally:
        pool.close()

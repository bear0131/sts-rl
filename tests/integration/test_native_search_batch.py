"""Thread scheduling must preserve every tree's leaf and completed decision."""
from dataclasses import asdict

import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('contract,language', [
    ('legacy', False), ('joint-v2', True), ('autoregressive-v2', True)])
def test_native_batch_preserves_each_leaf_and_result(combat_factory, threads, contract, language):
    deck = ['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 4 + ['BASH', 'BATTLE_TRANCE', 'TRUE_GRIT']
    def environments():
        return [combat_factory(deck, seed=42+i, relics=(['Burning Blood'] if i % 2 else []),
                    combat_action_contract=contract, card_language=language) for i in range(4)]
    native, serial = environments(), environments()
    pool = NativeSearchBatch(native, threads)
    config = NativeSearchConfig(simulations=128, fixed_worldline=True, end_turn_as_leaf=True)
    calls = {i: ((np.linspace(-.3, .7, e.action_space.n),), dict(
        **asdict(config), root_value=.5, state_nonce=1729+i, compact_observation=True))
        for i, e in enumerate(native)}
    try:
        # Two roots on unchanged states exercise reuse and release of old sessions.
        for _ in range(2):
            pool.begin(calls)
            responses = {i: e.nosl_is_az_cpp_begin(*calls[i][0], **calls[i][1])
                         for i, e in enumerate(serial)}
            total = 0
            while not pool.done:
                owners = pool.payload['owners']
                assert owners == [i for i, r in responses.items() if not r['done']]
                obs = pool.observations()
                for row, i in enumerate(owners):
                    reference = responses[i]['observation_batch']
                    assert obs.keys() == reference.keys()
                    for key in obs:
                        np.testing.assert_array_equal(obs[key][row], reference[key][0], err_msg=f'{i}: {key}')
                    assert bool(pool.payload['needs_policy'][row]) == responses[i]['leaf_meta'][0]['needs_policy']
                values = np.asarray(.2 + .6 * obs['player_scalars'][:, 0], dtype=np.float32)
                logits = np.broadcast_to(np.linspace(-.2, .8, native[0].action_space.n, dtype=np.float32),
                                          (len(owners), native[0].action_space.n)).copy()
                for row, i in enumerate(owners):
                    leaf = responses[i]['leaf_meta'][0]
                    policy = logits[row] if leaf['needs_policy'] else np.empty(0)
                    responses[i] = serial[i].nosl_is_az_cpp_backprop_and_leaf_batch(
                        [(leaf['request_id'], float(values[row]), policy)], max_leaves=1)
                total += len(owners)
                pool.advance(logits, values)
            assert total > 0
            assert sum(pool.leaves) == total
            actual = pool.results()
            for i in calls:
                assert responses[i]['done']
                for key in ('action', 'native_action_bits', 'root_value', 'simulations',
                            'visit_policy', 'root_visit_policy', 'root_visits'):
                    np.testing.assert_array_equal(actual[i][key], responses[i]['result'][key])
    finally:
        pool.close()


def test_native_batch_invalid_reply_can_abort_and_reuse(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5)
    pool = NativeSearchBatch([env], 2)
    calls = {0: ((np.zeros(env.action_space.n),), dict(simulations=8,
               compact_observation=True, leaf_batch_size=1, state_nonce=1729))}
    try:
        pool.begin(calls)
        with pytest.raises(ValueError, match='rows must match'):
            pool.advance(np.zeros((0, env.action_space.n), np.float32), np.zeros(0, np.float32))
        pool.clear()
        pool.begin(calls)
        while not pool.done:
            n = len(pool.payload['owners'])
            pool.advance(np.zeros((n, env.action_space.n), np.float32), np.full(n, .5, np.float32))
        assert pool.results()[0]['simulations'] == 8
    finally:
        pool.close()

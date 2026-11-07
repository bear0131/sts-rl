"""Actual native world dispatch preserves searches, labels and slot ownership."""
import time

import numpy as np
import pytest
import torch

import env.core
import slaythespire as sts
from env.search.small_playout import PAIRED_CONTRACT, PairedWorldPlayoutStream, SmallPlayoutStream
from rl.policies.small_playout import export_online_search_pair
from rl.policies.token_regret import OnlineTokenRegretPolicy, improved_log_target
from rl.search.paired_token_rollout import PairedTokenBatch


pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def same_search(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == 'stats':
            assert actual[key].keys() == expected[key].keys()
            for name in actual[key]:
                if name != 'search_seconds':
                    np.testing.assert_array_equal(actual[key][name], expected[key][name])
        else:
            np.testing.assert_array_equal(actual[key], expected[key])


def make_batch(envs, policy, pool):
    pair = export_online_search_pair(policy, collect_samples=True, guide_only=True)
    pair['search_contract'] = PAIRED_CONTRACT
    pair.update(common_simulations=17, elite_simulations=17, boss_simulations=31,
                playout_depth=12)
    return PairedTokenBatch(envs, pool, pair, policy)


def drain(stream):
    out, deadline = {}, time.monotonic() + 30
    while stream.pending:
        assert time.monotonic() < deadline
        ready = stream.receive(timeout=.05)
        assert not out.keys() & ready.keys()
        out.update(ready)
    return out


@pytest.mark.parametrize('threads', [1, 3, 24])
def test_world_partition_matches_serial_for_partial_and_forced_worlds(combat_factory, threads):
    torch.manual_seed(716)
    policy = OnlineTokenRegretPolicy(sts.small_playout_key_count()).eval()
    net = sts.SmallPlayoutNet(policy.export_guide())
    roots = [combat_factory(cards, seed=seed).thick.bc for cards, seed in [
        (['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3, 190),
        (['WOUND'] * 10, 191),
        (['BATTLE_TRANCE', 'POMMEL_STRIKE', 'DEFEND_RED'] * 3, 192)]]
    pool = sts.NeuralSearchBatch(threads)
    try:
        for budget in [1, 7, 17, 250, 1000]:
            configs = [sts.SmallPlayoutConfig() for _ in roots]
            for cfg in configs:
                cfg.paired_worldlines, cfg.simulations, cfg.playout_depth = True, budget, 12
            nonces = [31, 0, 2**64-1]
            expected = [net.search(r, c, n) for r, c, n in zip(roots, configs, nonces)]
            actual = sts.small_playout_world_batch(pool, net, roots, configs, nonces)
            for a, b in zip(actual, expected, strict=True):
                same_search(a, b)
    finally:
        pool.clear()


def test_stream_preserves_real_root_choices_and_learner_labels(combat_factory):
    torch.manual_seed(91)
    policy = OnlineTokenRegretPolicy(sts.small_playout_key_count()).eval()
    outputs = []
    pool = sts.NeuralSearchBatch(1)
    try:
        for stream_type in [SmallPlayoutStream, PairedWorldPlayoutStream]:
            envs = [combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3,
                                  seed=190+i, combat_action_contract='autoregressive-v2')
                    for i in range(6)]
            batch = make_batch(envs, policy, pool)
            options = dict(batch_limit=2) if stream_type is PairedWorldPlayoutStream else {}
            with stream_type(batch, threads=3, **options) as stream:
                stream.submit({i: ((), {'state_nonce': 90+i}) for i in range(3)})
                with pytest.raises(RuntimeError, match='outstanding'):
                    stream.submit({0: ((), {'state_nonce': 7})})
                stream.submit({i: ((), {'state_nonce': 90+i}) for i in range(3, 6)})
                assert set(drain(stream)) == set(range(6))
            assert not stream.pending
            with pytest.raises(RuntimeError, match='closed'):
                stream.receive()
            if stream_type is PairedWorldPlayoutStream:
                assert batch.stats['world_dispatch_roots'] == 6
                assert 1 <= batch.stats['world_dispatch_max_roots'] <= 2
            outputs.append({row['env']: row for row in batch.samples})
        assert outputs[0].keys() == outputs[1].keys() == set(range(6))
        for slot in outputs[0]:
            a, b = outputs[0][slot], outputs[1][slot]
            for key in ['action_bits', 'visits', 'values', 'recommended_action', 'state', 'keys']:
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_allclose(np.exp(improved_log_target(a)),
                                       np.exp(improved_log_target(b)), atol=2e-5, rtol=0)
    finally:
        pool.clear()


def test_dome_bypass_and_native_failure_close_without_pending_work(combat_factory):
    torch.manual_seed(92)
    policy = OnlineTokenRegretPolicy(sts.small_playout_key_count()).eval()
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3,
                         relics=['Runic Dome'], combat_action_contract='autoregressive-v2')
    pool = sts.NeuralSearchBatch(1)
    try:
        batch = make_batch([env], policy, pool)
        with PairedWorldPlayoutStream(batch, threads=2) as stream:
            stream.submit({0: ((), {'state_nonce': 3})})
            assert set(drain(stream)) == {0}
        assert batch.stats['dome_policy_decisions'] == 1
        assert batch.samples == []
        other = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3,
                               combat_action_contract='autoregressive-v2')
        batch = make_batch([other], policy, pool)
        batch.pair['common_simulations'] = 0
        stream = PairedWorldPlayoutStream(batch, threads=2)
        stream.submit({0: ((), {'state_nonce': 3})})
        with pytest.raises(ValueError, match='unsupported'):
            drain(stream)
        assert not stream.pending
        stream.close()
        with pytest.raises(RuntimeError, match='closed'):
            stream.submit({0: ((), {'state_nonce': 3})})
    finally:
        pool.clear()

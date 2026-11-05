"""Frozen evaluation uses the real production worker's search and thin actions."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import env.core
import slaythespire as sts

from env.core import native_actions as C
from env.search.small_playout import SEARCH_CONTRACT, SmallPlayoutBatch
from rl.evaluation.strategies import DecisionRequest, OnlinePlayoutEvaluationStrategy
from rl.policies.small_playout import SmallPlayoutPolicy, export_online_search_pair
from rl.training.small_playout import configure_collection

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


def request(env, seed=42, step=0):
    return DecisionRequest(env, {}, {'phase': 'combat'}, seed, step)


def test_frozen_export_matches_actual_training_configuration():
    policy = SmallPlayoutPolicy(sts.small_playout_key_count())
    model = SimpleNamespace(policy=SimpleNamespace(small_playout=policy),
                            small_playout_search_contract=SEARCH_CONTRACT,
                            nosl_is_az_replay_enabled=False, _frozen_upper=False)
    received = []
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    configure_collection(model, SimpleNamespace(configure_small_playout=received.append))
    frozen = export_online_search_pair(policy, collect_samples=False)
    train = received[0]
    np.testing.assert_array_equal(frozen['policy'], train['policy'])
    assert {k: v for k, v in frozen.items() if k not in ('online', 'policy')} == {
        k: v for k, v in train.items() if k not in ('online', 'policy')}
    assert train['online'] is True and frozen['online'] is False
    assert frozen['sha256'] == model.small_playout_collection_sha256
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize('paired', [False, True])
def test_batched_tiers_match_collection_actions_and_budget(combat_factory, monkeypatch, paired):
    policy = SmallPlayoutPolicy(sts.small_playout_key_count())
    strategy = OnlinePlayoutEvaluationStrategy(policy, threads=2, paired_worldlines=paired)
    pairs = []
    for room in (sts.Room.MONSTER, sts.Room.ELITE, sts.Room.BOSS):
        pair = [combat_factory(['STRIKE_RED', 'DEFEND_RED']*5,
                               combat_action_contract='autoregressive-v2') for _ in range(2)]
        for env in pair:
            env.thick.gc.cur_room = room
        pairs.append(pair)
    actual_envs, expected_envs = [list(group) for group in zip(*pairs)]
    pool = sts.NeuralSearchBatch(2)
    worker = SmallPlayoutBatch(expected_envs, pool, export_online_search_pair(policy, collect_samples=False, paired_worldlines=paired))
    seen = []
    native_search = sts.small_playout_search_batch
    def record(pool, net, roots, configs, nonces, prior):
        assert all(c.paired_worldlines == paired for c in configs)
        seen.append([(c.simulations, c.boss, c.use_prior, c.playout_depth, c.tree_depth) for c in configs])
        return native_search(pool, net, roots, configs, nonces, prior)
    monkeypatch.setattr(sts, 'small_playout_search_batch', record)
    try:
        strategy.bind_environments(actual_envs)
        # Reverse/subset request order must not change environment identity.
        indices = [2, 0, 1]
        decisions = strategy.decide([request(actual_envs[i]) for i in indices])
        expected = worker.decisions({i: ((), {'state_nonce': 42*1000000007}) for i in indices})
        assert [d.action for d in decisions] == [expected[i]['action'] for i in indices]
        assert all(d.native_action is None for d in decisions)
        assert seen == [[(1000, True, False, 200, 1024), (250, False, False, 200, 1024),
                         (250, False, False, 200, 1024)]]*2
        for i, decision in zip(indices, decisions):
            left = actual_envs[i].step(decision.action)
            right = expected_envs[i].step(expected[i]['action'])
            assert left[1:4] == right[1:4]
            assert actual_envs[i].thick.bc.to_dict() == expected_envs[i].thick.bc.to_dict()
        assert not strategy._batch.samples
        with pytest.raises(ValueError, match='duplicate'):
            strategy.decide([request(actual_envs[0])]*2)
    finally:
        strategy.unbind_environments()
        pool.clear()
    assert strategy.manifest()['search_statistics']['boss/decisions'] == 1
    assert strategy._pool is None and strategy._batch is None
    strategy.bind_environments(actual_envs)
    strategy.unbind_environments()
    strategy.unbind_environments()


def test_dome_never_calls_hidden_intent_search(combat_factory, monkeypatch):
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED']*5, relics=['Runic Dome'],
                         combat_action_contract='autoregressive-v2')
    strategy = OnlinePlayoutEvaluationStrategy(SmallPlayoutPolicy(sts.small_playout_key_count()))
    monkeypatch.setattr(sts, 'small_playout_search_batch', lambda *args: pytest.fail('Dome searched hidden intent'))
    rng = np.random.get_state()
    try:
        strategy.bind_environments([env])
        result = strategy.decide([request(env)])[0]
        assert env.action_masks()[result.action]
        assert result.diagnostics['small_dome_policy_decisions'] == 1
        assert strategy.manifest()['search_statistics']['dome_policy_decisions'] == 1
        assert not strategy._batch.samples
        current = np.random.get_state()
        assert rng[0] == current[0] and rng[2:] == current[2:]
        np.testing.assert_array_equal(rng[1], current[1])
    finally:
        strategy.unbind_environments()


def test_pending_multi_executes_prefix_then_stop_without_new_search(combat_factory, monkeypatch):
    env = combat_factory(['PURITY']*5, combat_action_contract='autoregressive-v2')
    env.thick.bc.apply_custom_state({'hand': ['PURITY']+['STRIKE_RED']*5})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    env._rulebase_pending_multi = 0b101
    original = env.thick.bc.to_dict()
    strategy = OnlinePlayoutEvaluationStrategy(SmallPlayoutPolicy(sts.small_playout_key_count()))
    monkeypatch.setattr(sts, 'small_playout_search_batch', lambda *args: pytest.fail('pending choice searched again'))
    try:
        strategy.bind_environments([env])
        for step, action in enumerate((C.AUTOREGRESSIVE_MULTI, C.AUTOREGRESSIVE_MULTI+2, C.AUTOREGRESSIVE_STOP)):
            decision = strategy.decide([request(env, step=step)])[0]
            assert decision.action == action and decision.native_action is None
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            if step < 2:
                assert env.thick.bc.to_dict() == original
        assert env._rulebase_pending_multi is None and env.thick.bc.to_dict() != original
    finally:
        strategy.unbind_environments()


def test_paired_worlds_compare_the_same_draws_and_keep_the_legal_instances(combat_factory):
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'POMMEL_STRIKE', 'SHRUG_IT_OFF', 'BASH']*2)
    bc = env.thick.bc
    state = bc.to_dict()
    policy = SmallPlayoutPolicy(sts.small_playout_key_count())
    net = sts.SmallPlayoutNet(policy.export())
    bits = sts.small_playout_features(bc)['action_bits']
    cfg = sts.SmallPlayoutConfig()
    count = len(bits)
    first = sts.small_paired_worldlines(bc, net, cfg, bits, count*7, 551)
    reversed_order = sts.small_paired_worldlines(bc, net, cfg, bits[::-1], count*7, 551)
    np.testing.assert_array_equal(first['values'], np.asarray(reversed_order['values'])[:, ::-1])
    assert first['root_transitions'] == count*7 and bc.to_dict() == state
    cfg.paired_worldlines = True
    logits = np.asarray(net.logits(bc))
    for budget in (1, count+1, 250):
        cfg.simulations = budget
        out = net.search(bc, cfg, 551)
        visits = np.asarray(out['visits'])
        values = np.asarray(out['values'])
        eligible = (visits > 0) & (values >= values[visits > 0].max() - .005)
        selected = bits.index(out['action'])
        assert eligible[selected] and logits[selected] == logits[eligible].max()
        assert visits.sum() == budget and visits.max()-visits.min() <= 1
        assert sum(out['terminal_visits']) + out['stats']['depth_cuts'] == budget
        assert out['stats']['tree_transitions'] == 0 and bc.to_dict() == state
    with pytest.raises(ValueError, match='duplicate'):
        sts.small_paired_worldlines(bc, net, cfg, [bits[0], bits[0]], 2, 551)
    with pytest.raises(ValueError, match='frozen'):
        export_online_search_pair(policy, collect_samples=True, paired_worldlines=True)

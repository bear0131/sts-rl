"""Real native small-policy inference/search/learner lifecycle, no mock labels."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import env.core
import slaythespire as sts

from rl.algorithms.small_playout import BattleCollector, BattleFixture, SmallPlayoutLearner
from rl.policies.small_playout import collate_features
from rl.search.small_playout import SmallSearchConfig
from rl.evaluation.strategies import DecisionRequest, SmallPlayoutStrategy

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def action_identity(action):
    return (int(action.action_type), action.source_idx, action.target_idx, tuple(action.selected_indices))


def assert_legal(bc, bits):
    assert action_identity(sts.Action(bits)) in {action_identity(a) for a in bc.legal_actions()}


def test_forced_playout_skips_network_without_reducing_simulations(combat_factory):
    bc = combat_factory(['WOUND'] * 10).thick.bc
    features = sts.small_playout_features(bc)
    assert len(features['action_bits']) == 1
    assert sts.Action(features['action_bits'][0]).action_type == sts.ActionType.END_TURN
    learner = SmallPlayoutLearner(seed=42, config=SmallSearchConfig(simulations=8))
    result = learner.search.search([bc], [42])[0]
    stats = result['stats']
    assert stats['simulations'] == 8
    assert stats['policy_calls'] == stats['action_scores'] == 0
    assert stats['forced_policy_skips'] == stats['playout_transitions'] > 0
    assert stats['terminal_losses'] == 8
    assert result['visits'] == [8] and result['action'] == features['action_bits'][0]


def test_torch_native_logits_and_weight_refresh(combat_factory):
    learner = SmallPlayoutLearner(seed=19)
    env = combat_factory(['POMMEL_STRIKE', 'DEFEND_RED', 'BASH', 'BATTLE_TRANCE', 'SHRUG_IT_OFF'] * 2)
    bc = env.thick.bc
    for step in range(20):
        if bc.is_battle_over:
            break
        features = sts.small_playout_features(bc)
        with torch.inference_mode():
            expected = learner.policy(*collate_features([features])).numpy()[0]
        native = learner.search.net.logits(bc)
        # Different float reduction order; ~1e-7 observed on CPU, bounded at 2e-5.
        np.testing.assert_allclose(native, expected, atol=2e-5, rtol=2e-5)
        for bits, row in zip(features['action_bits'], features['actions']):
            assert_legal(bc, bits)
            if not bc.is_targeted_action(bits):
                np.testing.assert_array_equal(row[16:23], 0)
        bc.execute(sts.Action(features['action_bits'][step % len(native)]))
    before = learner.policy.export()
    assert before.size == 4984 + 33 * sts.small_playout_key_count()
    root = BattleFixture().create(42)
    old = learner.search.net.logits(root)
    with torch.no_grad():
        learner.policy.linear.weight[0, 0] += 1
    assert learner.search.net.logits(root) == old  # native is an immutable snapshot
    learner.search.refresh(learner.policy)
    assert learner.search.net.logits(root) != old


@pytest.mark.parametrize('layer,parameter', [('fc1', 'weight'), ('fc1', 'bias'),
                                           ('fc2', 'weight'), ('fc2', 'bias'),
                                           ('projection', 'weight'), ('linear', 'weight'),
                                           ('keys', 'weight'), ('bias', 'weight')])
@pytest.mark.parametrize('selection', [False, True])
def test_weight_refresh_preserves_existing_snapshot(combat_factory, layer, parameter, selection):
    learner = SmallPlayoutLearner(seed=19)
    root = BattleFixture().create(42)
    if selection:
        root = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED'] * 3).thick.bc
        root.apply_custom_state({'hand': ['PURITY', 'STRIKE_RED', 'DEFEND_RED']})
        root.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    features = sts.small_playout_features(root)
    snapshot = learner.search.net
    old = snapshot.logits(root)
    with torch.no_grad():
        getattr(getattr(learner.policy, layer), parameter).add_(.25)
    assert snapshot.logits(root) == old
    learner.search.refresh(learner.policy)
    assert snapshot.logits(root) == old
    actual = learner.search.net.logits(root)
    assert actual != old
    with torch.inference_mode():
        expected = learner.policy(*collate_features([features])).numpy()[0]
    # Torch uses a different reduction layout; native-to-native equivalence is
    # separately checked with fixed searches and complete collection traces.
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('selection', [False, True])
def test_zero_action_features_ignore_unused_coefficients(combat_factory, selection):
    bc = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED'] * 3).thick.bc
    bc.apply_custom_state({'hand': ['PURITY', 'STRIKE_RED', 'DEFEND_RED']})
    if selection:
        bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    features = sts.small_playout_features(bc)
    inactive = np.all(features['actions'] == 0, axis=0)
    assert inactive.any() and not inactive.all()
    if selection:
        assert set((features['keys'] >= 0).sum(axis=1)) == {1, 2}
    learner = SmallPlayoutLearner(seed=19)
    expected = learner.search.net.logits(bc)
    with torch.no_grad():
        # Coefficients of absent features cannot affect candidate scores, even
        # when their magnitude dwarfs all active features and card embeddings.
        learner.policy.projection.weight[:, inactive] = 1e20
        learner.policy.linear.weight[:, inactive] = -1e20
    learner.search.refresh(learner.policy)
    assert learner.search.net.logits(bc) == expected


def test_selected_card_embeddings_are_averaged_not_summed(combat_factory):
    bc = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED'] * 3).thick.bc
    bc.apply_custom_state({'hand': ['PURITY', 'STRIKE_RED', 'DEFEND_RED']})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    features = sts.small_playout_features(bc)
    learner = SmallPlayoutLearner(seed=19)
    with torch.no_grad():
        for parameter in learner.policy.parameters():
            parameter.zero_()
        learner.policy.fc2.bias[0] = 1
        learner.policy.keys.weight[:, 0] = torch.arange(learner.policy.key_count) / 4
        learner.policy.bias.weight[:, 0] = torch.arange(learner.policy.key_count) / 8
    learner.search.refresh(learner.policy)
    # Binary fractions keep this independent single/two-key oracle exact.
    expected = [float(row[row >= 0].mean() * 3 / 8) for row in features['keys']]
    assert set((features['keys'] >= 0).sum(axis=1)) == {1, 2}
    assert learner.search.net.logits(bc) == expected


def test_all_ten_card_keys_survive_complete_subset_features(combat_factory):
    bc = combat_factory(['STRIKE_RED'] * 5).thick.bc
    bc.apply_custom_state({'hand': ['STRIKE_RED', 'DEFEND_RED'] * 5, 'potions': ['Gamblers Brew']})
    bc.execute(sts.Action(sts.ActionType.POTION, 0))
    features = sts.small_playout_features(bc)
    assert features['keys'].shape == (1024, 10)
    all_selected = features['action_bits'].index((int(sts.ActionType.MULTI_CARD_SELECT) << 29) | 1023)
    assert np.all(features['keys'][all_selected] >= 0)
    assert len(set(features['keys'][all_selected])) == 2
    assert np.array_equal(features['keys'][all_selected][::2], np.repeat(features['keys'][all_selected][0], 5))
    learner = SmallPlayoutLearner(seed=19)
    with torch.no_grad():
        for parameter in learner.policy.parameters():
            parameter.zero_()
        learner.policy.fc2.bias[0] = 1
        learner.policy.keys.weight[:, 0] = torch.arange(learner.policy.key_count) / 4
        learner.policy.bias.weight[:, 0] = torch.arange(learner.policy.key_count) / 8
    learner.search.refresh(learner.policy)
    expected = [float(row[row >= 0].mean() * 3 / 8) for row in features['keys']]
    np.testing.assert_allclose(learner.search.net.logits(bc), expected, rtol=1e-6, atol=1e-6)


def test_zero_state_and_inactive_hidden_channels_ignore_coefficients():
    learner = SmallPlayoutLearner(seed=19)
    root = BattleFixture().create(42)
    inactive = sts.small_playout_features(root)['state'] == 0
    assert inactive.any() and not inactive.all()
    with torch.no_grad():
        learner.policy.fc1.weight[:16].zero_()
        learner.policy.fc1.bias[:16].fill_(-1.)
    learner.search.refresh(learner.policy)
    expected = learner.search.net.logits(root)
    with torch.no_grad():
        learner.policy.fc1.weight[:, inactive] = 1e20
        learner.policy.fc2.weight[:, :16] = -1e20
    learner.search.refresh(learner.policy)
    assert learner.search.net.logits(root) == expected


def test_action_projection_handles_input_eviction_and_dynamic_targets(combat_factory):
    bc = combat_factory(['STRIKE_RED'] * 5).thick.bc
    bc.apply_custom_state({'hand': ['STRIKE_RED']})
    learner = SmallPlayoutLearner(seed=19)
    original = learner.search.net.logits(bc)
    state = bc.card_state()
    # More distinct card feature prefixes than the bounded inference cache.
    # Repeated calls must remain correct both before and after collisions.
    for growth in range(4097):
        state['hand'][0]['special_data'] = growth
        bc.set_card_state(state)
        actual = learner.search.net.logits(bc)
        assert learner.search.net.logits(bc) == actual
        if growth % 257 == 0:
            with torch.inference_mode():
                expected = learner.policy(*collate_features([sts.small_playout_features(bc)])).numpy()[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    state['hand'][0]['special_data'] = 0
    bc.set_card_state(state)
    assert learner.search.net.logits(bc) == original
    for hp in [17, 31, 29, 17]:
        bc.apply_custom_state({'monsters': [{'hp': hp}]})
        with torch.inference_mode():
            expected = learner.policy(*collate_features([sts.small_playout_features(bc)])).numpy()[0]
        np.testing.assert_allclose(learner.search.net.logits(bc), expected, atol=2e-5, rtol=2e-5)


def test_released_native_snapshots_cannot_reuse_old_action_projections():
    import gc
    root = BattleFixture().create(42)
    learner = SmallPlayoutLearner(seed=19)
    features = collate_features([sts.small_playout_features(root)])
    for _ in range(64):
        with torch.no_grad():
            learner.policy.keys.weight.add_(1 / 128)
            learner.policy.linear.weight.add_(1 / 128)
            expected = learner.policy(*features).numpy()[0]
        snapshot = sts.SmallPlayoutNet(learner.policy.export())
        np.testing.assert_allclose(snapshot.logits(root), expected, atol=2e-5, rtol=2e-5)
        del snapshot
        gc.collect()


@pytest.mark.parametrize('nonce', [0, 1, 73, 2**63 + 7])
def test_small_and_general_resampling_share_future_rng_without_hidden_cards(combat_factory, nonce):
    root = combat_factory(['STRIKE_RED', 'DEFEND_RED'] * 5).thick.bc
    root.apply_custom_state({'hand': ['WILD_STRIKE', 'TRUE_GRIT', 'DEFEND_RED'],
                             'draw': [], 'discard': [], 'exhaust': []})
    expected, actual = root.clone(), root.clone()
    expected.nosl_redeterminize(nonce)
    sts.small_playout_redeterminize(actual, nonce)
    for step in range(100):
        assert actual.to_dict() == expected.to_dict()
        assert actual.nosl_future_key() == expected.nosl_future_key()
        if expected.is_battle_over:
            break
        actions = expected.legal_actions()
        action = actions[step % len(actions)]
        expected.execute(action)
        actual.execute(action)


@pytest.mark.parametrize('threads', [1, 4])
def test_parallel_search_callbacks_preserve_results_and_release_failed_batch(threads):
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=16), threads=threads)
    roots = [BattleFixture().create(seed) for seed in (42, 73, 81)]
    configs = [learner.search.config.native()] * len(roots)
    nonces = [7, 9, 11]
    pool, net = learner.search.pool, learner.search.net
    expected = sts.small_playout_search_batch(pool, net, roots, configs, nonces)
    started, completed = [], {}
    def ready(index, action):
        assert index not in completed
        completed[index] = action
        # Roots were copied before callbacks start. Advancing the real game
        # cannot alter a different search's retained snapshot.
        roots[index].execute(sts.Action(action))
    actual = sts.small_playout_search_batch(pool, net, roots, configs, nonces,
        on_ready=ready, on_start=lambda: started.append(True))
    assert started == [True]
    assert completed == {i: result['action'] for i, result in enumerate(expected)}
    for left, right in zip(expected, actual, strict=True):
        for key in left:
            if key == 'stats':
                assert {k:v for k,v in left[key].items() if k != 'search_seconds'} == {
                    k:v for k,v in right[key].items() if k != 'search_seconds'}
            else:
                np.testing.assert_equal(left[key], right[key])
    roots = [BattleFixture().create(seed) for seed in (42, 73, 81)]
    def fail(index, action):
        if index == 0:
            raise ValueError('callback diagnostic failure')
    with pytest.raises(ValueError, match='callback diagnostic failure'):
        sts.small_playout_search_batch(pool, net, roots, configs, nonces, on_ready=fail)
    repeated = sts.small_playout_search_batch(pool, net, roots, configs, nonces)
    assert [r['visits'] for r in repeated] == [r['visits'] for r in expected]
    assert [r['values'] for r in repeated] == [r['values'] for r in expected]


def test_callbacks_are_serial_on_one_consumer_and_failures_release_pool():
    from threading import get_ident
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=16), threads=4)
    roots = [BattleFixture().create(seed) for seed in (42, 73, 81)]
    configs = [learner.search.config.native()] * len(roots)
    calls = []
    def ordinary():
        calls.append(('start', get_ident()))
    def ready(index, action):
        assert calls[0][0] == 'start'
        calls.append((index, get_ident()))
    sts.small_playout_search_batch(learner.search.pool, learner.search.net,
        roots, configs, [7, 9, 11], on_start=ordinary, on_ready=ready)
    assert sorted(c[0] for c in calls[1:]) == [0, 1, 2]
    assert len({c[1] for c in calls}) == 1
    def fail():
        raise ValueError('ordinary callback failed')
    with pytest.raises(ValueError, match='ordinary callback failed'):
        sts.small_playout_search_batch(learner.search.pool, learner.search.net,
            roots, configs, [7, 9, 11], on_start=fail)
    actual = sts.small_playout_search_batch(learner.search.pool, learner.search.net,
        roots, configs, [7, 9, 11], on_start=ordinary, on_ready=ready)
    assert len(actual) == len(roots)
    bad = learner.search.config.native()
    bad.simulations = 0
    with pytest.raises(ValueError, match='budget'):
        sts.small_playout_search_batch(learner.search.pool, learner.search.net,
            roots, [bad] * len(roots), [7, 9, 11], on_start=ordinary, on_ready=ready)
    repeated = sts.small_playout_search_batch(learner.search.pool, learner.search.net,
        roots, configs, [7, 9, 11], on_ready=ready)
    assert [r['visits'] for r in actual] == [r['visits'] for r in repeated]


def test_search_determinism_hidden_state_and_parallel_equivalence():
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=32), threads=2)
    bc = BattleFixture().create(42)
    key = bc.nosl_future_key()
    clone = bc.clone()
    clone.nosl_redeterminize(1717)
    for field in ('state', 'actions', 'keys'):
        np.testing.assert_array_equal(sts.small_playout_features(bc)[field], sts.small_playout_features(clone)[field])
    results = learner.search.search([bc, clone], [81, 81])
    single = learner.search.net.search(bc, learner.search.config.native(), 81)
    for result in [*results, single]:
        assert result['visits'] == results[0]['visits']
        assert result['values'] == results[0]['values']
        assert result['action'] in result['action_bits']
        assert sum(result['visits']) == 32
        stats = result['stats']
        assert sum(stats[k] for k in ('terminal_wins', 'terminal_losses', 'terminal_escapes', 'depth_cuts', 'tree_cuts')) == 32
        assert stats['playout_transitions'] == sum(i*n for i, n in enumerate(stats['playout_lengths']))
    assert bc.nosl_future_key() == key
    # Searching must not consume the live game's future RNG.
    untouched = BattleFixture().create(42)
    for bits in results[0]['action_bits']:
        a, b = bc.clone(), untouched.clone()
        a.execute(sts.Action(bits))
        b.execute(sts.Action(bits))
        assert a.nosl_future_key() == b.nosl_future_key()


@pytest.mark.parametrize('nonce', [0, 91])
def test_playout_trace_keeps_independent_features_across_selection_and_hand_changes(combat_factory, nonce):
    bc = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED'] * 3).thick.bc
    bc.apply_custom_state({'hand': ['PURITY', 'BASH', 'DEFEND_RED']})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    learner = SmallPlayoutLearner(seed=19, threads=1)
    trace = sts.small_playout_collect_batch(
        learner.search.pool, learner.search.net, [bc], learner.search.config.native(), [nonce])[0]
    replay = bc.clone()
    sts.small_playout_redeterminize(replay, nonce)
    sizes = set()
    assert trace['steps'][0]['state'][31] == 1
    for step in trace['steps']:
        expected = sts.small_playout_features(replay)
        for field in ('state', 'actions', 'keys'):
            np.testing.assert_array_equal(step[field], expected[field])
        assert step['action_bits'] == expected['action_bits']
        sizes.add(len(step['action_bits']))
        replay.execute(sts.Action(step['action_bits'][step['choice']]))
    assert len(sizes) > 1
    assert len(trace['steps']) == trace['transitions']
    if trace['censored']:
        assert trace['value'] is None
        with pytest.raises(ValueError, match='unfinished'):
            sts.small_playout_value(replay)
    else:
        assert sts.small_playout_value(replay) == trace['value']


def test_caps_are_not_terminal_and_invalid_budgets_fail():
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=16, playout_depth=1))
    root = BattleFixture().create(42)
    stats = learner.search.search([root], [4])[0]['stats']
    assert stats['depth_cuts'] > 0
    assert stats['terminal_wins'] == 0
    cfg = replace(learner.search.config, tree_depth=1).native()
    result = learner.search.net.search(root, cfg, 4)
    stats = result['stats']
    assert stats['tree_cuts'] == 16
    assert stats['policy_calls'] == 0
    assert sum(result['terminal_visits']) == 0
    assert all(value == 0 for value in result['values'])
    assert result['value_statistic'] == 'terminal-return-lower-bound'
    assert stats['terminal_losses'] == stats['terminal_wins'] == stats['terminal_escapes'] == 0
    cfg.playout_depth = 201
    with pytest.raises(ValueError, match='budget'):
        learner.search.net.search(root, cfg, 4)


def test_independent_stream_joins_native_failure_and_retains_root(combat_factory):
    import threading
    from env.search.small_playout import SmallPlayoutBatch, SmallPlayoutStream
    from rl.policies.small_playout import export_online_search_pair
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED']*5,
                         combat_action_contract='autoregressive-v2')
    pair = export_online_search_pair(SmallPlayoutLearner(seed=19).policy, collect_samples=True)
    pair.update(common_simulations=0, playout_depth=8)
    batch = SmallPlayoutBatch([env], None, pair)
    before = env.thick.bc.to_dict()
    previous_threads = {t.ident for t in threading.enumerate()}
    stream = SmallPlayoutStream(batch, threads=2)
    stream.submit({0: ((), dict(state_nonce=19))})
    with pytest.raises(ValueError, match='budget'):
        while stream.pending:
            stream.receive(timeout=.1)
    assert not stream.pending
    assert env.thick.bc.to_dict() == before
    assert not [t for t in threading.enumerate()
                if t.name.startswith('sts-search') and t.ident not in previous_threads]
    with pytest.raises(RuntimeError, match='closed'):
        stream.submit({0: ((), dict(state_nonce=19))})
    pair['common_simulations'] = 8
    with SmallPlayoutStream(batch, threads=2) as retry:
        retry.submit({0: ((), dict(state_nonce=19))})
        result = {}
        while retry.pending:
            result.update(retry.receive(timeout=.1))
    assert set(result) == {0} and batch.samples


def test_two_rounds_checkpoint_and_exact_continuation(tmp_path):
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=12), threads=2)
    collector = BattleCollector(n_envs=2)
    first, report = collector.collect(learner.search, 16)
    assert report['counts']['real_transitions'] == 32
    update = learner.update(first, batch_size=8)
    assert update['parameter_l2_change'] > 0
    learner.save(tmp_path / 'small.pt', collector)
    restored, pending = learner.load(tmp_path / 'small.pt', threads=2)
    second, expected = collector.collect(learner.search, 16)
    repeated, actual = pending.collect(restored.search, 16)
    for a, b in zip(second, repeated):
        for key in ('state', 'actions', 'keys', 'visits', 'action_bits'):
            np.testing.assert_array_equal(a[key], b[key])
    assert expected['battles'] == actual['battles']
    assert expected['counts']['completed_battles'] > 0
    learner.update(second, batch_size=8)
    restored.update(repeated, batch_size=8)
    np.testing.assert_array_equal(learner.policy.export(), restored.policy.export())
    assert learner.updates == 2 and restored.optimizer_steps == 8
    restored.save(tmp_path / 'second.pt', pending)
    final, resumed = restored.load(tmp_path / 'second.pt')
    assert final.updates == 2 and resumed.steps == 64


def test_evaluation_strategy_uses_complete_actions(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='native-v1')
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=8))
    strategy = SmallPlayoutStrategy(learner.search)
    result = strategy.decide([DecisionRequest(env, {}, {'phase': 'combat'}, 42, 0)])[0]
    assert result.action is None
    assert_legal(env.thick.bc, result.native_action)
    _, _, _, truncated, info = env.step_native_combat(result.native_action)
    assert not truncated and info['native_action'] == result.native_action
    with pytest.raises(ValueError, match='combat'):
        strategy.decide([DecisionRequest(env, {}, {'phase': 'map'}, 42, 0)])
    legacy = combat_factory(['STRIKE_RED'] * 5)
    with pytest.raises(ValueError, match='native-v1'):
        strategy.decide([DecisionRequest(legacy, {}, {'phase': 'combat'}, 42, 0)])


@pytest.mark.parametrize('card,zone', [('HEADBUTT', 'discard'), ('EXHUME', 'exhaust'),
                                     ('SECRET_TECHNIQUE', 'draw'), ('DISCOVERY', 'generated'), ('PURITY', 'hand')])
def test_selection_regions_and_complete_subsets(combat_factory, card, zone):
    env = combat_factory([card] * 5, combat_action_contract='native-v1')
    bc = env.thick.bc
    bc.apply_custom_state({'hand': [card, 'STRIKE_RED', 'DEFEND_RED'],
                          'draw': ['DEFEND_RED'] * 100 + ['SHRUG_IT_OFF'],
                          'discard': ['BASH', 'DEFEND_RED'], 'exhaust': ['STRIKE_RED', 'DEFEND_RED']})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    assert bc.select_info()['zone'] == zone
    features = sts.small_playout_features(bc)
    if card == 'SECRET_TECHNIQUE':
        assert set(features['action_bits']) == {2 << 29, (2 << 29) | 100}
    else:
        assert set(features['action_bits']) == set(env.native_combat_actions())
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=16, playout_depth=2))
    with torch.inference_mode():
        expected = learner.policy(*collate_features([features])).numpy()[0]
    np.testing.assert_allclose(learner.search.net.logits(bc), expected, atol=2e-5, rtol=2e-5)
    if zone == 'draw':
        assert any(sts.Action(bits).source_idx == 100 for bits in features['action_bits'])
    if card == 'PURITY':
        assert {bits & 0xffff for bits in features['action_bits']} == {0, 1, 2, 3}
        assert sorted((row >= 0).sum() for row in features['keys']) == [1, 1, 1, 2]
    chosen = learner.search.search([bc], [123])[0]['action']
    assert chosen in features['action_bits']
    bc.execute(sts.Action(chosen))


def test_target_slots_and_potion_features(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, encounter='Three Darklings', combat_action_contract='native-v1')
    bc = env.thick.bc
    bc.apply_custom_state({'monsters': [{'hp': 0, 'is_dead': True, 'gone': True}, {'hp': 17}, {'hp': 29}],
                          'potions': ['Fire Potion', 'Smoke Bomb']})
    env._invalidate_backend_cache()
    features = sts.small_playout_features(bc)
    assert set(features['action_bits']) == set(env.native_combat_actions())
    assert features['state'][19] == pytest.approx(17/200)
    assert features['state'][22] == pytest.approx(29/200)
    for bits, row in zip(features['action_bits'], features['actions']):
        if bc.is_targeted_action(bits):
            target = sts.Action(bits).target_idx
            assert target in (1, 2)
            assert row[16] == pytest.approx((17 if target == 1 else 29)/200)
            assert row[22] == pytest.approx((target+1)/5)


@pytest.mark.parametrize('card,zone', [('HEADBUTT', 'discard'), ('EXHUME', 'exhaust')])
def test_dynamic_playout_preserves_choices_beyond_fixed_policy_slots(combat_factory, card, zone):
    """SIM-SMALL-NATIVE-CHOICE-CAPACITY: no unique option may be truncated."""
    bc = combat_factory([card] * 5).thick.bc
    # 172 instances, 100 distinct effects, including handles beyond slot 95.
    cards = [('SEARING_BLOW', i) for i in range(100)] + [('SEARING_BLOW', 0)] * 72
    bc.apply_custom_state({'hand': [card], zone: cards})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    features = sts.small_playout_features(bc)
    assert [sts.Action(bits).source_idx for bits in features['action_bits']] == list(range(100))
    assert features['actions'].shape == (100, 24)
    np.testing.assert_allclose(features['actions'][:, 6], np.arange(100) / 10, atol=1e-6)
    # The fixed ABI continues to reject this state rather than aliasing picks
    # onto multi-select ids or changing historical checkpoint dimensions.
    with pytest.raises(RuntimeError, match='100 distinct of 172'):
        bc.compact_leaf_arrays()
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=250, playout_depth=200))
    with torch.inference_mode():
        expected = learner.policy(*collate_features([features])).numpy()[0]
    np.testing.assert_allclose(learner.search.net.logits(bc), expected, atol=2e-5, rtol=2e-5)
    result = learner.search.search([bc], [123])[0]
    assert result['action'] in features['action_bits']
    np.testing.assert_array_equal(result['action_bits'], features['action_bits'])
    assert np.isfinite(result['visits']).all()
    # Last distinct option is executable even if the search chose another one.
    bc.execute(sts.Action(int(features['action_bits'][-1])))
    assert int(bc.input_state) != 2


@pytest.mark.parametrize('contract', ['native-v1', 'autoregressive-v2'])
def test_compact_search_uses_drink_only_contract_with_passive_potion(combat_factory, contract):
    env = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH'] * 4, combat_action_contract=contract)
    bc = env.thick.bc
    bc.apply_custom_state({'potions': ['Fairy Potion', 'Fire Potion']})
    env._invalidate_backend_cache()
    features = sts.small_playout_features(bc)
    legal = set(env.native_combat_actions())
    assert set(features['action_bits']) == legal
    assert not any(bits >> 29 == 1 and (bits & 0xffff) == 0 for bits in legal)
    assert any(bits >> 29 == 1 and (bits & 0xffff) == 1 for bits in legal)
    learner = SmallPlayoutLearner(seed=42, config=SmallSearchConfig(simulations=250))
    for nonce in range(8):
        result = learner.search.search([bc], [nonce])[0]
        assert set(result['action_bits']) == legal
        assert result['action'] in legal
        if contract == 'autoregressive-v2':
            mapped = env._apply_rulebase_turn_decision(result['action'], {})
            assert env.action_masks()[mapped['action']]


def test_dome_features_hide_intent_and_search_refuses_privileged_rollouts(combat_factory):
    bc = combat_factory(['STRIKE_RED'] * 5, relics=['Runic Dome']).thick.bc
    assert np.all(sts.small_playout_features(bc)['state'][18:31:3] == 0)
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=8))
    with pytest.raises(ValueError, match='public-intent belief'):
        learner.search.search([bc], [42])


def test_frozen_eye_and_placed_top_survive_resampling(combat_factory):
    frozen = combat_factory(['STRIKE_RED', 'BASH', 'DEFEND_RED'] * 5, relics=['Frozen Eye']).thick.bc
    before = frozen.to_dict()['draw_pile'] if 'draw_pile' in frozen.to_dict() else frozen.to_dict()['draw']
    for nonce in (1, 2, 3):
        copied = frozen.clone()
        sts.small_playout_redeterminize(copied, nonce)
        data = copied.to_dict()
        assert (data['draw_pile'] if 'draw_pile' in data else data['draw']) == before
    bc = combat_factory(['WARCRY'] * 5).thick.bc
    bc.apply_custom_state({'hand': ['WARCRY', 'BASH', 'DEFEND_RED'], 'draw': ['ANGER', 'CLEAVE'], 'discard': []})
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    pick = next(a for a in bc.legal_actions() if int(a.action_type) == 2)
    bc.execute(pick)
    assert sts.small_playout_known_top(bc) == 1
    # Drawing after each independent belief sample must return the placed card.
    tops = []
    for nonce in range(5):
        copied = bc.clone()
        sts.small_playout_redeterminize(copied, nonce)
        raw = copied.to_dict()
        pile = raw['draw_pile'] if 'draw_pile' in raw else raw['draw']
        tops.append(pile[-1])
    assert all(top == tops[0] for top in tops)


def test_separate_prior_strict_root_value_and_real_reinforce(tmp_path):
    from rl.algorithms.small_playout import leave_one_out_advantages, save_policy_pair, load_policy_pair
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=20), threads=2)
    prior = SmallPlayoutLearner(seed=7).policy
    root = BattleFixture().create(42)
    config = replace(learner.search.config, use_prior=True).native()
    native_prior = sts.SmallPlayoutNet(prior.export())
    with pytest.raises(ValueError, match='budget'):
        learner.search.net.search(root, config, 42)
    result = learner.search.net.search(root, config, 42, native_prior)
    assert result['stats']['prior_calls'] > 0
    assert sts.small_playout_root_pick([50, 2], [.1, .4], 42) == 1
    assert sts.small_playout_root_pick([50, 2], [.39, .4], 42) == 1
    ties = [sts.small_playout_root_pick([50, 2], [.4, .4], nonce) for nonce in range(100)]
    assert 30 < sum(ties) < 70
    assert sts.small_playout_root_pick([50, 0], [.4, .8], 42) == 0
    with pytest.raises(ValueError, match='visited'):
        sts.small_playout_root_pick([0, 0], [0., 0.], 42)
    a = leave_one_out_advantages([[1., 0., 0.], [-2., -2., -2.]])
    assert a[0, 0] > 0 and np.all(a[0, 1:] < 0) and np.all(a[1] == 0)
    before = prior.export().copy()
    update = learner.reinforce([root] * 8, nonces=[i*100 for i in range(8)], k=4, batch_size=128)
    assert update['accepted_groups'] > 0 and update['samples'] > 0
    assert update['parameter_l2_change'] > 0 and np.isfinite(update['loss'])
    np.testing.assert_array_equal(prior.export(), before)
    save_policy_pair(tmp_path/'pair.pt', learner.policy, prior, provenance={'source':'test-real-playouts'})
    policy2, prior2, meta = load_policy_pair(tmp_path/'pair.pt')
    np.testing.assert_array_equal(policy2.export(), learner.policy.export())
    np.testing.assert_array_equal(prior2.export(), prior.export())
    assert meta['source'] == 'test-real-playouts'


@pytest.mark.parametrize('escape', [False, True])
def test_terminal_value_uses_only_hp_or_boss_victory(run_combat_factory, escape):
    root = run_combat_factory(potions=['Smoke Bomb'])
    bc = root.bc
    bc.set_player_block(999)
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    stolen = 99 - bc.to_dict()['player']['gold']
    assert stolen > 0
    if escape:
        bc.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    else:
        bc.set_player_strength(999)
        bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert bc.is_battle_over
    raw = bc.to_dict()
    assert sts.small_playout_value(bc) == raw['player']['hp']
    assert sts.small_playout_value(bc, True) == (0 if escape else 1)


@pytest.mark.parametrize('threads', [1, 3])
def test_search_state_scratch_does_not_leak_between_different_piles(combat_factory, threads):
    small = combat_factory(['STRIKE_RED', 'DEFEND_RED', 'BASH'] * 3).thick.bc
    large = small.clone()
    large.apply_custom_state({
        'draw': ['STRIKE_RED', 'DEFEND_RED', 'BASH'] * 43,
        'discard': ['POMMEL_STRIKE'] * 17,
        'exhaust': ['ANGER'] * 7,
        'player': {'powers': [('STRENGTH', 3), ('ARTIFACT', 2), ('BARRICADE', 1)]},
    })
    before = [bc.to_dict() for bc in (small, large)]
    learner = SmallPlayoutLearner(
        config=SmallSearchConfig(simulations=16, tree_depth=4, playout_depth=8), threads=threads)
    expected = learner.search.search([small], [41])[0]
    # Reuse the pool after a different, larger state and after it is released.
    first = learner.search.search([large, small, large, small], [42, 41, 43, 41])
    assert [bc.to_dict() for bc in (small, large)] == before
    del large
    repeated = learner.search.search([small] * 6, [41] * 6)
    for actual in [first[1], first[3], *repeated]:
        assert actual.keys() == expected.keys()
        for field, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[field], value)
            elif field != 'stats':
                assert actual[field] == value
        assert {k: v for k, v in actual['stats'].items() if k != 'search_seconds'} == {
            k: v for k, v in expected['stats'].items() if k != 'search_seconds'}


@pytest.mark.parametrize('tree_depth', [1, 1024])
def test_chance_widening_and_worker_failure_recovery(combat_factory, tree_depth):
    bc = combat_factory(['SHRUG_IT_OFF'] * 5).thick.bc
    bc.apply_custom_state({'hand': ['BATTLE_TRANCE'], 'draw': [
        'STRIKE_RED', 'DEFEND_RED', 'BASH', 'INFLAME', 'HEAVY_BLADE', 'ANGER', 'DISARM', 'CLEAVE'],
        'discard': []})
    before = bc.to_dict()
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=500, tree_depth=tree_depth), threads=2)
    first = learner.search.search([bc, bc], [23, 24])
    assert sum(r['stats']['chance_replays'] for r in first) > 0
    if tree_depth == 1:
        assert all(r['stats']['tree_cuts'] == 500 for r in first)
    else:
        assert all(r['stats']['nodes'] > len(r['action_bits']) for r in first)
    bad = replace(learner.search.config, simulations=0).native()
    with pytest.raises(ValueError, match='budget'):
        sts.small_playout_search_batch(learner.search.pool, learner.search.net, [bc, bc], [bad, bad], [23, 24])
    after = learner.search.search([bc, bc], [23, 24])
    for initial, repeated in zip(first, after):
        for field in ('action', 'action_bits', 'visits', 'values'):
            assert initial[field] == repeated[field]
        assert {k: v for k, v in initial['stats'].items() if k != 'search_seconds'} == {
            k: v for k, v in repeated['stats'].items() if k != 'search_seconds'}
    assert bc.to_dict() == before


def test_codex_generated_choices_do_not_merge_colliding_legacy_keys():
    # A real simulator journal exposed a legacy hash-combine collision between
    # Codex triples (297,11,132) and (300,355,258). It previously reused an edge
    # for a card absent from the current choice screen. No model is restored.
    import json
    from pathlib import Path
    from env.core.run_env import SlayTheSpireEnv
    env = SlayTheSpireEnv(global_map_view=True, combat_action_contract='autoregressive-v2',
        outer_return_context=True, combat_value_objective='combat-outcome-means-v1',
        deck_size_observation=True, neow_observation=True, event_option_observation=True)
    try:
        journal = json.loads((Path(__file__).parents[1]/'fixtures/small_playout_codex_journal.json').read_text())
        env.restore_training_state(journal)
        root = env.thick.bc.clone()
        for bits in [3, 1, 2147483648, 1073741824, 8, 7]:
            root.execute(sts.Action(bits))
        before = root.to_dict()
        learner = SmallPlayoutLearner(seed=42, threads=2)
        results = learner.search.search([root, root], [42000001653]*2,
                                       [SmallSearchConfig(simulations=1000)]*2)
        assert results[0]['visits'] == results[1]['visits']
        assert sum(results[0]['visits']) == 1000
        assert results[0]['action'] in results[0]['action_bits']
        assert root.to_dict() == before
    finally:
        env.close()


def test_escaped_thief_gold_does_not_change_terminal_hp_value(run_combat_factory):
    bc = run_combat_factory().bc
    for _ in range(10):
        if bc.is_battle_over:
            break
        bc.set_player_block(999)
        bc.execute(sts.Action(sts.ActionType.END_TURN))
    assert bc.is_battle_over and int(bc.outcome) == 1
    raw = bc.to_dict()
    stolen = 99 - raw['player']['gold']
    assert stolen > 0
    assert sts.small_playout_value(bc) == raw['player']['hp']


def test_legacy_search_checkpoint_and_pair_fail_closed(tmp_path):
    from rl.algorithms.small_playout import save_policy_pair, load_policy_pair
    learner = SmallPlayoutLearner(seed=42)
    path = tmp_path/'pair.pt'
    save_policy_pair(path, learner.policy, learner.policy, provenance={})
    payload = torch.load(path, weights_only=True)
    payload.pop('search_contract')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='incompatible'):
        load_policy_pair(path)
    learner.save(tmp_path/'learner.pt', BattleCollector(n_envs=1))
    payload = torch.load(tmp_path/'learner.pt', weights_only=True)
    payload.pop('search_contract')
    torch.save(payload, tmp_path/'learner.pt')
    with pytest.raises(ValueError, match='incompatible'):
        learner.load(tmp_path/'learner.pt')


def test_real_free_damage_combo_remains_searchable(combat_factory):
    bc = combat_factory(['ANGER']*5, relics=['Unceasing Top']).thick.bc
    bc.apply_custom_state({'hand': ['ANGER'], 'draw': [], 'discard': [],
                           'monsters': [{'hp': 7}]})
    learner = SmallPlayoutLearner(seed=42, config=SmallSearchConfig(simulations=32))
    result = learner.search.search([bc], [91])[0]
    assert sum(result['terminal_visits']) > 0
    assert int(sts.Action(result['action']).action_type) == int(sts.ActionType.CARD)
    for _ in range(4):
        if bc.is_battle_over:
            break
        result = learner.search.search([bc], [91])[0]
        bc.execute(sts.Action(result['action']))
    assert bc.is_battle_over and int(bc.outcome) == 1


def test_turn_card_budget_finishes_pending_selection_then_ends_turn(combat_factory):
    bc = combat_factory(['PURITY', 'STRIKE_RED', 'DEFEND_RED']*3).thick.bc
    bc.apply_custom_state({'hand':['PURITY', 'STRIKE_RED', 'DEFEND_RED'],
                           'player':{'counters':{'cards_played':127}}})
    assert not sts.small_playout_turn_card_limit_reached(bc)
    assert any(sts.Action(b).action_type == sts.ActionType.CARD for b in sts.small_playout_features(bc)['action_bits'])
    bc.execute(sts.Action(sts.ActionType.CARD,0,0))
    # The 128th card may request a choice; never End Turn through that choice.
    assert not sts.small_playout_turn_card_limit_reached(bc)
    choices=sts.small_playout_features(bc)['action_bits']
    assert choices and all(sts.Action(b).action_type==sts.ActionType.MULTI_CARD_SELECT for b in choices)
    bc.execute(sts.Action(next(b for b in choices if not sts.Action(b).selected_indices)))
    assert sts.small_playout_turn_card_limit_reached(bc)
    limited = sts.small_playout_features(bc)['action_bits']
    assert len(limited)==1 and sts.Action(limited[0]).action_type==sts.ActionType.END_TURN
    learner=SmallPlayoutLearner(seed=42,config=SmallSearchConfig(simulations=8))
    before=bc.to_dict()['turn']
    result=learner.search.search([bc],[42])[0]
    assert result['stats']['turn_card_limit_decisions']==1
    assert result['stats']['turn_card_limit_simulated']>=8
    bc.execute(sts.Action(result['action']))
    assert bc.to_dict()['turn']>before and not sts.small_playout_turn_card_limit_reached(bc)
    assert any(sts.Action(b).action_type==sts.ActionType.CARD for b in sts.small_playout_features(bc)['action_bits'])

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


def test_caps_are_not_terminal_and_invalid_budgets_fail():
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=16, playout_depth=1))
    root = BattleFixture().create(42)
    stats = learner.search.search([root], [4])[0]['stats']
    assert stats['depth_cuts'] > 0
    assert stats['terminal_wins'] == 0
    cfg = replace(learner.search.config, tree_depth=1).native()
    stats = learner.search.net.search(root, cfg, 4)['stats']
    assert stats['tree_cuts'] == 16
    assert stats['policy_calls'] == 0
    cfg.playout_depth = 201
    with pytest.raises(ValueError, match='budget'):
        learner.search.net.search(root, cfg, 4)


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


def test_separate_prior_root_q_band_and_real_reinforce(tmp_path):
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
    bits = result['action_bits'][:2]
    assert sts.small_playout_root_pick(root, bits, [50, 2], [.1, .4], .02, False) == 1
    assert sts.small_playout_root_pick(root, bits, [50, 2], [.39, .4], .02, False) == 0
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
def test_manual_value_distinguishes_recovered_and_lost_gold(run_combat_factory, escape):
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
    hp = np.clip((raw['player']['hp'] + 4*sum(p not in ('EMPTY_POTION_SLOT', 'INVALID') for p in raw['potions'])
                  - .01*raw['turn'])/200, 0, 1)
    expected = (.5 + .49*hp - stolen/1000) if escape else (1 + .5*hp)
    assert sts.small_playout_value(bc) == pytest.approx(expected)


def test_chance_widening_and_worker_failure_recovery(combat_factory):
    bc = combat_factory(['SHRUG_IT_OFF'] * 5).thick.bc
    bc.apply_custom_state({'hand': ['BATTLE_TRANCE'], 'draw': [
        'STRIKE_RED', 'DEFEND_RED', 'BASH', 'INFLAME', 'HEAVY_BLADE', 'ANGER', 'DISARM', 'CLEAVE'],
        'discard': []})
    learner = SmallPlayoutLearner(config=SmallSearchConfig(simulations=500, tree_depth=1), threads=2)
    first = learner.search.search([bc, bc], [23, 24])
    assert sum(r['stats']['chance_replays'] for r in first) > 0
    assert all(r['stats']['tree_cuts'] == 500 for r in first)
    bad = replace(learner.search.config, simulations=0).native()
    with pytest.raises(ValueError, match='budget'):
        sts.small_playout_search_batch(learner.search.pool, learner.search.net, [bc, bc], [bad, bad], [23, 24])
    after = learner.search.search([bc, bc], [23, 24])
    assert [r['visits'] for r in first] == [r['visits'] for r in after]


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


def test_escaped_thief_gold_is_penalized(run_combat_factory):
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
    hp = (raw['player']['hp'] - .01*raw['turn']) / 200
    assert sts.small_playout_value(bc) == pytest.approx(1 + .5*hp - stolen/1000)

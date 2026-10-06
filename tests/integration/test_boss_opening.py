import copy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from env.core.boss_opening import (
    BOSSES, CONTRACT, BossOpeningPool, boss_episode_outcome, generate_opening,
    opening_digest, replay_opening,
)
from env.core.run_env import SlayTheSpireEnv


@pytest.fixture(scope='module')
def pool(tmp_path_factory):
    entries = [generate_opening(seed)[0] for seed in (0, 7, 1)]
    assert [entry['boss'] for entry in entries] == list(BOSSES)
    path = tmp_path_factory.mktemp('boss-openings') / 'dataset.json'
    path.write_text(json.dumps(dict(contract=CONTRACT, entries=entries)), encoding='utf-8')
    return BossOpeningPool(path)


def make_env(pool, index):
    return SlayTheSpireEnv(backend_factory=pool.factory(fixed_index=index),
                          global_map_view=True, card_language=True,
                          combat_action_contract='autoregressive-v2')


@pytest.fixture
def manual_watch(pool):
    from apps.dashboard import WatchSession

    class Watch(WatchSession):
        def _policy_outputs(self, observation, mask):
            probabilities = np.asarray(mask, dtype=np.float32)
            return probabilities / probabilities.sum(), 0.0, None

    env = make_env(pool, 0)
    model = SimpleNamespace(observation_space=env.observation_space,
                            policy=SimpleNamespace(set_training_mode=lambda enabled: None),
                            nosl_is_az_simulations=0)
    env.close()
    session = Watch(model, threading.Lock(), boss_opening_pool=pool)
    yield session
    session.close()


@pytest.mark.parametrize('index', range(3))
def test_dashboard_training_opening_matches_actor_and_restarts(pool, manual_watch, index):
    session = manual_watch
    seed = pool.entries[index]['source_seed']
    state = session.reset(seed, training_opening_seed=seed, manual_mode=True)
    actor = make_env(pool, index)
    try:
        expected, _ = actor.reset(seed=456)
        assert session.env.thick.bc.to_dict() == actor.thick.bc.to_dict()
        assert opening_digest(session.env.thick) == pool.entries[index]['opening_sha256']
        assert set(expected) == set(session.obs)
        for key in expected:
            np.testing.assert_array_equal(session.obs[key], expected[key])
        assert state['manual_mode'] and state['decision'] is None
        assert not state['route_preprocess']['enabled']
        assert state['training_opening']['dataset_sha256'] == pool.sha256
        assert {a['action'] for a in state['manual_actions']} == set(np.flatnonzero(actor.action_masks()))
        original = actor.thick.bc.to_dict()
        action = next(a['action'] for a in state['manual_actions'] if a['animation'] and a['animation']['kind'] == 'play_card')
        # A cached model decision with the same policy ID must not hijack the human action.
        session.env._nosl_cpp_pending = (action, 4 << 29)
        actor.step(action)
        after = session.manual_action(action, state['action_revision'])
        assert session.env.thick.bc.to_dict() == actor.thick.bc.to_dict()
        with pytest.raises(ValueError, match='画面已更新'):
            session.manual_action(action, state['action_revision'])
        assert session.position == 1
        session.undo()
        assert session.env.thick.bc.to_dict() == original
        session.manual_action(action, session.action_revision)
        assert session.env.thick.bc.to_dict() == actor.thick.bc.to_dict()
        session.reset(seed, training_opening_seed=seed, manual_mode=True)
        assert session.env.thick.bc.to_dict() == original
        assert opening_digest(pool.opening(index)) == pool.entries[index]['opening_sha256']
        session.preview_map(3)
        assert session.position == 0
        assert session.step()['step'] == 0  # autoplay cannot move manual state
        session.configure_search(manual_mode=False)
        assert session.decision is not None
        with pytest.raises(ValueError, match='手动操作'):
            session.manual_action(action, session.action_revision)
        session.configure_search(manual_mode=True)
        with pytest.raises(ValueError, match='当前状态'):
            session.manual_action(999999, session.action_revision)
        session.reset(0)
        assert session.training_opening is None
        assert session.env.thick.bc is None
    finally:
        actor.close()


def test_dashboard_training_opening_rejects_resource_editor_and_mixed_modes(pool, manual_watch):
    session = manual_watch
    seed = pool.entries[0]['source_seed']
    session.reset(seed, training_opening_seed=seed, manual_mode=True)
    before = session.env.thick.bc.to_dict()
    for kwargs in ({'precompute': True}, {'target_boss_act': 2}):
        with pytest.raises(ValueError, match='混用'):
            session.reset(seed, training_opening_seed=seed, **kwargs)
    with pytest.raises(ValueError, match='不在'):
        session.reset(seed, training_opening_seed=999999)
    with pytest.raises(ValueError, match='普通爬塔'):
        session.apply_custom_state({})
    assert session.env.thick.bc.to_dict() == before


def test_dashboard_search_retains_model_utility_and_hidden_search_settings(manual_watch, monkeypatch):
    from apps import dashboard
    session = manual_watch
    session.model.policy.entity_config = {'boss_win_weight': 0.9}
    session.model.nosl_is_az_merge_equivalent_actions = False
    seen = []
    def capture(model, env, observation, config, **kwargs):
        seen.append(config)
        return {}
    monkeypatch.setattr(dashboard, 'native_search', capture)
    session.configure_search(simulations=128, paper_full_gumbel=True,
                             gumbel_value_scale=1.0, gumbel_rescale_q=False, gumbel_scale=0.0)
    session._run_native_search(np.zeros(198), 0.0)
    assert seen[0].boss_win_weight == 0.9
    assert not seen[0].merge_equivalent_actions
    assert seen[0].gumbel_value_scale == 1 and not seen[0].gumbel_rescale_q
    assert seen[0].gumbel_scale == 0


def test_dashboard_manual_multi_selection(manual_watch, replayable_selection_env):
    session = manual_watch
    session.env.close()
    session.env = replayable_selection_env
    session._run_backend_factory = session.env._backend_factory
    state = session.reset(42, manual_mode=True)
    play = next(a for a in state['manual_actions'] if a['animation'] and a['animation']['kind'] == 'play_card' and a['animation']['slot'] == 0)
    state = session.manual_action(play['action'], state['action_revision'])
    choices = [a for a in state['manual_actions'] if a.get('choice', {}).get('kind') == 'select_hand']
    assert len(choices) == 5
    choice = choices[0]
    assert choice['choice']['card']['id'] == 'Strike_R'
    state = session.manual_action(choice['action'], state['action_revision'])
    confirm = next(a for a in state['manual_actions'] if a.get('choice', {}).get('kind') == 'confirm')
    assert confirm['choice']['selected'] == [choice['choice']['index']]
    state = session.manual_action(confirm['action'], state['action_revision'])
    assert len(state['canon']['hand']) == 4
    assert len(state['canon']['exhaust']) == 2  # Purity plus the chosen Strike
    session.undo()
    assert session.env._multi == {choice['choice']['index']}


@pytest.mark.parametrize('index', range(3))
def test_opening_journal_is_exact_and_no_resource_reset(pool, index):
    entry = pool.entries[index]
    original = pool.opening(index)
    reconstructed = replay_opening(entry)
    assert opening_digest(reconstructed) == opening_digest(original)
    assert reconstructed.bc.to_dict() == original.bc.to_dict()
    assert not reconstructed.gc.skip_battles
    assert reconstructed.bc.to_dict()['turn'] == 0  # simulator counts from zero
    assert reconstructed.gc.act == 2 and reconstructed.gc.floor_num == 33
    assert reconstructed.bc.to_dict()['player']['max_hp'] == reconstructed.gc.max_hp
    assert tuple(reconstructed.bc.potions) == tuple(original.bc.potions)


def test_openings_preserve_varying_hp_and_potions():
    _, backend = generate_opening(2)
    assert backend.gc.cur_hp == 72
    _, grown = generate_opening(5)
    assert grown.gc.max_hp == 94
    assert grown.gc.cur_hp == 94
    assert grown.gc.overworld_potions == grown.bc.potions


def test_skipped_fights_consume_neow_charges_before_boss():
    # Previously reached a Boss with Neow charges and Mercury Hourglass:
    # skipped fights never exited a BC, so the Boss was reduced to 1 HP.
    _, backend = generate_opening(300234)
    neow = [r for r in backend.gc.relic_state() if r['id'] == 'NeowsBlessing']
    assert len(neow) == 1 and neow[0]['counter'] == 0
    assert not backend.bc.is_battle_over
    alive = [m for m in backend.bc.to_dict()['monsters'] if m['hp'] > 0]
    assert alive and max(m['hp'] for m in alive) > 100


@pytest.mark.parametrize('index', range(3))
def test_actor_isolation_restore_and_true_loss(pool, index):
    env = make_env(pool, index)
    sibling = make_env(pool, index)
    observation, _ = env.reset(seed=123)
    sibling.reset(seed=123)
    initial = sibling.thick.bc.to_dict()
    opening = opening_digest(pool.opening(index))
    # Native end turn; it is a real transition, not an HP/phase fixture edit.
    def wait_action():
        legal = np.flatnonzero(env.action_masks())
        return 0 if 0 in legal else int(legal[0])
    env.step(wait_action())
    state = env.export_training_state()
    restored = make_env(pool, index)
    restored_obs, _ = restored.restore_training_state(state)
    assert env.thick.bc.to_dict() == restored.thick.bc.to_dict()
    for key, values in env._obs().items():
        np.testing.assert_array_equal(values, restored_obs[key])
    assert sibling.thick.bc.to_dict() == initial
    assert opening_digest(pool.opening(index)) == opening
    for _ in range(100):
        _, _, done, truncated, info = env.step(wait_action())
        if done:
            break
    assert done and not truncated
    assert info['nosl_value_terminal_outcome'] == 'loss'
    assert info['nosl_value_terminal_target'] == 0
    assert info['run_stats']['episode_scope'] == 'boss-opening'
    assert info['run_stats']['boss'] == BOSSES[index]
    assert info['run_stats']['combat_outcome'] == 'loss'
    assert boss_episode_outcome(env, info, done, truncated)['result'] == 'death'
    assert env.thick.gc.outcome.name == 'UNDECIDED'  # no mutation of cached GC
    assert opening_digest(pool.opening(index)) == opening
    env.reset(seed=123)
    assert env.thick.bc.to_dict() == initial


def test_rejects_dataset_change_and_post_opening_journal(pool, tmp_path):
    env = make_env(pool, 0)
    env.reset(seed=0)
    state = env.export_training_state()
    state['backend_identity']['dataset_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='identity mismatch'):
        env.restore_training_state(state)
    entry = copy.deepcopy(pool.entries[0])
    entry['actions'].append(entry['actions'][-1])
    with pytest.raises(ValueError, match='final route action'):
        replay_opening(entry)

import copy
import json

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

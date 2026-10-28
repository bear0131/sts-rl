"""Static map topology must not freeze position or live relic-assisted choices."""
import pytest
import numpy as np

from env.core.spiremap import RunMap

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_warmed_map_view_tracks_live_choices_without_changing_physical_edges():
    rmap = RunMap(42, 0, 1)
    rmap.full_view(-1, -1)
    for y in range(13):
        live = [x for x in range(7) if rmap.room_type(x, y + 1) not in ('NONE', 'INVALID')]
        roots = [x for x in range(7) if rmap.room_type(x, y) not in ('NONE', 'INVALID')
                 and set(rmap.reachable(x, y)) < set(live)]
        if roots:
            break
    else:
        pytest.fail('fixed map must contain an off-path room')
    x = roots[0]
    normal = rmap.reachable(x, y)
    off_path = next(c for c in live if c not in normal)
    ordinary = rmap.full_view(x, y, next_columns=normal)
    assisted = rmap.full_view(x, y, next_columns=live)
    assert [(n['x'], n['y']) for n in assisted if n['choice']] == [(c, y + 1) for c in live]
    assert next(n for n in assisted if n['current'])['children'] == normal
    assert not next(n for n in ordinary if (n['x'], n['y']) == (off_path, y + 1))['reachable']
    assert next(n for n in assisted if (n['x'], n['y']) == (off_path, y + 1))['reachable']
    assisted[0]['children'].append(999)
    assisted[0]['room'] = 'CORRUPTED'
    assert rmap.full_view(x, y, next_columns=normal) == ordinary


def test_map_view_rebuilds_when_native_map_identity_changes():
    rmap = RunMap(42, 0, 1)
    original = rmap.full_view(-1, -1)
    replacement = RunMap(87, 20, 2)
    expected = replacement.full_view(-1, -1)
    assert expected != original
    rmap._m = replacement._m
    assert rmap.full_view(-1, -1) == expected
    assert RunMap(42, 0, 1).full_view(-1, -1) == original


def test_warmed_map_survives_journal_restore_and_fresh_reset():
    from env.core.run_env import SlayTheSpireEnv
    env = SlayTheSpireEnv(global_map_view=True)
    restored = SlayTheSpireEnv(global_map_view=True)
    try:
        env.reset(seed=42)
        for _ in range(16):
            if env.thick.phase() == 'map':
                break
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert env.thick.phase() == 'map'
        before = env._obs()
        state = env.export_training_state()
        actual, _ = restored.restore_training_state(state)
        np.testing.assert_equal(actual, before)
        action = int(np.flatnonzero(env.action_masks())[0])
        np.testing.assert_equal(env.step(action), restored.step(action))
        np.testing.assert_equal(env.reset(seed=7), restored.reset(seed=7))
    finally:
        env.close()
        restored.close()

"""Natural native first-row overflow must remain visible and executable."""
import numpy as np
import pytest
import subprocess

from env.core import action_space as A, encode, game_actions as ga
from env.core.run_env import SlayTheSpireEnv
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_seed_one_four_start_nodes_are_all_visible_and_last_is_executable():
    env = SlayTheSpireEnv(global_map_view=True)
    try:
        env.reset(seed=1)
        for _ in range(8):
            if env.thick.phase() == 'map':
                break
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert env.thick.phase() == 'map'
        nodes = env.thick.obs_canonical()['options']['nodes']
        assert len(nodes) == 4  # Original BRIDGE40 reproduction, not a synthetic map.
        obs = env._obs()
        columns = [x for x, _ in nodes]
        assert np.flatnonzero(obs['map_full_features'][:, 3]).tolist() == columns
        for x in columns:
            assert ga.map_node(x) in env.thick.legal_semantic_actions()
            assert env.action_masks()[A.seg_base('MAP_NODE') + x]
        x = columns[-1]
        env.step(A.seg_base('MAP_NODE') + x)
        assert env.thick.gc.cur_map_node_x == x
        assert env.thick.gc.cur_map_node_y == 0
        assert env.thick.bc is not None
    finally:
        env.close()


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('charges', [-1, 0, 1, 2, 3])
def test_native_wing_boots_map_actions(game_context_executable, charges, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'wing-boots-map', str(charges), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.parametrize('charges', [0, 1, 2, 3])
def test_wing_boots_real_map_mask_full_graph_and_restored_charges(charges, monkeypatch):
    env = SlayTheSpireEnv(global_map_view=True)
    try:
        env.reset(seed=42)
        gc, rmap = env.thick.gc, env.thick.rmap
        # Choose an actual generated row with a room off the current path.
        for y in range(13):
            live = [c for c in range(7) if rmap.room_type(c, y + 1) not in ('NONE', 'INVALID')]
            roots = [c for c in range(7) if rmap.room_type(c, y) not in ('NONE', 'INVALID')
                     and set(rmap.reachable(c, y)) < set(live)]
            if roots:
                break
        else:
            pytest.fail('fixed map has no off-path room')
        x = roots[0]
        normal = rmap.reachable(x, y)
        off_path = next(c for c in live if c not in normal)
        gc.cur_map_node_x, gc.cur_map_node_y = x, y
        gc.floor_num, gc.screen_state = y + 1, sts.ScreenState.MAP_SCREEN
        gc.set_relic_state([{'id': 'WingedGreaves', 'counter': charges}])
        env._invalidate_backend_cache()
        expected = live if charges else normal
        canonical = env.thick.obs_canonical()
        assert [c for c, _ in canonical['options']['nodes']] == expected
        assert [a[1] for a in env.thick.legal_semantic_actions() if a[0] == ga.MAP] == expected
        graph = canonical['options']['full_map']
        assert [(n['x'], n['y']) for n in graph if n['choice']] == [(c, y + 1) for c in expected]
        assert next(n['children'] for n in graph if n['current']) == normal
        assert env.observation_space.contains(env._obs())
        # Charges can change at the same coordinate. No cached projection may
        # retain assisted choices after the last charge is consumed.
        for count in (charges, 1, 0, charges):
            gc.set_relic_state([{'id': 'WingedGreaves', 'counter': count}])
            env._invalidate_backend_cache()
            actual = env._obs()
            with monkeypatch.context() as patch:
                patch.setattr(env.thick, 'encoding_map_observation', lambda canon: None)
                reference = env._obs()
            for key in actual:
                assert actual[key].tobytes() == reference[key].tobytes(), (count, key)
        assert [c for c in range(7) if env.action_masks()[A.seg_base('MAP_NODE') + c]] == expected
        if charges:
            env.step(A.seg_base('MAP_NODE') + off_path)
            assert (gc.cur_map_node_x, gc.cur_map_node_y) == (off_path, y + 1)
            assert gc.relic_state() == [{'id': 'WingedGreaves', 'counter': charges - 1}]
            restored = sts.GameContext(sts.CharacterClass.IRONCLAD, rmap.seed, rmap.ascension)
            restored.set_relic_state(gc.relic_state())
            restored.cur_map_node_x, restored.cur_map_node_y = off_path, y + 1
            assert restored.legal_map_options() == gc.legal_map_options()
        else:
            with pytest.raises(ValueError, match='not connected'):
                gc.transition_to_map_node(off_path)
            assert (gc.cur_map_node_x, gc.cur_map_node_y) == (x, y)
    finally:
        env.close()


def test_map_projection_replacement_schema_order_and_output_ownership():
    import copy
    from env.core.spiremap import RunMap

    rmap = RunMap(42, 0, 1)
    rmap.encoded_view(-1, -1, next_columns=rmap.reachable(-1, -1), global_map_view=True)
    branch = copy.copy(rmap)
    for owner, replacement in ((rmap, None), (branch, None),
                               (branch, RunMap(43, 0, 2)), (rmap, None)):
        if replacement is not None:
            owner._m = replacement._m
        first = owner.reachable(-1, -1)
        last = next(x for x in range(7) if owner.room_type(x, 14) == 'REST')
        for x, y, columns, full in (
                (-1, -1, first, True), (-1, -1, first[::-1], True),
                (-1, -1, first[:3], False), (-1, -1, first, True),
                (last, 14, [last], True), (-1, -1, first, True)):
            options = dict(nodes=[(c, 'BOSS' if y == 14 else owner.room_type(c, 0))
                                  for c in columns])
            if full:
                options['full_map'] = owner.full_view(x, y, next_columns=columns)
            expected = encode.encode(dict(phase='map', options=options), global_map_view=full)
            for _ in range(2):
                projection = owner.encoded_view(x, y, next_columns=columns, global_map_view=full)
                actual = encode.encode(dict(phase='map', options=options), global_map_view=full,
                                       _map_observation=projection)
                for key in expected:
                    assert actual[key].tobytes() == expected[key].tobytes(), key
                for key in projection:
                    assert not projection[key].flags.writeable
                    actual[key].fill(0)


@pytest.mark.parametrize('full_map', [False, True])
def test_native_boss_entry_has_real_room_identity_and_enters_boss(full_map):
    env = SlayTheSpireEnv(global_map_view=full_map)
    try:
        env.reset(seed=42)
        # Controlled final-row root; execution still uses the real transition.
        gc, rmap = env.thick.gc, env.thick.rmap
        x = next(x for x in range(7) if rmap.room_type(x, 14) == 'REST')
        gc.cur_map_node_x, gc.cur_map_node_y = x, 14
        gc.floor_num, gc.screen_state = 15, sts.ScreenState.MAP_SCREEN
        env._invalidate_backend_cache()
        assert env.thick.obs_canonical()['options']['nodes'] == [(x, 'BOSS')]
        assert env.observation_space.contains(env._obs())
        if full_map:
            assert np.flatnonzero(env._obs()['map_full_features'][:, 3]).tolist() == [98 + x]
        action = A.seg_base('MAP_NODE') + x
        assert env.action_masks()[action]
        env.step(action)
        assert gc.cur_room == sts.Room.BOSS and gc.cur_map_node_y == 15
        assert env.thick.bc is not None
    finally:
        env.close()


@pytest.mark.parametrize('ascension', [0, 20])
def test_fourth_act_native_map_matches_rooms_without_extending_training_outcome(ascension):
    from env.core.spiremap import RunMap
    env = SlayTheSpireEnv(global_map_view=True, ascension=ascension)
    try:
        env.reset(seed=42)
        gc = env.thick.gc
        # Controlled victory entry: skip the earlier bosses, not their callbacks.
        gc.act = 3
        gc.red_key = gc.blue_key = gc.green_key = True
        gc.cur_map_node_x = next(x for x in range(7) if env.thick.rmap.room_type(x, 14) == 'REST')
        gc.cur_map_node_y, gc.floor_num = 14, 50
        gc.skip_battles = True
        gc.transition_to_map_node(gc.cur_map_node_x)
        env.thick._settle()
        assert gc.act == 4
        # The existing policy's whole-run contract ends after Act 3. Do not
        # change that reward/terminal definition to exercise an auxiliary map.
        assert env.thick.is_over and env.thick.legal_semantic_actions() == []
        rmap = RunMap(gc.seed, ascension, 4)
        expected = ['REST', 'SHOP', 'ELITE', 'BOSS']
        for y, room in enumerate(expected):
            assert gc.legal_map_options() == [3]
            graph = rmap.full_view(gc.cur_map_node_x, gc.cur_map_node_y)
            assert [(n['x'], n['y'], n['room']) for n in graph] == [
                (3, row, label) for row, label in enumerate(expected)]
            assert [(n['x'], n['y']) for n in graph if n['choice']] == [(3, y)]
            options = dict(nodes=[(3, room)], full_map=graph)
            assert encode.obs_space(global_map_view=True).contains(
                encode.encode(dict(phase='map', options=options), global_map_view=True))
            if y == 3:
                gc.skip_battles = False
            gc.transition_to_map_node(3)
            assert (gc.cur_map_node_x, gc.cur_map_node_y) == (3, y)
            assert gc.cur_room == getattr(sts.Room, room)
            if y == 0:
                gc.choose_campfire_option(0)
            elif y == 1:
                gc.shop_proceed()
            elif y == 2:
                gc.reward_proceed()
        assert gc.screen_state == sts.ScreenState.BATTLE
        assert gc.encounter.name == 'THE_HEART'
    finally:
        env.close()

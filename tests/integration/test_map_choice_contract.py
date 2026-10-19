"""Natural native first-row overflow must remain visible and executable."""
import numpy as np
import pytest

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

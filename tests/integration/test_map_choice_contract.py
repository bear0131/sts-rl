"""Natural native first-row overflow must remain visible and executable."""
import numpy as np
import pytest

from env.core import action_space as A, encode, game_actions as ga
from env.core.run_env import SlayTheSpireEnv

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

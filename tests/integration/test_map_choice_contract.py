"""Natural native first-row overflow must remain visible and executable."""
import numpy as np
import pytest

from env.core import action_space as A, encode, game_actions as ga
from env.core.run_env import SlayTheSpireEnv
from bridge.real_game_env import RealGameAdapter
from env.core.game_env import SimGameEnv
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('pick', range(4))
def test_seed_one_four_start_nodes_match_bridge_and_are_executable(pick):
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
        full_map = env.thick.obs_canonical()['options']['full_map']
        next_nodes = [dict(x=x, y=0, room=room) for x, room in reversed(nodes)]
        adapter = RealGameAdapter(global_map_view=True)
        adapter.update(dict(in_game=True, available_commands=['choose'], game_state=dict(
            screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[], map=full_map,
            choice_list=[str(n['x']) for n in next_nodes],
            screen_state=dict(next_nodes=next_nodes, first_node_chosen=False))))
        real_obs = encode.encode(dict(phase='map', options=adapter._options('map')),
                                 global_map_view=True)
        for key in ['map_full_rooms', 'map_full_features']:
            np.testing.assert_array_equal(obs[key], real_obs[key])
        x = columns[pick]
        command = adapter.command_for_semantic(ga.map_node(x))
        assert next_nodes[int(command.split()[1])]['x'] == x
        env.step(A.seg_base('MAP_NODE') + x)
        assert env.thick.gc.cur_map_node_x == x
        assert env.thick.gc.cur_map_node_y == 0
        assert env.thick.bc is not None
    finally:
        env.close()


@pytest.mark.parametrize('act', [1, 2, 3])
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('full', [False, True])
def test_final_row_boss_choice_has_actual_room_and_enters_battle(act, ascension, full):
    env = SimGameEnv(ascension=ascension, global_map_view=full)
    env.reset(seed=42)
    # Controlled map position, not a natural full-run claim.
    env.gc.act = act
    env.gc.boss = [sts.MonsterEncounter.THE_GUARDIAN,
                   sts.MonsterEncounter.CHAMP, sts.MonsterEncounter.TIME_EATER][act - 1]
    env.gc.screen_state = sts.ScreenState.MAP_SCREEN
    env._rebuild_map()
    x = next(x for x in range(7) if env.rmap.room_type(x, 14) == 'REST')
    env.gc.cur_map_node_x, env.gc.cur_map_node_y = x, 14
    canonical = env.obs_canonical()
    assert canonical['options']['nodes'] == [(x, 'BOSS')]
    obs = encode.encode(canonical, global_map_view=full)
    assert obs['map_node_rooms'][0] == encode.V.ROOM_VOCAB_INDEX['BOSS']
    assert env.legal_semantic_actions() == [ga.map_node(x)]
    if full:
        assert obs['map_full_features'][14 * 7 + x, 3] == 1
    env.step(ga.map_node(x))
    assert env.phase() == 'combat'
    assert env.bc is not None
    assert env.gc.cur_room == sts.Room.BOSS
    expected = ['THE_GUARDIAN', 'THE_CHAMP', 'TIME_EATER'][act - 1]
    assert expected in [m['id'] for m in env.bc.to_dict()['monsters']]

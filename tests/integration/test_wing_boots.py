"""BRIDGE44: charged flight changes choices, not the physical map edges."""
import subprocess

import numpy as np
import pytest
import slaythespire as sts

from bridge.real_game_env import RealGameAdapter
from bridge.replay import Driver
from env.core import action_space as A, encode, game_actions as ga
from env.core.run_env import SlayTheSpireEnv


@pytest.fixture
def wing_root():
    created = []

    def make(ascension=0, counter=3):
        env = SlayTheSpireEnv(ascension=ascension, global_map_view=True)
        env.reset(seed=42)
        created.append(env)
        gc, rmap = env.thick.gc, env.thick.rmap
        # Find a real ordinary edge and a real unconnected next-row node.
        for y in range(13):
            available = [x for x in range(7) if rmap.room_type(x, y + 1) != 'NONE']
            for x in range(7):
                if rmap.room_type(x, y) == 'NONE':
                    continue
                ordinary = rmap.reachable(x, y)
                cross = sorted(set(available) - set(ordinary))
                if ordinary and cross:
                    gc.cur_map_node_x, gc.cur_map_node_y = x, y
                    gc.floor_num = y + 1
                    gc.screen_state = sts.ScreenState.MAP_SCREEN
                    gc.set_relic_state([dict(id='WingedGreaves', counter=counter)])
                    env._invalidate_backend_cache()
                    return env, ordinary, cross, available
        raise AssertionError('fixed seed has no cross-route fixture')

    yield make
    for env in created:
        env.close()


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(9), ids=[
    'absent', 'zero', 'used-up', 'one', 'two', 'three', 'first-and-boss', 'act4', 'portal'])
def test_native_wing_boots_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'wing-boots', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.timeout(30)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('counter', [-2, 0, 1, 2, 3])
@pytest.mark.parametrize('jump', [False, True])
def test_map_actions_observation_and_actual_charge_agree(wing_root, ascension, counter, jump):
    env, ordinary, cross, available = wing_root(ascension, counter)
    gc = env.thick.gc
    expected = available if counter > 0 else ordinary
    assert gc.map_node_options() == expected
    assert env.thick.legal_semantic_actions() == [ga.map_node(x) for x in expected]
    canon = env.thick.obs_canonical()
    assert [x for x, _ in canon['options']['nodes']] == expected
    assert canon['global']['relic_counters'] == [counter]
    obs = env._obs()
    assert obs['relic_counters'][0] == pytest.approx(counter / encode.COUNTER_SCALE)
    y, x = gc.cur_map_node_y, gc.cur_map_node_x
    assert np.flatnonzero(obs['map_full_features'][:, 3]).tolist() == [7 * (y + 1) + c for c in expected]
    # Choices include flight, but the serialized physical edges remain ordinary.
    physical = next(n for n in canon['options']['full_map'] if (n['x'], n['y']) == (x, y))
    assert physical['children'] == ordinary
    for column in range(7):
        assert bool(env.action_masks()[A.seg_base('MAP_NODE') + column]) == (column in expected)
    target = cross[0] if jump else ordinary[0]
    if jump and counter <= 0:
        with pytest.raises(ValueError, match='not currently legal'):
            gc.choose_map_node(target)
        assert env.thick.obs_canonical() == canon
        return
    env.step(A.seg_base('MAP_NODE') + target)
    assert (gc.cur_map_node_x, gc.cur_map_node_y) == (target, y + 1)
    after = -2 if jump and counter == 1 else counter - int(jump)
    assert gc.relic_state() == [dict(id='WingedGreaves', counter=after)]
    assert env.thick.obs_canonical()['global']['relic_counters'] == [after]


@pytest.mark.timeout(30)
@pytest.mark.parametrize('counter', [-2, 1, 3])
def test_flight_candidates_match_client_commands_and_real_replay(wing_root, counter):
    env, ordinary, cross, available = wing_root(counter=counter)
    gc = env.thick.gc
    canonical = env.thick.obs_canonical()
    nodes = [dict(x=x, y=gc.cur_map_node_y + 1, room=room)
             for x, room in reversed(canonical['options']['nodes'])]
    gs = dict(screen_type='MAP', room_phase='COMPLETE', potions=[], relics=gc.relic_state(),
              map=canonical['options']['full_map'], choice_list=[f"x={n['x']}" for n in nodes],
              screen_state=dict(next_nodes=nodes, current_node=dict(x=gc.cur_map_node_x, y=gc.cur_map_node_y)))
    adapter = RealGameAdapter(global_map_view=True)
    adapter.update(dict(in_game=True, available_commands=['choose'], game_state=gs))
    real = adapter.observation()
    sim = env._obs()
    for key in ['map_full_rooms', 'map_full_features', 'relic_counters']:
        np.testing.assert_array_equal(real[key], sim[key])
    assert set(adapter.legal_semantic_actions()) == set(env.thick.legal_semantic_actions())
    target = cross[0] if counter > 0 else ordinary[0]
    command = adapter.command_for_semantic(ga.map_node(target))
    assert nodes[int(command.split()[1])]['x'] == target
    Driver(gc).overworld_cmd(gs, command)
    assert gc.cur_map_node_x == target
    after = -2 if counter == 1 else counter - int(counter > 0)
    assert gc.relic_state()[0]['counter'] == after


@pytest.mark.parametrize('x', [-1, 7, 256])
def test_invalid_map_column_does_not_consume_or_advance(wing_root, x):
    env, _, _, _ = wing_root()
    before = env.thick.obs_canonical()
    with pytest.raises(ValueError):
        env.thick.gc.choose_map_node(x)
    assert env.thick.obs_canonical() == before


def test_new_relic_defaults_and_wrong_screen_rejection(wing_root):
    env, ordinary, _, _ = wing_root()
    gc = env.thick.gc
    gc.set_relics(['WingedGreaves'])
    assert gc.relic_state() == [dict(id='WingedGreaves', counter=3)]
    gc.screen_state = sts.ScreenState.REST_ROOM
    before = (gc.floor_num, gc.cur_map_node_x, gc.cur_map_node_y, gc.relic_state())
    with pytest.raises(ValueError, match='not currently legal'):
        gc.choose_map_node(ordinary[0])
    assert (gc.floor_num, gc.cur_map_node_x, gc.cur_map_node_y, gc.relic_state()) == before

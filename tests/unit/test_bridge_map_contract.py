"""BRIDGE40: full candidates survive the frozen three-token map preview."""
from copy import deepcopy

import numpy as np
import pytest

from bridge.real_game_env import RealGameAdapter
from env.core import action_space as A, encode, game_actions as ga, vocab as V


def map_state(count):
    return dict(phase='map', options=dict(
        nodes=[(x, 'MONSTER') for x in range(count)],
        full_map=[dict(x=x, y=0, room='MONSTER', children=[],
                       current=False, choice=True, reachable=True) for x in range(count)]))


@pytest.mark.parametrize('count', range(1, 8))
def test_every_map_choice_is_encoded_without_changing_frozen_preview(count):
    state = map_state(count)
    before = deepcopy(state)
    obs = encode.encode(state, global_map_view=True)
    assert encode.obs_space(global_map_view=True).contains(obs)
    assert state == before
    assert np.flatnonzero(obs['map_full_features'][:, 3]).tolist() == list(range(count))
    assert np.count_nonzero(obs['map_full_rooms']) == count
    assert obs['map_node_rooms'][:min(3, count)].tolist() == [V.ROOM_VOCAB_INDEX['MONSTER']] * min(3, count)
    np.testing.assert_array_equal(obs['map_node_x'][:min(3, count)],
                                  np.arange(min(3, count), dtype=np.float32) / encode.MAP_X_SCALE)
    if count <= 3:
        legacy = encode.encode(state)
        for key in legacy:
            np.testing.assert_array_equal(legacy[key], obs[key], err_msg=key)
    else:
        # Main's legacy local-only schema is explicitly a partial preview.
        legacy = encode.encode(state)
        np.testing.assert_array_equal(legacy['map_node_rooms'], obs['map_node_rooms'])
        np.testing.assert_array_equal(legacy['map_node_x'], obs['map_node_x'])


@pytest.mark.parametrize('mutation', ['missing_graph', 'missing_choice', 'wrong_room',
    'duplicate_node', 'extra_choice', 'bad_edge', 'missing_flags', 'mixed_rows', 'absent_room'])
def test_incomplete_full_map_cannot_hide_the_fourth_choice(mutation):
    state = map_state(4)
    graph = state['options']['full_map']
    if mutation == 'missing_graph':
        del state['options']['full_map']
    elif mutation == 'missing_choice':
        graph[-1]['choice'] = False
    elif mutation == 'wrong_room':
        graph[-1]['room'] = 'REST'
    elif mutation == 'duplicate_node':
        graph.append(deepcopy(graph[0]))
    elif mutation == 'extra_choice':
        graph.append(dict(graph[-1], x=4))
    elif mutation == 'bad_edge':
        graph[-1]['children'] = [6]
    elif mutation == 'missing_flags':
        del graph[-1]['reachable']
    elif mutation == 'mixed_rows':
        graph[-1]['y'] = 1
    else:
        graph[-1]['room'] = 'PAD'
    with pytest.raises(ValueError):
        encode.encode(state, global_map_view=True)


def test_boss_choice_reuses_the_actual_final_row_node():
    state = map_state(1)
    state['options']['nodes'] = [(0, 'BOSS')]
    state['options']['full_map'][0].update(y=14, room='REST', current=True)
    obs = encode.encode(state, global_map_view=True)
    assert obs['map_full_features'][98, 3] == 1
    assert obs['map_node_rooms'][0] == V.ROOM_VOCAB_INDEX['BOSS']
    state['options']['full_map'][0]['current'] = False
    with pytest.raises(ValueError, match='room'):
        encode.encode(state, global_map_view=True)


@pytest.mark.parametrize('count', range(4, 8))
def test_bridge_keeps_all_columns_and_original_ui_order(count):
    # Deliberately reversed UI ordering: a physical column is not a menu index.
    nodes = [dict(x=x, y=0, symbol='M', children=[]) for x in reversed(range(count))]
    adapter = RealGameAdapter(global_map_view=True)
    adapter.update(dict(in_game=True, available_commands=['choose'], game_state=dict(
        screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[], map=nodes,
        choice_list=[str(x) for x in reversed(range(count))],
        screen_state=dict(next_nodes=nodes, first_node_chosen=False, boss_available=False))))
    assert adapter._map_nodes() == [(x, 'MONSTER') for x in reversed(range(count))]
    legal = adapter.legal_semantic_actions()
    assert legal == [ga.map_node(x) for x in reversed(range(count))]
    mask = adapter.action_mask()
    for x in range(count):
        assert mask[A.seg_base('MAP_NODE') + x]
        assert adapter.command_for_semantic(ga.map_node(x)) == f'CHOOSE {count - x - 1}'
    obs = encode.encode(dict(phase='map', options=adapter._options('map')), global_map_view=True)
    assert np.flatnonzero(obs['map_full_features'][:, 3]).tolist() == list(range(count))


@pytest.mark.parametrize('bad', [None, {}, dict(x=1, y=0), dict(x=7, y=0, symbol='M'),
    dict(x=1.0, y=0, symbol='M'), dict(x=True, y=0, symbol='M'),
    dict(x=2, y=0, symbol='M'), dict(x=1, y=1, symbol='M'),
    dict(x=1, y=0, symbol='unknown')])
def test_bad_candidate_is_rejected_without_shifting_ui_indices(bad):
    nodes = [bad, dict(x=2, y=0, symbol='M')]
    frame = dict(in_game=True, available_commands=['choose'], game_state=dict(
        screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[],
        choice_list=['first', 'second'], screen_state=dict(next_nodes=nodes)))
    before = deepcopy(frame)
    adapter = RealGameAdapter()
    adapter.update(frame)
    for operation in [adapter.legal_semantic_actions, adapter.action_mask,
                      lambda: adapter.command_for_semantic(ga.map_node(2))]:
        with pytest.raises(ValueError):
            operation()
    assert frame == before


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_room', 'bad_child',
    'duplicate_child', 'missing_children', 'invalid_child', 'nonadjacent_child',
    'bad_coordinate', 'bad_current'])
def test_bridge_rejects_incomplete_or_contradictory_graph(mutation):
    nodes = [dict(x=0, y=0, symbol='M', children=[])]
    graph = deepcopy(nodes)
    ss = dict(next_nodes=nodes, first_node_chosen=False)
    if mutation == 'missing':
        graph = []
    elif mutation == 'duplicate':
        graph.append(deepcopy(graph[0]))
    elif mutation == 'wrong_room':
        graph[0]['symbol'] = 'R'
    elif mutation == 'bad_child':
        graph[0]['children'] = [2]
    elif mutation == 'duplicate_child':
        graph[0]['children'] = [2, 2]
    elif mutation == 'missing_children':
        del graph[0]['children']
    elif mutation == 'invalid_child':
        graph[0]['children'] = [dict(x=2)]
    elif mutation == 'nonadjacent_child':
        graph[0]['children'] = [dict(x=0, y=0)]
    elif mutation == 'bad_coordinate':
        graph.append(dict(x=8, y=0, symbol='M', children=[]))
    else:
        ss.update(first_node_chosen=True, current_node=dict(x=4, y=3))
    adapter = RealGameAdapter(global_map_view=True)
    adapter.update(dict(in_game=True, available_commands=['choose'], game_state=dict(
        screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[],
        map=graph, choice_list=['first'], screen_state=ss)))
    with pytest.raises(ValueError):
        adapter._options('map')


def test_map_labels_cannot_substitute_for_structured_coordinates():
    adapter = RealGameAdapter()
    adapter.update(dict(in_game=True, available_commands=['choose'], game_state=dict(
        screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[],
        choice_list=['Monster', 'Rest'], screen_state={})))
    with pytest.raises(ValueError, match='structured next_nodes'):
        adapter.legal_semantic_actions()


def test_bridge_boss_retains_current_column_and_ui_index_zero():
    node = dict(x=4, y=14, symbol='R', children=[])
    adapter = RealGameAdapter(global_map_view=True)
    frame = dict(in_game=True, available_commands=['choose'], game_state=dict(
        screen_type='MAP', room_phase='COMPLETE', potions=[], relics=[], map=[node],
        choice_list=['boss'], screen_state=dict(current_node=node, boss_available=True)))
    adapter.update(frame)
    assert adapter.command_for_semantic(ga.map_node(4)) == 'CHOOSE 0'
    obs = encode.encode(dict(phase='map', options=adapter._options('map')), global_map_view=True)
    assert obs['map_full_features'][102, 3] == 1
    for current in [None, dict(x=4, y=13), dict(x=-1, y=14)]:
        bad = deepcopy(frame)
        bad['game_state']['screen_state']['current_node'] = current
        adapter.update(bad)
        with pytest.raises(ValueError):
            adapter.legal_semantic_actions()

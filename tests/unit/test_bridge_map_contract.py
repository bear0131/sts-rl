"""BRIDGE40: full candidates survive the frozen three-token map preview."""
from copy import deepcopy

import numpy as np
import pytest

from bridge.real_game_env import RealDecision, RealGameAdapter
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
        with pytest.raises(ValueError, match='capacity 3'):
            encode.encode(state)


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


def test_map_word_table_documents_preview_vs_complete_candidates():
    from bridge.schemas.codec_contract import complete_word_tables
    fields = complete_word_tables()['map_transport_fields']
    assert set(fields) == {'next_nodes', 'next_nodes_order', 'x', 'y', 'symbol',
                           'children', 'current_node', 'boss_available',
                           'map_full_rooms', 'map_full_features', 'map_node_rooms', 'map_node_x'}


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


def flight_map_frame(counter=1):
    nodes = [dict(x=x, y=6, symbol='R') for x in (6, 1)]
    return dict(in_game=True, available_commands=['choose'], game_state=dict(
        seed=42, act=1, floor=6, screen_type='MAP', room_phase='COMPLETE', potions=[],
        relics=[] if counter is None else [dict(id='WingedGreaves', counter=counter)],
        map=[dict(x=1, y=5, symbol='R', children=[dict(x=1, y=6)])]
            + [dict(n, children=[]) for n in nodes],
        choice_list=['x=6', 'x=1'], screen_state=dict(
            next_nodes=nodes, first_node_chosen=True, boss_available=False,
            current_node=dict(x=1, y=5, symbol='R'))))


@pytest.mark.parametrize('counter,column,after', [(1, 6, -2), (2, 6, 1), (3, 6, 2),
    (None, 1, None), (0, 1, 0), (-2, 1, -2), (1, 1, 1), (3, 1, 3)])
def test_map_receipt_waits_for_move_and_exact_charge_effect(counter, column, after):
    from bridge.client.agent_orchestrator import MapCommandAck
    frame = flight_map_frame(counter)
    adapter = RealGameAdapter()
    adapter.update(frame)
    decision = RealDecision(adapter.command_for_semantic(ga.map_node(column)),
                            'test', semantic=ga.map_node(column))
    ack = MapCommandAck()
    ack.arm(frame, decision, 0, adapter)
    assert ack.waiting(frame, 1)  # Never click twice on a delayed old frame.
    post = deepcopy(frame)
    post['game_state'].update(floor=7, screen_type='REST')
    if after is not None:
        post['game_state']['relics'][0]['counter'] = after
    assert not ack.waiting(post, 2)
    assert ack.pending is None


@pytest.mark.parametrize('counter', [None, 0, -2])
def test_map_receipt_rejects_free_off_path_move(counter):
    adapter = RealGameAdapter()
    adapter.update(flight_map_frame(counter))
    with pytest.raises(ValueError, match='requires a Wing Boots charge'):
        adapter.map_move_expectation(6)


@pytest.mark.parametrize('mutation', ['counter_missing', 'counter_unknown', 'duplicate_relic',
    'missing_edges', 'missing_flags', 'wrong_row'])
def test_map_receipt_does_not_guess_missing_connection_or_counter(mutation):
    frame = flight_map_frame()
    gs = frame['game_state']
    if mutation == 'counter_missing':
        del gs['relics'][0]['counter']
    elif mutation == 'counter_unknown':
        gs['relics'][0]['counter'] = 42
    elif mutation == 'duplicate_relic':
        gs['relics'] *= 2
    elif mutation == 'missing_edges':
        del gs['map'][0]['children']
    elif mutation == 'missing_flags':
        del gs['screen_state']['first_node_chosen']
    else:
        for node in gs['screen_state']['next_nodes']:
            node['y'] = 7
    adapter = RealGameAdapter()
    adapter.update(frame)
    with pytest.raises(ValueError):
        adapter.map_move_expectation(6)


def test_map_receipt_rejects_wrong_ui_index_wrong_cost_and_timeout():
    from bridge.client.agent_orchestrator import MapCommandAck
    frame = flight_map_frame()
    adapter = RealGameAdapter()
    adapter.update(frame)
    ack = MapCommandAck()
    with pytest.raises(ValueError, match='semantic destination'):
        ack.arm(frame, RealDecision('CHOOSE 1', 'test', semantic=ga.map_node(6)), 0, adapter)
    ack.arm(frame, RealDecision('CHOOSE 0', 'test', semantic=ga.map_node(6)), 0, adapter)
    post = deepcopy(frame)
    post['game_state']['floor'] += 1
    with pytest.raises(ValueError, match='connection cost'):
        ack.waiting(post, 1)
    assert ack.pending is not None
    with pytest.raises(RuntimeError, match='10 seconds'):
        ack.waiting(frame, 11)


@pytest.mark.parametrize('boss', [False, True])
def test_first_room_and_boss_move_never_charge_wing_boots(boss):
    frame = flight_map_frame()
    ss = frame['game_state']['screen_state']
    ss.update(first_node_chosen=boss, boss_available=boss,
              current_node=dict(x=1 if boss else 0, y=14 if boss else -1,
                                symbol='R', children=[]))
    if not boss:
        for node in ss['next_nodes']:
            node['y'] = 0
    adapter = RealGameAdapter()
    adapter.update(frame)
    assert adapter.map_move_expectation(1)['flight_counter'] == 1

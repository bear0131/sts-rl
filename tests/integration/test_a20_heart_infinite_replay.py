"""Natural A20 Heart run: renewable draw/energy/block, then banked-block victory."""
import json
from pathlib import Path

import pytest

from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv

pytestmark = pytest.mark.native


def test_natural_a20_heart_infinite_and_barricade_victory():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures' /
                          'a20_heart_infinite_seed4214060967.json').read_text())
    env = SlayTheSpireEnv(
        backend_factory=lambda: SimGameEnv(ascension=20, stop_after_act=4,
                                           global_map_view=True),
        combat_action_contract='native-v1', global_map_view=True)
    loop, bosses, heart_turns = [], [], set()
    carried = None
    try:
        env.reset(seed=fixture['reset_seed'])
        assert env.thick.gc.seed == fixture['game_seed']
        for index, (kind, action) in enumerate(fixture['actions']):
            assert not env.thick.is_over
            assert env.thick.gc.ascension == 20 and not env.thick.gc.skip_battles
            if env.thick.bc is not None:
                state = env.thick.obs_canonical()
                ids = tuple(m['id'] for m in env.thick.bc.to_dict()['monsters'])
                if env.thick.gc.floor_num >= 50 and (not bosses or ids != bosses[-1]):
                    bosses.append(ids)
                if ids == ('CORRUPT_HEART',):
                    assert env.thick.gc.blue_key and env.thick.gc.green_key and env.thick.gc.red_key
                    heart_turns.add(state['turn'])
                    if index >= fixture['cycle_start']:
                        assert state['player']['hp'] == 22
                    if state['turn'] == 6 and carried is None:
                        carried = state['player']['block']
                if fixture['cycle_start'] <= index <= fixture['cycle_end']:
                    assert state['global']['floor'] == 56 and state['turn'] == 5
                    assert state['player']['energy'] == 1
                    flags = env.thick.bc.public_player_power_flags()['flags']
                    assert flags['BARRICADE']
                    raw = env.thick.bc.to_dict()
                    assert dict(raw['player']['powers'])['Rage'] == 3
                    assert dict(raw['monsters'][0]['powers'])['Beat Of Death'] == 2
                    loop.append(state)
                    if index < fixture['cycle_end']:
                        assert kind == 'native' and action >> 29 == 0
                        card = state['hand'][action & 0xffff]
                        expected = (('Dropkick', 1) if (index - fixture['cycle_start']) % 2 == 0
                                    else ('Pommel Strike', 0))
                        assert (card['id'], card['cost']) == expected
                        assert card['up'] == 1

            if kind == 'native':
                assert action in env.native_combat_actions(), index
                result = env.step_native_combat(action)
            elif kind == 'key':
                assert action in env.thick.legal_key_rewards(), index
                result = env.step_key_reward(action)
            else:
                assert kind == 'thin' and env.action_masks()[action], index
                result = env.step(action)
            assert not result[3]
            assert result[2] == (index == len(fixture['actions']) - 1)

        assert bosses == [('CULTIST', 'CULTIST', 'AWAKENED_ONE'), ('DECA', 'DONU'),
                          ('SPIRE_SHIELD', 'SPIRE_SPEAR'), ('CORRUPT_HEART',)]
        assert heart_turns == set(range(1, 9))
        assert env.thick.run_stats() == fixture['expected']
        assert fixture['expected']['outcome'] == 'PLAYER_VICTORY'
        assert len(loop) == 601  # 600 plays, including the state after the last.
        assert (loop[0]['player']['block'], loop[-1]['player']['block']) == (56, 656)
        assert carried == 594  # Barricade retains block after Burn and 15 hits.

        def card_zones(state):
            return tuple(tuple(sorted((c['id'], c['up'], c.get('cost'))
                                      for c in state[zone]))
                         for zone in ('hand', 'draw', 'discard', 'exhaust'))

        # A pair restores every card-zone multiset, preserves energy and HP,
        # and gains two block despite Beat of Death. The damage cap can stop
        # damage, but does not stop renewable block production.
        steady = loop[fixture['steady_cycle_start'] - fixture['cycle_start']::2]
        assert len(steady) == 300
        assert all(card_zones(s) == card_zones(steady[0]) for s in steady)
        assert all(s['player']['hp'] == 22 and s['player']['energy'] == 1 for s in steady)
        assert all(b['player']['block'] == a['player']['block'] + 2
                   for a, b in zip(steady, steady[1:]))
        enemy_hp = [s['monsters'][0]['hp'] for s in steady]
        assert enemy_hp[0] > enemy_hp[-1] == 472
        assert all(a >= b for a, b in zip(enemy_hp, enemy_hp[1:]))
        assert enemy_hp[-100:] == [472] * 100
    finally:
        env.close()

"""Natural fixed-seed run: final A20 boss dies to a repeatable resource loop."""
import json
from pathlib import Path

import pytest

from env.core.run_env import SlayTheSpireEnv

pytestmark = pytest.mark.native


def test_natural_a20_run_finishes_second_boss_with_infinite():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures' /
                          'a20_infinite_seed4214060967.json').read_text())
    env = SlayTheSpireEnv(ascension=fixture['ascension'],
                         combat_action_contract='native-v1', global_map_view=True)
    loop = []
    bosses = []
    try:
        env.reset(seed=fixture['reset_seed'])
        assert env.thick.gc.seed == fixture['game_seed']
        for index, (kind, action) in enumerate(fixture['actions']):
            assert not env.thick.is_over
            assert env.thick.gc.ascension == 20 and not env.thick.gc.skip_battles
            if env.thick.bc is not None and env.thick.gc.floor_num >= 50:
                ids = tuple(m['id'] for m in env.thick.bc.to_dict()['monsters'])
                if not bosses or ids != bosses[-1]:
                    bosses.append(ids)
            if index >= fixture['cycle_start']:
                state = env.thick.obs_canonical()
                assert state['global']['floor'] == 51
                assert state['turn'] == 7 and state['player']['hp'] == 10
                assert kind == 'native' and action >> 29 == 0
                card = state['hand'][action & 0xffff]
                expected_card = ('Dropkick' if (index - fixture['cycle_start']) % 2 == 0
                                 else 'Pommel Strike')
                assert (card['id'], card['up'], card['cost']) == (expected_card, 1, 0)
                loop.append(state)
            if kind == 'native':
                assert action in env.native_combat_actions()
                _, _, terminal, truncated, _ = env.step_native_combat(action)
            else:
                assert kind == 'thin' and env.action_masks()[action]
                _, _, terminal, truncated, _ = env.step(action)
            assert not truncated
            assert terminal == (index == len(fixture['actions']) - 1)

        assert bosses == [('CULTIST', 'CULTIST', 'AWAKENED_ONE'), ('DECA', 'DONU')]
        assert env.thick.run_stats() == fixture['expected']
        assert fixture['expected']['outcome'] == 'PLAYER_VICTORY'
        assert len(loop) == 20

        # After bootstrap, every two plays recover the same card-zone multisets.
        # Energy grows by one, HP/turn are unchanged, and enemy HP decreases.
        def resource_cards(state):
            return tuple(tuple(sorted((c['id'], c['up'], c.get('cost'))
                                      for c in state[zone]))
                         for zone in ('hand', 'draw', 'discard', 'exhaust'))

        steady = loop[fixture['steady_cycle_start'] - fixture['cycle_start']::2]
        assert len(steady) == 9
        assert all(resource_cards(s) == resource_cards(steady[0]) for s in steady)
        assert [s['player']['energy'] for s in steady] == list(range(2, 11))
        hp = [sum(m['hp'] for m in s['monsters']) for s in steady]
        assert all(a > b for a, b in zip(hp, hp[1:]))
    finally:
        env.close()

"""RELIC01: Du-Vu Doll uses all curses in the current master deck."""
import subprocess

import pytest
import slaythespire as sts

from env.core import encode
from schemas.combat_state import normalize_sim_global
from tests.bridge_fixtures import native_master_card


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('variant', range(4), ids=[
    'relic-first', 'deck-first', 'relic-first-omamori', 'deck-first-omamori'])
def test_duvu_native_deck_lifecycle(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'duvu', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.mark.parametrize('ascension', [0, 20])
@pytest.mark.parametrize('setter', ['set_deck', 'set_deck_state'])
@pytest.mark.parametrize('relic_first', [False, True])
@pytest.mark.parametrize('counter', [-1, 0, 99])
def test_duvu_restored_deck_controls_export_and_attack(ascension, setter, relic_first, counter):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, ascension)
    gc.floor_num = 1
    relics = [dict(id='Du-Vu Doll', counter=counter), dict(id='Omamori', counter=2)]

    def restore(curses):
        ids = ['STRIKE_RED'] + curses
        if setter == 'set_deck':
            gc.set_deck([(c, 0) for c in ids])
        else:
            gc.set_deck_state([native_master_card(c) for c in ids])

    if relic_first:
        gc.set_relic_state(relics)
    else:
        restore(['PAIN'])
        gc.set_relic_state(relics)
    # Restore neither consumes Omamori nor replays acquisition effects.
    for curses in [['PAIN'], ['ASCENDERS_BANE', 'CURSE_OF_THE_BELL', 'NECRONOMICURSE'], []]:
        restore(curses)
        assert gc.relic_state() == [dict(id='Du-Vu Doll', counter=len(curses)), relics[1]]
        public = normalize_sim_global(gc)
        assert public['relic_counters'] == [len(curses), 2]
        obs = encode.encode({'phase': 'map', 'global': public})
        assert obs['relic_counters'][0] == pytest.approx(len(curses) / encode.COUNTER_SCALE)
        battle = sts.create_battle(gc, 'Cultist')
        assert dict(battle.to_dict()['player']['powers']).get('Strength', 0) == len(curses)
        hand = battle.card_state()['hand']
        strike = next(i for i, c in enumerate(hand) if c['id'] == 'STRIKE_RED')
        before = battle.to_dict()['monsters'][0]['hp']
        battle.execute(sts.Action(sts.ActionType.CARD, strike, 0))
        assert battle.to_dict()['monsters'][0]['hp'] == before - 6 - len(curses)

"""Native evidence behind public relic valuation; no whole-game claims."""
import pytest
import slaythespire as sts

from rl.evaluation.baselines.overworld_rules import relic_strength_sources


@pytest.mark.parametrize('cid', ['CORRUPTION', 'DARK_EMBRACE', 'ENTRENCH'])
@pytest.mark.parametrize('seed', [31, 42, 93])
def test_snecko_draw_randomizes_base_and_upgraded_cost_identically(combat_factory, cid, seed):
    hands = []
    for up in (0, 1):
        env = combat_factory([cid] * 5, seed=seed, relics=['Snecko Eye'])
        env.thick.gc.set_deck([(cid, up)] * 5)
        env.thick.bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
        hands.append(env.thick.bc.to_dict()['hand'])
    assert [c[1] for c in hands[0]] == [0] * 5
    assert [c[1] for c in hands[1]] == [1] * 5
    assert [c[2] for c in hands[0]] == [c[2] for c in hands[1]]
    assert all(0 <= c[2] <= 3 for c in hands[0])


@pytest.mark.parametrize('relic,counter,expected_strength,expected_sources', [
    ('Girya', 0, 0, 0), ('Girya', 3, 3, 1), ('Du-Vu Doll', 1, 1, 1),
    ('Brimstone', 0, 2, 1),
])
def test_native_public_relic_sources_match_starting_strength(
        combat_factory, relic, counter, expected_strength, expected_sources):
    env = combat_factory(['ASCENDERS_BANE'] + ['STRIKE_RED'] * 4)
    env.thick.gc.set_relic_state([{'id': relic, 'counter': counter}])
    env.thick.bc = sts.create_battle(env.thick.gc, 'Jaw Worm')
    env._invalidate_backend_cache()
    raw = env.thick.bc.to_dict()
    assert dict(raw['player']['powers']).get('Strength', 0) == expected_strength
    assert relic_strength_sources(env.thick.obs_canonical()) == expected_sources

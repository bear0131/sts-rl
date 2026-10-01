"""SIM01: preserve the missing power-card retaliation as an explicit failure."""
import pytest
import slaythespire as sts

pytestmark = pytest.mark.native


@pytest.mark.parametrize('ascension,expected_gain', [(0, 1), (20, 2)])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                  reason='SIM01: onUsePowerCard leaves the Curiosity trigger commented out')
def test_awakened_curiosity_gains_strength_after_player_power(
        combat_factory, ascension, expected_gain):
    env = combat_factory(['INFLAME'] * 10, encounter='Awakened One')
    if ascension:
        env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, ascension)
        env.thick.gc.set_deck([('INFLAME', 0)] * 10)
        env.thick.bc = sts.create_battle(env.thick.gc, 'Awakened One')
    battle = env.thick.bc
    battle.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'block': 0},
        'hand': ['INFLAME'], 'draw': ['DEFEND_RED'] * 5,
        'discard': [], 'exhaust': [],
    })
    before = dict(battle.to_dict()['monsters'][2]['powers'])
    assert before['Curiosity'] == expected_gain
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    after = battle.to_dict()
    assert dict(after['player']['powers'])['Strength'] == 2
    assert dict(after['monsters'][2]['powers']).get('Strength', 0) == (
        before.get('Strength', 0) + expected_gain
    )

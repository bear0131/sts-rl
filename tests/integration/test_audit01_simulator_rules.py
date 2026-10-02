"""Read-only AUDIT01 findings; unfixed rules are explicit strict xfails."""
import pytest
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _battle(combat_factory, hand):
    env = combat_factory(['STRIKE_RED'] * 10)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': hand, 'draw': ['DEFEND_RED'] * 10,
        'discard': [], 'exhaust': [], 'player': {'energy': 20},
        'monsters': [{'max_hp': 500}]})
    battle.apply_custom_state({'monsters': [{'hp': 500}]})
    return battle


@pytest.mark.parametrize('upgraded,amount', [(0, 9), (1, 15)])
@pytest.mark.xfail(strict=True, reason='SIM02: Dark Shackles adds positive permanent Strength')
def test_dark_shackles_reduces_strength_for_one_turn(combat_factory, upgraded, amount):
    battle = _battle(combat_factory, [('DARK_SHACKLES', upgraded)])
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    powers = dict(battle.to_dict()['monsters'][0]['powers'])
    assert powers.get('Strength', 0) == -amount
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert dict(battle.to_dict()['monsters'][0]['powers']).get('Strength', 0) == 0


@pytest.mark.xfail(strict=True, reason='SIM03: new Panache counter starts at zero')
def test_panache_does_not_trigger_on_first_following_card(combat_factory):
    battle = _battle(combat_factory, ['PANACHE', 'ANGER'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    before = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert before - battle.to_dict()['monsters'][0]['hp'] == 6


@pytest.mark.xfail(strict=True, reason='SIM03: Panache fails to reset after each five cards')
def test_panache_triggers_only_on_each_fifth_card(combat_factory):
    battle = _battle(combat_factory, ['PANACHE'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({'hand': ['ANGER'] * 10})
    damage = []
    for _ in range(10):
        before = battle.to_dict()['monsters'][0]['hp']
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        damage.append(before - battle.to_dict()['monsters'][0]['hp'])
    assert damage == [6, 6, 6, 6, 16, 6, 6, 6, 6, 16]


@pytest.mark.xfail(strict=True, reason='SIM04: upgraded Forethought still selects exactly one card')
def test_upgraded_forethought_allows_multiple_cards(combat_factory):
    battle = _battle(combat_factory, [('FORETHOUGHT', 1), 'STRIKE_RED', 'DEFEND_RED'])
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    select = battle.to_dict()['select']
    assert select['context'] == 'FORETHOUGHT'
    assert select['can_pick_any'] and select['multi']


@pytest.mark.parametrize('card', [
    'ANGER', 'BASH', 'BODY_SLAM', 'CARNAGE', 'CLOTHESLINE', 'HEAVY_BLADE',
    'IRON_WAVE', 'RAMPAGE', 'THUNDERCLAP', 'TWIN_STRIKE', 'UPPERCUT',
    'DEFEND_RED', 'BLOODLETTING', 'DISARM', 'ENTRENCH', 'FLAME_BARRIER',
    'FLEX', 'GHOSTLY_ARMOR', 'IMPERVIOUS', 'METALLICIZE', 'INFLAME',
    'BARRICADE', 'CORRUPTION', 'DEMON_FORM',
])
@pytest.mark.parametrize('upgraded', [0, 1])
def test_equivalent_deterministic_plays_reach_same_public_future(combat_factory, card, upgraded):
    battle = _battle(combat_factory, [(card, upgraded), 'DEFEND_RED', (card, upgraded)])
    assert battle.equivalent_play_slots()[2] == 0
    left, right = battle.clone(), battle.clone()
    left.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    right.execute(sts.Action(sts.ActionType.CARD, 2, 0))
    assert left.nosl_future_key() == right.nosl_future_key()
    assert left.to_dict()['player'] == right.to_dict()['player']
    assert left.to_dict()['monsters'] == right.to_dict()['monsters']

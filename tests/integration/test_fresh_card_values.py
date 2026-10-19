"""BRIDGE28: fresh values and upgrade counts survive construction and injection."""
import pytest
import slaythespire as sts

from bridge.schemas.combat_state import normalize_sim


@pytest.mark.parametrize('card,misc', [('RITUAL_DAGGER', 15), ('GENETIC_ALGORITHM', 1),
                                     ('RAMPAGE', 0), ('SEARING_BLOW', 0)])
def test_fresh_persistent_card_value(card, misc):
    assert sts.Card(getattr(sts.CardId, card)).misc == misc


@pytest.mark.parametrize('upgrade', [0, 1])
@pytest.mark.parametrize('injected', [False, True])
def test_fresh_ritual_dagger_deals_fifteen_damage(combat_factory, upgrade, injected):
    env = combat_factory(['RITUAL_DAGGER'] * 5, 'Cultist')
    env.thick.gc.set_deck([('RITUAL_DAGGER', upgrade)] * 5)
    battle = sts.create_battle(env.thick.gc, 'Cultist')
    if injected:
        battle.apply_custom_state({'hand': [('RITUAL_DAGGER', upgrade)]})
    battle.apply_custom_state({'monsters': [{'hp': 40, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 25


@pytest.mark.parametrize('upgrade,grown', [(0, 18), (1, 20)])
def test_ritual_dagger_kill_growth_commits_to_master_card(short_combat_env_type, upgrade, grown):
    env = short_combat_env_type()
    try:
        env.reset(seed=901)
        env.thick.gc.set_deck([('RITUAL_DAGGER', upgrade)])
        battle = sts.create_battle(env.thick.gc, 'Cultist')
        battle.apply_custom_state({'monsters': [{'hp': 1, 'block': 0}]})
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        assert battle.to_dict()['monsters'][0]['hp'] <= 0
        battle.exit_battle(env.thick.gc)
        assert env.thick.gc.deck[0].misc == grown
    finally:
        env.close()


@pytest.mark.parametrize('upgrade,damage', [(0, 12), (1, 16), (2, 21), (3, 27)])
@pytest.mark.parametrize('injected', [False, True])
def test_searing_blow_upgrade_count_controls_actual_damage(combat_factory, upgrade, damage, injected):
    env = combat_factory(['SEARING_BLOW'] * 5, 'Cultist')
    env.thick.gc.set_deck([('SEARING_BLOW', upgrade)] * 5)
    battle = sts.create_battle(env.thick.gc, 'Cultist')
    if injected:
        battle.apply_custom_state({'hand': [('SEARING_BLOW', upgrade)]})
    assert battle.to_dict()['hand'][0][1] == upgrade
    battle.apply_custom_state({'monsters': [{'hp': 40, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 40 - damage


@pytest.mark.parametrize('pile', ['hand', 'draw', 'discard', 'exhaust'])
@pytest.mark.parametrize('upgrade', [0, 1, 3])
def test_injected_piles_preserve_searing_count_in_bridge_schema(combat_factory, pile, upgrade):
    battle = combat_factory(['DEFEND_RED'] * 5, 'Cultist').thick.bc
    battle.apply_custom_state({pile: [('SEARING_BLOW', upgrade)]})
    normalized = normalize_sim(battle.to_dict())
    assert normalized[pile][0]['id'] == 'Searing Blow'
    assert normalized[pile][0]['up'] == upgrade


@pytest.mark.parametrize('seed,relics,expected', [(32, [], 1), (54, [], 0),
                                               (54, ['Molten Egg 2'], 1)])
def test_generated_searing_reward_retains_random_or_egg_upgrade(seed, relics, expected):
    # Fixed card RNG exercises Card(id, upgraded), unlike set_deck's upgrade loop.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    gc.act = 3
    gc.set_relics(relics)
    reward = gc._dbg_card_reward()
    assert next(up for card, up in reward if card == 'SEARING_BLOW') == expected


@pytest.mark.parametrize('upgrade', [0, 1])
def test_existing_ritual_dagger_growth_is_not_reset_to_fresh_value(combat_factory, upgrade):
    gc = combat_factory(['DEFEND_RED'] * 5, 'Cultist').thick.gc
    gc.set_deck([])
    card = sts.Card(sts.CardId.RITUAL_DAGGER)
    card.misc = 30
    if upgrade:
        card.upgrade()
    gc.obtain_card(card)
    battle = sts.create_battle(gc, 'Cultist')
    battle.apply_custom_state({'monsters': [{'hp': 40, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 10
    assert gc.deck[0].misc == 30


@pytest.mark.parametrize('initial_upgrades,expected,damage', [(0, 1, 16), (1, 1, 16), (3, 3, 27)])
def test_molten_egg_obtain_preserves_existing_searing_upgrades(initial_upgrades, expected, damage):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.set_deck([])
    gc.set_relics(['Molten Egg 2'])
    card = sts.Card(sts.CardId.SEARING_BLOW)
    for _ in range(initial_upgrades):
        card.upgrade()
    gc.obtain_card(card)
    assert gc.deck_ids() == [('SEARING_BLOW', expected)]
    battle = sts.create_battle(gc, 'Cultist')
    battle.apply_custom_state({'monsters': [{'hp': 40, 'block': 0}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert battle.to_dict()['monsters'][0]['hp'] == 40 - damage

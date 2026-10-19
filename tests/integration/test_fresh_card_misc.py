"""BRIDGE28: fresh persistent card values follow the actual card constructor."""
import pytest
import slaythespire as sts

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('card,misc', [('RITUAL_DAGGER', 15), ('GENETIC_ALGORITHM', 1)])
def test_fresh_persistent_card_misc_matches_java_constructor(card, misc):
    assert sts.Card(getattr(sts.CardId, card)).misc == misc


def test_every_native_fresh_card_misc_has_an_explicit_matching_word_table():
    from bridge.schemas.codec_contract import complete_word_tables
    for native, row in complete_word_tables()['native_fresh_card_misc'].items():
        assert sts.Card(getattr(sts.CardId, native)).misc == row['initial_misc']


@pytest.mark.parametrize('upgrade', [0, 1])
def test_fresh_ritual_dagger_damage_is_not_its_growth_amount(combat_factory, upgrade):
    env = combat_factory(['RITUAL_DAGGER'] * 5, 'Cultist')
    env.thick.gc.set_deck([('RITUAL_DAGGER', upgrade)] * 5)
    battle = sts.create_battle(env.thick.gc, 'Cultist')
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

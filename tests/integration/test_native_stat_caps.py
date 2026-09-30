"""Player Strength and Dexterity use the game's bounded power contract."""

import slaythespire as sts


def test_limit_break_respects_strength_cap():
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.floor_num = 1
    gc.set_deck([("LIMIT_BREAK", 1)] * 10)
    battle = sts.create_battle(gc, "Cultist")
    battle.set_player_strength(998)
    action = next(
        action for action in battle.legal_actions()
        if action.action_type == sts.ActionType.CARD
    )
    battle.execute(action)
    assert dict(battle.to_dict()["player"]["powers"])["Strength"] == 999


def test_external_strength_setter_respects_upper_bound():
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.floor_num = 1
    gc.set_deck([("STRIKE_RED", 0)] * 10)
    battle = sts.create_battle(gc, "Cultist")
    battle.set_player_strength(2_000_000)
    assert dict(battle.to_dict()["player"]["powers"])["Strength"] == 999
    battle.set_player_status("DEXTERITY", 2_000_000)
    assert dict(battle.to_dict()["player"]["powers"])["Dexterity"] == 999
    battle.set_monster_status(0, "STRENGTH", 2_000_000)
    assert dict(battle.to_dict()["monsters"][0]["powers"])["Strength"] == 999

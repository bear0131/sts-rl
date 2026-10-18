import pytest

from bridge.real_game_env import RealGameAdapter, _neow_option_from_text


@pytest.mark.parametrize('text,bonus', [
    ('Choose a Card to obtain', 0), ('Obtain a random rare Card', 1),
    ('Remove a Card from your deck', 2), ('Upgrade a Card', 3),
    ('Transform a Card', 4), ('Choose a colorless Card to obtain', 5),
    ('Obtain 3 random Potions', 6), ('Obtain a random common Relic', 7),
    ('Max HP +8', 8), ('Enemies in your next three combats have 1 HP', 9),
    ('Obtain 100 Gold', 10), ('Choose a rare colorless Card to obtain', 11),
    ('Remove 2 Cards', 12), ('Obtain a random rare Relic', 13),
    ('Choose a rare Card to obtain', 14), ('Gain 250 Gold', 15),
    ('Transform 2 Cards', 16),
])
def test_stock_english_neow_bonus_ids(text, bonus):
    assert _neow_option_from_text(text) == {'bonus_id': bonus, 'drawback_id': 1}


@pytest.mark.parametrize('text,bonus,drawback', [
    ('[ #g升级一张牌 ]', 3, 1),
    ('[ 接下来 3 场战斗中的敌人将只有 1 点生命 ]', 9, 1),
    ('[ 失去所有金币。 去除 2 张牌 ]', 12, 3),
    ('[ 随机获得一个普通遗物 ]', 7, 1),
    ('[ 获得 100 金币 ]', 10, 1),
    ('[ 失去所有金币。 获得一个稀有遗物 ]', 13, 3),
    ('[ 失去你的初始遗物， 获得一件随机Boss遗物。 ]', 18, 6),
    ('[ Lose your starting Relic Obtain a random boss Relic ]', 18, 6),
    ('[ Lose 8 Max HP Max HP +16 ]', 17, 2),
    ('[ 受到 24 点伤害。 最大生命 +16 ]', 17, 5),
    ('[ Obtain a Curse Choose a rare Card to obtain ]', 14, 4),
])
def test_neow_drawbacks_and_real_chinese_frames(text, bonus, drawback):
    assert _neow_option_from_text(text) == {'bonus_id': bonus, 'drawback_id': drawback}


def test_neow_adapter_uses_real_text_and_preserves_structured_ids():
    adapter = RealGameAdapter(neow_observation=True)
    opts = adapter._neow_options({'options': [
        {'text': 'Upgrade a Card'}, {'text': 'Obtain 100 Gold'},
        {'bonus_id': 17, 'drawback_id': 4},
        {'text': 'Lose your starting Relic Obtain a random boss Relic'},
    ]})
    assert opts == [dict(bonus_id=b, drawback_id=d) for b,d in [(3,1),(10,1),(17,4),(18,6)]]


def test_unknown_neow_offer_fails_instead_of_silent_zeros():
    with pytest.raises(ValueError, match='unsupported Neow'):
        RealGameAdapter(neow_observation=True)._neow_options({'options':[{'text':'unknown offer'}]})

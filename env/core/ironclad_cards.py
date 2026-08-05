"""Wiki-backed Ironclad card annotations used by the observation encoder.

The source of truth is the Slay the Spire Wiki (wiki.gg), specifically
``Module:Cards/data``. Values are written as ``(base, upgraded)``. Damage is
per hit and block is per application; dynamic behavior is described in the
bilingual effect text. X-cost is represented by ``X_COST``.
"""

from dataclasses import dataclass


ATTACK = "attack"
SKILL = "skill"
POWER = "power"
X_COST = -1


@dataclass(frozen=True)
class CardInfo:
    name_zh: str
    kind: str
    cost: tuple[int, int]
    damage: tuple[int, int]
    block: tuple[int, int]
    effect_en: str
    effect_zh: str


def _c(name_zh, kind, cost, damage=(0, 0), block=(0, 0), en="", zh=""):
    return CardInfo(name_zh, kind, cost, damage, block, en, zh)


# CommunicationMod card id -> wiki-backed annotation. This is the single
# machine-readable card-information table; docs/ironclad_card_table.md is its
# human-readable bilingual rendering.
IRONCLAD_CARDS: dict[str, CardInfo] = {
    "Bash": _c("痛击", ATTACK, (2, 2), (8, 10), en="Deal 8/10 damage. Apply 2/3 Vulnerable.", zh="造成8/10点伤害。施加2/3层易伤。"),
    "Defend_R": _c("防御", SKILL, (1, 1), block=(5, 8), en="Gain 5/8 Block.", zh="获得5/8点格挡。"),
    "Strike_R": _c("打击", ATTACK, (1, 1), (6, 9), en="Deal 6/9 damage.", zh="造成6/9点伤害。"),
    "Anger": _c("愤怒", ATTACK, (0, 0), (6, 8), en="Deal 6/8 damage. Add a copy to the discard pile.", zh="造成6/8点伤害。将一张本牌的复制加入弃牌堆。"),
    "Armaments": _c("武装", SKILL, (1, 1), block=(5, 5), en="Gain 5 Block. Upgrade one/all cards in hand for this combat.", zh="获得5点格挡。本场战斗中升级手牌中的一张/所有牌。"),
    "Body Slam": _c("全身撞击", ATTACK, (1, 0), en="Deal damage equal to current Block.", zh="造成等同于当前格挡值的伤害。"),
    "Clash": _c("交锋", ATTACK, (0, 0), (14, 18), en="Playable only if every card in hand is an Attack. Deal 14/18 damage.", zh="仅当所有手牌均为攻击牌时可打出。造成14/18点伤害。"),
    "Cleave": _c("顺劈斩", ATTACK, (1, 1), (8, 11), en="Deal 8/11 damage to all enemies.", zh="对所有敌人造成8/11点伤害。"),
    "Clothesline": _c("金刚臂", ATTACK, (2, 2), (12, 14), en="Deal 12/14 damage. Apply 2/3 Weak.", zh="造成12/14点伤害。施加2/3层虚弱。"),
    "Flex": _c("活动肌肉", SKILL, (0, 0), en="Gain 2/4 Strength; lose it at end of turn.", zh="获得2/4点力量，并在回合结束时失去这些力量。"),
    "Havoc": _c("破灭", SKILL, (1, 0), en="Play the top draw-pile card and Exhaust it.", zh="打出抽牌堆顶牌并将其消耗。"),
    "Headbutt": _c("头槌", ATTACK, (1, 1), (9, 12), en="Deal 9/12 damage. Put a discard-pile card on top of draw pile.", zh="造成9/12点伤害。将弃牌堆中的一张牌置于抽牌堆顶。"),
    "Heavy Blade": _c("重刃", ATTACK, (2, 2), (14, 14), en="Deal 14 damage. Strength affects it 3/5 times.", zh="造成14点伤害。力量对本牌生效3/5倍。"),
    "Iron Wave": _c("铁斩波", ATTACK, (1, 1), (5, 7), (5, 7), en="Gain 5/7 Block. Deal 5/7 damage.", zh="获得5/7点格挡。造成5/7点伤害。"),
    "Perfected Strike": _c("完美打击", ATTACK, (2, 2), (6, 6), en="Deal 6 damage, plus 2/3 per card containing 'Strike'.", zh="造成6点伤害。每张名称中含“打击”的牌使伤害增加2/3点。"),
    "Pommel Strike": _c("剑柄打击", ATTACK, (1, 1), (9, 10), en="Deal 9/10 damage. Draw 1/2 cards.", zh="造成9/10点伤害。抽1/2张牌。"),
    "Shrug It Off": _c("耸肩无视", SKILL, (1, 1), block=(8, 11), en="Gain 8/11 Block. Draw 1 card.", zh="获得8/11点格挡。抽1张牌。"),
    "Sword Boomerang": _c("飞剑回旋镖", ATTACK, (1, 1), (3, 3), en="Deal 3 damage to a random enemy 3/4 times.", zh="对随机敌人造成3点伤害，共3/4次。"),
    "Thunderclap": _c("闪电霹雳", ATTACK, (1, 1), (4, 7), en="Deal 4/7 damage and apply 1 Vulnerable to all enemies.", zh="对所有敌人造成4/7点伤害并施加1层易伤。"),
    "True Grit": _c("坚毅", SKILL, (1, 1), block=(7, 9), en="Gain 7/9 Block. Exhaust one random/chosen card.", zh="获得7/9点格挡。随机/选择消耗一张牌。"),
    "Twin Strike": _c("双重打击", ATTACK, (1, 1), (5, 7), en="Deal 5/7 damage twice.", zh="造成2次5/7点伤害。"),
    "Warcry": _c("战吼", SKILL, (0, 0), en="Draw 1/2 cards. Put a hand card on top of draw pile. Exhaust.", zh="抽1/2张牌。将一张手牌置于抽牌堆顶。消耗。"),
    "Wild Strike": _c("狂野打击", ATTACK, (1, 1), (12, 17), en="Deal 12/17 damage. Shuffle a Wound into draw pile.", zh="造成12/17点伤害。将一张伤口洗入抽牌堆。"),
    "Battle Trance": _c("战斗专注", SKILL, (0, 0), en="Draw 3/4 cards. Cannot draw more cards this turn.", zh="抽3/4张牌。本回合不能再抽牌。"),
    "Blood For Blood": _c("以血还血", ATTACK, (4, 3), (18, 22), en="Costs 1 less each time HP is lost this combat. Deal 18/22 damage.", zh="本场战斗中每次失去生命，费用降低1。造成18/22点伤害。"),
    "Bloodletting": _c("放血", SKILL, (0, 0), en="Lose 3 HP. Gain 2/3 Energy.", zh="失去3点生命。获得2/3点能量。"),
    "Burning Pact": _c("燃烧契约", SKILL, (1, 1), en="Exhaust 1 card. Draw 2/3 cards.", zh="消耗1张牌。抽2/3张牌。"),
    "Carnage": _c("残杀", ATTACK, (2, 2), (20, 28), en="Ethereal. Deal 20/28 damage.", zh="虚无。造成20/28点伤害。"),
    "Combust": _c("燃烧", POWER, (1, 1), en="At end of turn lose 1 HP and deal 5/7 damage to all enemies.", zh="回合结束时失去1点生命，并对所有敌人造成5/7点伤害。"),
    "Dark Embrace": _c("黑暗之拥", POWER, (2, 1), en="Whenever a card is Exhausted, draw 1 card.", zh="每当一张牌被消耗时，抽1张牌。"),
    "Disarm": _c("缴械", SKILL, (1, 1), en="Enemy loses 2/3 Strength. Exhaust.", zh="敌人失去2/3点力量。消耗。"),
    "Dropkick": _c("飞身踢", ATTACK, (1, 1), (5, 8), en="Deal 5/8 damage. If target is Vulnerable, gain 1 Energy and draw 1.", zh="造成5/8点伤害。若目标有易伤，获得1点能量并抽1张牌。"),
    "Dual Wield": _c("双持", SKILL, (1, 1), en="Choose an Attack or Power; add 1/2 copies to hand.", zh="选择一张攻击牌或能力牌，将1/2张复制加入手牌。"),
    "Entrench": _c("巩固", SKILL, (2, 1), en="Double current Block.", zh="使当前格挡翻倍。"),
    "Evolve": _c("进化", POWER, (1, 1), en="Whenever a Status is drawn, draw 1/2 additional cards.", zh="每当抽到状态牌时，额外抽1/2张牌。"),
    "Feel No Pain": _c("无惧疼痛", POWER, (1, 1), en="Whenever a card is Exhausted, gain 3/4 Block.", zh="每当一张牌被消耗时，获得3/4点格挡。"),
    "Fire Breathing": _c("火焰吐息", POWER, (1, 1), en="When a Status or Curse is drawn, deal 6/10 damage to all enemies.", zh="每当抽到状态牌或诅咒牌时，对所有敌人造成6/10点伤害。"),
    "Flame Barrier": _c("火焰屏障", SKILL, (2, 2), block=(12, 16), en="Gain 12/16 Block. This turn, attackers take 4/6 damage.", zh="获得12/16点格挡。本回合每次受到攻击时，对攻击者造成4/6点伤害。"),
    "Ghostly Armor": _c("幽灵铠甲", SKILL, (1, 1), block=(10, 13), en="Ethereal. Gain 10/13 Block.", zh="虚无。获得10/13点格挡。"),
    "Hemokinesis": _c("御血术", ATTACK, (1, 1), (15, 20), en="Lose 2 HP. Deal 15/20 damage.", zh="失去2点生命。造成15/20点伤害。"),
    "Infernal Blade": _c("地狱之刃", SKILL, (1, 0), en="Add a random Attack to hand; it costs 0 this turn. Exhaust.", zh="将一张随机攻击牌加入手牌；其本回合费用为0。消耗。"),
    "Inflame": _c("燃烧", POWER, (1, 1), en="Gain 2/3 Strength.", zh="获得2/3点力量。"),
    "Intimidate": _c("威吓", SKILL, (0, 0), en="Apply 1/2 Weak to all enemies. Exhaust.", zh="对所有敌人施加1/2层虚弱。消耗。"),
    "Metallicize": _c("金属化", POWER, (1, 1), en="At end of turn, gain 3/4 Block.", zh="回合结束时获得3/4点格挡。"),
    "Power Through": _c("硬撑", SKILL, (1, 1), block=(15, 20), en="Add 2 Wounds to hand. Gain 15/20 Block.", zh="将2张伤口加入手牌。获得15/20点格挡。"),
    "Pummel": _c("连续拳", ATTACK, (1, 1), (2, 2), en="Deal 2 damage 4/5 times. Exhaust.", zh="造成4/5次2点伤害。消耗。"),
    "Rage": _c("狂怒", SKILL, (0, 0), en="This turn, gain 3/5 Block whenever an Attack is played.", zh="本回合每打出一张攻击牌，获得3/5点格挡。"),
    "Rampage": _c("暴走", ATTACK, (1, 1), (8, 8), en="Deal 8 damage. Increase this card's damage by 5/8 this combat.", zh="造成8点伤害。本场战斗中本牌每次打出后伤害增加5/8点。"),
    "Reckless Charge": _c("鲁莽冲锋", ATTACK, (0, 0), (7, 10), en="Deal 7/10 damage. Shuffle a Dazed into draw pile.", zh="造成7/10点伤害。将一张晕眩洗入抽牌堆。"),
    "Rupture": _c("撕裂", POWER, (1, 1), en="Whenever a card causes HP loss, gain 1/2 Strength.", zh="每当因卡牌效果失去生命时，获得1/2点力量。"),
    "Searing Blow": _c("灼热攻击", ATTACK, (2, 2), (12, 16), en="Deal 12/16 damage. Can be upgraded repeatedly.", zh="造成12/16点伤害。可重复升级。"),
    "Second Wind": _c("重振精神", SKILL, (1, 1), block=(5, 7), en="Exhaust all non-Attacks in hand; gain 5/7 Block per card.", zh="消耗手牌中所有非攻击牌；每张获得5/7点格挡。"),
    "Seeing Red": _c("盛怒", SKILL, (1, 0), en="Gain 2 Energy. Exhaust.", zh="获得2点能量。消耗。"),
    "Sentinel": _c("哨卫", SKILL, (1, 1), block=(5, 8), en="Gain 5/8 Block. If Exhausted, gain 2/3 Energy.", zh="获得5/8点格挡。若本牌被消耗，获得2/3点能量。"),
    "Sever Soul": _c("断魂斩", ATTACK, (2, 2), (16, 22), en="Exhaust all non-Attacks in hand. Deal 16/22 damage.", zh="消耗手牌中所有非攻击牌。造成16/22点伤害。"),
    "Shockwave": _c("震荡波", SKILL, (2, 2), en="Apply 3/5 Weak and Vulnerable to all enemies. Exhaust.", zh="对所有敌人施加3/5层虚弱和易伤。消耗。"),
    "Spot Weakness": _c("观察弱点", SKILL, (1, 1), en="If target intends to attack, gain 3/4 Strength.", zh="若目标意图攻击，获得3/4点力量。"),
    "Uppercut": _c("上勾拳", ATTACK, (2, 2), (13, 13), en="Deal 13 damage. Apply 1/2 Weak and 1/2 Vulnerable.", zh="造成13点伤害。施加1/2层虚弱和1/2层易伤。"),
    "Whirlwind": _c("旋风斩", ATTACK, (X_COST, X_COST), (5, 8), en="Deal 5/8 damage to all enemies X times.", zh="对所有敌人造成X次5/8点伤害。"),
    "Barricade": _c("壁垒", POWER, (3, 2), en="Block is not removed at start of turn.", zh="回合开始时不再失去格挡。"),
    "Berserk": _c("狂暴", POWER, (0, 0), en="Gain 2/1 Vulnerable. At start of turn, gain 1 Energy.", zh="获得2/1层易伤。每回合开始时获得1点能量。"),
    "Bludgeon": _c("重锤", ATTACK, (3, 3), (32, 42), en="Deal 32/42 damage.", zh="造成32/42点伤害。"),
    "Brutality": _c("残暴", POWER, (0, 0), en="Upgraded is Innate. At start of turn lose 1 HP and draw 1.", zh="升级后固有。每回合开始时失去1点生命并抽1张牌。"),
    "Corruption": _c("腐化", POWER, (3, 2), en="Skills cost 0 and are Exhausted when played.", zh="技能牌费用变为0，但打出后会被消耗。"),
    "Demon Form": _c("恶魔形态", POWER, (3, 3), en="At start of turn, gain 2/3 Strength.", zh="每回合开始时获得2/3点力量。"),
    "Double Tap": _c("双发", SKILL, (1, 1), en="This turn, the next 1/2 Attacks are played twice.", zh="本回合接下来1/2张攻击牌会打出两次。"),
    "Exhume": _c("发掘", SKILL, (1, 0), en="Put a non-Exhume card from Exhaust pile into hand. Exhaust.", zh="从消耗堆选择一张非发掘牌加入手牌。消耗。"),
    "Feed": _c("吞噬", ATTACK, (1, 1), (10, 12), en="Deal 10/12 damage. Fatal: gain 3/4 Max HP. Exhaust.", zh="造成10/12点伤害。致命：最大生命增加3/4。消耗。"),
    "Fiend Fire": _c("恶魔之焰", ATTACK, (2, 2), (7, 10), en="Exhaust the rest of hand; deal 7/10 damage per card. Exhaust.", zh="消耗其余所有手牌；每张造成7/10点伤害。消耗。"),
    "Immolate": _c("燔祭", ATTACK, (2, 2), (21, 28), en="Deal 21/28 damage to all enemies. Add a Burn to discard pile.", zh="对所有敌人造成21/28点伤害。将一张灼伤加入弃牌堆。"),
    "Impervious": _c("岿然不动", SKILL, (2, 2), block=(30, 40), en="Gain 30/40 Block. Exhaust.", zh="获得30/40点格挡。消耗。"),
    "Juggernaut": _c("势不可挡", POWER, (2, 2), en="Whenever Block is gained, deal 5/7 damage to a random enemy.", zh="每当获得格挡时，对随机敌人造成5/7点伤害。"),
    "Limit Break": _c("突破极限", SKILL, (1, 1), en="Double Strength. Base Exhausts; upgraded does not.", zh="使力量翻倍。未升级时消耗；升级后不消耗。"),
    "Offering": _c("祭品", SKILL, (0, 0), en="Lose 6 HP. Gain 2 Energy. Draw 3/5 cards. Exhaust.", zh="失去6点生命。获得2点能量。抽3/5张牌。消耗。"),
    "Reaper": _c("死神", ATTACK, (2, 2), (4, 5), en="Deal 4/5 damage to all enemies; heal unblocked damage. Exhaust.", zh="对所有敌人造成4/5点伤害；回复等同于未被格挡伤害的生命。消耗。"),
}


def card_numbers(card_id: str, upgrades: int = 0) -> tuple[int, int, int] | None:
    """Return ``(cost, base damage, base block)`` for an Ironclad card."""
    info = IRONCLAD_CARDS.get(card_id)
    if info is None:
        return None
    level = int(upgrades > 0)
    damage = info.damage[level]
    if card_id == "Searing Blow":
        n = max(0, int(upgrades))
        damage = n * (n + 7) // 2 + 12
    return info.cost[level], damage, info.block[level]


assert len(IRONCLAD_CARDS) == 75

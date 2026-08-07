"""Structured effect language for all Ironclad and common Status cards.

The language is deliberately small and compositional: card metadata, actions,
targets, conditions, triggers, durations, and numeric arguments are separate
tokens. Programs are authored from the same bilingual card table used by the
observation encoder; no natural-language parsing happens at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import re

from .cards_vocab import CARD_INDEX, CARD_VOCAB
from .ironclad_cards import ATTACK, POWER, SKILL, X_COST, IRONCLAD_CARDS


PAD = "PAD"
CARD_BEGIN = "CARD_BEGIN"
CARD_END = "CARD_END"
STMT_END = "STMT_END"
BLOCK_END = "BLOCK_END"


def _n(value: int) -> str:
    return "X" if value == X_COST else f"N{value}"


def _expand_number_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    """Apply a digit-level numeric tokenizer, analogous to byte-level fallback."""
    out = []
    for token in tokens:
        match = re.fullmatch(r"N([0-9]+)", token)
        if match:
            out.extend(("NUM_BEGIN", *(f"DIGIT_{d}" for d in match.group(1)),
                        "NUM_END"))
        elif token == "X":
            out.extend(("NUM_BEGIN", "VALUE_X", "NUM_END"))
        else:
            out.append(token)
    return tuple(out)


def _v(values: tuple[int, int], upgraded: bool) -> str:
    return _n(values[int(upgraded)])


def _do(operation: str, *args: str) -> tuple[str, ...]:
    return ("DO", operation, *args, STMT_END)


def _trigger(event: str, *body: tuple[str, ...]) -> tuple[str, ...]:
    return ("TRIGGER", event, *_join(body), BLOCK_END)


def _if(condition: str, *body: tuple[str, ...]) -> tuple[str, ...]:
    return ("IF", condition, *_join(body), BLOCK_END)


def _trait(name: str) -> tuple[str, ...]:
    return ("TRAIT", name, STMT_END)


def _join(parts: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(token for part in parts for token in part)


def _effects(*parts: tuple[str, ...]) -> tuple[str, ...]:
    return _join(parts)


RARITY_BASIC = {"Bash", "Defend_R", "Strike_R"}
RARITY_COMMON = {
    "Anger", "Armaments", "Body Slam", "Clash", "Cleave", "Clothesline",
    "Flex", "Havoc", "Headbutt", "Heavy Blade", "Iron Wave",
    "Perfected Strike", "Pommel Strike", "Shrug It Off", "Sword Boomerang",
    "Thunderclap", "True Grit", "Twin Strike", "Warcry", "Wild Strike",
}
RARITY_RARE = {
    "Barricade", "Berserk", "Bludgeon", "Brutality", "Corruption",
    "Demon Form", "Double Tap", "Exhume", "Feed", "Fiend Fire", "Immolate",
    "Impervious", "Juggernaut", "Limit Break", "Offering", "Reaper",
}

STATUS_CARDS = {
    "Burn": {
        "name_zh": "灼伤",
        "effect_en": "Unplayable. At end of turn, take 2 damage (4 for Burn+).",
        "effect_zh": "无法打出。回合结束时受到2点伤害；灼伤+为4点。",
    },
    "Dazed": {
        "name_zh": "晕眩",
        "effect_en": "Unplayable. Ethereal.",
        "effect_zh": "无法打出。虚无。",
    },
    "Slimed": {
        "name_zh": "粘液",
        "effect_en": "Costs 1. Exhaust.",
        "effect_zh": "耗费1点能量。消耗。",
    },
    "Void": {
        "name_zh": "虚空",
        "effect_en": "Unplayable. Ethereal. When drawn, lose 1 Energy.",
        "effect_zh": "无法打出。虚无。抽到时失去1点能量。",
    },
    "Wound": {
        "name_zh": "伤口",
        "effect_en": "Unplayable.",
        "effect_zh": "无法打出。",
    },
}
LANGUAGE_CARDS = frozenset(IRONCLAD_CARDS) | frozenset(STATUS_CARDS)


def card_rarity(card_id: str) -> str:
    if card_id in STATUS_CARDS:
        return "SPECIAL"
    if card_id in RARITY_BASIC:
        return "BASIC"
    if card_id in RARITY_COMMON:
        return "COMMON"
    if card_id in RARITY_RARE:
        return "RARE"
    if card_id in IRONCLAD_CARDS:
        return "UNCOMMON"
    raise KeyError(card_id)


EffectBuilder = Callable[[bool], tuple[str, ...]]


def _simple_damage(values):
    return lambda u: _do("DEAL_DAMAGE", _v(values, u), "TARGET_ENEMY")


def _simple_block(values):
    return lambda u: _do("GAIN_BLOCK", _v(values, u), "SELF")


_EFFECTS: dict[str, EffectBuilder] = {
    "Bash": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((8, 10), u), "TARGET_ENEMY"),
        _do("APPLY", "VULNERABLE", _v((2, 3), u), "TARGET_ENEMY")),
    "Defend_R": _simple_block((5, 8)),
    "Strike_R": _simple_damage((6, 9)),
    "Anger": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((6, 8), u), "TARGET_ENEMY"),
        _do("CREATE_COPY", "THIS_CARD", "N1", "DISCARD_PILE")),
    "Armaments": lambda u: _effects(
        _do("GAIN_BLOCK", "N5", "SELF"),
        _do("UPGRADE_CARD", "ALL" if u else "CHOOSE", *(() if u else ("N1",)),
            "HAND", "THIS_COMBAT")),
    "Body Slam": lambda u: _do(
        "DEAL_DAMAGE_FROM", "CURRENT_BLOCK", "TARGET_ENEMY"),
    "Clash": lambda u: _effects(
        _trait("PLAYABLE_IF_ALL_HAND_ATTACK"),
        _do("DEAL_DAMAGE", _v((14, 18), u), "TARGET_ENEMY")),
    "Cleave": lambda u: _do("DEAL_DAMAGE", _v((8, 11), u), "ALL_ENEMIES"),
    "Clothesline": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((12, 14), u), "TARGET_ENEMY"),
        _do("APPLY", "WEAK", _v((2, 3), u), "TARGET_ENEMY")),
    "Flex": lambda u: _do(
        "GAIN_TEMPORARY", "STRENGTH", _v((2, 4), u), "SELF", "UNTIL_END_TURN"),
    "Havoc": lambda u: _do(
        "PLAY_CARD", "TOP", "DRAW_PILE", "THEN_EXHAUST", "IGNORE_COST"),
    "Headbutt": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((9, 12), u), "TARGET_ENEMY"),
        _do("MOVE_CARD", "CHOOSE", "N1", "DISCARD_PILE", "TOP", "DRAW_PILE")),
    "Heavy Blade": lambda u: _effects(
        _do("DEAL_DAMAGE", "N14", "TARGET_ENEMY"),
        _do("SET_STRENGTH_MULTIPLIER", _v((3, 5), u), "THIS_CARD")),
    "Iron Wave": lambda u: _effects(
        _do("GAIN_BLOCK", _v((5, 7), u), "SELF"),
        _do("DEAL_DAMAGE", _v((5, 7), u), "TARGET_ENEMY")),
    "Perfected Strike": lambda u: _do(
        "DEAL_DAMAGE", "N6", "PLUS", _v((2, 3), u), "PER",
        "CARD_WITH_STRIKE", "ALL_COMBAT_PILES", "TARGET_ENEMY"),
    "Pommel Strike": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((9, 10), u), "TARGET_ENEMY"),
        _do("DRAW", _v((1, 2), u), "SELF")),
    "Shrug It Off": lambda u: _effects(
        _do("GAIN_BLOCK", _v((8, 11), u), "SELF"), _do("DRAW", "N1", "SELF")),
    "Sword Boomerang": lambda u: _do(
        "DEAL_DAMAGE", "N3", "RANDOM_ENEMY", "TIMES", _v((3, 4), u)),
    "Thunderclap": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((4, 7), u), "ALL_ENEMIES"),
        _do("APPLY", "VULNERABLE", "N1", "ALL_ENEMIES")),
    "True Grit": lambda u: _effects(
        _do("GAIN_BLOCK", _v((7, 9), u), "SELF"),
        _do("EXHAUST_CARD", "CHOOSE" if u else "RANDOM", "N1", "HAND")),
    "Twin Strike": lambda u: _do(
        "DEAL_DAMAGE", _v((5, 7), u), "TARGET_ENEMY", "TIMES", "N2"),
    "Warcry": lambda u: _effects(
        _do("DRAW", _v((1, 2), u), "SELF"),
        _do("MOVE_CARD", "CHOOSE", "N1", "HAND", "TOP", "DRAW_PILE"),
        _trait("EXHAUST")),
    "Wild Strike": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((12, 17), u), "TARGET_ENEMY"),
        _do("CREATE_CARD", "WOUND", "N1", "DRAW_PILE", "SHUFFLE")),
    "Battle Trance": lambda u: _effects(
        _do("DRAW", _v((3, 4), u), "SELF"),
        _do("APPLY", "NO_DRAW", "N1", "SELF", "UNTIL_END_TURN")),
    "Blood For Blood": lambda u: _effects(
        _trigger("ON_HP_LOSS", _do("REDUCE_COST", "N1", "THIS_CARD", "THIS_COMBAT")),
        _do("DEAL_DAMAGE", _v((18, 22), u), "TARGET_ENEMY")),
    "Bloodletting": lambda u: _effects(
        _do("LOSE_HP", "N3", "SELF"), _do("GAIN_ENERGY", _v((2, 3), u), "SELF")),
    "Burning Pact": lambda u: _effects(
        _do("EXHAUST_CARD", "CHOOSE", "N1", "HAND"),
        _do("DRAW", _v((2, 3), u), "SELF")),
    "Carnage": lambda u: _effects(
        _trait("ETHEREAL"), _do("DEAL_DAMAGE", _v((20, 28), u), "TARGET_ENEMY")),
    "Combust": lambda u: _trigger(
        "END_TURN", _do("LOSE_HP", "N1", "SELF"),
        _do("DEAL_DAMAGE", _v((5, 7), u), "ALL_ENEMIES")),
    "Dark Embrace": lambda u: _trigger(
        "ON_ANY_CARD_EXHAUSTED", _do("DRAW", "N1", "SELF")),
    "Disarm": lambda u: _effects(
        _do("LOSE", "STRENGTH", _v((2, 3), u), "TARGET_ENEMY"), _trait("EXHAUST")),
    "Dropkick": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((5, 8), u), "TARGET_ENEMY"),
        _if("TARGET_HAS_VULNERABLE", _do("GAIN_ENERGY", "N1", "SELF"),
            _do("DRAW", "N1", "SELF"))),
    "Dual Wield": lambda u: _do(
        "CREATE_COPY", "CHOOSE_ATTACK_OR_POWER", _v((1, 2), u), "HAND"),
    "Entrench": lambda u: _do("DOUBLE", "CURRENT_BLOCK", "SELF"),
    "Evolve": lambda u: _trigger(
        "ON_DRAW_STATUS", _do("DRAW", _v((1, 2), u), "SELF")),
    "Feel No Pain": lambda u: _trigger(
        "ON_ANY_CARD_EXHAUSTED", _do("GAIN_BLOCK", _v((3, 4), u), "SELF")),
    "Fire Breathing": lambda u: _trigger(
        "ON_DRAW_STATUS_OR_CURSE",
        _do("DEAL_DAMAGE", _v((6, 10), u), "ALL_ENEMIES")),
    "Flame Barrier": lambda u: _effects(
        _do("GAIN_BLOCK", _v((12, 16), u), "SELF"),
        _trigger("ON_ATTACKED_THIS_TURN",
                 _do("DEAL_DAMAGE", _v((4, 6), u), "ATTACKER"))),
    "Ghostly Armor": lambda u: _effects(
        _trait("ETHEREAL"), _do("GAIN_BLOCK", _v((10, 13), u), "SELF")),
    "Hemokinesis": lambda u: _effects(
        _do("LOSE_HP", "N2", "SELF"),
        _do("DEAL_DAMAGE", _v((15, 20), u), "TARGET_ENEMY")),
    "Infernal Blade": lambda u: _effects(
        _do("CREATE_RANDOM_CARD", "TYPE_ATTACK", "N1", "HAND"),
        _do("SET_COST", "N0", "CREATED_CARD", "UNTIL_END_TURN"),
        _trait("EXHAUST")),
    "Inflame": lambda u: _do("GAIN", "STRENGTH", _v((2, 3), u), "SELF"),
    "Intimidate": lambda u: _effects(
        _do("APPLY", "WEAK", _v((1, 2), u), "ALL_ENEMIES"), _trait("EXHAUST")),
    "Metallicize": lambda u: _trigger(
        "END_TURN", _do("GAIN_BLOCK", _v((3, 4), u), "SELF")),
    "Power Through": lambda u: _effects(
        _do("CREATE_CARD", "WOUND", "N2", "HAND"),
        _do("GAIN_BLOCK", _v((15, 20), u), "SELF")),
    "Pummel": lambda u: _effects(
        _do("DEAL_DAMAGE", "N2", "TARGET_ENEMY", "TIMES", _v((4, 5), u)),
        _trait("EXHAUST")),
    "Rage": lambda u: _trigger(
        "ON_PLAY_ATTACK_THIS_TURN", _do("GAIN_BLOCK", _v((3, 5), u), "SELF")),
    "Rampage": lambda u: _effects(
        _do("DEAL_DAMAGE", "N8", "TARGET_ENEMY"),
        _do("INCREASE_BASE_DAMAGE", _v((5, 8), u), "THIS_CARD", "THIS_COMBAT")),
    "Reckless Charge": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((7, 10), u), "TARGET_ENEMY"),
        _do("CREATE_CARD", "DAZED", "N1", "DRAW_PILE", "SHUFFLE")),
    "Rupture": lambda u: _trigger(
        "ON_CARD_HP_LOSS", _do("GAIN", "STRENGTH", _v((1, 2), u), "SELF")),
    "Searing Blow": lambda u: _effects(
        _do("DEAL_DAMAGE_FORMULA", "SEARING_BLOW_UPGRADES", "UPGRADE_COUNT",
            _v((0, 1), u), "TARGET_ENEMY"),
        _trait("REPEATABLE_UPGRADE")),
    "Second Wind": lambda u: _effects(
        _do("EXHAUST_CARD", "ALL_NON_ATTACK", "HAND"),
        _do("GAIN_BLOCK", _v((5, 7), u), "PER", "EXHAUSTED_CARD", "SELF")),
    "Seeing Red": lambda u: _effects(
        _do("GAIN_ENERGY", "N2", "SELF"), _trait("EXHAUST")),
    "Sentinel": lambda u: _effects(
        _do("GAIN_BLOCK", _v((5, 8), u), "SELF"),
        _trigger("ON_THIS_CARD_EXHAUSTED",
                 _do("GAIN_ENERGY", _v((2, 3), u), "SELF"))),
    "Sever Soul": lambda u: _effects(
        _do("EXHAUST_CARD", "ALL_NON_ATTACK", "HAND"),
        _do("DEAL_DAMAGE", _v((16, 22), u), "TARGET_ENEMY")),
    "Shockwave": lambda u: _effects(
        _do("APPLY", "WEAK", _v((3, 5), u), "ALL_ENEMIES"),
        _do("APPLY", "VULNERABLE", _v((3, 5), u), "ALL_ENEMIES"),
        _trait("EXHAUST")),
    "Spot Weakness": lambda u: _if(
        "TARGET_INTENDS_ATTACK", _do("GAIN", "STRENGTH", _v((3, 4), u), "SELF")),
    "Uppercut": lambda u: _effects(
        _do("DEAL_DAMAGE", "N13", "TARGET_ENEMY"),
        _do("APPLY", "WEAK", _v((1, 2), u), "TARGET_ENEMY"),
        _do("APPLY", "VULNERABLE", _v((1, 2), u), "TARGET_ENEMY")),
    "Whirlwind": lambda u: _do(
        "DEAL_DAMAGE", _v((5, 8), u), "ALL_ENEMIES", "TIMES", "X"),
    "Barricade": lambda u: _trait("KEEP_BLOCK_BETWEEN_TURNS"),
    "Berserk": lambda u: _effects(
        _do("APPLY", "VULNERABLE", _v((2, 1), u), "SELF"),
        _trigger("START_TURN", _do("GAIN_ENERGY", "N1", "SELF"))),
    "Bludgeon": _simple_damage((32, 42)),
    "Brutality": lambda u: _effects(
        *([_trait("INNATE")] if u else []),
        _trigger("START_TURN", _do("LOSE_HP", "N1", "SELF"),
                 _do("DRAW", "N1", "SELF"))),
    "Corruption": lambda u: _effects(
        _do("SET_COST", "N0", "ALL_SKILLS", "THIS_COMBAT"),
        _trigger("ON_PLAY_SKILL", _do("EXHAUST_CARD", "PLAYED_CARD"))),
    "Demon Form": lambda u: _trigger(
        "START_TURN", _do("GAIN", "STRENGTH", _v((2, 3), u), "SELF")),
    "Double Tap": lambda u: _do(
        "PLAY_TWICE", _v((1, 2), u), "NEXT_ATTACKS", "UNTIL_END_TURN"),
    "Exhume": lambda u: _effects(
        _do("MOVE_CARD", "CHOOSE_NON_EXHUME", "EXHAUST_PILE", "HAND"),
        _trait("EXHAUST")),
    "Feed": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((10, 12), u), "TARGET_ENEMY"),
        _trigger("FATAL", _do("GAIN", "MAX_HP", _v((3, 4), u), "SELF")),
        _trait("EXHAUST")),
    "Fiend Fire": lambda u: _effects(
        _do("EXHAUST_CARD", "REST_OF_HAND"),
        _do("DEAL_DAMAGE", _v((7, 10), u), "PER", "EXHAUSTED_CARD", "TARGET_ENEMY"),
        _trait("EXHAUST")),
    "Immolate": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((21, 28), u), "ALL_ENEMIES"),
        _do("CREATE_CARD", "BURN", "N1", "DISCARD_PILE")),
    "Impervious": lambda u: _effects(
        _do("GAIN_BLOCK", _v((30, 40), u), "SELF"), _trait("EXHAUST")),
    "Juggernaut": lambda u: _trigger(
        "ON_GAIN_BLOCK", _do("DEAL_DAMAGE", _v((5, 7), u), "RANDOM_ENEMY")),
    "Limit Break": lambda u: _effects(
        _do("DOUBLE", "STRENGTH", "SELF"), *([] if u else [_trait("EXHAUST")])),
    "Offering": lambda u: _effects(
        _do("LOSE_HP", "N6", "SELF"), _do("GAIN_ENERGY", "N2", "SELF"),
        _do("DRAW", _v((3, 5), u), "SELF"), _trait("EXHAUST")),
    "Reaper": lambda u: _effects(
        _do("DEAL_DAMAGE", _v((4, 5), u), "ALL_ENEMIES"),
        _do("HEAL_FROM", "UNBLOCKED_DAMAGE_DEALT", "SELF"), _trait("EXHAUST")),
}


def _status_effects(card_id: str, upgraded: bool) -> tuple[str, ...]:
    medical_kit = (
        _trait("UNPLAYABLE_NORMALLY"),
        _trait("PLAYABLE_WITH_MEDICAL_KIT"),
        _trait("EXHAUST_WITH_MEDICAL_KIT"),
    )
    if card_id == "Burn":
        return _effects(
            _trigger("END_TURN_IN_HAND", _do("LOSE_HP", "N4" if upgraded else "N2", "SELF")),
            *medical_kit,
        )
    if card_id == "Dazed":
        return _effects(_trait("ETHEREAL"), *medical_kit)
    if card_id == "Slimed":
        return _trait("EXHAUST")
    if card_id == "Void":
        return _effects(
            _trigger("ON_DRAW", _do("LOSE_ENERGY", "N1", "SELF")),
            _trait("ETHEREAL"),
            *medical_kit,
        )
    if card_id == "Wound":
        return _effects(*medical_kit)
    raise KeyError(card_id)


_TYPE_TOKEN = {
    ATTACK: "TYPE_ATTACK", SKILL: "TYPE_SKILL", POWER: "TYPE_POWER",
    "status": "TYPE_STATUS",
}


def card_program(card_id: str, upgraded: bool = False) -> tuple[str, ...]:
    """Compile one card/version to a complete language-token program."""
    if card_id in STATUS_CARDS:
        kind = "status"
        cost = "N1" if card_id == "Slimed" else "NOT_APPLICABLE"
        effects = _status_effects(card_id, bool(upgraded))
    else:
        info = IRONCLAD_CARDS[card_id]
        kind = info.kind
        cost = _v(info.cost, upgraded)
        effects = _EFFECTS[card_id](bool(upgraded))
    raw = (
        CARD_BEGIN,
        _TYPE_TOKEN[kind],
        f"RARITY_{card_rarity(card_id)}",
        "COST", cost,
        *effects,
        CARD_END,
    )
    return _expand_number_tokens(raw)


def combined_card_program(card_id: str) -> tuple[str, ...]:
    """Both versions in one sequence, used as static semantic card context."""
    return (
        "VERSIONS_BEGIN", "BASE_VERSION", *card_program(card_id, False),
        "UPGRADED_VERSION", *card_program(card_id, True), "VERSIONS_END",
    )


TOKEN_DOC: dict[str, tuple[str, str]] = {
    PAD: ("Padding; ignored by the model.", "填充；模型忽略。"),
    CARD_BEGIN: ("Start of one card program.", "一张卡牌程序的开始。"),
    CARD_END: ("End of one card program.", "一张卡牌程序的结束。"),
    STMT_END: ("End of one statement.", "一条语句的结束。"),
    BLOCK_END: ("End of a trigger or condition body.", "触发器或条件作用域结束。"),
    "VERSIONS_BEGIN": ("Start of the base/upgraded pair.", "基础版/升级版组合开始。"),
    "VERSIONS_END": ("End of the base/upgraded pair.", "基础版/升级版组合结束。"),
    "BASE_VERSION": ("The following program is the base card.", "以下程序描述基础版。"),
    "UPGRADED_VERSION": ("The following program is upgraded.", "以下程序描述升级版。"),
    "DO": ("Execute an effect operation.", "执行一个效果操作。"),
    "TRIGGER": ("Execute the body when the following event occurs.", "当随后事件发生时执行作用域。"),
    "IF": ("Execute the body if the following condition is true.", "若随后条件成立则执行作用域。"),
    "TRAIT": ("Declare an intrinsic card property.", "声明卡牌固有属性。"),
    "COST": ("Printed energy cost; VALUE_X denotes X-cost.", "印刷能量费用；VALUE_X 表示 X 费。"),
    "NUM_BEGIN": ("Start of a numeric literal.", "一个数值字面量的开始。"),
    "NUM_END": ("End of a numeric literal.", "一个数值字面量的结束。"),
    "VALUE_X": ("Energy spent on an X-cost card.", "X 费牌实际消耗的能量。"),
}


_TOKEN_GROUPS = {
    "Card metadata": {
        "TYPE_ATTACK": "Attack card / 攻击牌", "TYPE_SKILL": "Skill card / 技能牌",
        "TYPE_POWER": "Power card / 能力牌", "RARITY_BASIC": "Basic rarity / 基础",
        "TYPE_STATUS": "Status card / 状态牌",
        "RARITY_COMMON": "Common rarity / 普通", "RARITY_UNCOMMON": "Uncommon rarity / 罕见",
        "RARITY_RARE": "Rare rarity / 稀有",
        "RARITY_SPECIAL": "Special rarity / 特殊",
    },
    "Operations": {
        "DEAL_DAMAGE": "Deal the stated damage / 造成给定伤害",
        "DEAL_DAMAGE_FROM": "Damage equals a referenced value / 伤害等于引用值",
        "DEAL_DAMAGE_FORMULA": "Damage uses a named formula / 伤害使用指定公式",
        "GAIN_BLOCK": "Gain Block / 获得格挡", "DRAW": "Draw cards / 抽牌",
        "APPLY": "Apply a status or debuff / 施加状态或减益",
        "GAIN": "Gain a stat / 获得属性", "LOSE": "Lose a stat / 失去属性",
        "LOSE_HP": "Lose HP directly / 直接失去生命",
        "LOSE_ENERGY": "Lose Energy / 失去能量",
        "GAIN_ENERGY": "Gain energy / 获得能量", "GAIN_TEMPORARY": "Gain a temporary stat / 获得临时属性",
        "CREATE_CARD": "Create a named card / 生成指定卡",
        "CREATE_RANDOM_CARD": "Create a random matching card / 生成随机匹配卡",
        "CREATE_COPY": "Create copies / 生成复制",
        "MOVE_CARD": "Move a card between zones / 在区域间移动卡牌",
        "PLAY_CARD": "Play a card from a zone / 从区域打出卡牌",
        "EXHAUST_CARD": "Exhaust matching cards / 消耗匹配卡牌",
        "UPGRADE_CARD": "Upgrade matching cards / 升级匹配卡牌",
        "SET_COST": "Set card cost / 设置卡牌费用",
        "REDUCE_COST": "Reduce card cost / 降低卡牌费用",
        "DOUBLE": "Double the referenced value / 将引用值翻倍",
        "PLAY_TWICE": "Play upcoming cards twice / 重复打出后续卡牌",
        "HEAL_FROM": "Heal from a referenced amount / 按引用数值回复生命",
        "SET_STRENGTH_MULTIPLIER": "Set Strength scaling multiplier / 设置力量加成倍率",
        "INCREASE_BASE_DAMAGE": "Permanently increase this instance's combat damage / 本场战斗增加此实例伤害",
    },
    "Targets and zones": {
        "SELF": "Player / 玩家", "TARGET_ENEMY": "Chosen enemy / 选定敌人",
        "ALL_ENEMIES": "All enemies / 所有敌人", "RANDOM_ENEMY": "Random enemy / 随机敌人",
        "ATTACKER": "Enemy that attacked / 攻击者", "HAND": "Hand / 手牌",
        "REST_OF_HAND": "Every other card in hand / 其余手牌",
        "DRAW_PILE": "Draw pile / 抽牌堆", "DISCARD_PILE": "Discard pile / 弃牌堆",
        "EXHAUST_PILE": "Exhaust pile / 消耗堆", "ALL_COMBAT_PILES": "All combat card zones / 所有战斗牌区",
        "THIS_CARD": "This card instance / 本牌实例", "CREATED_CARD": "The card just created / 刚生成的牌",
        "PLAYED_CARD": "The card just played / 刚打出的牌", "NEXT_ATTACKS": "Upcoming Attack cards / 接下来攻击牌",
        "ALL_SKILLS": "All Skill cards / 所有技能牌",
    },
    "Selection and quantities": {
        "ALL": "All matching objects / 所有匹配对象", "ALL_NON_ATTACK": "All non-Attacks / 所有非攻击牌",
        "CHOOSE": "Choose matching objects / 选择匹配对象", "RANDOM": "Choose randomly / 随机选择",
        "CHOOSE_ATTACK_OR_POWER": "Choose an Attack or Power / 选择攻击或能力牌",
        "CHOOSE_NON_EXHUME": "Choose a card other than Exhume / 选择非发掘牌",
        "TOP": "Top of pile / 牌堆顶", "PER": "For each / 每个",
        "TIMES": "Repeat count / 重复次数", "PLUS": "Add the following term / 加上后续项",
    },
    "Statuses and values": {
        "STRENGTH": "Strength / 力量", "WEAK": "Weak / 虚弱",
        "VULNERABLE": "Vulnerable / 易伤", "NO_DRAW": "No Draw / 无法抽牌",
        "MAX_HP": "Maximum HP / 最大生命", "CURRENT_BLOCK": "Current Block / 当前格挡",
        "UNBLOCKED_DAMAGE_DEALT": "Total unblocked damage dealt / 实际造成的未格挡伤害",
        "EXHAUSTED_CARD": "Each card exhausted by this effect / 被本效果消耗的每张牌",
        "CARD_WITH_STRIKE": "Card whose name contains Strike / 名称含“打击”的牌",
        "SEARING_BLOW_UPGRADES": "12 + n(n+7)/2 for upgrade count n / 升级 n 次时为 12+n(n+7)/2",
        "UPGRADE_COUNT": "Number of upgrades applied to this card / 本牌的升级次数",
        "WOUND": "Wound status card / 伤口", "DAZED": "Dazed status card / 晕眩",
        "BURN": "Burn status card / 灼伤",
        "NOT_APPLICABLE": "No printed energy cost / 无印刷能量费用",
    },
    "Timing and modifiers": {
        "THIS_COMBAT": "For this combat / 本场战斗", "UNTIL_END_TURN": "Until turn end / 直到回合结束",
        "SHUFFLE": "Shuffle into destination / 洗入目标牌堆", "IGNORE_COST": "Do not pay printed cost / 不支付印刷费用",
        "THEN_EXHAUST": "Exhaust after playing / 打出后消耗",
    },
    "Events and conditions": {
        "START_TURN": "At start of turn / 回合开始", "END_TURN": "At end of turn / 回合结束",
        "ON_DRAW": "When this card is drawn / 抽到此牌时",
        "END_TURN_IN_HAND": "At end of turn while this card is in hand / 此牌在手中时的回合结束",
        "ON_HP_LOSS": "Whenever HP is lost / 每次失去生命",
        "ON_CARD_HP_LOSS": "When a card causes HP loss / 卡牌导致生命损失时",
        "ON_ANY_CARD_EXHAUSTED": "Whenever any card is exhausted / 任意牌被消耗时",
        "ON_THIS_CARD_EXHAUSTED": "When this card is exhausted / 本牌被消耗时",
        "ON_DRAW_STATUS": "When a Status is drawn / 抽到状态牌时",
        "ON_DRAW_STATUS_OR_CURSE": "When a Status or Curse is drawn / 抽到状态或诅咒时",
        "ON_ATTACKED_THIS_TURN": "Whenever attacked this turn / 本回合每次受到攻击",
        "ON_PLAY_ATTACK_THIS_TURN": "Whenever an Attack is played this turn / 本回合每次打出攻击牌",
        "ON_PLAY_SKILL": "Whenever a Skill is played / 每次打出技能牌",
        "ON_GAIN_BLOCK": "Whenever Block is gained / 每次获得格挡",
        "FATAL": "If this damage kills the target / 若此次伤害击杀目标",
        "TARGET_HAS_VULNERABLE": "Target currently has Vulnerable / 目标当前有易伤",
        "TARGET_INTENDS_ATTACK": "Target intends to attack / 目标意图攻击",
    },
    "Traits": {
        "EXHAUST": "Exhaust after use / 使用后消耗", "ETHEREAL": "Ethereal / 虚无",
        "UNPLAYABLE_NORMALLY": "Normally cannot be played / 通常无法打出",
        "PLAYABLE_WITH_MEDICAL_KIT": "Can be played with Medical Kit / 拥有医疗箱时可以打出",
        "EXHAUST_WITH_MEDICAL_KIT": "Exhausts when played with Medical Kit / 通过医疗箱打出时消耗",
        "INNATE": "Innate / 固有", "REPEATABLE_UPGRADE": "May be upgraded repeatedly / 可重复升级",
        "KEEP_BLOCK_BETWEEN_TURNS": "Block is retained between turns / 回合间保留格挡",
        "PLAYABLE_IF_ALL_HAND_ATTACK": "Playable only if all hand cards are Attacks / 仅当全部手牌为攻击牌时可打出",
    },
}

for descriptions in _TOKEN_GROUPS.values():
    for token, bilingual in descriptions.items():
        en, zh = bilingual.split(" / ", 1)
        TOKEN_DOC[token] = (en + ".", zh + "。")
for digit in range(10):
    TOKEN_DOC[f"DIGIT_{digit}"] = (
        f"Decimal digit {digit}.", f"十进制数字 {digit}。"
    )


ALL_PROGRAM_TOKENS = {
    token
    for card_id in LANGUAGE_CARDS
    for upgraded in (False, True)
    for token in card_program(card_id, upgraded)
} | {"VERSIONS_BEGIN", "VERSIONS_END", "BASE_VERSION", "UPGRADED_VERSION"}

missing_docs = ALL_PROGRAM_TOKENS - TOKEN_DOC.keys()
assert not missing_docs, f"undocumented card-language tokens: {sorted(missing_docs)}"
assert set(_EFFECTS) == set(IRONCLAD_CARDS)
assert set(STATUS_CARDS) == {"Burn", "Dazed", "Slimed", "Void", "Wound"}
assert len(RARITY_BASIC | RARITY_COMMON | RARITY_RARE) == 39

TOKEN_VOCAB = (PAD, *sorted(ALL_PROGRAM_TOKENS - {PAD}))
TOKEN_INDEX = {token: idx for idx, token in enumerate(TOKEN_VOCAB)}
MAX_PROGRAM_TOKENS = max(
    len(combined_card_program(card_id)) for card_id in LANGUAGE_CARDS
)
MAX_VERSION_PROGRAM_TOKENS = max(
    len(card_program(card_id, upgraded))
    for card_id in LANGUAGE_CARDS
    for upgraded in (False, True)
)


def encoded_card_programs() -> tuple[tuple[int, ...], ...]:
    """One PAD-padded combined program per global card-vocabulary row."""
    rows = []
    for card_id in CARD_VOCAB:
        tokens = combined_card_program(card_id) if card_id in LANGUAGE_CARDS else ()
        ids = tuple(TOKEN_INDEX[token] for token in tokens)
        rows.append(ids + (0,) * (MAX_PROGRAM_TOKENS - len(ids)))
    return tuple(rows)


def encoded_card_version_programs() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """PAD-padded base/upgraded programs for every global card-vocab row."""
    rows = []
    for card_id in CARD_VOCAB:
        versions = []
        for upgraded in (False, True):
            tokens = card_program(card_id, upgraded) if card_id in LANGUAGE_CARDS else ()
            ids = tuple(TOKEN_INDEX[token] for token in tokens)
            versions.append(
                ids + (0,) * (MAX_VERSION_PROGRAM_TOKENS - len(ids))
            )
        rows.append(tuple(versions))
    return tuple(rows)


def render_markdown() -> str:
    lines = [
        "# Ironclad and Status Card Effect Language / 战士与状态牌效果语言",
        "",
        "This document is generated from `env/core/card_language.py`. Each card is "
        "compiled independently for its base and upgraded versions.",
        "",
        "本文由 `env/core/card_language.py` 生成。每张牌的基础版和升级版分别编译，"
        "不存在依赖自然语言解析的隐式差值。",
        "",
        "## Grammar / 语法",
        "",
        "```text",
        "card      := CARD_BEGIN TYPE_* RARITY_* COST value statement* CARD_END",
        "statement := DO operation argument* STMT_END",
        "           | TRIGGER event statement* BLOCK_END",
        "           | IF condition statement* BLOCK_END",
        "           | TRAIT property STMT_END",
        "value     := NUM_BEGIN DIGIT_0...DIGIT_9+ NUM_END",
        "           | NUM_BEGIN VALUE_X NUM_END",
        "```",
        "",
        "## Model Encoding / 模型编码",
        "",
        "Enable with `--card-language`. The model keeps the learned card-ID "
        "embedding and adds an effect-language vector produced by token embedding, "
        "explicit token positions, a local order-sensitive 1-D encoder, and masked "
        "pooling. Draw, discard, and full-deck cards are connected again as ordered "
        "card-language sequences. The same effective "
        "card vectors are used for hand, reward, selection, shop, draw, discard, "
        "exhaust, and full-deck observations.",
        "",
        "使用 `--card-language` 开启。模型保留原有卡牌 ID embedding，并叠加由 "
        "token embedding、显式 token 位置编码、保持局部顺序的一维编码器和掩码池化"
        "得到的效果语言向量。抽牌堆、弃牌堆和完整牌组中的卡牌语言会按牌序再次连接。"
        "手牌、奖励、选牌、商店和各牌堆共用同一语义卡表。",
        "",
        f"Language version / 语言版本: `4`; vocabulary / 词表大小: "
        f"`{len(TOKEN_VOCAB)}`; maximum combined program length / 最长组合程序: "
        f"`{MAX_PROGRAM_TOKENS}`.",
        "",
        "## Token Dictionary / Token 词典",
        "",
        "| Token | English | 中文 |",
        "|---|---|---|",
    ]
    for token in TOKEN_VOCAB:
        en, zh = TOKEN_DOC[token]
        lines.append(f"| `{token}` | {en} | {zh} |")

    lines += ["", "## Cards / 卡牌翻译", ""]
    for card_id, info in IRONCLAD_CARDS.items():
        lines += [
            f"### {info.name_zh} / {card_id}",
            "",
            f"- Type / 类型: `{info.kind}`",
            f"- Rarity / 稀有度: `{card_rarity(card_id).lower()}`",
            f"- Cost / 费用: `{info.cost[0]}` -> `{info.cost[1]}`",
            f"- Damage / 伤害: `{info.damage[0]}` -> `{info.damage[1]}`",
            f"- Block / 格挡: `{info.block[0]}` -> `{info.block[1]}`",
            f"- English: {info.effect_en}",
            f"- 中文: {info.effect_zh}",
            "- Base DSL / 基础版:",
            "",
            "```text",
            " ".join(card_program(card_id, False)),
            "```",
            "",
            "- Upgraded DSL / 升级版:",
            "",
            "```text",
            " ".join(card_program(card_id, True)),
            "```",
            "",
        ]
    lines += ["", "## Status Cards / 状态牌", ""]
    for card_id, info in STATUS_CARDS.items():
        lines += [
            f"### {info['name_zh']} / {card_id}",
            "",
            f"- English: {info['effect_en']}",
            f"- 中文: {info['effect_zh']}",
            "- Base DSL / 基础版:",
            "",
            "```text",
            " ".join(card_program(card_id, False)),
            "```",
            "",
            "- Upgraded DSL / 升级版:",
            "",
            "```text",
            " ".join(card_program(card_id, True)),
            "```",
            "",
        ]
    return "\n".join(lines)

"""Token language for monster intents and live entity status bars.

Unlike card programs, entity programs contain runtime values.  Numbers use the
same digit-level representation as ``card_language`` so 12 and 13 share useful
subtokens rather than requiring unrelated categorical rows.
"""

from __future__ import annotations

from . import vocab as V
from . import intents


PAD = "PAD"


# Authoritative attack set: mirrors Monster::getMoveBaseDamage.  An explicit
# set is preferable to guessing from names (COUNT and DARK_ECHO are attacks).
ATTACK_MOVES = frozenset("""
ACID_SLIME_L_CORROSIVE_SPIT ACID_SLIME_L_TACKLE ACID_SLIME_M_CORROSIVE_SPIT
ACID_SLIME_M_TACKLE ACID_SLIME_S_TACKLE AWAKENED_ONE_SLASH
AWAKENED_ONE_SOUL_STRIKE AWAKENED_ONE_DARK_ECHO AWAKENED_ONE_SLUDGE
AWAKENED_ONE_TACKLE BEAR_LUNGE BEAR_MAUL BLUE_SLAVER_RAKE BLUE_SLAVER_STAB
BOOK_OF_STABBING_MULTI_STAB BOOK_OF_STABBING_SINGLE_STAB
BRONZE_AUTOMATON_FLAIL BRONZE_AUTOMATON_HYPER_BEAM BRONZE_ORB_BEAM
BYRD_HEADBUTT BYRD_PECK BYRD_SWOOP CENTURION_FURY CENTURION_SLASH
CHOSEN_POKE CHOSEN_ZAP CHOSEN_DEBILITATE CORRUPT_HEART_BLOOD_SHOTS
CORRUPT_HEART_ECHO CULTIST_DARK_STRIKE DAGGER_STAB DAGGER_EXPLODE
DARKLING_NIP DARKLING_CHOMP DECA_BEAM DONU_BEAM EXPLODER_SLAM
EXPLODER_EXPLODE FAT_GREMLIN_SMASH FUNGI_BEAST_BITE GIANT_HEAD_COUNT
GIANT_HEAD_IT_IS_TIME GREEN_LOUSE_BITE GREMLIN_LEADER_STAB GREMLIN_NOB_RUSH
GREMLIN_NOB_SKULL_BASH GREMLIN_WIZARD_ULTIMATE_BLAST HEXAGHOST_DIVIDER
HEXAGHOST_INFERNO HEXAGHOST_SEAR HEXAGHOST_TACKLE JAW_WORM_CHOMP
JAW_WORM_THRASH LAGAVULIN_ATTACK LOOTER_LUNGE LOOTER_MUG
MAD_GREMLIN_SCRATCH MUGGER_MUG MUGGER_LUNGE MYSTIC_ATTACK_DEBUFF
NEMESIS_ATTACK NEMESIS_SCYTHE ORB_WALKER_LASER ORB_WALKER_CLAW
POINTY_ATTACK RED_LOUSE_BITE RED_SLAVER_SCRAPE RED_SLAVER_STAB
REPTOMANCER_BIG_BITE REPTOMANCER_SNAKE_STRIKE REPULSOR_BASH
ROMEO_AGONIZING_SLASH ROMEO_CROSS_SLASH SENTRY_BEAM
SHELLED_PARASITE_DOUBLE_STRIKE SHELLED_PARASITE_FELL SHELLED_PARASITE_SUCK
SHIELD_GREMLIN_SHIELD_BASH SLIME_BOSS_SLAM SNAKE_PLANT_CHOMP
SNEAKY_GREMLIN_PUNCTURE SNECKO_TAIL_WHIP SNECKO_BITE
SPHERIC_GUARDIAN_SLAM SPHERIC_GUARDIAN_HARDEN
SPHERIC_GUARDIAN_ATTACK_DEBUFF SPIKER_CUT SPIKE_SLIME_L_FLAME_TACKLE
SPIKE_SLIME_M_FLAME_TACKLE SPIKE_SLIME_S_TACKLE
SPIRE_GROWTH_QUICK_TACKLE SPIRE_GROWTH_SMASH SPIRE_SHIELD_BASH
SPIRE_SHIELD_SMASH SPIRE_SPEAR_BURN_STRIKE SPIRE_SPEAR_SKEWER
TASKMASTER_SCOURING_WHIP TORCH_HEAD_TACKLE THE_CHAMP_FACE_SLAP
THE_CHAMP_HEAVY_SLASH THE_CHAMP_EXECUTE THE_COLLECTOR_FIREBALL
THE_GUARDIAN_FIERCE_BASH THE_GUARDIAN_WHIRLWIND
THE_GUARDIAN_ROLL_ATTACK THE_GUARDIAN_TWIN_SLAM THE_MAW_SLAM THE_MAW_NOM
TIME_EATER_REVERBERATE TIME_EATER_HEAD_SLAM TRANSIENT_ATTACK
WRITHING_MASS_FLAIL WRITHING_MASS_MULTI_STRIKE WRITHING_MASS_STRONG_STRIKE
""".split())


def number_tokens(value: int) -> tuple[str, ...]:
    value = int(value)
    sign = ("SIGN_NEGATIVE",) if value < 0 else ()
    return ("NUM_BEGIN", *sign, *(f"DIGIT_{d}" for d in str(abs(value))), "NUM_END")


def _move_effects(move: str) -> list[str]:
    """Return compositional effect tokens in addition to the exact move token."""
    category = "attack" if move in ATTACK_MOVES else intents.derive(move, 0)
    exact = _EXACT_EFFECTS.get(move, ())
    if category == "unknown" and exact:
        if "TARGET_PLAYER" in exact and ("APPLY" in exact or "REDUCE" in exact):
            category = "debuff"
        elif any(op in exact for op in ("GAIN", "HEAL", "APPLY")):
            category = "buff"
    effects: list[str] = [f"INTENT_KIND_{category.upper()}", "STMT_END"]
    if move in ATTACK_MOVES:
        effects += ["DO", "DEAL_DAMAGE", "CURRENT_DAMAGE", "TARGET_PLAYER",
                    "TIMES", "CURRENT_HITS", "STMT_END"]

    # Explicit semantic vocabulary for non-damage and hybrid portions. Amounts
    # that depend on ascension/internal state are represented honestly as
    # VALUE_DYNAMIC and current attacks receive exact runtime values separately.
    rules = (
        (("DEFEND", "PROTECT", "FORTIFY", "HARDEN", "CHARGING_UP"),
         ("DO", "GAIN_BLOCK", "VALUE_DYNAMIC", "TARGET_ALLY_OR_SELF", "STMT_END")),
        (("RAKE", "AGONIZING_SLASH"),
         ("DO", "APPLY", "WEAK", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("DEBILITATE",),
         ("DO", "APPLY", "VULNERABLE", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("ATTACK_DEBUFF",),
         ("DO", "APPLY", "FRAIL", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("ENTANGLE",),
         ("DO", "APPLY", "ENTANGLED", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("HEX",),
         ("DO", "APPLY", "HEX", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("CONSTRICT",),
         ("DO", "APPLY", "CONSTRICTED", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("SIPHON_SOUL",),
         ("DO", "REDUCE", "STRENGTH_AND_DEXTERITY", "VALUE_DYNAMIC", "TARGET_PLAYER", "STMT_END")),
        (("BUFF", "BOOST", "BELLOW", "INCANTATION", "GROW", "CAW", "GLOAT", "ANGER"),
         ("DO", "APPLY", "STRENGTH", "VALUE_DYNAMIC", "TARGET_ALLY_OR_SELF", "STMT_END")),
        (("HEAL",),
         ("DO", "HEAL", "VALUE_DYNAMIC", "TARGET_ALLY_OR_SELF", "STMT_END")),
        (("SPAWN", "SUMMON"),
         ("DO", "SUMMON_MONSTER", "VALUE_DYNAMIC", "STMT_END")),
        (("SPLIT",), ("DO", "SPLIT_MONSTER", "CURRENT_HP", "STMT_END")),
        (("ESCAPE", "SMOKE_BOMB"), ("DO", "ESCAPE", "SELF", "STMT_END")),
        (("STUNNED",), ("DO", "SKIP_TURN", "SELF", "STMT_END")),
        (("SLEEP",), ("DO", "SLEEP", "SELF", "STMT_END")),
        (("CORROSIVE_SPIT", "GOOP_SPRAY"),
         ("DO", "ADD_CARD", "SLIMED", "VALUE_DYNAMIC", "DISCARD_PILE", "STMT_END")),
        (("SEAR", "INFERNO", "BURN_STRIKE", "LASER"),
         ("DO", "ADD_CARD", "BURN", "VALUE_DYNAMIC", "PILE_DYNAMIC", "STMT_END")),
        (("BOLT",),
         ("DO", "ADD_CARD", "DAZED", "VALUE_DYNAMIC", "DISCARD_PILE", "STMT_END")),
        (("SCOURING_WHIP",),
         ("DO", "ADD_CARD", "WOUND", "VALUE_DYNAMIC", "DISCARD_PILE", "STMT_END")),
        (("STASIS",), ("DO", "STEAL_CARD", "VALUE_DYNAMIC", "STASIS", "STMT_END")),
        (("REBIRTH", "REINCARNATE", "REGROW"),
         ("DO", "REVIVE", "SELF", "STMT_END")),
    )
    for needles, tokens in rules:
        if any(needle in move for needle in needles):
            effects.extend(tokens)
    effects.extend(exact)
    if len(effects) == 2:
        effects += ["DO", "SPECIAL_MOVE", "VALUE_DYNAMIC", "STMT_END"]
    return effects


def _effect(operation: str, name: str, target="SELF") -> tuple[str, ...]:
    return ("DO", operation, name, "VALUE_DYNAMIC", target, "STMT_END")


# Effects whose meaning cannot be recovered safely from a verb substring.
# Values stay dynamic because many change at A17/A18/A19 or use miscInfo.
_EXACT_EFFECTS = {
    "BEAR_BEAR_HUG": _effect("REDUCE", "DEXTERITY", "TARGET_PLAYER"),
    "ACID_SLIME_L_LICK": _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "ACID_SLIME_M_LICK": _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "ACID_SLIME_S_LICK": _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "SPIKE_SLIME_L_LICK": _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "SPIKE_SLIME_M_LICK": _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "CULTIST_INCANTATION": _effect("APPLY", "RITUAL"),
    "BRONZE_ORB_SUPPORT_BEAM": _effect("GAIN", "BLOCK", "TARGET_ALLY"),
    "CHOSEN_DRAIN": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "STRENGTH"),
    "GIANT_HEAD_GLARE": _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "GREEN_LOUSE_SPIT_WEB": _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "GREMLIN_NOB_BELLOW": _effect("APPLY", "ANGER"),
    "GREMLIN_WIZARD_CHARGING": _effect("CHARGE", "ULTIMATE_BLAST"),
    "BYRD_FLY": _effect("APPLY", "FLIGHT"),
    "JAW_WORM_BELLOW": _effect("APPLY", "STRENGTH") + _effect("GAIN", "BLOCK"),
    "HEXAGHOST_INFLAME": _effect("APPLY", "STRENGTH"),
    "LAGAVULIN_SLEEP": ("DO", "SLEEP", "SELF", "STMT_END"),
    "DECA_SQUARE_OF_PROTECTION": _effect("GAIN", "BLOCK", "ALL_ALLIES"),
    "DONU_CIRCLE_OF_POWER": _effect("APPLY", "STRENGTH", "ALL_ALLIES"),
    "GREMLIN_LEADER_ENCOURAGE": _effect("APPLY", "STRENGTH", "ALL_ALLIES") + _effect("GAIN", "BLOCK"),
    "GREMLIN_LEADER_RALLY": _effect("SUMMON", "GREMLIN", "EMPTY_SLOTS"),
    "MYSTIC_BUFF": _effect("APPLY", "STRENGTH", "ALL_ALLIES"),
    "MYSTIC_HEAL": _effect("HEAL", "HP", "ALL_ALLIES"),
    "NEMESIS_DEBUFF": _effect("ADD_CARD", "BURN", "DISCARD_PILE"),
    "REPULSOR_REPULSE": _effect("ADD_CARD", "DAZED", "DRAW_PILE"),
    "ROMEO_MOCK": _effect("PREPARE", "AGONIZING_SLASH"),
    "SENTRY_BOLT": _effect("ADD_CARD", "DAZED", "DISCARD_PILE"),
    "SLIME_BOSS_PREPARING": _effect("PREPARE", "SLAM"),
    "SNAKE_PLANT_ENFEEBLING_SPORES": _effect("APPLY", "FRAIL", "TARGET_PLAYER") + _effect("APPLY", "WEAK", "TARGET_PLAYER"),
    "SNECKO_PERPLEXING_GLARE": _effect("APPLY", "CONFUSED", "TARGET_PLAYER"),
    "SPHERIC_GUARDIAN_ACTIVATE": _effect("GAIN", "BLOCK"),
    "SPIKER_SPIKE": _effect("APPLY", "THORNS"),
    "SPIRE_GROWTH_CONSTRICT": _effect("APPLY", "CONSTRICTED", "TARGET_PLAYER"),
    "SPIRE_SPEAR_PIERCER": _effect("APPLY", "STRENGTH"),
    "THE_CHAMP_TAUNT": _effect("APPLY", "VULNERABLE", "TARGET_PLAYER"),
    "THE_CHAMP_GLOAT": _effect("APPLY", "STRENGTH"),
    "THE_CHAMP_ANGER": _effect("APPLY", "STRENGTH"),
    "THE_CHAMP_DEFENSIVE_STANCE": _effect("APPLY", "METALLICIZE") + _effect("CLEANSE", "DEBUFFS"),
    "THE_COLLECTOR_BUFF": _effect("APPLY", "STRENGTH", "ALL_ALLIES"),
    "THE_COLLECTOR_MEGA_DEBUFF": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "VULNERABLE", "TARGET_PLAYER") + _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "THE_GUARDIAN_VENT_STEAM": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "VULNERABLE", "TARGET_PLAYER"),
    "THE_GUARDIAN_DEFENSIVE_MODE": _effect("GAIN", "BLOCK") + _effect("APPLY", "SHARP_HIDE"),
    "THE_MAW_ROAR": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "THE_MAW_DROOL": _effect("APPLY", "STRENGTH"),
    "TIME_EATER_RIPPLE": _effect("APPLY", "VULNERABLE", "TARGET_PLAYER") + _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "TIME_EATER_HASTE": _effect("HEAL", "HP") + _effect("GAIN", "BLOCK") + _effect("CLEANSE", "DEBUFFS"),
    "WRITHING_MASS_IMPLANT": _effect("ADD_CARD", "PARASITE", "DECK"),
    "WRITHING_MASS_WITHER": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "VULNERABLE", "TARGET_PLAYER"),
    "CORRUPT_HEART_DEBILITATE": _effect("APPLY", "WEAK", "TARGET_PLAYER") + _effect("APPLY", "VULNERABLE", "TARGET_PLAYER") + _effect("APPLY", "FRAIL", "TARGET_PLAYER"),
    "CORRUPT_HEART_BUFF": _effect("APPLY", "STRENGTH") + _effect("APPLY", "ARTIFACT"),
}


def intent_program(move: str) -> tuple[str, ...]:
    if not move or move in {"PAD", "INVALID"}:
        return ()
    return ("INTENT_BEGIN", f"MOVE_{move}", *_move_effects(move), "INTENT_END")


def status_program(entity: str, hp: int, max_hp: int, block: int,
                   powers: dict[str, int]) -> tuple[str, ...]:
    out = ["ENTITY_BEGIN", f"ENTITY_{entity}"]
    for field, value in (("HP", hp), ("MAX_HP", max_hp), ("BLOCK", block)):
        out.extend(("FIELD", field, *number_tokens(value), "STMT_END"))
    for power, amount in sorted(powers.items()):
        if amount:
            out.extend(("STATUS", f"STATUS_{power}", *number_tokens(amount), "STMT_END"))
    out.append("ENTITY_END")
    return tuple(out)


ALL_PROGRAM_TOKENS = {
    token for move in V.MOVE_VOCAB[1:] for token in intent_program(move)
}
ALL_PROGRAM_TOKENS.update({
    "ENTITY_BEGIN", "ENTITY_END", "ENTITY_PLAYER", "ENTITY_MONSTER", "FIELD",
    "HP", "MAX_HP", "BLOCK", "STATUS", "NUM_BEGIN", "NUM_END",
    "SIGN_NEGATIVE", "STMT_END", *(f"DIGIT_{d}" for d in range(10)),
    *(f"STATUS_{p}" for p in V.PLAYER_BUFF_VOCAB),
    *(f"STATUS_{p}" for p in V.MONSTER_BUFF_VOCAB),
})
TOKEN_VOCAB = (PAD, *sorted(ALL_PROGRAM_TOKENS - {PAD}))
TOKEN_INDEX = {token: i for i, token in enumerate(TOKEN_VOCAB)}
MAX_INTENT_TOKENS = max(map(len, (intent_program(m) for m in V.MOVE_VOCAB[1:])))
MAX_LIVE_INTENT_TOKENS = MAX_INTENT_TOKENS + 12
MAX_PLAYER_STATUS_TOKENS = 768
MAX_MONSTER_STATUS_TOKENS = 384


def live_intent_program(move: str, damage: int, hits: int) -> tuple[str, ...]:
    out = []
    for token in intent_program(move):
        if token == "CURRENT_DAMAGE":
            out.extend(number_tokens(damage))
        elif token == "CURRENT_HITS":
            out.extend(number_tokens(hits))
        else:
            out.append(token)
    return tuple(out)


def encoded_intent_programs() -> tuple[tuple[int, ...], ...]:
    rows = []
    for move in V.MOVE_VOCAB:
        ids = tuple(TOKEN_INDEX[t] for t in intent_program(move))
        rows.append(ids + (0,) * (MAX_INTENT_TOKENS - len(ids)))
    # Capacity rows are PAD so future vocab appends retain the tensor contract.
    rows.extend([(0,) * MAX_INTENT_TOKENS] * (V.MOVE_VOCAB_CAPACITY - len(rows)))
    return tuple(rows)


assert ATTACK_MOVES <= set(V.MOVE_VOCAB)
assert len(encoded_intent_programs()) == V.MOVE_VOCAB_CAPACITY

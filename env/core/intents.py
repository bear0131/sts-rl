"""Derive an INTENT_CATEGORY from a monster's current move — backend-agnostic.

The sim exposes the rolled move as a MonsterMoveId enum-name string (e.g.
"JAW_WORM_CHOMP", "CENTURION_DEFEND") plus the move's base damage. It does NOT
expose a rich intent-type enum (the upstream `isMoveAttack` is the only built-in
classifier — see MonsterMoves.h), so we classify here, in pure Python, shared by
the obs encoder and the renderer.

Two signals, in priority order:
  1. base_damage > 0  -> ATTACK. This is the authoritative attack signal (it is
     what the game's intent arrow is keyed on); a move that deals damage is an
     attack regardless of its name. Many moves are attack+buff/debuff hybrids
     (e.g. SPHERIC_GUARDIAN_ATTACK_DEBUFF) — we surface the attack, and the buff
     side is already visible via the target's buff vector next turn.
  2. otherwise, keyword-match the move name's action verb against the buckets
     below. Unmatched non-attack moves fall back to "buff" if the name looks
     like a self/ally power move, else "unknown".

This is a heuristic, deliberately coarse: it groups moves into the 9 categories
of vocab.INTENT_CATEGORY. The exact move id is ALSO in the obs (MOVE_VOCAB), so
the model can learn finer distinctions; this category is the cheap prior.
"""

# keyword (substring of the enum-name's action part) -> category. Checked in
# order; first hit wins. Tokens are matched against the full UPPER enum name.
_RULES = [
    ("stun",          ("STUNNED",)),
    ("sleep",         ("SLEEP", "ASLEEP", "DORMANT")),
    ("escape",        ("ESCAPE", "SMOKE_BOMB")),
    # strong/multi-target debuffs (Collector mega-debuff, Chosen hex, etc.)
    ("strong_debuff", ("MEGA_DEBUFF", "HEX", "DEBILITATE")),
    # ordinary debuffs applied to the player
    ("debuff",        ("DEBUFF", "ENTANGLE", "SPORES", "ENFEEBL", "CONSTRICT",
                       "SPIT_WEB", "GLARE", "MOCK", "TAUNT", "SCREAM", "WITHER",
                       "PERPLEXING")),
    # defensive moves (gain block / harden / shields, no damage)
    ("defend",        ("DEFEND", "HARDEN", "PROTECT", "FORTIFY", "DEFENSIVE",
                       "STANCE")),
    # self/ally buffs, summons, ramp-up, and heals (no damage to the player)
    ("buff",          ("BUFF", "BELLOW", "INCANTATION", "RITUAL", "GROW",
                       "BOOST", "CHARG", "RALLY", "ENCOURAGE", "PREPARING",
                       "ACTIVATE", "REGROW", "REINCARNATE", "REBIRTH", "SPLIT",
                       "SPAWN", "SUMMON", "HEAL", "HASTE", "ANGER", "GLOAT",
                       "FURY", "CIRCLE_OF_POWER", "SQUARE_OF_PROTECTION",
                       "VENT_STEAM", "COUNT", "IT_IS_TIME", "ROAR", "DROOL",
                       "FLY", "CAW", "IMPLANT", "ECHO")),
]


def derive(move_name, base_damage):
    """(move enum-name or None, base_damage) -> INTENT_CATEGORY string.

    Returns 'unknown' when there is no move info (move_name falsy / 'INVALID')
    — that is the PAD/not-yet-rolled case (e.g. a backend that has not exported
    the move id), which encodes to INTENT_CATEGORY index 0."""
    if base_damage and base_damage > 0:
        return "attack"
    if not move_name or move_name == "INVALID":
        return "unknown"
    name = move_name.upper()
    for category, tokens in _RULES:
        if any(tok in name for tok in tokens):
            return category
    return "unknown"

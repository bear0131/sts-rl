"""Combat observation encoder: canonical combat dict -> fixed-length float32 vector.

This module is the single source of truth for the obs layout. It is a PURE
function of the canonical combat state (the dict produced by
bridge.schemas.combat_state.normalize_sim / normalize_real), so the SAME encoder
serves both the sim backend (training) and the real-game backend (inference via
bridge). It must NOT import slaythespire.

============================ obs layout (v1) ============================
Everything is fixed-length with reserved/zero slots, so adding content later
(e.g. monster intent category) does NOT change the vector length — it only fills
a slot that was zero before. Length only changes when OBS_VERSION bumps.

  player block            (PLAYER_SCALARS)
    hp / max_hp / block / energy / turn          [scaled]
    + PLAYER_BUFFS buff amounts (fixed vocab)
  per-monster block x MAX_MONSTERS (rows zero-padded; absent monster = all zero)
    present / hp / max_hp / block                [scaled]
    base_damage / hits                           [this turn's incoming attack]
    intent_is_attack                             [derived: base_damage > 0]
    intent_category one-hot (INTENT_SLOTS)       [RESERVED — all zero in v1,
                                                  needs a cpp binding to fill]
    + MONSTER_BUFFS buff amounts (fixed vocab)
  hand: HAND_SLOTS slots, each a card index into CARD_VOCAB  [positional —
        the action "play hand[i]" maps to slot i, so order matters here]
  draw / discard / exhaust: count vector over CARD_VOCAB     [order-agnostic]

The exact sizes are computed below; OBS_SIZE is asserted stable in tests.
========================================================================
"""

import numpy as np

from .cards_vocab import CARD_VOCAB, CARD_INDEX, EMPTY

# Bump whenever the layout below changes in a way that invalidates checkpoints
# (reorder, resize, change scaling). Filling a previously-reserved zero slot with
# real data does NOT require a bump.
OBS_VERSION = 1

# ---- capacities (fixed; act-1 fits in 5 monsters, headroom for later acts) ----
MAX_MONSTERS = 5
HAND_SLOTS = 10          # real hand cap is 10
N_CARDS = len(CARD_VOCAB)

# ---- normalization scales (divisors; keep obs roughly in [0,1]) ----
HP_SCALE = 200.0
BLOCK_SCALE = 100.0
ENERGY_SCALE = 10.0
TURN_SCALE = 30.0
DMG_SCALE = 100.0
HITS_SCALE = 10.0
BUFF_SCALE = 30.0        # signed buffs (e.g. Strength) divided by this

# ---- fixed buff vocabularies (canonical power ids). Selected, not exhaustive:
# the common act-1 combat buffs. Unknown powers are dropped (they do not change
# the vector length). Extend append-only, same rule as the card vocab. ----
PLAYER_BUFFS = [
    "Strength", "Dexterity", "Focus", "Artifact",
    "Vulnerable", "Weakened", "Frail",          # canonical: Weak->Weakened
    "Ritual", "Metallicize", "Plated Armor", "Thorns",
    "Demon Form", "Flame Barrier", "Feel No Pain", "Rage",
]
MONSTER_BUFFS = [
    "Strength", "Artifact", "Vulnerable", "Weakened", "Frail",
    "Ritual", "Metallicize", "Plated Armor", "Thorns",
    "Anger",        # Gremlin Nob (sim: Enrage)
    "Curl Up", "Spore Cloud", "Angry", "Split",
]
PLAYER_BUFF_INDEX = {b: i for i, b in enumerate(PLAYER_BUFFS)}
MONSTER_BUFF_INDEX = {b: i for i, b in enumerate(MONSTER_BUFFS)}

# reserved one-hot for an intent category we cannot fill until the sim exports it
# (see DEVNOTES: monster intent/move_id not exported). Zeros in v1.
INTENT_SLOTS = 8

# ---- block sizes ----
PLAYER_SCALARS = 5                       # hp, max_hp, block, energy, turn
PLAYER_BLOCK = PLAYER_SCALARS + len(PLAYER_BUFFS)

MON_SCALARS = 7                          # present, hp, max_hp, block, base_dmg, hits, is_attack
MON_BLOCK = MON_SCALARS + INTENT_SLOTS + len(MONSTER_BUFFS)

HAND_BLOCK = HAND_SLOTS                   # one card-index per slot (kept as a value, see note)
PILE_BLOCK = N_CARDS                      # count vector, one per pile

OBS_SIZE = (
    PLAYER_BLOCK
    + MAX_MONSTERS * MON_BLOCK
    + HAND_BLOCK
    + 3 * PILE_BLOCK                       # draw, discard, exhaust
)


def _card_idx(card):
    return CARD_INDEX.get(card["id"], CARD_INDEX[EMPTY])


def encode(canon):
    """canonical combat dict -> np.float32 vector of length OBS_SIZE.

    NOTE on hand encoding: hand slots store the raw card index (a small int cast
    to float). That is deliberately a placeholder — a single scalar id is a poor
    network input (implies false ordinal structure between unrelated cards). v1
    keeps it simple to get the loop running; a later version should one-hot or
    embed each hand slot (that bump changes OBS_SIZE -> OBS_VERSION += 1). The
    actions module, not this scalar, is what the policy uses to pick a card.
    """
    out = np.zeros(OBS_SIZE, dtype=np.float32)
    off = 0

    # ---- player ----
    p = canon["player"]
    out[off + 0] = min(p["hp"], HP_SCALE) / HP_SCALE
    out[off + 1] = min(p["max_hp"], HP_SCALE) / HP_SCALE
    out[off + 2] = p["block"] / BLOCK_SCALE
    out[off + 3] = p["energy"] / ENERGY_SCALE
    out[off + 4] = canon["turn"] / TURN_SCALE
    for name, amt in p["powers"].items():
        j = PLAYER_BUFF_INDEX.get(name)
        if j is not None:
            out[off + PLAYER_SCALARS + j] = amt / BUFF_SCALE
    off += PLAYER_BLOCK

    # ---- monsters (positional, zero-padded) ----
    live = [m for m in canon["monsters"] if not m["gone"]]
    for i in range(MAX_MONSTERS):
        base = off + i * MON_BLOCK
        if i < len(live):
            m = live[i]
            out[base + 0] = 1.0
            out[base + 1] = min(m["hp"], HP_SCALE) / HP_SCALE
            out[base + 2] = min(m["max_hp"], HP_SCALE) / HP_SCALE
            out[base + 3] = m["block"] / BLOCK_SCALE
            out[base + 4] = m["base_damage"] / DMG_SCALE
            out[base + 5] = m["hits"] / HITS_SCALE
            out[base + 6] = 1.0 if m["base_damage"] > 0 else 0.0
            # INTENT_SLOTS one-hot: reserved, left zero in v1
            bbase = base + MON_SCALARS + INTENT_SLOTS
            for name, amt in m["powers"].items():
                j = MONSTER_BUFF_INDEX.get(name)
                if j is not None:
                    out[bbase + j] = amt / BUFF_SCALE
    off += MAX_MONSTERS * MON_BLOCK

    # ---- hand (positional) ----
    hand = canon["hand"]
    for i in range(HAND_SLOTS):
        out[off + i] = float(_card_idx(hand[i])) if i < len(hand) else 0.0
    off += HAND_BLOCK

    # ---- draw / discard / exhaust (count vectors) ----
    for pile in ("draw", "discard", "exhaust"):
        for c in canon[pile]:
            out[off + _card_idx(c)] += 1.0
        off += PILE_BLOCK

    assert off == OBS_SIZE, (off, OBS_SIZE)
    return out

"""Observation encoder v2: canonical state dict -> structured Dict observation.

This module is the single source of truth for the obs layout and the RL model's
input contract (see doc/day_0623/obs_design.md). It is a PURE function of the
canonical state — it does NOT import slaythespire — so the SAME encoder serves
the sim backend (training) and a real-game backend (inference via bridge).

Why a Dict, not a flat vector (obs_design.md §0.2): the model must know which
fields are categorical ids (card/monster/move/relic/...) vs continuous scalars,
and which ids share a vocabulary. A flat float vector throws that away — and it
is wrong to feed a categorical id as a float scalar (it invents a fake ordering:
"card 130 > card 25"). So:

  - categorical id groups -> int arrays of vocabulary indices (the model embeds
    them; embedding is a *model* parameter, never baked into the env — §0.1).
  - scalar groups          -> float arrays, normalized to roughly [0, 1].

Stability (§0.3): OBS_VERSION bumps whenever the key set, any field's shape, a
vocabulary's order, or a scalar's normalization changes — that invalidates old
checkpoints. Filling a reserved/PAD slot with real data is NOT a bump. The id
vocabularies are append-only and carry headroom (vocab.*_CAPACITY); the buff
groups are dense vectors of reserved length (vocab.*_BUFF_VEC).

The encoder accepts a single "obs canonical" dict assembled by the env, with
these optional top-level keys (all default to PAD/zero so a combat-only or
overworld-only canonical both encode):

    phase     : semantic phase string (vocab.PHASE_VOCAB)
    turn      : combat turn (int)
    player    : {hp,max_hp,block,energy,powers}
    monsters  : [{id,hp,max_hp,block,gone,base_damage,hits,powers,move,move_id}]
    hand/draw/discard/exhaust : [{id,up}, ...]
    global    : {character,ascension,act,floor,boss,relics,potions,gold,deck,boss_relics}
    options   : phase-specific decision context (map nodes / card offers / ...)
"""

import numpy as np
from gymnasium import spaces

from .cards_vocab import CARD_VOCAB, CARD_INDEX, EMPTY
from . import vocab as V
from . import intents

# Bump on any layout/order/scaling change (see module docstring). v1 was a flat
# Box; v2 is the structured Dict below.
OBS_VERSION = 2

# ---- capacities (fixed slot counts; reserved, PAD-filled) ----
MAX_MONSTERS = 5          # must equal actions.MAX_MONSTERS (slot i == target i)
HAND_SLOTS = 10           # real hand cap
N_CARDS = len(CARD_VOCAB)
R_MAX = 30                # held-relic id slots (PAD); see decision B (id list)
P_MAX = 5                 # potion slots
G_MAX = 12                # card-select grid candidate slots
OFFER_SLOTS = 3           # combat card-reward choices
MAP_EDGES = 3             # reachable next-row nodes (<=3 edges out of a node)

# ---- normalization scales (divisors; keep scalars ~[0,1]) ----
HP_SCALE = 200.0
BLOCK_SCALE = 100.0
ENERGY_SCALE = 10.0
TURN_SCALE = 30.0
DMG_SCALE = 100.0
HITS_SCALE = 10.0
BUFF_SCALE = 30.0         # signed buff amounts (e.g. Strength can be negative)
UPGRADE_SCALE = 5.0       # hand-card upgrade count (Searing Blow can stack)
ASCENSION_SCALE = 20.0
ACT_SCALE = 4.0
FLOOR_SCALE = 56.0
GOLD_SCALE = 999.0
MAP_X_SCALE = 6.0         # map columns 0..6

# ---- per-group scalar widths ----
PLAYER_SCALARS = 6        # hp, max_hp, block, energy, turn, hp_frac
MON_SCALARS = 6           # present, hp, hp_frac, block, base_damage, hits

# dtypes: categorical ids are integer indices, scalars are float32.
_CAT = np.int64
_SCALAR = np.float32


# ============================ observation space ============================

def obs_space():
    """The gymnasium spaces.Dict matching encode()'s output. Train with SB3
    MultiInputPolicy (decision C). Categorical groups are integer Boxes bounded
    by their vocabulary capacity; the model embeds them (env stays parameter-free)."""
    def cat(shape, n):
        return spaces.Box(low=0, high=n - 1, shape=shape, dtype=_CAT)

    def scal(shape, high=np.inf):
        return spaces.Box(low=-np.inf, high=high, shape=shape, dtype=_SCALAR)

    return spaces.Dict({
        # ---- player ----
        "player_scalars": scal((PLAYER_SCALARS,)),
        "player_buffs": scal((V.PLAYER_BUFF_VEC,)),
        # ---- monsters (fixed slots; slot i == action target i) ----
        "monster_id": cat((MAX_MONSTERS,), V.MONSTER_VOCAB_CAPACITY),
        "monster_move": cat((MAX_MONSTERS,), V.MOVE_VOCAB_CAPACITY),
        "monster_intent_cat": cat((MAX_MONSTERS,), len(V.INTENT_CATEGORY)),
        "monster_scalars": scal((MAX_MONSTERS, MON_SCALARS)),
        "monster_buffs": scal((MAX_MONSTERS, V.MONSTER_BUFF_VEC)),
        # ---- combat piles ----
        "hand_ids": cat((HAND_SLOTS,), N_CARDS),
        "hand_upgrades": scal((HAND_SLOTS,)),
        "draw_counts": scal((N_CARDS,)),
        "discard_counts": scal((N_CARDS,)),
        "exhaust_counts": scal((N_CARDS,)),
        # ---- run-global (every phase) ----
        "phase": cat((1,), len(V.PHASE_VOCAB)),
        "character": cat((1,), len(V.CHARACTER_VOCAB)),
        "ascension": scal((1,)),
        "act": scal((1,)),
        "floor": scal((1,)),
        "next_boss": cat((1,), V.BOSS_VOCAB_CAPACITY),
        "relics": cat((R_MAX,), V.RELIC_VOCAB_CAPACITY),
        "potions": cat((P_MAX,), V.POTION_VOCAB_CAPACITY),
        "gold": scal((1,)),
        "deck_counts": scal((N_CARDS,)),
        # ---- overworld decision context (phase-gated; PAD otherwise) ----
        "offer_card_ids": cat((OFFER_SLOTS,), N_CARDS),
        "select_card_ids": cat((G_MAX,), N_CARDS),
        "map_node_rooms": cat((MAP_EDGES,), len(V.ROOM_VOCAB)),
        "map_node_x": scal((MAP_EDGES,)),
        "boss_relic_ids": cat((OFFER_SLOTS,), V.RELIC_VOCAB_CAPACITY),
        "event_id": cat((1,), V.EVENT_VOCAB_CAPACITY),
    })


# ============================== index helpers ==============================

def _card_idx(card):
    return CARD_INDEX.get(card["id"], CARD_INDEX[EMPTY])


def _idx(index_map, key):
    return index_map.get(key, 0)            # 0 == PAD for every id vocabulary


def _empty():
    """A zero/PAD-filled obs dict (every absent group encodes to this)."""
    return {
        "player_scalars": np.zeros(PLAYER_SCALARS, _SCALAR),
        "player_buffs": np.zeros(V.PLAYER_BUFF_VEC, _SCALAR),
        "monster_id": np.zeros(MAX_MONSTERS, _CAT),
        "monster_move": np.zeros(MAX_MONSTERS, _CAT),
        "monster_intent_cat": np.zeros(MAX_MONSTERS, _CAT),
        "monster_scalars": np.zeros((MAX_MONSTERS, MON_SCALARS), _SCALAR),
        "monster_buffs": np.zeros((MAX_MONSTERS, V.MONSTER_BUFF_VEC), _SCALAR),
        "hand_ids": np.zeros(HAND_SLOTS, _CAT),
        "hand_upgrades": np.zeros(HAND_SLOTS, _SCALAR),
        "draw_counts": np.zeros(N_CARDS, _SCALAR),
        "discard_counts": np.zeros(N_CARDS, _SCALAR),
        "exhaust_counts": np.zeros(N_CARDS, _SCALAR),
        "phase": np.zeros(1, _CAT),
        "character": np.zeros(1, _CAT),
        "ascension": np.zeros(1, _SCALAR),
        "act": np.zeros(1, _SCALAR),
        "floor": np.zeros(1, _SCALAR),
        "next_boss": np.zeros(1, _CAT),
        "relics": np.zeros(R_MAX, _CAT),
        "potions": np.zeros(P_MAX, _CAT),
        "gold": np.zeros(1, _SCALAR),
        "deck_counts": np.zeros(N_CARDS, _SCALAR),
        "offer_card_ids": np.zeros(OFFER_SLOTS, _CAT),
        "select_card_ids": np.zeros(G_MAX, _CAT),
        "map_node_rooms": np.zeros(MAP_EDGES, _CAT),
        "map_node_x": np.zeros(MAP_EDGES, _SCALAR),
        "boss_relic_ids": np.zeros(OFFER_SLOTS, _CAT),
        "event_id": np.zeros(1, _CAT),
    }


# ================================ encoders ================================

def _encode_combat(o, canon):
    """Fill the combat groups in-place from a combat canonical (player/monsters/
    hand/piles). No-op if `player` is absent (non-combat phase)."""
    p = canon.get("player")
    if p is None:
        return
    hp, max_hp = p["hp"], p["max_hp"]
    o["player_scalars"][:] = (
        min(hp, HP_SCALE) / HP_SCALE,
        min(max_hp, HP_SCALE) / HP_SCALE,
        p["block"] / BLOCK_SCALE,
        p["energy"] / ENERGY_SCALE,
        canon.get("turn", 0) / TURN_SCALE,
        (hp / max_hp) if max_hp else 0.0,
    )
    for name, amt in p["powers"].items():
        j = V.PLAYER_BUFF_VOCAB_INDEX.get(name)
        if j is not None:
            o["player_buffs"][j] = amt / BUFF_SCALE

    # Monsters keep FIXED slots (obs_design.md §2.2): monster i stays in slot i;
    # a dead one is marked present=0 rather than filtered (filtering shifts slots
    # and breaks the slot<->action-target correspondence). We therefore index by
    # position, not by a "live" filter.
    monsters = canon.get("monsters", [])
    for i in range(MAX_MONSTERS):
        if i >= len(monsters):
            break
        m = monsters[i]
        present = 0.0 if m["gone"] else 1.0
        o["monster_id"][i] = _idx(V.MONSTER_VOCAB_INDEX, m["id"])
        o["monster_move"][i] = m.get("move_id", 0) or 0
        o["monster_intent_cat"][i] = V.INTENT_CATEGORY_INDEX[
            intents.derive(m.get("move"), m["base_damage"])
        ]
        mhp, mmax = m["hp"], m["max_hp"]
        o["monster_scalars"][i] = (
            present,
            min(mhp, HP_SCALE) / HP_SCALE,
            (mhp / mmax) if mmax else 0.0,
            m["block"] / BLOCK_SCALE,
            m["base_damage"] / DMG_SCALE,
            m["hits"] / HITS_SCALE,
        )
        for name, amt in m["powers"].items():
            j = V.MONSTER_BUFF_VOCAB_INDEX.get(name)
            if j is not None:
                o["monster_buffs"][i, j] = amt / BUFF_SCALE

    hand = canon.get("hand", [])
    for i in range(HAND_SLOTS):
        if i < len(hand):
            o["hand_ids"][i] = _card_idx(hand[i])
            o["hand_upgrades"][i] = hand[i].get("up", 0) / UPGRADE_SCALE
    for pile, key in (("draw", "draw_counts"), ("discard", "discard_counts"),
                      ("exhaust", "exhaust_counts")):
        for c in canon.get(pile, []):
            o[key][_card_idx(c)] += 1.0


def _encode_global(o, canon):
    """Fill the run-global groups from canon['global'] (present every phase)."""
    o["phase"][0] = V.PHASE_VOCAB_INDEX.get(canon.get("phase", "none"), 0)
    g = canon.get("global")
    if g is None:
        return
    o["character"][0] = V.CHARACTER_VOCAB_INDEX.get(g.get("character", "INVALID"), 0)
    o["ascension"][0] = g.get("ascension", 0) / ASCENSION_SCALE
    o["act"][0] = g.get("act", 0) / ACT_SCALE
    o["floor"][0] = g.get("floor", 0) / FLOOR_SCALE
    o["next_boss"][0] = _idx(V.BOSS_VOCAB_INDEX, g.get("boss", "PAD"))
    o["gold"][0] = g.get("gold", 0) / GOLD_SCALE
    # relics: multi-hot id LIST padded to R_MAX (decision B). Order = hold order.
    for i, rid in enumerate(g.get("relics", [])[:R_MAX]):
        o["relics"][i] = _idx(V.RELIC_VOCAB_INDEX, rid)
    # potions: slot-aligned ids; empty/INVALID names -> PAD 0.
    for i, pid in enumerate(g.get("potions", [])[:P_MAX]):
        o["potions"][i] = _idx(V.POTION_VOCAB_INDEX, pid)
    for cid, _up in g.get("deck", []):
        o["deck_counts"][CARD_INDEX.get(cid, CARD_INDEX[EMPTY])] += 1.0


def _encode_options(o, canon):
    """Fill phase-gated decision-context groups from canon['options'] /
    canon['global']['boss_relics']. Each group is only meaningful in its phase;
    encode() leaves the rest PAD."""
    phase = canon.get("phase", "none")
    opts = canon.get("options") or {}

    if phase == "combat_reward":
        for j, (cid, _up) in enumerate(opts.get("cards", [])[:OFFER_SLOTS]):
            o["offer_card_ids"][j] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
    elif phase == "card_select":
        for k, (cid, _up) in enumerate(opts.get("cards", [])[:G_MAX]):
            o["select_card_ids"][k] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
    elif phase == "map":
        for j, (x, room) in enumerate(opts.get("nodes", [])[:MAP_EDGES]):
            o["map_node_rooms"][j] = V.ROOM_VOCAB_INDEX.get(room, 0)
            o["map_node_x"][j] = x / MAP_X_SCALE
    elif phase == "event":
        g = canon.get("global") or {}
        # event id lives in the options (env reads gc.cur_event_name into it)
        o["event_id"][0] = _idx(V.EVENT_VOCAB_INDEX, opts.get("event", "PAD"))

    if phase == "boss_relic":
        g = canon.get("global") or {}
        for j, rid in enumerate(g.get("boss_relics", [])[:OFFER_SLOTS]):
            o["boss_relic_ids"][j] = _idx(V.RELIC_VOCAB_INDEX, rid)


def encode(canon):
    """obs canonical dict -> obs dict (numpy arrays matching obs_space()).

    `canon` may carry only the combat block (a normalize_sim result), only the
    overworld block, or the full assembled dict — absent groups encode to PAD."""
    o = _empty()
    _encode_combat(o, canon)
    _encode_global(o, canon)
    _encode_options(o, canon)
    return o

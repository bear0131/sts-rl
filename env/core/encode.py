"""Observation encoder v3: canonical state dict -> structured Dict observation.

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
    arg_step  : thin-env autoregressive sub-step (vocab.ARG_STEP_VOCAB; PAD=root)
    turn      : combat turn (int)
    player    : {hp,max_hp,block,energy,powers}
    monsters  : [{id,hp,max_hp,block,gone,base_damage,adjusted_damage,
                  hits,powers,move,move_id}]
    hand      : [{id,up,cost}, ...] (cost is current cost for this turn)
    draw/discard/exhaust : [{id,up}, ...]
    global    : {character,ascension,act,floor,boss,relics,potions,gold,deck,boss_relics}
    options   : phase-specific decision context (map nodes / card offers / ...)
"""

import numpy as np
from gymnasium import spaces

from .cards_vocab import CARD_VOCAB, CARD_INDEX, EMPTY
from .ironclad_cards import card_numbers
from .card_language import LANGUAGE_CARDS
from . import vocab as V
from . import intents
from . import combat_language as CL

# Bump on any layout/order/scaling change (see module docstring). v1 was a flat
# Box; v2 is the structured Dict below. v3 adds `arg_step` (the thin
# SlayTheSpireEnv's autoregressive sub-step, action_design.md §1). v4 adds the
# shop shelf groups (shop_*: card/relic/potion ids + prices + remove cost). v5
# adds the card-select screen contract (obs_design §4.1): select_context /
# pick_remaining / can_pick_zero / can_pick_any / selected_card_ids, and widens
# select_card_ids G_MAX 12->64 (full-deck select screens). v6 adds
# select_hand_mask (obs_design §4.2): a hand-aligned 0/1 vector of the cards the
# thin env has selected on a combat MULTI screen (Exhaust Many / Gamble), so the
# feed-forward policy sees its own pending pick (mask != obs, §0.5).
# The optional global-map layout is version 8: it appends the full 15x7 map DAG
# only when ``global_map_view=True``. The default v7 layout stays compatible
# with existing combat and whole-run checkpoints. The optional card-language
# layout is version 9: it appends compact, upgrade-aware language-card sequences
# for draw/discard/deck while retaining the complete count vectors.
# The optional deck-size layout is version 10: it adds the explicit normalized
# total held-card count while retaining deck_counts as the composition signal.
OBS_VERSION = 7
GLOBAL_MAP_OBS_VERSION = 8
CARD_LANGUAGE_OBS_VERSION = 9
DECK_SIZE_OBS_VERSION = 10

# ---- capacities (fixed slot counts; reserved, PAD-filled) ----
MAX_MONSTERS = 5          # must equal actions.MAX_MONSTERS (slot i == target i)
HAND_SLOTS = 10           # real hand cap
N_CARDS = len(CARD_VOCAB)
R_MAX = 30                # held-relic id slots (PAD); see decision B (id list)
P_MAX = 5                 # potion slots
G_MAX = 64                # card-select grid candidate slots (full deck: a Remove/
                          # Transform/Smith grid is the whole deck, 30-40+ cards
                          # mid/late run; 12 truncated it. 64 is the gen-1 cap.)
SELECT_PICK_SLOTS = 3     # overworld grid's max simultaneous picks (sim's
                          # haveSelectedCards is fixed_list<3>: Astrolabe=3 max)
OFFER_SLOTS = 3           # combat card-reward choices
MAP_EDGES = 3             # reachable next-row nodes (<=3 edges out of a node)
MAP_ROWS = 15
MAP_COLS = 7
MAP_FULL_SLOTS = MAP_ROWS * MAP_COLS
MAP_FULL_FEATURES = 12    # x, y, current, choice, reachable, 7 outgoing edges
SHOP_CARD_SLOTS = 7       # shop shelf: cards[7] (slot == buy action id)
SHOP_RELIC_SLOTS = 3      # shop shelf: relics[3]
SHOP_POTION_SLOTS = 3     # shop shelf: potions[3]
NEOW_SLOTS = 4            # Neow always offers four blessing choices
NEOW_BONUS_CAPACITY = 32
NEOW_DRAWBACK_CAPACITY = 8
PILE_LANGUAGE_SLOTS = len(LANGUAGE_CARDS)  # one slot per language-covered card id

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
DECK_SIZE_SCALE = 100.0
MAP_X_SCALE = 6.0         # map columns 0..6
PICK_SCALE = 10.0         # cards-remaining-to-pick (pickCount up to a handful)

# ---- per-group scalar widths ----
PLAYER_SCALARS = 6        # hp, max_hp, block, energy, turn, hp_frac
MON_SCALARS = 6           # present, hp, hp_frac, block, adjusted_damage, hits

# dtypes: categorical ids are integer indices, scalars are float32.
_CAT = np.int64
_SCALAR = np.float32


# ============================ observation space ============================

def obs_space(*, global_map_view=False, card_language=False,
              combat_language=False, deck_size_observation=False,
              neow_observation=False):
    """The gymnasium spaces.Dict matching encode()'s output. Train with SB3
    MultiInputPolicy (decision C). Categorical groups are integer Boxes bounded
    by their vocabulary capacity; the model embeds them (env stays parameter-free)."""
    def cat(shape, n):
        return spaces.Box(low=0, high=n - 1, shape=shape, dtype=_CAT)

    def scal(shape, high=np.inf):
        return spaces.Box(low=-np.inf, high=high, shape=shape, dtype=_SCALAR)

    groups = {
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
        "hand_costs": scal((HAND_SLOTS,)),
        "hand_damage": scal((HAND_SLOTS,)),
        "hand_block": scal((HAND_SLOTS,)),
        "draw_counts": scal((N_CARDS,)),
        "discard_counts": scal((N_CARDS,)),
        "exhaust_counts": scal((N_CARDS,)),
        # ---- run-global (every phase) ----
        "phase": cat((1,), len(V.PHASE_VOCAB)),
        # autoregressive sub-step (action_design.md §1): which step of a multi-step
        # action the thin env is on (root / pick_target / pick_card / ...). Tells
        # the model WHY the current action menu is shaped the way it is. PAD = root.
        "arg_step": cat((1,), len(V.ARG_STEP_VOCAB)),
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
        # card-select screen semantics + progress (obs_design §4.1). select_context
        # says WHAT this screen is for (upgrade/exhaust/transform/remove/...) — the
        # legal card set can be identical across two opposite-intent screens, so the
        # purpose must enter obs, not just the mask. pick_remaining/can_pick_* tell
        # the model how many more to pick and whether it may stop. selected_card_ids
        # are the cards already picked THIS overworld grid (the sim moves a picked
        # card out of the grid into here; combat never accumulates -> PAD there).
        "select_context": cat((1,), V.SELECT_CONTEXT_VOCAB_CAPACITY),
        "pick_remaining": scal((1,)),
        "can_pick_zero": scal((1,)),
        "can_pick_any": scal((1,)),
        "selected_card_ids": cat((SELECT_PICK_SLOTS,), N_CARDS),
        # combat MULTI screen (Exhaust Many / Gamble): hand-aligned 0/1 vector of
        # the cards the thin env has selected so far (obs_design §4.2). slot h ==
        # the PICK_CARD[h] id == hand slot h. PAD (all 0) off a MULTI screen.
        "select_hand_mask": scal((HAND_SLOTS,)),
        "map_node_rooms": cat((MAP_EDGES,), len(V.ROOM_VOCAB)),
        "map_node_x": scal((MAP_EDGES,)),
        "boss_relic_ids": cat((OFFER_SLOTS,), V.RELIC_VOCAB_CAPACITY),
        "event_id": cat((1,), V.EVENT_VOCAB_CAPACITY),
        # ---- shop shelf (phase-gated; slot i == buy action id; PAD otherwise) ----
        "shop_card_ids": cat((SHOP_CARD_SLOTS,), N_CARDS),
        "shop_card_prices": scal((SHOP_CARD_SLOTS,)),
        "shop_relic_ids": cat((SHOP_RELIC_SLOTS,), V.RELIC_VOCAB_CAPACITY),
        "shop_relic_prices": scal((SHOP_RELIC_SLOTS,)),
        "shop_potion_ids": cat((SHOP_POTION_SLOTS,), V.POTION_VOCAB_CAPACITY),
        "shop_potion_prices": scal((SHOP_POTION_SLOTS,)),
        "shop_remove_cost": scal((1,)),
    }
    if neow_observation:
        groups.update({
            # ---- Neow choices (slot i == NEOW[i] action id) ----
            "neow_bonus_ids": cat((NEOW_SLOTS,), NEOW_BONUS_CAPACITY),
            "neow_drawback_ids": cat((NEOW_SLOTS,), NEOW_DRAWBACK_CAPACITY),
            "neow_features": scal((NEOW_SLOTS, 2)),
        })
    if global_map_view:
        groups.update({
            "map_full_rooms": cat((MAP_FULL_SLOTS,), len(V.ROOM_VOCAB)),
            "map_full_features": scal(
                (MAP_FULL_SLOTS, MAP_FULL_FEATURES), high=1.0
            ),
        })
    if card_language:
        for pile in ("draw", "discard", "deck"):
            groups[f"{pile}_card_ids"] = cat((PILE_LANGUAGE_SLOTS,), N_CARDS)
            groups[f"{pile}_card_upgrades"] = scal((PILE_LANGUAGE_SLOTS,))
            groups[f"{pile}_card_counts"] = scal((PILE_LANGUAGE_SLOTS,))
            groups[f"{pile}_card_upgraded_counts"] = scal(
                (PILE_LANGUAGE_SLOTS,)
            )
    if combat_language:
        groups.update({
            "player_status_language": cat(
                (CL.MAX_PLAYER_STATUS_TOKENS,), len(CL.TOKEN_VOCAB)
            ),
            "monster_status_language": cat(
                (MAX_MONSTERS, CL.MAX_MONSTER_STATUS_TOKENS),
                len(CL.TOKEN_VOCAB),
            ),
            "monster_intent_language": cat(
                (MAX_MONSTERS, CL.MAX_LIVE_INTENT_TOKENS),
                len(CL.TOKEN_VOCAB),
            ),
        })
    if deck_size_observation:
        groups["deck_size"] = scal((1,), high=1.0)
    return spaces.Dict(groups)


# ============================== index helpers ==============================

def _card_idx(card):
    return CARD_INDEX.get(card["id"], CARD_INDEX[EMPTY])


def _idx(index_map, key):
    return index_map.get(key, 0)            # 0 == PAD for every id vocabulary


def _encode_card_sequence(o, pile, cards):
    """Compact every language-covered card ID without truncating duplicates."""
    ids_key = f"{pile}_card_ids"
    if ids_key not in o:
        return
    grouped = {}
    for card in cards:
        if isinstance(card, dict):
            card_id, upgrades = card["id"], card.get("up", 0)
        else:
            card_id, upgrades = card
        if card_id not in LANGUAGE_CARDS:
            continue
        stats = grouped.setdefault(card_id, [0, 0, 0])
        stats[0] += 1
        stats[1] += int(upgrades > 0)
        stats[2] += upgrades
    assert len(grouped) <= PILE_LANGUAGE_SLOTS
    for i, (card_id, (count, upgraded_count, upgrade_sum)) in enumerate(
        grouped.items()
    ):
        o[ids_key][i] = CARD_INDEX.get(card_id, CARD_INDEX[EMPTY])
        o[f"{pile}_card_counts"][i] = count
        o[f"{pile}_card_upgraded_counts"][i] = upgraded_count
        o[f"{pile}_card_upgrades"][i] = upgrade_sum / UPGRADE_SCALE


def _empty(*, global_map_view=False, card_language=False,
           combat_language=False, deck_size_observation=False,
           neow_observation=False):
    """A zero/PAD-filled obs dict (every absent group encodes to this)."""
    o = {
        "player_scalars": np.zeros(PLAYER_SCALARS, _SCALAR),
        "player_buffs": np.zeros(V.PLAYER_BUFF_VEC, _SCALAR),
        "monster_id": np.zeros(MAX_MONSTERS, _CAT),
        "monster_move": np.zeros(MAX_MONSTERS, _CAT),
        "monster_intent_cat": np.zeros(MAX_MONSTERS, _CAT),
        "monster_scalars": np.zeros((MAX_MONSTERS, MON_SCALARS), _SCALAR),
        "monster_buffs": np.zeros((MAX_MONSTERS, V.MONSTER_BUFF_VEC), _SCALAR),
        "hand_ids": np.zeros(HAND_SLOTS, _CAT),
        "hand_upgrades": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_costs": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_damage": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_block": np.zeros(HAND_SLOTS, _SCALAR),
        "draw_counts": np.zeros(N_CARDS, _SCALAR),
        "discard_counts": np.zeros(N_CARDS, _SCALAR),
        "exhaust_counts": np.zeros(N_CARDS, _SCALAR),
        "phase": np.zeros(1, _CAT),
        "arg_step": np.zeros(1, _CAT),
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
        "select_context": np.zeros(1, _CAT),
        "pick_remaining": np.zeros(1, _SCALAR),
        "can_pick_zero": np.zeros(1, _SCALAR),
        "can_pick_any": np.zeros(1, _SCALAR),
        "selected_card_ids": np.zeros(SELECT_PICK_SLOTS, _CAT),
        "select_hand_mask": np.zeros(HAND_SLOTS, _SCALAR),
        "map_node_rooms": np.zeros(MAP_EDGES, _CAT),
        "map_node_x": np.zeros(MAP_EDGES, _SCALAR),
        "boss_relic_ids": np.zeros(OFFER_SLOTS, _CAT),
        "event_id": np.zeros(1, _CAT),
        "shop_card_ids": np.zeros(SHOP_CARD_SLOTS, _CAT),
        "shop_card_prices": np.zeros(SHOP_CARD_SLOTS, _SCALAR),
        "shop_relic_ids": np.zeros(SHOP_RELIC_SLOTS, _CAT),
        "shop_relic_prices": np.zeros(SHOP_RELIC_SLOTS, _SCALAR),
        "shop_potion_ids": np.zeros(SHOP_POTION_SLOTS, _CAT),
        "shop_potion_prices": np.zeros(SHOP_POTION_SLOTS, _SCALAR),
        "shop_remove_cost": np.zeros(1, _SCALAR),
    }
    if neow_observation:
        o["neow_bonus_ids"] = np.zeros(NEOW_SLOTS, _CAT)
        o["neow_drawback_ids"] = np.zeros(NEOW_SLOTS, _CAT)
        o["neow_features"] = np.zeros((NEOW_SLOTS, 2), _SCALAR)
    if global_map_view:
        o["map_full_rooms"] = np.zeros(MAP_FULL_SLOTS, _CAT)
        o["map_full_features"] = np.zeros(
            (MAP_FULL_SLOTS, MAP_FULL_FEATURES), _SCALAR
        )
    if card_language:
        for pile in ("draw", "discard", "deck"):
            o[f"{pile}_card_ids"] = np.zeros(PILE_LANGUAGE_SLOTS, _CAT)
            o[f"{pile}_card_upgrades"] = np.zeros(
                PILE_LANGUAGE_SLOTS, _SCALAR
            )
            o[f"{pile}_card_counts"] = np.zeros(
                PILE_LANGUAGE_SLOTS, _SCALAR
            )
            o[f"{pile}_card_upgraded_counts"] = np.zeros(
                PILE_LANGUAGE_SLOTS, _SCALAR
            )
    if combat_language:
        o["player_status_language"] = np.zeros(
            CL.MAX_PLAYER_STATUS_TOKENS, _CAT
        )
        o["monster_status_language"] = np.zeros(
            (MAX_MONSTERS, CL.MAX_MONSTER_STATUS_TOKENS), _CAT
        )
        o["monster_intent_language"] = np.zeros(
            (MAX_MONSTERS, CL.MAX_LIVE_INTENT_TOKENS), _CAT
        )
    if deck_size_observation:
        o["deck_size"] = np.zeros(1, _SCALAR)
    return o


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
    if "player_status_language" in o:
        program = CL.status_program("PLAYER", hp, max_hp, p["block"], p["powers"])
        ids = [CL.TOKEN_INDEX.get(token, 0) for token in program]
        ids = ids[:CL.MAX_PLAYER_STATUS_TOKENS]
        o["player_status_language"][:len(ids)] = ids

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
            m.get("adjusted_damage", m["base_damage"]) / DMG_SCALE,
            m["hits"] / HITS_SCALE,
        )
        for name, amt in m["powers"].items():
            j = V.MONSTER_BUFF_VOCAB_INDEX.get(name)
            if j is not None:
                o["monster_buffs"][i, j] = amt / BUFF_SCALE
        if "monster_status_language" in o:
            status = CL.status_program(
                "MONSTER", mhp, mmax, m["block"], m["powers"]
            )
            status_ids = [CL.TOKEN_INDEX.get(token, 0) for token in status]
            status_ids = status_ids[:CL.MAX_MONSTER_STATUS_TOKENS]
            o["monster_status_language"][i, :len(status_ids)] = status_ids
            intent = CL.live_intent_program(
                m.get("move", "PAD"),
                m.get("adjusted_damage", m["base_damage"]),
                m["hits"],
            )
            intent_ids = [CL.TOKEN_INDEX.get(token, 0) for token in intent]
            intent_ids = intent_ids[:CL.MAX_LIVE_INTENT_TOKENS]
            o["monster_intent_language"][i, :len(intent_ids)] = intent_ids

    hand = canon.get("hand", [])
    for i in range(HAND_SLOTS):
        if i < len(hand):
            o["hand_ids"][i] = _card_idx(hand[i])
            upgrades = hand[i].get("up", 0)
            o["hand_upgrades"][i] = upgrades / UPGRADE_SCALE
            numbers = card_numbers(hand[i]["id"], upgrades)
            if numbers is not None:
                printed_cost, damage, block = numbers
                o["hand_costs"][i] = hand[i].get("cost", printed_cost) / ENERGY_SCALE
                o["hand_damage"][i] = damage / DMG_SCALE
                o["hand_block"][i] = block / BLOCK_SCALE
            elif "cost" in hand[i]:
                o["hand_costs"][i] = hand[i]["cost"] / ENERGY_SCALE
    for pile, key in (("draw", "draw_counts"), ("discard", "discard_counts"),
                      ("exhaust", "exhaust_counts")):
        for c in canon.get(pile, []):
            o[key][_card_idx(c)] += 1.0
    _encode_card_sequence(o, "draw", canon.get("draw", []))
    _encode_card_sequence(o, "discard", canon.get("discard", []))


def _encode_global(o, canon):
    """Fill the run-global groups from canon['global'] (present every phase)."""
    o["phase"][0] = V.PHASE_VOCAB_INDEX.get(canon.get("phase", "none"), 0)
    # arg_step rides at top level (the thin env sets it; combat-only canonicals
    # omit it -> root, index 0). Kept here so it is filled every phase.
    o["arg_step"][0] = V.ARG_STEP_VOCAB_INDEX.get(canon.get("arg_step", "root"), 0)
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
    if "deck_size" in o:
        o["deck_size"][0] = min(
            len(g.get("deck", [])) / DECK_SIZE_SCALE, 1.0
        )
    _encode_card_sequence(o, "deck", g.get("deck", []))


def _encode_options(o, canon):
    """Fill phase-gated decision-context groups from canon['options'] /
    canon['global']['boss_relics']. Each group is only meaningful in its phase;
    encode() leaves the rest PAD."""
    phase = canon.get("phase", "none")
    opts = canon.get("options") or {}

    if phase == "combat_reward":
        for j, (cid, _up) in enumerate(opts.get("cards", [])[:OFFER_SLOTS]):
            o["offer_card_ids"][j] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
    elif phase == "event" and opts.get("neow") and "neow_bonus_ids" in o:
        for i, option in enumerate(opts.get("neow", [])[:NEOW_SLOTS]):
            bonus_id = int(option.get("bonus_id", 0))
            drawback_id = int(option.get("drawback_id", 0))
            o["neow_bonus_ids"][i] = max(
                0, min(NEOW_BONUS_CAPACITY - 1, bonus_id)
            )
            o["neow_drawback_ids"][i] = max(
                0, min(NEOW_DRAWBACK_CAPACITY - 1, drawback_id)
            )
            o["neow_features"][i] = (
                i / max(1, NEOW_SLOTS - 1),
                1.0 if drawback_id > 1 else 0.0,
            )
    elif phase == "card_select":
        for k, (cid, _up) in enumerate(opts.get("cards", [])[:G_MAX]):
            o["select_card_ids"][k] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
    elif phase == "map":
        for j, (x, room) in enumerate(opts.get("nodes", [])[:MAP_EDGES]):
            o["map_node_rooms"][j] = V.ROOM_VOCAB_INDEX.get(room, 0)
            o["map_node_x"][j] = x / MAP_X_SCALE
        if "map_full_rooms" in o:
            for node in opts.get("full_map", []):
                x, y = int(node["x"]), int(node["y"])
                if not (0 <= x < MAP_COLS and 0 <= y < MAP_ROWS):
                    continue
                slot = y * MAP_COLS + x
                o["map_full_rooms"][slot] = V.ROOM_VOCAB_INDEX.get(
                    node["room"], 0
                )
                features = o["map_full_features"][slot]
                features[:5] = (
                    x / MAP_X_SCALE,
                    y / (MAP_ROWS - 1),
                    float(bool(node.get("current"))),
                    float(bool(node.get("choice"))),
                    float(bool(node.get("reachable"))),
                )
                for child_x in node.get("children", []):
                    if 0 <= child_x < MAP_COLS:
                        features[5 + child_x] = 1.0
    elif phase == "event":
        g = canon.get("global") or {}
        # event id lives in the options (env reads gc.cur_event_name into it)
        o["event_id"][0] = _idx(V.EVENT_VOCAB_INDEX, opts.get("event", "PAD"))
    elif phase == "shop":
        # Positional shelf: slot i lines up with the SHOP_* buy action id. A sold
        # slot has price -1 (shop_contents) -> PAD id + price 0 (decision: a shop
        # has no "originally empty" slot, so -1 means "already bought").
        for i, (cid, price) in enumerate(opts.get("shop_cards", [])[:SHOP_CARD_SLOTS]):
            if price != -1:
                o["shop_card_ids"][i] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
                o["shop_card_prices"][i] = price / GOLD_SCALE
        for i, (rid, price) in enumerate(opts.get("shop_relics", [])[:SHOP_RELIC_SLOTS]):
            if price != -1:
                o["shop_relic_ids"][i] = _idx(V.RELIC_VOCAB_INDEX, rid)
                o["shop_relic_prices"][i] = price / GOLD_SCALE
        for i, (pid, price) in enumerate(opts.get("shop_potions", [])[:SHOP_POTION_SLOTS]):
            if price != -1:
                o["shop_potion_ids"][i] = _idx(V.POTION_VOCAB_INDEX, pid)
                o["shop_potion_prices"][i] = price / GOLD_SCALE
        rc = opts.get("shop_remove_cost", -1)
        if rc != -1:
            o["shop_remove_cost"][0] = rc / GOLD_SCALE

    if phase == "boss_relic":
        g = canon.get("global") or {}
        for j, rid in enumerate(g.get("boss_relics", [])[:OFFER_SLOTS]):
            o["boss_relic_ids"][j] = _idx(V.RELIC_VOCAB_INDEX, rid)


def encode(
    canon, *, global_map_view=False, card_language=False,
    combat_language=False, deck_size_observation=False, neow_observation=False,
):
    """obs canonical dict -> obs dict (numpy arrays matching obs_space()).

    `canon` may carry only the combat block (a normalize_sim result), only the
    overworld block, or the full assembled dict — absent groups encode to PAD."""
    o = _empty(
        global_map_view=global_map_view,
        card_language=card_language,
        combat_language=combat_language,
        deck_size_observation=deck_size_observation,
        neow_observation=neow_observation,
    )
    _encode_combat(o, canon)
    _encode_global(o, canon)
    _encode_options(o, canon)
    _encode_select(o, (canon.get("options") or {}).get("select"))
    return o


def _encode_select(o, sel):
    """Card-select screen semantics + progress (obs_design §4.1). Phase-agnostic:
    the overworld card_select phase AND an in-combat card-select sub-screen (phase
    stays 'combat') both carry options['select'], so this is driven by the block's
    presence, not by phase. Absent -> everything stays PAD/zero (NONE context)."""
    if not sel:
        return
    o["select_context"][0] = _idx(V.SELECT_CONTEXT_VOCAB_INDEX, sel.get("context", "PAD"))
    o["pick_remaining"][0] = sel.get("pick_remaining", 0) / PICK_SCALE
    o["can_pick_zero"][0] = 1.0 if sel.get("can_pick_zero") else 0.0
    o["can_pick_any"][0] = 1.0 if sel.get("can_pick_any") else 0.0
    for i, cid in enumerate(sel.get("selected", [])[:SELECT_PICK_SLOTS]):
        o["selected_card_ids"][i] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
    # combat MULTI screen: thin-env selected hand indices -> hand-aligned 0/1.
    for h in sel.get("hand_selected", []):
        if 0 <= h < HAND_SLOTS:
            o["select_hand_mask"][h] = 1.0

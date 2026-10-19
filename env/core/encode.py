"""Observation encoder v3: canonical state dict -> structured Dict observation.

This module is the single source of truth for the obs layout and the RL model's
input contract (see docs/archive/doc/day_0623/obs_design.md). It is a PURE function of the
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
from bridge.schemas.entity_ids import (
    LIVE_RELIC_COUNTER_FIELDS, live_relic_counter_value,
    NEOW_BONUS_IDS, NEOW_DRAWBACK_IDS, NEOW_IMPLICIT_DRAWBACK_IDS,
)

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
# v11 is an intentional checkpoint break for the split combat/overworld policy.
# It adds a lossless per-card master-deck sequence (including upgrade count),
# action-aligned upgrade fields for every overworld card choice, and persistent
# run HP. The simulator's Deck::MAX_SIZE is the capacity source of truth.
OBS_VERSION = 20  # owned/claimable Act-4 keys are explicit, not inferred
GLOBAL_MAP_OBS_VERSION = 12
CARD_LANGUAGE_OBS_VERSION = 13
DECK_SIZE_OBS_VERSION = 14
INTERACTIVE_REASONING_OBS_VERSION = 15
# Search-only value leaves immediately after the enemy turn are represented by
# a dedicated scalar boundary bit.  This is a model-input contract change.
POST_ENEMY_TURN_BOUNDARY_OBS_VERSION = 16

# ---- capacities (fixed slot counts; reserved, PAD-filled) ----
MAX_MONSTERS = 5          # must equal actions.MAX_MONSTERS (slot i == target i)
HAND_SLOTS = 10           # real hand cap
N_CARDS = len(CARD_VOCAB)
# One slot per possible held relic. Runtime encoders trim trailing PAD slots
# before attention, so this complete contract does not make ordinary runs pay
# for all reserved slots.
R_MAX = V.RELIC_VOCAB_CAPACITY
P_MAX = 5                 # potion slots
DECK_SLOTS = 96           # sts::Deck::MAX_SIZE
G_MAX = DECK_SLOTS        # full-deck select grids use the same hard capacity
SELECT_NATIVE_HANDLE_CAPACITY = 65537  # 0 absent; native_index+1 in 1..65536
SELECT_PICK_SLOTS = 3     # overworld grid's max simultaneous picks (sim's
                          # haveSelectedCards is fixed_list<3>: Astrolabe=3 max)
OFFER_SLOTS = 5           # aligns with the reserved REWARD_CARD action segment
BOSS_RELIC_SLOTS = 3
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
EVENT_OPTION_SLOTS = 10   # slot i == EVENT_OPT[i]; 0 is PAD, option i stores i+1
EVENT_OPTION_CAPACITY = EVENT_OPTION_SLOTS + 1
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
COUNTER_SCALE = 10.0      # relic / per-turn counters are generally 0..10
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
RUN_HP_SCALARS = 3        # current hp, max hp, hp fraction (all phases)
COMBAT_COUNTER_KEYS = (
    "happy_flower", "incense_burner", "ink_bottle", "inserter",
    "nunchaku", "pen_nib", "sundial", "necronomicon_used",
    "cards_played", "attacks_played", "skills_played",
    "orange_attack", "orange_skill", "orange_power",
    "combust_hp_loss", "deva_form_energy_per_turn",
    "echo_form_cards_doubled", "panache_counter",
    "cards_discarded", "times_damaged",
    "bomb_1", "bomb_2", "bomb_3", "last_attack_unblocked_damage",
)
COMBAT_COUNTERS = len(COMBAT_COUNTER_KEYS)

_BINARY_COMBAT_COUNTERS = frozenset({
    "necronomicon_used", "orange_attack", "orange_skill", "orange_power",
})

_LIVE_RELIC_COUNTER = LIVE_RELIC_COUNTER_FIELDS

_BINARY_LIVE_RELICS = frozenset({
    "Centennial Puzzle",
    "Lizard Tail",
})

# dtypes: categorical ids are integer indices, scalars are float32.
_CAT = np.int64
_SCALAR = np.float32


# ============================ observation space ============================

def obs_space(*, global_map_view=False, card_language=False,
              combat_language=False, deck_size_observation=False,
              neow_observation=False, event_option_observation=False,
              interactive_reasoning=False):
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
        "combat_counters": scal((COMBAT_COUNTERS,)),
        "post_enemy_turn_boundary": scal((1,)),
        "combat_is_boss": scal((1,)),
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
        "relic_counters": scal((R_MAX,)),
        "potions": cat((P_MAX,), V.POTION_VOCAB_CAPACITY),
        "gold": scal((1,)),
        "run_hp": scal((RUN_HP_SCALARS,)),
        # ruby, emerald, sapphire ownership.  These determine Act-4 access and
        # cannot be reconstructed from relics or the current screen.
        "run_keys": scal((3,)),
        # Lossless master deck: slot i is one physical card, including its
        # upgrade count. This is the overworld value model's deck signal.
        "deck_ids": cat((DECK_SLOTS,), N_CARDS),
        "deck_upgrades": scal((DECK_SLOTS,)),
        # ---- overworld decision context (phase-gated; PAD otherwise) ----
        "offer_card_ids": cat((OFFER_SLOTS,), N_CARDS),
        "offer_card_upgrades": scal((OFFER_SLOTS,)),
        "reward_gold": scal((2,)),
        "reward_relic_ids": cat((3,), V.RELIC_VOCAB_CAPACITY),
        "reward_potion_ids": cat((5,), V.POTION_VOCAB_CAPACITY),
        # emerald, sapphire key offers on the current reward screen.
        "reward_keys": scal((2,)),
        "select_card_ids": cat((G_MAX,), N_CARDS),
        "select_card_upgrades": scal((G_MAX,)),
        "select_card_costs": scal((G_MAX,)),
        # Handles are for codec alignment only, never neural positional features.
        # 0 is padding; native index n is stored as n+1, including explicit skip.
        "select_native_indices": cat((G_MAX,), SELECT_NATIVE_HANDLE_CAPACITY),
        "select_zone": cat((1,), 7),
        "pending_card_slot": cat((1,), HAND_SLOTS + 1),
        "pending_potion_slot": cat((1,), P_MAX + 1),
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
        "selected_card_upgrades": scal((SELECT_PICK_SLOTS,)),
        # combat MULTI screen (Exhaust Many / Gamble): hand-aligned 0/1 vector of
        # the cards the thin env has selected so far (obs_design §4.2). slot h ==
        # the PICK_CARD[h] id == hand slot h. PAD (all 0) off a MULTI screen.
        "select_hand_mask": scal((HAND_SLOTS,)),
        "map_node_rooms": cat((MAP_EDGES,), len(V.ROOM_VOCAB)),
        "map_node_x": scal((MAP_EDGES,)),
        "boss_relic_ids": cat((BOSS_RELIC_SLOTS,), V.RELIC_VOCAB_CAPACITY),
        "event_id": cat((1,), V.EVENT_VOCAB_CAPACITY),
        # ---- shop shelf (phase-gated; slot i == buy action id; PAD otherwise) ----
        "shop_card_ids": cat((SHOP_CARD_SLOTS,), N_CARDS),
        "shop_card_upgrades": scal((SHOP_CARD_SLOTS,)),
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
    if event_option_observation:
        groups.update({
            # ---- normal event choices (slot i == EVENT_OPT[i] action id) ----
            # event_option_ids uses 0 as PAD; real sim option index i is i + 1.
            "event_option_ids": cat((EVENT_OPTION_SLOTS,), EVENT_OPTION_CAPACITY),
            "event_option_event_ids": cat(
                (EVENT_OPTION_SLOTS,), V.EVENT_VOCAB_CAPACITY
            ),
            # index_norm, is_legal. Option text is not exported by sts_lightspeed,
            # so this is the stable information available at the decision point.
            "event_option_features": scal((EVENT_OPTION_SLOTS, 2)),
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
    return _card_id_idx(card["id"])


def _card_id_idx(card_id):
    if card_id == EMPTY:
        raise ValueError("an actual card entry cannot use the EMPTY/PAD sentinel")
    try:
        return CARD_INDEX[card_id]
    except KeyError as exc:
        raise ValueError(f"unknown card id: {card_id!r}") from exc


def _idx(index_map, key):
    try:
        return index_map[key]
    except KeyError as exc:
        raise ValueError(f"unknown categorical value: {key!r}") from exc


def _card_tuple(card):
    if isinstance(card, dict):
        if "up" in card:
            upgrades = card["up"]
        elif "upgrades" in card:
            upgrades = card["upgrades"]
        else:
            raise ValueError(f"card entry is missing its upgrade count: {card!r}")
        return card["id"], int(upgrades)
    if len(card) < 2:
        raise ValueError(f"card entry needs id and upgrade count: {card!r}")
    return card[0], int(card[1])


def _require_capacity(name, values, capacity):
    if len(values) > capacity:
        raise ValueError(
            f"{name} contains {len(values)} entries, exceeding capacity {capacity}"
        )


def _require_keys(mapping, keys, context):
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be an object, got {mapping!r}")
    missing = set(keys) - set(mapping)
    if missing:
        raise ValueError(f"{context} is incomplete; missing {sorted(missing)!r}")


def _validate_complete_canonical(canon):
    """Validate a live real-game frame before any encoder defaults run.

    The general encoder deliberately supports partial simulator/test projections.
    RealGameAdapter marks authoritative frames strict_complete, so a missing
    value cannot be converted into a plausible zero/PAD observation.
    """
    _require_keys(canon, ("phase", "global", "options"), "canonical state")
    phase = canon["phase"]
    if phase not in V.PHASE_VOCAB_INDEX or phase in {"none", "invalid"}:
        raise ValueError(f"strict live frame has no actionable phase: {phase!r}")

    g = canon["global"]
    _require_keys(
        g,
        (
            "character", "ascension", "act", "floor", "combat_is_boss",
            "boss", "relics", "relic_counters", "potions", "gold", "hp",
            "max_hp", "deck", "boss_relics", "keys",
        ),
        "canonical global state",
    )
    if g["character"] == "INVALID":
        raise ValueError("strict live frame cannot use INVALID character")
    _idx(V.CHARACTER_VOCAB_INDEX, g["character"])
    if g["boss"] == "PAD":
        raise ValueError("strict live frame cannot use PAD boss")
    _idx(V.BOSS_VOCAB_INDEX, g["boss"])
    if len(g["relics"]) != len(g["relic_counters"]):
        raise ValueError(
            "global relic ids and counters are not slot-aligned: "
            f"{len(g['relics'])} != {len(g['relic_counters'])}"
        )
    _require_keys(g["keys"], ("ruby", "emerald", "sapphire"), "global keys")
    for name, value in g["keys"].items():
        if type(value) is not bool:
            raise ValueError(f"global key {name!r} must be a boolean, got {value!r}")
    for rid in g["relics"]:
        if rid == "PAD":
            raise ValueError("real relic list cannot contain PAD")
        _idx(V.RELIC_VOCAB_INDEX, rid)
    for pid in g["potions"]:
        _idx(V.POTION_VOCAB_INDEX, pid)  # PAD is legal only for an empty slot.
    for card in g["deck"]:
        cid, _up = _card_tuple(card)
        _card_id_idx(cid)

    opts = canon["options"]
    if phase == "combat":
        _require_keys(
            canon,
            ("turn", "player", "monsters", "hand", "draw", "discard", "exhaust"),
            "canonical combat state",
        )
        player = canon["player"]
        _require_keys(
            player,
            (
                "hp", "max_hp", "block", "energy", "gold", "powers",
                "energy_per_turn", "combat_counters",
            ),
            "canonical player state",
        )
        missing_counters = set(COMBAT_COUNTER_KEYS) - set(player["combat_counters"])
        if missing_counters:
            raise ValueError(
                "canonical combat counters are incomplete: "
                f"{sorted(missing_counters)!r}"
            )
        for index, monster in enumerate(canon["monsters"]):
            _require_keys(
                monster,
                (
                    "id", "hp", "max_hp", "block", "gone", "half_dead",
                    "base_damage", "adjusted_damage", "hits", "powers",
                    "move", "move_id", "last_move", "last_move_id",
                    "second_last_move", "second_last_move_id",
                ),
                f"canonical monster[{index}]",
            )
            if monster["id"] == "PAD":
                raise ValueError("real monster records cannot use PAD slots")
        for index, card in enumerate(canon["hand"]):
            _require_keys(card, ("id", "up", "cost"), f"canonical hand[{index}]")
    else:
        required_options = {
            "combat_reward": (
                "gold", "relics", "potions", "keys", "card_count", "cards"
            ),
            "card_select": ("cards", "select"),
            "map": ("nodes",),
            "event": ("event",),
            "shop": ("shop_cards", "shop_relics", "shop_potions", "shop_remove_cost"),
        }.get(phase, ())
        _require_keys(opts, required_options, f"canonical {phase} options")

    select = opts.get("select")
    if select is not None:
        _require_keys(
            select,
            (
                "context", "zone", "candidates", "selected", "pick_remaining",
                "pick_min", "pick_max", "can_pick_zero", "can_pick_any", "multi",
            ),
            "canonical selection descriptor",
        )
        if select["context"] == "PAD":
            raise ValueError("live card selection cannot use PAD context")
        _idx(V.SELECT_CONTEXT_VOCAB_INDEX, select["context"])


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
        o[ids_key][i] = _card_id_idx(card_id)
        o[f"{pile}_card_counts"][i] = count
        o[f"{pile}_card_upgraded_counts"][i] = upgraded_count
        o[f"{pile}_card_upgrades"][i] = upgrade_sum / UPGRADE_SCALE


def _empty(*, global_map_view=False, card_language=False,
           combat_language=False, deck_size_observation=False,
           neow_observation=False, event_option_observation=False,
           interactive_reasoning=False):
    """A zero/PAD-filled obs dict (every absent group encodes to this)."""
    o = {
        "player_scalars": np.zeros(PLAYER_SCALARS, _SCALAR),
        "player_buffs": np.zeros(V.PLAYER_BUFF_VEC, _SCALAR),
        "combat_counters": np.zeros(COMBAT_COUNTERS, _SCALAR),
        "post_enemy_turn_boundary": np.zeros(1, _SCALAR),
        "combat_is_boss": np.zeros(1, _SCALAR),
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
        "relic_counters": np.zeros(R_MAX, _SCALAR),
        "potions": np.zeros(P_MAX, _CAT),
        "gold": np.zeros(1, _SCALAR),
        "run_hp": np.zeros(RUN_HP_SCALARS, _SCALAR),
        "run_keys": np.zeros(3, _SCALAR),
        "deck_ids": np.zeros(DECK_SLOTS, _CAT),
        "deck_upgrades": np.zeros(DECK_SLOTS, _SCALAR),
        "offer_card_ids": np.zeros(OFFER_SLOTS, _CAT),
        "offer_card_upgrades": np.zeros(OFFER_SLOTS, _SCALAR),
        "reward_gold": np.zeros(2, _SCALAR),
        "reward_relic_ids": np.zeros(3, _CAT),
        "reward_potion_ids": np.zeros(5, _CAT),
        "reward_keys": np.zeros(2, _SCALAR),
        "select_card_ids": np.zeros(G_MAX, _CAT),
        "select_card_upgrades": np.zeros(G_MAX, _SCALAR),
        "select_card_costs": np.zeros(G_MAX, _SCALAR),
        "select_native_indices": np.zeros(G_MAX, _CAT),
        "select_zone": np.zeros(1, _CAT),
        "pending_card_slot": np.zeros(1, _CAT),
        "pending_potion_slot": np.zeros(1, _CAT),
        "select_context": np.zeros(1, _CAT),
        "pick_remaining": np.zeros(1, _SCALAR),
        "can_pick_zero": np.zeros(1, _SCALAR),
        "can_pick_any": np.zeros(1, _SCALAR),
        "selected_card_ids": np.zeros(SELECT_PICK_SLOTS, _CAT),
        "selected_card_upgrades": np.zeros(SELECT_PICK_SLOTS, _SCALAR),
        "select_hand_mask": np.zeros(HAND_SLOTS, _SCALAR),
        "map_node_rooms": np.zeros(MAP_EDGES, _CAT),
        "map_node_x": np.zeros(MAP_EDGES, _SCALAR),
        "boss_relic_ids": np.zeros(BOSS_RELIC_SLOTS, _CAT),
        "event_id": np.zeros(1, _CAT),
        "shop_card_ids": np.zeros(SHOP_CARD_SLOTS, _CAT),
        "shop_card_upgrades": np.zeros(SHOP_CARD_SLOTS, _SCALAR),
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
    if event_option_observation:
        o["event_option_ids"] = np.zeros(EVENT_OPTION_SLOTS, _CAT)
        o["event_option_event_ids"] = np.zeros(EVENT_OPTION_SLOTS, _CAT)
        o["event_option_features"] = np.zeros(
            (EVENT_OPTION_SLOTS, 2), _SCALAR
        )
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


def _empty_combat(*, card_language=False, combat_language=False):
    """Allocate only the fields consumed by the split combat specialist.

    This is an internal transport representation for neural search leaves.  It
    It mirrors the public combat fields, including the explicit
    post-enemy-turn boundary bit.
    """
    o = {
        "select_card_ids": np.zeros(G_MAX, _CAT),
        "select_card_upgrades": np.zeros(G_MAX, _SCALAR),
        "select_card_costs": np.zeros(G_MAX, _SCALAR),
        "select_native_indices": np.zeros(G_MAX, _CAT),
        "select_zone": np.zeros(1, _CAT),
        "pending_card_slot": np.zeros(1, _CAT),
        "pending_potion_slot": np.zeros(1, _CAT),
        "arg_step": np.zeros(1, _CAT),
        "ascension": np.zeros(1, _SCALAR),
        "can_pick_any": np.zeros(1, _SCALAR),
        "can_pick_zero": np.zeros(1, _SCALAR),
        "character": np.zeros(1, _CAT),
        "combat_counters": np.zeros(COMBAT_COUNTERS, _SCALAR),
        "post_enemy_turn_boundary": np.zeros(1, _SCALAR),
        "combat_is_boss": np.zeros(1, _SCALAR),
        "discard_counts": np.zeros(N_CARDS, _SCALAR),
        "draw_counts": np.zeros(N_CARDS, _SCALAR),
        "exhaust_counts": np.zeros(N_CARDS, _SCALAR),
        "gold": np.zeros(1, _SCALAR),
        "run_keys": np.zeros(3, _SCALAR),
        "hand_block": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_costs": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_damage": np.zeros(HAND_SLOTS, _SCALAR),
        "hand_ids": np.zeros(HAND_SLOTS, _CAT),
        "hand_upgrades": np.zeros(HAND_SLOTS, _SCALAR),
        "monster_buffs": np.zeros(
            (MAX_MONSTERS, V.MONSTER_BUFF_VEC), _SCALAR
        ),
        "monster_id": np.zeros(MAX_MONSTERS, _CAT),
        "monster_intent_cat": np.zeros(MAX_MONSTERS, _CAT),
        "monster_move": np.zeros(MAX_MONSTERS, _CAT),
        "monster_scalars": np.zeros(
            (MAX_MONSTERS, MON_SCALARS), _SCALAR
        ),
        "pick_remaining": np.zeros(1, _SCALAR),
        "player_buffs": np.zeros(V.PLAYER_BUFF_VEC, _SCALAR),
        "player_scalars": np.zeros(PLAYER_SCALARS, _SCALAR),
        "potions": np.zeros(P_MAX, _CAT),
        "relic_counters": np.zeros(R_MAX, _SCALAR),
        "relics": np.zeros(R_MAX, _CAT),
        "select_context": np.zeros(1, _CAT),
        "select_hand_mask": np.zeros(HAND_SLOTS, _SCALAR),
    }
    if card_language:
        for pile in ("draw", "discard"):
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
        j = _idx(V.PLAYER_BUFF_VOCAB_INDEX, name)
        o["player_buffs"][j] = amt / BUFF_SCALE
    counters = p.get("combat_counters", {})
    for i, name in enumerate(COMBAT_COUNTER_KEYS):
        value = counters.get(name, 0)
        o["combat_counters"][i] = (
            float(bool(value)) if name in _BINARY_COMBAT_COUNTERS
            else value / COUNTER_SCALE
        )
    if "player_status_language" in o:
        program = CL.status_program("PLAYER", hp, max_hp, p["block"], p["powers"])
        _require_capacity(
            "player status language", program, CL.MAX_PLAYER_STATUS_TOKENS
        )
        ids = [_idx(CL.TOKEN_INDEX, token) for token in program]
        o["player_status_language"][:len(ids)] = ids

    # Monsters keep FIXED slots (obs_design.md §2.2): monster i stays in slot i;
    # a dead one is marked present=0 rather than filtered (filtering shifts slots
    # and breaks the slot<->action-target correspondence). We therefore index by
    # position, not by a "live" filter.
    monsters = canon.get("monsters", [])
    _require_capacity("monsters", monsters, MAX_MONSTERS)
    for i in range(MAX_MONSTERS):
        if i >= len(monsters):
            break
        m = monsters[i]
        if m["id"] == "PAD":
            continue
        present = 0.0 if m["gone"] else 1.0
        o["monster_id"][i] = _idx(V.MONSTER_VOCAB_INDEX, m["id"])
        move = m.get("move")
        if move == "INVALID" and (
            m["gone"] or canon.get("post_enemy_turn_boundary", False)
        ):
            move_id = 0
        else:
            move_id = _idx(V.MOVE_VOCAB_INDEX, move)
            if move_id == 0:
                raise ValueError(
                    f"live monster {m['id']!r} cannot use PAD as its move"
                )
        if "move_id" not in m:
            raise ValueError(f"monster {m['id']!r} is missing move_id")
        if int(m["move_id"]) != move_id:
            raise ValueError(
                f"monster move index mismatch for {move!r}: "
                f"got {m['move_id']!r}, expected {move_id}"
            )
        o["monster_move"][i] = move_id
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
            j = _idx(V.MONSTER_BUFF_VOCAB_INDEX, name)
            o["monster_buffs"][i, j] = amt / BUFF_SCALE
        if "monster_status_language" in o:
            status = CL.status_program(
                "MONSTER", mhp, mmax, m["block"], m["powers"]
            )
            _require_capacity(
                f"monster[{i}] status language",
                status,
                CL.MAX_MONSTER_STATUS_TOKENS,
            )
            status_ids = [_idx(CL.TOKEN_INDEX, token) for token in status]
            o["monster_status_language"][i, :len(status_ids)] = status_ids
            intent = CL.live_intent_program(
                m.get("move", "PAD"),
                m.get("adjusted_damage", m["base_damage"]),
                m["hits"],
            )
            _require_capacity(
                f"monster[{i}] intent language",
                intent,
                CL.MAX_LIVE_INTENT_TOKENS,
            )
            intent_ids = [_idx(CL.TOKEN_INDEX, token) for token in intent]
            o["monster_intent_language"][i, :len(intent_ids)] = intent_ids

    hand = canon.get("hand", [])
    _require_capacity("hand", hand, HAND_SLOTS)
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


def _encode_combat_global(o, canon):
    """Fill the run fields that the combat specialist actually consumes."""
    o["post_enemy_turn_boundary"][0] = float(
        bool(canon.get("post_enemy_turn_boundary", False))
    )
    o["arg_step"][0] = _idx(
        V.ARG_STEP_VOCAB_INDEX, canon.get("arg_step", "root")
    )
    g = canon.get("global")
    o["combat_is_boss"][0] = float(bool((g or {}).get("combat_is_boss", False)))
    if g is None:
        return
    o["character"][0] = _idx(
        V.CHARACTER_VOCAB_INDEX, g.get("character", "INVALID")
    )
    o["ascension"][0] = g.get("ascension", 0) / ASCENSION_SCALE
    gold = g.get("gold", 0)
    if canon.get("phase") == "combat" and canon.get("player") is not None:
        gold = canon["player"].get("gold", gold)
    o["gold"][0] = gold / GOLD_SCALE
    keys = g.get("keys")
    if keys is not None and "run_keys" in o:
        _require_keys(keys, ("ruby", "emerald", "sapphire"), "global keys")
        o["run_keys"][:] = tuple(
            float(bool(keys[name])) for name in ("ruby", "emerald", "sapphire")
        )

    relics = g.get("relics", [])
    relic_counters = g.get("relic_counters", [])
    _require_capacity("global.relics", relics, R_MAX)
    _require_capacity("global.relic_counters", relic_counters, R_MAX)
    if len(relics) != len(relic_counters):
        raise ValueError("global relic ids and counters are not slot-aligned; missing counters cannot become zero")
    live_counters = canon.get("player", {}).get("combat_counters", {})
    for i, rid in enumerate(relics):
        o["relics"][i] = _idx(V.RELIC_VOCAB_INDEX, rid)
        value = relic_counters[i]
        if type(value) is not int or value < 0:
            raise ValueError("global native relic counter must be an explicit nonnegative integer")
        o["relic_counters"][i] = value / COUNTER_SCALE
        live_key = _LIVE_RELIC_COUNTER.get(rid)
        if (
            canon.get("phase") == "combat"
            and live_key is not None
            and ((live_key == "$turn" and "turn" in canon) or live_key in live_counters)
        ):
            live_value = live_relic_counter_value(rid, canon)
            o["relic_counters"][i] = (
                float(bool(live_value))
                if rid in _BINARY_LIVE_RELICS
                else live_value / COUNTER_SCALE
            )
    potions = g.get("potions", [])
    _require_capacity("global.potions", potions, P_MAX)
    for i, pid in enumerate(potions):
        o["potions"][i] = _idx(V.POTION_VOCAB_INDEX, pid)


def _encode_global(o, canon):
    """Fill the run-global groups from canon['global'] (present every phase)."""
    o["phase"][0] = _idx(
        V.PHASE_VOCAB_INDEX, canon.get("phase", "none")
    )
    if "post_enemy_turn_boundary" in o:
        o["post_enemy_turn_boundary"][0] = float(
            bool(canon.get("post_enemy_turn_boundary", False))
        )
    # arg_step rides at top level (the thin env sets it; combat-only canonicals
    # omit it -> root, index 0). Kept here so it is filled every phase.
    o["arg_step"][0] = _idx(
        V.ARG_STEP_VOCAB_INDEX, canon.get("arg_step", "root")
    )
    g = canon.get("global")
    o["combat_is_boss"][0] = float(bool((g or {}).get("combat_is_boss", False)))
    if g is None:
        return
    o["character"][0] = _idx(
        V.CHARACTER_VOCAB_INDEX, g.get("character", "INVALID")
    )
    o["ascension"][0] = g.get("ascension", 0) / ASCENSION_SCALE
    o["act"][0] = g.get("act", 0) / ACT_SCALE
    o["floor"][0] = g.get("floor", 0) / FLOOR_SCALE
    o["next_boss"][0] = _idx(V.BOSS_VOCAB_INDEX, g.get("boss", "PAD"))
    gold = g.get("gold", 0)
    if canon.get("phase") == "combat" and canon.get("player") is not None:
        gold = canon["player"].get("gold", gold)
    o["gold"][0] = gold / GOLD_SCALE
    hp = canon.get("hp", g.get("hp", 0))
    max_hp = canon.get("max_hp", g.get("max_hp", 0))
    if canon.get("phase") == "combat" and canon.get("player") is not None:
        hp = canon["player"].get("hp", hp)
        max_hp = canon["player"].get("max_hp", max_hp)
    o["run_hp"][:] = (
        min(hp, HP_SCALE) / HP_SCALE,
        min(max_hp, HP_SCALE) / HP_SCALE,
        (hp / max_hp) if max_hp else 0.0,
    )
    keys = g.get("keys")
    if keys is not None:
        _require_keys(keys, ("ruby", "emerald", "sapphire"), "global keys")
        o["run_keys"][:] = tuple(
            float(bool(keys[name])) for name in ("ruby", "emerald", "sapphire")
        )
    # Relic ids and persistent counters share slots. During combat, overwrite
    # the counters that live in BattleContext with their current values.
    relics = g.get("relics", [])
    relic_counters = g.get("relic_counters", [])
    _require_capacity("global.relics", relics, R_MAX)
    _require_capacity("global.relic_counters", relic_counters, R_MAX)
    if len(relics) != len(relic_counters):
        raise ValueError("global relic ids and counters are not slot-aligned; missing counters cannot become zero")
    for i, rid in enumerate(relics):
        o["relics"][i] = _idx(V.RELIC_VOCAB_INDEX, rid)
        value = relic_counters[i]
        if type(value) is not int or value < 0:
            raise ValueError("global native relic counter must be an explicit nonnegative integer")
        o["relic_counters"][i] = value / COUNTER_SCALE
        live_key = _LIVE_RELIC_COUNTER.get(rid)
        live_counters = canon.get("player", {}).get("combat_counters", {})
        if (
            canon.get("phase") == "combat"
            and live_key is not None
            and ((live_key == "$turn" and "turn" in canon) or live_key in live_counters)
        ):
            live_value = live_relic_counter_value(rid, canon)
            o["relic_counters"][i] = (
                float(bool(live_value))
                if rid in _BINARY_LIVE_RELICS
                else live_value / COUNTER_SCALE
            )
    # potions: slot-aligned ids; empty/INVALID names -> PAD 0.
    potions = g.get("potions", [])
    _require_capacity("global.potions", potions, P_MAX)
    for i, pid in enumerate(potions):
        o["potions"][i] = _idx(V.POTION_VOCAB_INDEX, pid)
    deck = g.get("deck", [])
    _require_capacity("global.deck", deck, DECK_SLOTS)
    for i, card in enumerate(deck):
        cid, up = _card_tuple(card)
        o["deck_ids"][i] = _card_id_idx(cid)
        o["deck_upgrades"][i] = up / UPGRADE_SCALE
    if "deck_size" in o:
        o["deck_size"][0] = min(
            len(deck) / DECK_SIZE_SCALE, 1.0
        )
    _encode_card_sequence(o, "deck", deck)


def _encode_options(o, canon):
    """Fill phase-gated decision-context groups from canon['options'] /
    canon['global']['boss_relics']. Each group is only meaningful in its phase;
    encode() leaves the rest PAD."""
    phase = canon.get("phase", "none")
    opts = canon.get("options") or {}

    if phase == "combat_reward":
        cards = opts.get("cards", [])
        _require_capacity("options.cards", cards, OFFER_SLOTS)
        for j, card in enumerate(cards):
            cid, up = _card_tuple(card)
            o["offer_card_ids"][j] = _card_id_idx(cid)
            o["offer_card_upgrades"][j] = up / UPGRADE_SCALE
        gold = opts.get("gold", [])
        relics = opts.get("relics", [])
        potions = opts.get("potions", [])
        keys = opts.get("keys", [])
        _require_capacity("options.gold", gold, 2)
        _require_capacity("options.relics", relics, 3)
        _require_capacity("options.potions", potions, 5)
        _require_capacity("options.keys", keys, 1)
        for j, amount in enumerate(gold):
            o["reward_gold"][j] = amount / GOLD_SCALE
        for j, rid in enumerate(relics):
            o["reward_relic_ids"][j] = _idx(V.RELIC_VOCAB_INDEX, rid)
        for j, pid in enumerate(potions):
            o["reward_potion_ids"][j] = _idx(V.POTION_VOCAB_INDEX, pid)
        for key in keys:
            if key == "EMERALD_KEY":
                o["reward_keys"][0] = 1.0
            elif key == "SAPPHIRE_KEY":
                o["reward_keys"][1] = 1.0
            else:
                raise ValueError(f"unknown reward key identity: {key!r}")
    elif phase == "event" and opts.get("neow") and "neow_bonus_ids" in o:
        neow = opts.get("neow", [])
        _require_capacity("options.neow", neow, NEOW_SLOTS)
        for i, option in enumerate(neow):
            _require_keys(option, ("bonus_id", "drawback_id"), "Neow offer")
            bonus_id = option["bonus_id"]
            drawback_id = option["drawback_id"]
            if type(bonus_id) is not int or bonus_id not in NEOW_BONUS_IDS.values():
                raise ValueError(f"invalid Neow bonus id: {bonus_id}")
            if (type(drawback_id) is not int or drawback_id not in
                    (*NEOW_DRAWBACK_IDS.values(), *NEOW_IMPLICIT_DRAWBACK_IDS.values())
                    or (bonus_id == 18) != (drawback_id == 6)):
                raise ValueError(f"invalid Neow drawback id: {drawback_id}")
            o["neow_bonus_ids"][i] = bonus_id
            o["neow_drawback_ids"][i] = drawback_id
            o["neow_features"][i] = (
                i / max(1, NEOW_SLOTS - 1),
                1.0 if drawback_id > 1 else 0.0,
            )
    elif phase == "card_select":
        cards = opts.get("cards", [])
        _require_capacity("options.cards", cards, G_MAX)
        o["select_zone"][0] = 6
        for k, card in enumerate(cards):
            cid, up = _card_tuple(card)
            o["select_card_ids"][k] = _card_id_idx(cid)
            o["select_card_upgrades"][k] = up / UPGRADE_SCALE
            o["select_native_indices"][k] = k + 1
    elif phase == "map":
        nodes = opts.get("nodes", [])
        _require_capacity("options.nodes", nodes, MAP_EDGES)
        for j, (x, room) in enumerate(nodes):
            o["map_node_rooms"][j] = _idx(V.ROOM_VOCAB_INDEX, room)
            o["map_node_x"][j] = x / MAP_X_SCALE
        if "map_full_rooms" in o:
            for node in opts.get("full_map", []):
                x, y = int(node["x"]), int(node["y"])
                if not (0 <= x < MAP_COLS and 0 <= y < MAP_ROWS):
                    raise ValueError(f"map node outside observation grid: {(x, y)!r}")
                slot = y * MAP_COLS + x
                o["map_full_rooms"][slot] = _idx(
                    V.ROOM_VOCAB_INDEX, node["room"]
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
                    if not 0 <= child_x < MAP_COLS:
                        raise ValueError(f"map child outside observation grid: {child_x!r}")
                    features[5 + child_x] = 1.0
    elif phase == "event":
        g = canon.get("global") or {}
        # event id lives in the options (env reads gc.cur_event_name into it)
        event_id = _idx(V.EVENT_VOCAB_INDEX, opts.get("event", "PAD"))
        o["event_id"][0] = event_id
        if "event_option_ids" in o:
            event_options = opts.get("event_options", [])
            _require_capacity(
                "options.event_options", event_options, EVENT_OPTION_SLOTS
            )
            for option_idx in event_options:
                option_idx = int(option_idx)
                if not (0 <= option_idx < EVENT_OPTION_SLOTS):
                    raise ValueError(f"invalid event option index: {option_idx}")
                o["event_option_ids"][option_idx] = option_idx + 1
                o["event_option_event_ids"][option_idx] = event_id
                o["event_option_features"][option_idx] = (
                    option_idx / max(1, EVENT_OPTION_SLOTS - 1),
                    1.0,
                )
    elif phase == "shop":
        # Positional shelf: slot i lines up with the SHOP_* buy action id. A sold
        # slot has price -1 (shop_contents) -> PAD id + price 0 (decision: a shop
        # has no "originally empty" slot, so -1 means "already bought").
        shop_cards = opts.get("shop_cards", [])
        _require_capacity("options.shop_cards", shop_cards, SHOP_CARD_SLOTS)
        for i, (cid, up, price) in enumerate(shop_cards):
            if price != -1:
                o["shop_card_ids"][i] = _card_id_idx(cid)
                o["shop_card_upgrades"][i] = up / UPGRADE_SCALE
                o["shop_card_prices"][i] = price / GOLD_SCALE
        shop_relics = opts.get("shop_relics", [])
        shop_potions = opts.get("shop_potions", [])
        _require_capacity("options.shop_relics", shop_relics, SHOP_RELIC_SLOTS)
        _require_capacity("options.shop_potions", shop_potions, SHOP_POTION_SLOTS)
        for i, (rid, price) in enumerate(shop_relics):
            if price != -1:
                o["shop_relic_ids"][i] = _idx(V.RELIC_VOCAB_INDEX, rid)
                o["shop_relic_prices"][i] = price / GOLD_SCALE
        for i, (pid, price) in enumerate(shop_potions):
            if price != -1:
                o["shop_potion_ids"][i] = _idx(V.POTION_VOCAB_INDEX, pid)
                o["shop_potion_prices"][i] = price / GOLD_SCALE
        rc = opts.get("shop_remove_cost", -1)
        if rc != -1:
            o["shop_remove_cost"][0] = rc / GOLD_SCALE

    if phase == "boss_relic":
        g = canon.get("global") or {}
        boss_relics = g.get("boss_relics", [])
        _require_capacity("global.boss_relics", boss_relics, BOSS_RELIC_SLOTS)
        for j, rid in enumerate(boss_relics):
            o["boss_relic_ids"][j] = _idx(V.RELIC_VOCAB_INDEX, rid)


def encode(
    canon, *, global_map_view=False, card_language=False,
    combat_language=False, deck_size_observation=False, neow_observation=False,
    event_option_observation=False, interactive_reasoning=False,
):
    """obs canonical dict -> obs dict (numpy arrays matching obs_space()).

    `canon` may carry only the combat block (a normalize_sim result), only the
    overworld block, or the full assembled dict — absent groups encode to PAD."""
    if canon.get("strict_complete"):
        _validate_complete_canonical(canon)
    o = _empty(
        global_map_view=global_map_view,
        card_language=card_language,
        combat_language=combat_language,
        deck_size_observation=deck_size_observation,
        neow_observation=neow_observation,
        event_option_observation=event_option_observation,
        interactive_reasoning=interactive_reasoning,
    )
    if "pending_card_slot" in o:
        o["pending_card_slot"][0] = int(canon.get("pending_card_slot", 0))
    o["pending_potion_slot"][0] = int(canon.get("pending_potion_slot", 0))
    _encode_combat(o, canon)
    _encode_global(o, canon)
    _encode_options(o, canon)
    _encode_select(o, (canon.get("options") or {}).get("select"))
    return o


def encode_combat(canon, *, card_language=False, combat_language=False):
    """Encode only fields consumed by the split combat specialist.

    This compact representation is used only to transport neural-search
    leaves. Fields follow the same versioned public schema. Inference adapters
    project onto a historical checkpoint's declared observation space.
    """
    if canon.get("strict_complete"):
        _validate_complete_canonical(canon)
    o = _empty_combat(
        card_language=card_language,
        combat_language=combat_language,
    )
    o["pending_card_slot"][0] = int(canon.get("pending_card_slot", 0))
    o["pending_potion_slot"][0] = int(canon.get("pending_potion_slot", 0))
    _encode_combat(o, canon)
    _encode_combat_global(o, canon)
    _encode_select(o, (canon.get("options") or {}).get("select"))
    return o


def _encode_select(o, sel):
    """Card-select screen semantics + progress (obs_design §4.1). Phase-agnostic:
    the overworld card_select phase AND an in-combat card-select sub-screen (phase
    stays 'combat') both carry options['select'], so this is driven by the block's
    presence, not by phase. Absent -> everything stays PAD/zero (NONE context)."""
    if not sel:
        return
    zones = {"none": 0, "hand": 1, "draw": 2, "discard": 3,
             "exhaust": 4, "generated": 5, "grid": 6}
    o["select_zone"][0] = zones[sel.get("zone", "none")]
    candidates = sel.get("candidates", ())
    _require_capacity("options.select.candidates", candidates, G_MAX)
    for index, candidate in enumerate(candidates):
        o["select_native_indices"][index] = int(candidate["native_index"]) + 1
        card = candidate["card"]
        if card is not None:
            o["select_card_ids"][index] = _card_idx(card)
            o["select_card_upgrades"][index] = card.get("up", 0) / UPGRADE_SCALE
            o["select_card_costs"][index] = card.get("cost", 0) / ENERGY_SCALE
    o["select_context"][0] = _idx(V.SELECT_CONTEXT_VOCAB_INDEX, sel.get("context", "PAD"))
    o["pick_remaining"][0] = sel.get("pick_remaining", 0) / PICK_SCALE
    o["can_pick_zero"][0] = 1.0 if sel.get("can_pick_zero") else 0.0
    o["can_pick_any"][0] = 1.0 if sel.get("can_pick_any") else 0.0
    selected = sel.get("selected", [])
    _require_capacity("options.select.selected", selected, SELECT_PICK_SLOTS)
    if "selected_card_ids" in o:
        for i, card in enumerate(selected):
            cid, up = _card_tuple(card)
            o["selected_card_ids"][i] = _card_id_idx(cid)
            o["selected_card_upgrades"][i] = up / UPGRADE_SCALE
    # combat MULTI screen: thin-env selected hand indices -> hand-aligned 0/1.
    for h in sel.get("hand_selected", []):
        if 0 <= h < HAND_SLOTS:
            o["select_hand_mask"][h] = 1.0

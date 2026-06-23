"""Fixed-length action id space v2 — the thin layer's model contract.

This replaces the flat 71-id combat-only `actions.py` space (kept for the
combat-internal sts<->id mapping, which this module reuses) with a
**phase-partitioned, autoregressive** id space spanning the whole run
(action_design.md). Like `actions.py`/`encode.py` it is PURE: it imports only the
other pure modules (`actions`, `game_actions`, `vocab`) and the backend-agnostic
semantic-action vocabulary. The thin `SlayTheSpireEnv` (run_env.py) owns the
live sub-state and calls into here; the thick `SimGameEnv` is never touched.

============================ design (action_design.md) ============================

§0.2/0.3  The model decides AUTOREGRESSIVELY, not as a flat cartesian product:
  step 1 (ROOT)       pick the main action: end turn / play card h / map node / ...
  step 2 (PARAM)      only if that action needs a parameter: pick the target / ...
A targeted card play is therefore TWO timesteps — PLAY_CARD[h] then PICK_TARGET[t]
— and the thin env emits ONE semantic action `(COMBAT, encode_play(h,t))` to the
thick layer only once both are in. To the thick layer / sim it is opaque.

§2/§3  ONE big table, phase-gated. Every id permanently belongs to ONE segment
(append-only, never renumbered) so the table can be sliced into per-phase
sub-partitions for a specialist/HRL reading without renumbering. The mask zeros
everything not legal in the current (phase, arg_step).

Reserved-but-masked segments (USE_POTION, SHOP_*, the wide EVENT_OPT tail): the
deferred C++ binding batch (action_design §5) will make these legal; reserving
their ids NOW means turning them on later fills reserved width and does NOT bump
ACTION_VERSION. Until then the mask simply never sets them (the thick layer does
not enumerate those semantic actions yet).
"""

from . import actions
from . import game_actions as ga
from . import vocab as V

ACTION_VERSION = 2

# ---- capacities (must match encode.py slot counts) ----
HAND_SLOTS = actions.HAND_SLOTS        # 10  (combat hand; play slot == hand idx)
MAX_MONSTERS = actions.MAX_MONSTERS    # 5   (target slot == monster slot in obs)
P_MAX = 5                              # potion slots (reserved segment)
G_MAX = 12                             # card-select / grid pick slots
N_COLS = 7                             # map columns 0..6
REWARD_GOLD = 2                        # Rewards.gold is array<int,2>
REWARD_RELIC = 3                       # Rewards.relics is array<,3>
REWARD_POTION = 5                      # Rewards.potions is array<,5>
REWARD_CARD = 5                        # Rewards.cardRewards is array<,5>
CAMPFIRE_W = len(ga.CAMPFIRE_OPTIONS)  # 6: rest/smith/recall/lift/toke/dig
EVENT_OPT_W = 10                       # reserved width (thick gives 1 today, §5#7)
NEOW_W = 4                             # the 4 Neow rewards
CHEST_W = 2                            # open / skip
BOSS_RELIC_W = 3                       # 3 boss relics offered
SHOP_CARD = 7                          # Shop.cards[7]   (reserved, §5#6)
SHOP_RELIC = 3                         # Shop.relics[3]  (reserved)
SHOP_POTION = 3                        # Shop.potions[3] (reserved)

# ---- segment table (append-only; widths above; bases computed below) ----
# Each entry: (NAME, width). ORDER IS FROZEN — append new segments at the END only.
_SEGMENTS = [
    ("END_TURN",      1),
    ("PLAY_CARD",     HAND_SLOTS),
    ("USE_POTION",    P_MAX),          # reserved (no thick potion verb yet)
    ("PICK_TARGET",   MAX_MONSTERS),
    ("PICK_CARD",     G_MAX),
    ("MAP_NODE",      N_COLS),
    ("REWARD_GOLD",   REWARD_GOLD),
    ("REWARD_RELIC",  REWARD_RELIC),
    ("REWARD_POTION", REWARD_POTION),
    ("REWARD_CARD",   REWARD_CARD),
    ("REWARD_SKIP",   1),
    ("CAMPFIRE",      CAMPFIRE_W),
    ("EVENT_OPT",     EVENT_OPT_W),
    ("NEOW",          NEOW_W),
    ("CHEST",         CHEST_W),
    ("BOSS_RELIC",    BOSS_RELIC_W),
    ("SHOP_CARD",     SHOP_CARD),       # reserved
    ("SHOP_RELIC",    SHOP_RELIC),      # reserved
    ("SHOP_POTION",   SHOP_POTION),     # reserved
    ("SHOP_REMOVE",   1),               # reserved
    ("PROCEED",       1),
]

# base id of each segment + total width, derived from the frozen order/widths.
_BASE = {}
_acc = 0
for _name, _w in _SEGMENTS:
    _BASE[_name] = _acc
    _acc += _w
N_ACTIONS = _acc

# convenient module-level base constants (END_TURN == 0)
END_TURN = _BASE["END_TURN"]
_PLAY = _BASE["PLAY_CARD"]
_POTION = _BASE["USE_POTION"]
_TARGET = _BASE["PICK_TARGET"]
_PICK = _BASE["PICK_CARD"]
_MAP = _BASE["MAP_NODE"]
_RGOLD = _BASE["REWARD_GOLD"]
_RRELIC = _BASE["REWARD_RELIC"]
_RPOTION = _BASE["REWARD_POTION"]
_RCARD = _BASE["REWARD_CARD"]
_RSKIP = _BASE["REWARD_SKIP"]
_CAMP = _BASE["CAMPFIRE"]
_EVENT = _BASE["EVENT_OPT"]
_NEOW = _BASE["NEOW"]
_CHEST = _BASE["CHEST"]
_BOSS = _BASE["BOSS_RELIC"]
_PROCEED = _BASE["PROCEED"]

# campfire id offset <-> option name (the thick layer keys on names).
_CAMP_NAME_BY_IDX = {idx: name for name, idx in ga.CAMPFIRE_OPTIONS.items()}

# arg-step vocab names (mirror vocab.ARG_STEP_VOCAB) the encoder reads.
ROOT = "root"
PICK_TARGET = "pick_target"
PICK_CARD = "pick_card"


# ============================ segment id helpers ============================

def seg_base(name):
    return _BASE[name]


def _play_targeting(legal_semantic):
    """From the legal semantic actions, classify each combat hand slot's play:
    returns (untargeted: set[h], targets: dict[h -> set[t]]). A card is one or the
    other (the sim never lists both for the same slot)."""
    untargeted, targets = set(), {}
    for a in legal_semantic:
        if a[0] != ga.COMBAT:
            continue
        kind = actions.decode(a[1])
        if kind[0] != actions.KIND_PLAY:
            continue
        _, h, t = kind
        if t is None:
            untargeted.add(h)
        else:
            targets.setdefault(h, set()).add(t)
    return untargeted, targets


def _combat_select_only(legal_semantic, phase):
    """True when the only combat decisions on offer are SINGLE_CARD_SELECT picks
    (the sim opened an in-combat card-select sub-screen, e.g. Armaments/Headbutt).
    The sim never mixes these with normal plays, so 'all combat ids are SELECT'
    is a reliable signal."""
    if phase != "combat":
        return False
    kinds = [actions.decode(a[1])[0] for a in legal_semantic if a[0] == ga.COMBAT]
    return bool(kinds) and all(k == actions.KIND_SELECT for k in kinds)


# ============================== arg_step ==============================

def arg_step_name(pending, legal_semantic, phase):
    """Which autoregressive sub-step the thin env is on (vocab.ARG_STEP_VOCAB).

    pending is None at ROOT, or ('play', hand_idx) once a targeted card has been
    chosen and we await its target. A card-select screen (combat sub-screen OR
    overworld grid) is a forced single pick -> pick_card."""
    if pending is not None and pending[0] == "play":
        return PICK_TARGET
    if phase == "card_select":
        return PICK_CARD
    if _combat_select_only(legal_semantic, phase):
        return PICK_CARD
    return ROOT


# ============================== mask ==============================

def legal_ids(legal_semantic, pending, phase):
    """The thin action ids that are legal RIGHT NOW (a sorted list of ints the
    caller flips True in the bool mask). Branches on the sub-step:

      - pending=('play',h): only PICK_TARGET[t] for targets legal for card h.
      - otherwise (ROOT or a card-select screen): every main action on offer,
        each mapped to its segment id.
    """
    ids = set()

    if pending is not None and pending[0] == "play":
        h = pending[1]
        _, targets = _play_targeting(legal_semantic)
        for t in targets.get(h, ()):
            ids.add(_TARGET + t)
        return sorted(ids)

    untargeted, targets = _play_targeting(legal_semantic)
    for a in legal_semantic:
        verb = a[0]
        if verb == ga.COMBAT:
            kind = actions.decode(a[1])
            if kind[0] == actions.KIND_END:
                ids.add(END_TURN)
            elif kind[0] == actions.KIND_PLAY:
                ids.add(_PLAY + kind[1])          # one root id per hand slot
            elif kind[0] == actions.KIND_SELECT:
                ids.add(_PICK + kind[1])          # in-combat card-select pick
        elif verb == ga.NEOW:
            ids.add(_NEOW + a[1])
        elif verb == ga.MAP:
            ids.add(_MAP + a[1])
        elif verb == ga.CLAIM:
            kind, idx = a[1], a[2]
            base = {ga.CLAIM_GOLD: _RGOLD, ga.CLAIM_RELIC: _RRELIC,
                    ga.CLAIM_POTION: _RPOTION}[kind]
            ids.add(base + idx)
        elif verb == ga.REWARD_CARD:
            ids.add(_RCARD + a[1])
        elif verb == ga.SKIP_CARDS:
            ids.add(_RSKIP)
        elif verb == ga.SELECT:
            ids.add(_PICK + a[1])                 # overworld grid pick
        elif verb == ga.CAMPFIRE:
            ids.add(_CAMP + ga.CAMPFIRE_OPTIONS[a[1]])
        elif verb == ga.EVENT:
            ids.add(_EVENT + a[1])
        elif verb == ga.CHEST:
            ids.add(_CHEST + (0 if a[1] else 1))
        elif verb == ga.BOSS_RELIC:
            ids.add(_BOSS + a[1])
        elif verb in (ga.PROCEED, ga.SHOP_PROCEED):
            ids.add(_PROCEED)
        # SHOP buy/remove + potion use are not enumerated by the thick layer yet
        # (reserved segments); they simply never become legal here.
    return sorted(ids)


# ============================== resolve ==============================

def resolve(chosen_id, legal_semantic, pending, phase):
    """Advance the autoregressive machine by one chosen id.

    Returns (semantic_action_or_None, new_pending):
      - a semantic-action tuple (game_actions) to hand the thick layer + the new
        pending (usually None), OR
      - (None, ('play', h)) when the chosen id only PARTIALLY specifies an action
        (a targeted card was picked; we now await PICK_TARGET) — the thin env must
        NOT call the backend this step.

    Raises ValueError on an id that does not correspond to any reachable action
    (the mask should make that impossible; this is defense in depth)."""
    cid = int(chosen_id)

    # ---- parameter step: filling the target of a pending card play ----
    if pending is not None and pending[0] == "play":
        if _TARGET <= cid < _TARGET + MAX_MONSTERS:
            t = cid - _TARGET
            return ga.combat(actions.encode_play(pending[1], t)), None
        raise ValueError(f"expected PICK_TARGET id, got {cid}")

    # ---- root / card-select step ----
    if cid == END_TURN:
        return ga.combat(actions.END_TURN), None

    if _PLAY <= cid < _PLAY + HAND_SLOTS:
        h = cid - _PLAY
        untargeted, targets = _play_targeting(legal_semantic)
        if h in targets and h not in untargeted:
            return None, ("play", h)              # need a target -> step 2
        return ga.combat(actions.encode_play(h, None)), None

    if _PICK <= cid < _PICK + G_MAX:
        k = cid - _PICK
        if phase == "combat":
            return ga.combat(actions.encode_select(k)), None
        return ga.select(k), None                 # overworld grid

    if _MAP <= cid < _MAP + N_COLS:
        return ga.map_node(cid - _MAP), None
    if _RGOLD <= cid < _RGOLD + REWARD_GOLD:
        return ga.claim(ga.CLAIM_GOLD, cid - _RGOLD), None
    if _RRELIC <= cid < _RRELIC + REWARD_RELIC:
        return ga.claim(ga.CLAIM_RELIC, cid - _RRELIC), None
    if _RPOTION <= cid < _RPOTION + REWARD_POTION:
        return ga.claim(ga.CLAIM_POTION, cid - _RPOTION), None
    if _RCARD <= cid < _RCARD + REWARD_CARD:
        return ga.reward_card(cid - _RCARD), None
    if cid == _RSKIP:
        return ga.skip_cards(), None
    if _CAMP <= cid < _CAMP + CAMPFIRE_W:
        return ga.campfire(_CAMP_NAME_BY_IDX[cid - _CAMP]), None
    if _EVENT <= cid < _EVENT + EVENT_OPT_W:
        return ga.event(cid - _EVENT), None
    if _NEOW <= cid < _NEOW + NEOW_W:
        return ga.neow(cid - _NEOW), None
    if _CHEST <= cid < _CHEST + CHEST_W:
        return ga.chest(cid == _CHEST), None      # _CHEST+0 = open, +1 = skip
    if _BOSS <= cid < _BOSS + BOSS_RELIC_W:
        return ga.boss_relic(cid - _BOSS), None
    if cid == _PROCEED:
        # PROCEED leaves whatever screen we are on; pick the legal flavor.
        if any(a[0] == ga.SHOP_PROCEED for a in legal_semantic):
            return ga.shop_proceed(), None
        return ga.proceed(), None

    raise ValueError(f"action id {cid} out of range / in a reserved segment")


# ============================== introspection ==============================

def segment_of(action_id):
    """(segment_name, offset) for an id — for debugging/rendering."""
    for name, w in _SEGMENTS:
        b = _BASE[name]
        if b <= action_id < b + w:
            return name, action_id - b
    raise ValueError(action_id)

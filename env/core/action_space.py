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

Reserved-but-masked -> LIVE (USE_POTION): the deferred C++ binding batch
(action_design §5) is now wired, so this segment is LIVE. Turning it on filled
reserved width and did NOT bump ACTION_VERSION (the segment was already part of
N_ACTIONS). The thick layer enumerates real potion uses (game_env.legal_potions /
legal_overworld_potions, sim-gated by search::Action::isValidPotionAction); a
targeted potion (Fire/Fear/Poison/Weak) reuses the PICK_TARGET substep exactly
like a targeted card play, so the autoregressive machine stays uniform. EVENT_OPT
is live for the low option ids (the thick layer enumerates real event options)
with a reserved wide tail; SHOP_* likewise went LIVE this way (the thick layer
enumerates real buys/removes) — both filling reserved width.

ACTION_VERSION 4 (was 3): appended a CONFIRM_SELECT segment (1 id) at the END of
the table for the combat MULTI card-select screens (Exhaust Many / Gamble,
action_design §1.1). Those resolve the WHOLE screen with one hand-index bitset,
so the thin env TOGGLES hand cards across steps (reusing the PICK_CARD ids over
hand indices) and CONFIRMs once — CONFIRM_SELECT is that commit. Appending at the
end renumbers NOTHING before it (every prior base is frozen); it only widens
N_ACTIONS by 1, which still bumps ACTION_VERSION (the policy's output head grows).
No new OBS-less state: the toggled set rides in obs as select_hand_mask
(OBS_VERSION 6) so the feed-forward policy sees what it has already picked.

ACTION_VERSION 3 (was 2): the PICK_CARD segment widened G_MAX 12->64 to fit a
full-deck card-select grid (Remove/Transform/Smith candidates = the whole deck).
That renumbers every segment after PICK_CARD, so it is a real version bump — but
it lands in the same change as OBS_VERSION 4->5 (the select-screen contract,
obs_design §4.1), which already invalidates old checkpoints, so renumbering costs
nothing extra here. The old G_MAX=12 was also a latent bug: a pick idx >=12
overflowed PICK_CARD into the MAP_NODE segment.
"""

from . import actions
from . import game_actions as ga
from . import vocab as V

ACTION_VERSION = 4

# ---- capacities (must match encode.py slot counts) ----
HAND_SLOTS = actions.HAND_SLOTS        # 10  (combat hand; play slot == hand idx)
MAX_MONSTERS = actions.MAX_MONSTERS    # 5   (target slot == monster slot in obs)
P_MAX = 5                              # potion slots (reserved segment)
G_MAX = 64                             # card-select / grid pick slots (full deck;
                                       # must equal encode.G_MAX — a Remove/Transform
                                       # grid is the whole deck. Was 12: a >12 pick
                                       # idx overflowed PICK_CARD into MAP_NODE.)
N_COLS = 7                             # map columns 0..6
REWARD_GOLD = 2                        # Rewards.gold is array<int,2>
REWARD_RELIC = 3                       # Rewards.relics is array<,3>
REWARD_POTION = 5                      # Rewards.potions is array<,5>
REWARD_CARD = 5                        # Rewards.cardRewards is array<,5>
CAMPFIRE_W = len(ga.CAMPFIRE_OPTIONS)  # 6: rest/smith/recall/lift/toke/dig
EVENT_OPT_W = 10                       # reserved width (thick enumerates real opts, §5#7)
NEOW_W = 4                             # the 4 Neow rewards
CHEST_W = 2                            # open / skip
BOSS_RELIC_W = 3                       # 3 boss relics offered
SHOP_CARD = 7                          # Shop.cards[7]   (live, §5#6)
SHOP_RELIC = 3                         # Shop.relics[3]  (live)
SHOP_POTION = 3                        # Shop.potions[3] (live)

# ---- segment table (append-only; widths above; bases computed below) ----
# Each entry: (NAME, width). ORDER IS FROZEN — append new segments at the END only.
_SEGMENTS = [
    ("END_TURN",      1),
    ("PLAY_CARD",     HAND_SLOTS),
    ("USE_POTION",    P_MAX),          # LIVE (thick enumerates real potion uses)
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
    ("SHOP_CARD",     SHOP_CARD),
    ("SHOP_RELIC",    SHOP_RELIC),
    ("SHOP_POTION",   SHOP_POTION),
    ("SHOP_REMOVE",   1),
    ("PROCEED",       1),
    ("CONFIRM_SELECT", 1),            # commit a combat MULTI pick (Exhaust Many /
                                      # Gamble); appended last so prior ids freeze.
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
_SHOP_CARD = _BASE["SHOP_CARD"]
_SHOP_RELIC = _BASE["SHOP_RELIC"]
_SHOP_POTION = _BASE["SHOP_POTION"]
_SHOP_REMOVE = _BASE["SHOP_REMOVE"]
_PROCEED = _BASE["PROCEED"]
_CONFIRM = _BASE["CONFIRM_SELECT"]

# shop buy-kind -> its segment base (the thick layer keys buys on these strings).
_SHOP_BASE_BY_KIND = {
    ga.SHOP_CARD: _SHOP_CARD,
    ga.SHOP_RELIC: _SHOP_RELIC,
    ga.SHOP_POTION: _SHOP_POTION,
}

# campfire id offset <-> option name (the thick layer keys on names).
_CAMP_NAME_BY_IDX = {idx: name for name, idx in ga.CAMPFIRE_OPTIONS.items()}

# arg-step vocab names (mirror vocab.ARG_STEP_VOCAB) the encoder reads.
ROOT = "root"
PICK_TARGET = "pick_target"
PICK_CARD = "pick_card"
PICK_HAND = "pick_hand"        # combat MULTI screen: toggling hand cards to commit


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


def _potion_targeting(legal_semantic):
    """From the legal semantic actions, classify each held potion SLOT's use:
    returns (untargeted: set[slot], targets: dict[slot -> set[t]]). Mirrors
    _play_targeting exactly so the autoregressive PICK_TARGET substep is shared
    between targeted card plays and targeted potions (Fire/Fear/Poison/Weak): a
    potion is either single-step (untargeted) or needs a target, never both, and
    the thick layer already enumerated one (USE_POTION, slot, t) per legal target."""
    untargeted, targets = set(), {}
    for a in legal_semantic:
        if a[0] != ga.USE_POTION:
            continue
        slot, t = a[1], a[2]
        if t is None:
            untargeted.add(slot)
        else:
            targets.setdefault(slot, set()).add(t)
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
    chosen and we await its target, or ('potion', slot) once a targeted potion has
    been chosen and we await its target. Both targeted cases share the PICK_TARGET
    substep (a monster pick). A card-select screen (combat sub-screen OR overworld
    grid) is a forced single pick -> pick_card."""
    if pending is not None and pending[0] in ("play", "potion"):
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
      - pending=('potion',slot): only PICK_TARGET[t] for targets legal for that
        potion — the SAME PICK_TARGET segment as a card play, so the target-pick
        obs/mask path is shared (a potion's target is just a monster, like a card's).
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

    if pending is not None and pending[0] == "potion":
        slot = pending[1]
        _, targets = _potion_targeting(legal_semantic)
        for t in targets.get(slot, ()):
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
        elif verb == ga.SHOP_BUY:
            ids.add(_SHOP_BASE_BY_KIND[a[1]] + a[2])   # (kind, slot) -> segment id
        elif verb == ga.SHOP_REMOVE:
            ids.add(_SHOP_REMOVE)
        elif verb == ga.USE_POTION:
            # One ROOT id per potion SLOT (USE_POTION segment), regardless of
            # whether it is targeted: a targeted potion picks its monster in the
            # PICK_TARGET substep (see resolve), exactly like a targeted card play
            # emits one PLAY_CARD[h] root id and defers the target. The thick layer
            # may list several (slot, t) actions for one targeted slot; they all
            # collapse to the single _POTION+slot root id here (the set dedupes).
            ids.add(_POTION + a[1])
        # all enumerated verbs are now handled above.
    return sorted(ids)


# ============================== resolve ==============================

def resolve(chosen_id, legal_semantic, pending, phase):
    """Advance the autoregressive machine by one chosen id.

    Returns (semantic_action_or_None, new_pending):
      - a semantic-action tuple (game_actions) to hand the thick layer + the new
        pending (usually None), OR
      - (None, ('play', h)) when the chosen id only PARTIALLY specifies an action
        (a targeted card was picked; we now await PICK_TARGET), OR
      - (None, ('potion', slot)) likewise for a targeted potion — the thin env must
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

    # ---- parameter step: filling the target of a pending potion use ----
    # Same PICK_TARGET segment as a card play; we just route the chosen monster
    # into a USE_POTION semantic action instead of a COMBAT play.
    if pending is not None and pending[0] == "potion":
        if _TARGET <= cid < _TARGET + MAX_MONSTERS:
            t = cid - _TARGET
            return ga.use_potion(pending[1], t), None
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

    if _POTION <= cid < _POTION + P_MAX:
        slot = cid - _POTION
        untargeted, targets = _potion_targeting(legal_semantic)
        if slot in targets and slot not in untargeted:
            return None, ("potion", slot)         # targeted potion -> await target
        return ga.use_potion(slot, None), None    # untargeted: resolves in one step

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
    if _SHOP_CARD <= cid < _SHOP_CARD + SHOP_CARD:
        return ga.shop_buy(ga.SHOP_CARD, cid - _SHOP_CARD), None
    if _SHOP_RELIC <= cid < _SHOP_RELIC + SHOP_RELIC:
        return ga.shop_buy(ga.SHOP_RELIC, cid - _SHOP_RELIC), None
    if _SHOP_POTION <= cid < _SHOP_POTION + SHOP_POTION:
        return ga.shop_buy(ga.SHOP_POTION, cid - _SHOP_POTION), None
    if cid == _SHOP_REMOVE:
        return ga.shop_remove(), None
    if cid == _PROCEED:
        # PROCEED leaves whatever screen we are on; pick the legal flavor.
        if any(a[0] == ga.SHOP_PROCEED for a in legal_semantic):
            return ga.shop_proceed(), None
        return ga.proceed(), None

    raise ValueError(f"action id {cid} out of range / in a reserved segment")


# ===================== combat MULTI card-select (Exhaust Many / Gamble) =====
# These screens resolve the WHOLE screen with one hand-index bitset, so the thin
# env runs a toggle+confirm machine: each step either TOGGLES a hand card in/out
# of the pending set (a PICK_CARD id over the hand index) or CONFIRMs the set
# (CONFIRM_SELECT). The pending set lives in the thin env; these helpers are pure
# over (descriptor, selected) so the codec stays backend-free. `descriptor` is the
# combat select block (combat_select_options): {multi, cards_in_hand, pick_max,
# can_pick_zero, ...}. `selected` is the current set of toggled hand indices.

def is_multi_select(descriptor):
    """True when the live combat card-select screen is a MULTI (Exhaust Many /
    Gamble) screen — the only one the toggle+confirm machine drives."""
    return bool(descriptor) and bool(descriptor.get("multi"))


def multi_legal_ids(descriptor, selected):
    """Thin ids legal on a MULTI screen given the already-toggled `selected` set:

      - PICK_CARD[h] for every hand index h in 0..cards_in_hand-1 that can still
        be toggled. An already-selected card can always be toggled OFF; an unselected
        one only if we are under the pick_max cap (Exhaust Many caps; Gamble's cap
        is the whole hand, so it never blocks).
      - CONFIRM_SELECT, always (an empty pick is valid — it just dismisses the
        screen; chooseExhaust/GambleCards early-return on empty).
    """
    ids = set()
    n = int(descriptor.get("cards_in_hand", 0))
    cap = int(descriptor.get("pick_max", 0))
    for h in range(min(n, G_MAX)):
        if h in selected:
            ids.add(_PICK + h)                 # toggle OFF always allowed
        elif len(selected) < cap:
            ids.add(_PICK + h)                 # toggle ON only under the cap
    ids.add(_CONFIRM)
    return sorted(ids)


def multi_toggle(chosen_id, selected):
    """Apply one MULTI-screen id to the toggled set. Returns (new_selected, commit)
    where commit is None for a toggle (stay on the screen) or the sorted tuple of
    chosen hand indices when CONFIRM_SELECT was chosen (hand it to the backend as
    a MULTI_SELECT semantic action).

    selected is a Python set of hand indices; we return a fresh set so the caller
    can keep it immutable per step."""
    cid = int(chosen_id)
    if cid == _CONFIRM:
        return set(selected), tuple(sorted(selected))
    if _PICK <= cid < _PICK + G_MAX:
        h = cid - _PICK
        new = set(selected)
        new.discard(h) if h in new else new.add(h)
        return new, None
    raise ValueError(f"expected a MULTI toggle/confirm id, got {cid}")


# ============================== introspection ==============================
def segment_of(action_id):
    """(segment_name, offset) for an id — for debugging/rendering."""
    for name, w in _SEGMENTS:
        b = _BASE[name]
        if b <= action_id < b + w:
            return name, action_id - b
    raise ValueError(action_id)

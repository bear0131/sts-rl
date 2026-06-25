"""Semantic action protocol for whole-run play — backend-agnostic.

A *semantic action* is a small tuple `(verb, *args)` describing a game-level
decision in terms the player would recognize ("open map node x", "claim the
relic", "pick reward card j"), NOT in terms of any backend's internals. Both
SimGameEnv (sim) and a future RealGameEnv (bridge) speak this same vocabulary:
each enumerates the currently-legal semantic actions and knows how to apply one
against its own backend.

Design split (see doc/plan_simgameenv.md):
  - This module = the vocabulary + tiny builders + a human-readable label.
  - SimGameEnv.apply() = the dispatch that turns a semantic action into sim calls
    (it owns gc/bc). RealGameEnv.apply() would map the same tuples to CommMod
    commands.
  - Fixed-length action ids + a global mask are deliberately NOT defined here —
    that compression is the thin SlayTheSpireEnv's job, so it can choose the
    encoding without the backend locking it in. The thick layer only exposes
    `legal_semantic_actions() -> list[tuple]` and `apply(action)`.

The combat phase does not re-invent anything: a combat decision is
`(COMBAT, combat_action_id)` where combat_action_id indexes the existing,
RNG-verified 71-action space in actions.py.
"""

from . import actions

# ---- verbs ----
NEOW = "neow"               # (NEOW, option_idx)            0..3
MAP = "map"                 # (MAP, x)                      next-row column to enter
COMBAT = "combat"           # (COMBAT, action_id)           delegates to actions.py (0..N_ACTIONS)
CLAIM = "claim"             # (CLAIM, kind, idx)            kind in {"gold","relic","potion"}
REWARD_CARD = "reward_card" # (REWARD_CARD, j)              pick card j on a combat card-reward
SKIP_CARDS = "skip_cards"   # (SKIP_CARDS,)                 skip the card reward
SELECT = "select"           # (SELECT, k)                   pick k on a CARD_SELECT/grid screen
MULTI_SELECT = "multi_select"  # (MULTI_SELECT, (h0,h1,...)) resolve a combat MULTI screen
                            #                               (Exhaust Many / Gamble) with the
                            #                               chosen hand indices in one shot
CAMPFIRE = "campfire"       # (CAMPFIRE, option)            option in CAMPFIRE_OPTIONS
EVENT = "event"             # (EVENT, i)                    event option i
CHEST = "chest"             # (CHEST, open_bool)            open/skip a treasure chest
BOSS_RELIC = "boss_relic"   # (BOSS_RELIC, i)               pick boss relic i
PROCEED = "proceed"         # (PROCEED,)                    leave a reward/rest/chest screen
SHOP_PROCEED = "shop_proceed"  # (SHOP_PROCEED,)            leave the shop
SHOP_BUY = "shop_buy"       # (SHOP_BUY, kind, idx)         kind in {"card","relic","potion"}; idx = shelf slot
SHOP_REMOVE = "shop_remove" # (SHOP_REMOVE,)                pay to remove a card (opens a card_select pick)
USE_POTION = "use_potion"   # (USE_POTION, slot, target)    drink the held potion in slot (0-based);
                            #                               target is a monster idx for a targeted
                            #                               potion (Fire/Fear/Poison/Weak) or None for
                            #                               an untargeted/overworld one. Mirrors COMBAT's
                            #                               targeted-vs-untargeted split so the thin
                            #                               layer's autoregressive PICK_TARGET substep is
                            #                               shared with card plays (see action_space.py).

# claimable kinds on a combat-reward screen
CLAIM_GOLD = "gold"
CLAIM_RELIC = "relic"
CLAIM_POTION = "potion"

# shop buy kinds (keyed to the sim's shop_buy_card/relic/potion + the
# gc.legal_shop_options() "kind" strings)
SHOP_CARD = "card"
SHOP_RELIC = "relic"
SHOP_POTION = "potion"

# campfire options -> the sim's fixed chooseCampfireOption index (replay._CAMPFIRE)
CAMPFIRE_OPTIONS = {"rest": 0, "smith": 1, "recall": 2, "lift": 3, "toke": 4, "dig": 5}


# ---- builders (clarity at call sites; tuples are the canonical form) ----
def neow(i):            return (NEOW, i)
def map_node(x):        return (MAP, x)
def combat(action_id):  return (COMBAT, int(action_id))
def claim(kind, idx=0): return (CLAIM, kind, idx)
def reward_card(j):     return (REWARD_CARD, j)
def skip_cards():       return (SKIP_CARDS,)
def select(k):          return (SELECT, k)
def multi_select(idxs): return (MULTI_SELECT, tuple(int(h) for h in idxs))
def campfire(opt):      return (CAMPFIRE, opt)
def event(i):           return (EVENT, i)
def chest(open_):       return (CHEST, bool(open_))
def boss_relic(i):      return (BOSS_RELIC, i)
def proceed():          return (PROCEED,)
def shop_proceed():     return (SHOP_PROCEED,)
def shop_buy(kind, idx): return (SHOP_BUY, kind, idx)
def shop_remove():      return (SHOP_REMOVE,)
def use_potion(slot, target=None):
    # target is a monster idx (targeted potion in combat) or None (untargeted /
    # overworld). int(slot) keeps the tuple hashable + backend-symmetric.
    return (USE_POTION, int(slot), None if target is None else int(target))


def describe(action, canon=None):
    """Human-readable label for a semantic action. For COMBAT, defer to the
    combat renderer. For card-bearing overworld actions (reward_card / select)
    look up the actual card from the overworld canonical's options when given."""
    verb = action[0]
    if verb == COMBAT:
        from . import render
        # canon here is the combat-phase canonical dict, if available
        return f"combat: {render.render_action(action[1], canon)}"
    if verb == NEOW:        return f"Neow option {action[1]}"
    if verb == MAP:         return f"go to map node x={action[1]}"
    if verb == CLAIM:       return f"claim {action[1]} #{action[2]}"
    if verb == REWARD_CARD: return f"take card: {_card_at(canon, 'cards', action[1])}"
    if verb == SKIP_CARDS:  return "skip card reward"
    if verb == SELECT:      return f"select: {_card_at(canon, 'cards', action[1])}"
    if verb == MULTI_SELECT:
        idxs = action[1]
        if not idxs:
            return "confirm select: none"
        return "confirm select: hand " + ",".join(f"#{h}" for h in idxs)
    if verb == CAMPFIRE:    return f"campfire: {action[1]}"
    if verb == EVENT:       return f"event option {action[1]}"
    if verb == CHEST:       return "open chest" if action[1] else "skip chest"
    if verb == BOSS_RELIC:  return f"take boss relic #{action[1]}"
    if verb == PROCEED:     return "proceed"
    if verb == SHOP_PROCEED: return "leave shop"
    if verb == SHOP_BUY:    return f"buy {action[1]} #{action[2]}"
    if verb == SHOP_REMOVE: return "pay to remove a card"
    if verb == USE_POTION:
        slot, tgt = action[1], action[2]
        name = _potion_at(canon, slot)
        return (f"use potion {name} (slot #{slot})" if tgt is None
                else f"use potion {name} (slot #{slot}) on monster #{tgt}")
    return str(action)


def _card_at(canon, key, idx):
    """Look up the (id, up) card at options[key][idx] in an overworld canonical
    and render it as 'Panacea' / 'Strike_R+'. Falls back to '#idx' if the canon
    isn't available (e.g. labelling without state)."""
    try:
        cid, up = canon["options"][key][idx]
    except (TypeError, KeyError, IndexError):
        return f"#{idx}"
    if not up:
        return cid
    return f"{cid}+" if up == 1 else f"{cid}+{up}"


def _potion_at(canon, slot):
    """The held-potion name in `slot`, read from the canonical's global block
    (normalize_sim_global's 'potions' list, slot-aligned with the sim). Combat and
    overworld canonicals both carry it under 'global'; fall back to '#slot' when
    labelling without state."""
    try:
        return canon["global"]["potions"][slot]
    except (TypeError, KeyError, IndexError):
        return f"#{slot}"


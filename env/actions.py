"""Fixed combat action space: stable integer action ids <-> game actions.

Like encode.py, this module is PURE and does NOT import slaythespire. It speaks
abstract action ids and the canonical legal-action list; the env layer is the
only thing that turns a decoded action into an actual sts.Action (or, later, a
real-game command string via bridge). That keeps one action space shared across
the sim and real backends.

============================ action layout (v1) ============================
A flat, fixed id space. The policy outputs one id; a legal-action mask zeros out
everything not currently playable.

  END_TURN                         1 id
  PLAY card h on monster t         HAND_SLOTS * MAX_MONSTERS   (targeted cards)
  PLAY card h (no target)          HAND_SLOTS                  (untargeted cards)
  CARD_SELECT pick k               SELECT_SLOTS                (hand-select screens)

Potions are NOT in v1 (sim doesn't enumerate potion actions cleanly — see
DEVNOTES). Reserved to append later without disturbing existing ids.
===========================================================================
"""

from cards_vocab import EMPTY  # noqa: F401  (kept for parity; layout below is self-contained)

ACTION_VERSION = 1

# must match encode.py capacities
HAND_SLOTS = 10
MAX_MONSTERS = 5
SELECT_SLOTS = 10        # max cards on a hand-select screen we address

# ---- id ranges (append-only; never renumber) ----
END_TURN = 0

_PLAY_TGT_BASE = 1                               # HAND_SLOTS * MAX_MONSTERS ids
_PLAY_NOTGT_BASE = _PLAY_TGT_BASE + HAND_SLOTS * MAX_MONSTERS   # HAND_SLOTS ids
_SELECT_BASE = _PLAY_NOTGT_BASE + HAND_SLOTS                    # SELECT_SLOTS ids

N_ACTIONS = _SELECT_BASE + SELECT_SLOTS

# abstract action kinds the env will translate into sts.Action
KIND_END = "end"
KIND_PLAY = "play"            # (KIND_PLAY, hand_idx, target_idx_or_None)
KIND_SELECT = "select"        # (KIND_SELECT, select_idx)


def decode(action_id):
    """action id -> abstract action tuple. Raises on an out-of-range id."""
    if action_id == END_TURN:
        return (KIND_END,)
    if _PLAY_TGT_BASE <= action_id < _PLAY_NOTGT_BASE:
        k = action_id - _PLAY_TGT_BASE
        return (KIND_PLAY, k // MAX_MONSTERS, k % MAX_MONSTERS)
    if _PLAY_NOTGT_BASE <= action_id < _SELECT_BASE:
        return (KIND_PLAY, action_id - _PLAY_NOTGT_BASE, None)
    if _SELECT_BASE <= action_id < N_ACTIONS:
        return (KIND_SELECT, action_id - _SELECT_BASE)
    raise ValueError(f"action id {action_id} out of range [0,{N_ACTIONS})")


def encode_play(hand_idx, target_idx):
    """abstract play -> action id (inverse of decode, for building the mask)."""
    if target_idx is None:
        return _PLAY_NOTGT_BASE + hand_idx
    return _PLAY_TGT_BASE + hand_idx * MAX_MONSTERS + target_idx


def encode_select(select_idx):
    return _SELECT_BASE + select_idx

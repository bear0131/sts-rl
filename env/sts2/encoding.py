"""STS2 visible-state v1. Feature hashing is a baseline, not an identity codec.

Native actions remain exact dictionaries; hashes are used ONLY by the policy.
Draw order, RNG and enemy private AI state never enter this encoder.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

OBS_VERSION = "sts2-visible-hash-v1"
ACTION_VERSION = "sts2-native-candidates-v1"
STATE_DIM = 1024
ACTION_DIM = 256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def public_card(card):
    # UID/index are execution locators, not properties of a visible card face.
    fields = ("id", "kind", "type", "upgraded", "cost_override", "base_cost",
              "base_damage", "base_block", "flags", "enchantment", "affliction",
              "damage", "block", "combat_vars", "pile")
    return {k: deepcopy(card[k]) for k in fields if k in card}


def visible_combat(raw, pending=None):
    player_keys = ("class", "hp", "max_hp", "block", "energy", "max_energy",
                   "energy_regen_per_turn", "powers", "relics", "potions",
                   "stars", "osty", "orbs", "orb_slots", "cards_played_this_turn")
    result = {k: raw[k] for k in ("turn", "round", "phase", "side", "is_elite") if k in raw}
    result["player"] = {k: deepcopy(raw["player"][k]) for k in player_keys if k in raw["player"]}
    result["encounter"] = [{k: deepcopy(m[k]) for k in
                           ("kind", "hp", "max_hp", "block", "powers", "intents", "is_dead") if k in m}
                          for m in raw.get("encounter", [])]
    # Preserve every enemy slot, including dead/empty slots. Only the draw pile
    # is unordered: the other public pile orders are allowed information.
    result["deck"] = {k: [public_card(c) for c in raw.get("deck", {}).get(k, [])]
                      for k in ("hand", "draw_pile", "discard_pile", "exhaust_pile")}
    result["deck"]["draw_pile"].sort(key=canonical)
    if pending is not None:
        result["pending"] = {k: deepcopy(v) for k, v in pending.items() if k != "cards"}
        result["pending"]["cards"] = sorted([public_card(c) for c in pending["cards"]], key=canonical)
    return result


@lru_cache(maxsize=32768)
def _bucket(key, width):
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % width, (1.0 if digest[4] & 1 else -1.0)


def features(value, width):
    """Deterministic signed hashing of all scalar paths; no silent truncation."""
    out = np.zeros(width, dtype=np.float32)

    def add(key, amount):
        i, sign = _bucket(key, width)
        out[i] += sign * amount

    def visit(item, path):
        if isinstance(item, dict):
            for k in sorted(item):
                visit(item[k], f"{path}/{k}")
        elif isinstance(item, (list, tuple)):
            add(f"{path}/length", math.log1p(len(item)))
            for i, child in enumerate(item):
                visit(child, f"{path}/{i}")
        elif isinstance(item, bool):
            add(f"{path}={item}", 1)
        elif isinstance(item, (int, float)):
            if not math.isfinite(item):
                raise ValueError(f"Nonfinite observation at {path}")
            add(path, math.copysign(math.log1p(abs(item)), item))
        else:
            add(f"{path}={item}", 1)

    visit(value, "")
    return out

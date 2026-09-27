"""Sparse, human-readable run-history collection for whole-run evaluation.

The collector deliberately records decision boundaries and resource deltas, not
combat actions.  It consumes the backend's canonical state so that event/shop
metadata remains typed and action-aligned; missing native metadata is retained
as an option index/label instead of being guessed from localized text.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import re
import numpy as np


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _snapshot(env):
    """Copy the backend canonical state without retaining native objects."""
    thick = getattr(getattr(env, "unwrapped", env), "thick", None)
    if thick is None:
        return {}
    return _jsonable(copy.deepcopy(thick.obs_canonical()))


def _global(state):
    return state.get("global", state)


def _counter_delta(before, after):
    left = Counter(json.dumps(item, sort_keys=True) for item in (before or []))
    right = Counter(json.dumps(item, sort_keys=True) for item in (after or []))
    gained, lost = [], []
    for item, count in (right - left).items():
        gained.extend([json.loads(item)] * count)
    for item, count in (left - right).items():
        lost.extend([json.loads(item)] * count)
    return gained, lost


def state_diff(before, after):
    """Return typed resource changes used by event/combat/shop summaries."""
    bg, ag = _global(before), _global(after)
    result = {}
    for key in ("hp", "max_hp", "gold"):
        if bg.get(key) != ag.get(key):
            result[key] = {"before": bg.get(key), "after": ag.get(key),
                           "delta": (ag.get(key) or 0) - (bg.get(key) or 0)}
    for key in ("deck", "relics", "potions"):
        gained, lost = _counter_delta(bg.get(key, []), ag.get(key, []))
        if gained or lost:
            result[key] = {"gained": gained, "lost": lost}
    return result


def _phase(state):
    return state.get("phase")


def _position(state):
    current = _global(state)
    return {"act": current.get("act"), "floor": current.get("floor", state.get("floor"))}


def _persistent_state(state):
    current = _global(state)
    return {"phase": state.get("phase"), **{
        key: copy.deepcopy(current.get(key))
        for key in ("act", "floor", "hp", "max_hp", "gold", "deck", "relics", "potions")
    }}


def _action_label(env, action, native_action):
    if native_action is not None:
        return "native combat action"
    try:
        return str(env.describe_action(int(action)))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return f"action {action}"


def _event_index(label):
    match = re.search(r"event option (\d+)", label)
    if match:
        return int(match.group(1))
    match = re.search(r"Neow option (\d+)", label)
    return int(match.group(1)) - 1 if match else None


class EpisodeTraceCollector:
    """Collect one sparse run history.

    ``before_step`` and ``after_step`` are called by :func:`runner.evaluate`.
    A new combat/event/shop record is opened lazily from canonical phase state,
    which also makes this work with native complete combat actions.
    """

    def __init__(self, env, episode: int, seed: int):
        self.env = env
        self.episode = int(episode)
        self.seed = int(seed)
        self.map_path = []
        self.events = []
        self.combats = []
        self.shops = []
        self.rewards = []
        self.chests = []
        self.boss_relics = []
        self.actions = []
        self._active_combat = None
        self._last_before = None
        self._last_after = None
        self._last_entry = None
        self._pending_shop_remove = None
        self._started = _snapshot(env)

    def before_step(self, action, native_action=None):
        state = _snapshot(self.env)
        self._last_before = state
        phase = _phase(state)
        label = _action_label(self.env, action, native_action)
        entry = {"phase": phase, "action": label, **_position(state)}
        if action is not None:
            entry["action_id"] = int(action)
        if native_action is not None:
            entry["native"] = True
        if phase == "map":
            node = _map_node(state, label)
            if node is not None:
                node.update(_position(state))
                self.map_path.append(node)
                entry["node"] = node
        elif phase == "event":
            entry.update(_event_entry(state, label))
        elif phase == "shop":
            entry["item"] = _shop_item(state, label)
        elif phase == "combat_reward":
            entry["reward"] = _reward_item(state, label)
        elif phase == "boss_relic":
            entry["candidates"] = list(_global(state).get("boss_relics", []))
            index = _boss_index(label)
            if index is not None and index < len(entry["candidates"]):
                entry["selected"] = entry["candidates"][index]
        self._last_entry = entry
        # Combat details are represented by start/end snapshots only.
        if phase != "combat":
            self.actions.append(entry)

        if phase == "combat" and self._active_combat is None:
            monsters = [m.get("id") for m in state.get("monsters", [])]
            self._active_combat = {
                "act": _global(state).get("act"),
                "floor": _global(state).get("floor", state.get("floor")),
                "encounter": monsters,
                "pre_hp": _global(state).get("hp", state.get("player", {}).get("hp")),
                "pre_state": state,
            }

    def after_step(self, action, reward, terminated, truncated, info,
                   native_action=None):
        state = _snapshot(self.env)
        self._last_after = state
        before = self._last_before or {}
        phase = _phase(before)
        after_phase = _phase(state)
        diff = state_diff(before, state)
        if self._last_entry is not None and phase != "combat":
            self._last_entry["diff"] = diff
        if phase == "event":
            event = self._last_entry if self._last_entry and self._last_entry.get("phase") == "event" else {}
            self.events.append({**_position(before),
                "event": (before.get("options") or {}).get("event"),
                "action": event.get("action"),
                "action_id": event.get("action_id"),
                "option_index": event.get("option_index"),
                "choices": event.get("choices", []),
                "diff": diff,
            })
        if phase == "shop":
            entry = self._last_entry or {}
            item = entry.get("item")
            if item is not None or "buy" in str(entry.get("action", "")):
                transaction = {**_position(before), "action": entry.get("action"),
                               "item": item, "diff": diff}
                self.shops.append(transaction)
                if (item or {}).get("kind") == "remove":
                    self._pending_shop_remove = transaction
        elif phase == "card_select" and self._pending_shop_remove is not None:
            self._pending_shop_remove["selection"] = (self._last_entry or {}).get("action")
            self._pending_shop_remove["selection_diff"] = diff
            self._pending_shop_remove = None
        if phase == "combat_reward":
            entry = self._last_entry or {}
            item = entry.get("reward")
            self.rewards.append({**_position(before), "action": entry.get("action"),
                                 "item": item,
                                 "offers": copy.deepcopy(before.get("options") or {}),
                                 "diff": diff})
        if phase == "chest":
            entry = self._last_entry or {}
            self.chests.append({**_position(before), "action": entry.get("action"), "diff": diff})
        if phase == "boss_relic":
            entry = self._last_entry or {}
            self.boss_relics.append({**_position(before), "action": entry.get("action"),
                                    "candidates": entry.get("candidates", []),
                                    "selected": entry.get("selected"),
                                    "diff": diff})

        if self._active_combat is not None and after_phase != "combat":
            record = self._active_combat
            record.update({
                "post_hp": _global(state).get("hp", state.get("player", {}).get("hp")),
                "hp_loss": (record.get("pre_hp") or 0) - (_global(state).get("hp", 0) or 0),
                "reward_candidates": (copy.deepcopy(state.get("options") or {})
                                      if after_phase == "combat_reward" else {}),
                "post_state": state,
                "diff": state_diff(record["pre_state"], state),
            })
            record.pop("pre_state", None)
            record.pop("post_state", None)
            self.combats.append(record)
            self._active_combat = None

    def finish(self, result=None):
        final = self._last_after or _snapshot(self.env)
        if self._active_combat is not None:
            # A truncated evaluator may stop while a battle is live; preserve a
            # clear incomplete marker instead of inventing a post-combat HP.
            self._active_combat["incomplete"] = True
            self.combats.append(self._active_combat)
            self._active_combat = None
        return {
            "seed": self.seed,
            "result": result,
            "map_path": self.map_path,
            "question_mark_events": [event for event in self.events if event.get("event") != "NEOW"],
            "events": self.events,
            "combats": self.combats,
            "shops": self.shops,
            "rewards": self.rewards,
            "chests": self.chests,
            "boss_relics": self.boss_relics,
            "actions": self.actions,
            "final_state": _persistent_state(final),
        }


def _map_node(state, label):
    match = re.search(r"x=(-?\d+)", label)
    if not match:
        return None
    x = int(match.group(1))
    floor = _global(state).get("floor", state.get("floor"))
    for node in (state.get("options") or {}).get("nodes", []):
        if isinstance(node, (list, tuple)) and int(node[0]) == x:
            return {"x": x, "room": node[1], "from_floor": floor,
                    "floor": (floor or 0) + 1}
    return {"x": x, "from_floor": floor, "floor": (floor or 0) + 1}


def _event_entry(state, label):
    options = state.get("options") or {}
    choices = options.get("event_choices") or options.get("neow", [])
    return {"event": options.get("event"), "option_index": _event_index(label),
            "choices": choices,
            "event_options": options.get("event_options", [])}


def _shop_item(state, label):
    if label == "pay to remove a card":
        return {"kind": "remove", "cost": (state.get("options") or {}).get("shop_remove_cost")}
    match = re.search(r"buy (card|relic|potion) #(\d+)", label)
    if not match:
        return None
    kind, raw_index = match.group(1), int(match.group(2))
    key = {"card": "shop_cards", "relic": "shop_relics", "potion": "shop_potions"}[kind]
    values = (state.get("options") or {}).get(key, [])
    return {"kind": kind, "index": raw_index,
            "offer": values[raw_index] if raw_index < len(values) else None}


def _reward_item(state, label):
    options = state.get("options") or {}
    match = re.search(r"claim (gold|relic|potion) #(\d+)", label)
    if match:
        kind, index = match.group(1), int(match.group(2))
        values = options.get({"gold": "gold", "relic": "relics", "potion": "potions"}[kind], [])
        return {"kind": kind, "index": index,
                "offer": values[index] if index < len(values) else None}
    match = re.search(r"take card: (.+)$", label)
    if match:
        return {"kind": "card", "offer": match.group(1)}
    return None


def _boss_index(label):
    match = re.search(r"boss relic #(\d+)", label)
    return int(match.group(1)) if match else None


def _item_text(item):
    if item is None:
        return "unknown"
    if isinstance(item, list):
        if len(item) >= 2 and isinstance(item[1], int):
            return f"{item[0]}{'+' * item[1]}"
        return "/".join(map(str, item))
    return str(item)


def _diff_text(diff):
    parts = []
    for key in ("hp", "max_hp", "gold"):
        change = diff.get(key)
        if change:
            parts.append(f"{key} {change['before']}->{change['after']} ({change['delta']:+})")
    for key, singular in (("deck", "card"), ("relics", "relic"), ("potions", "potion")):
        change = diff.get(key, {})
        if change.get("gained"):
            parts.append(f"{singular}+ " + ", ".join(_item_text(x) for x in change["gained"]))
        if change.get("lost"):
            parts.append(f"{singular}- " + ", ".join(_item_text(x) for x in change["lost"]))
    return "; ".join(parts) or "no persistent-state change"


def render_episode_markdown(trajectory):
    """Render one compact, outside-combat history suitable for seed comparison."""
    final = trajectory.get("final_state", {})
    lines = [
        f"## Seed {trajectory.get('seed')} — {trajectory.get('result')}", "",
        f"Final: Act {final.get('act')} floor {final.get('floor')}, "
        f"HP {final.get('hp')}/{final.get('max_hp')}, gold {final.get('gold')}", "",
    ]
    route = trajectory.get("map_path", [])
    if route:
        lines.extend(["Route: " + " -> ".join(
            f"F{n.get('floor')} {n.get('room', '?')}" for n in route), ""])
    lines.append("Events:")
    for event in trajectory.get("events", []):
        choice = ""
        choices, index = event.get("choices") or [], event.get("option_index")
        if index is not None:
            for detail in choices:
                if detail.get("index", detail.get("option", detail.get("option_index"))) == index:
                    choice = f"; typed choice={json.dumps(detail, ensure_ascii=False)}"
                    break
        lines.append(f"- A{event.get('act')}F{event.get('floor')} {event.get('event')}: "
                     f"{event.get('action')}{choice}; {_diff_text(event.get('diff', {}))}")
    if not trajectory.get("events"):
        lines.append("- none")
    lines.extend(["", "Combats:"])
    rewards_by_floor = {}
    for reward in trajectory.get("rewards", []):
        rewards_by_floor.setdefault((reward.get("act"), reward.get("floor")), []).append(reward)
    for combat in trajectory.get("combats", []):
        suffix = " incomplete" if combat.get("incomplete") else ""
        lines.append(f"- A{combat.get('act')}F{combat.get('floor')} "
                     f"{', '.join(combat.get('encounter') or ['unknown'])}: "
                     f"HP {combat.get('pre_hp')}->{combat.get('post_hp', '?')} "
                     f"(loss {combat.get('hp_loss', '?')}){suffix}; "
                     f"combat delta: {_diff_text(combat.get('diff', {}))}")
        candidates = combat.get("reward_candidates") or {}
        if candidates:
            lines.append("  - offers: " + json.dumps(candidates, ensure_ascii=False))
        for reward in rewards_by_floor.get((combat.get("act"), combat.get("floor")), []):
            if reward.get("item") is not None or reward.get("diff"):
                lines.append(f"  - reward {reward.get('action')}: "
                             f"{json.dumps(reward.get('item'), ensure_ascii=False)}; "
                             f"{_diff_text(reward.get('diff', {}))}")
    lines.extend(["", "Shops / chests / boss relics:"])
    for shop in trajectory.get("shops", []):
        detail = _diff_text(shop.get("diff", {}))
        if shop.get("selection_diff"):
            detail += "; " + _diff_text(shop["selection_diff"])
        lines.append(f"- shop A{shop.get('act')}F{shop.get('floor')}: {shop.get('action')}; "
                     f"{json.dumps(shop.get('item'), ensure_ascii=False)}; {detail}")
    for chest in trajectory.get("chests", []):
        lines.append(f"- chest A{chest.get('act')}F{chest.get('floor')}: "
                     f"{chest.get('action')}; {_diff_text(chest.get('diff', {}))}")
    for boss in trajectory.get("boss_relics", []):
        lines.append(f"- boss relic A{boss.get('act')}F{boss.get('floor')}: "
                     f"offers={boss.get('candidates')}; selected={boss.get('selected')}; "
                     f"{_diff_text(boss.get('diff', {}))}")
    if not any(trajectory.get(key) for key in ("shops", "chests", "boss_relics")):
        lines.append("- none")
    return "\n".join(lines) + "\n"

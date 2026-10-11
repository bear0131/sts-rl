"""Real-game bridge adapter for whole-run policy inference.

CommunicationMod is state-driven: it sends one JSON frame and expects one command
line back.  The training environment is action-driven: the policy sees a
canonical observation plus a legal mask over ``env.core.action_space`` ids.  This
module is the thin adapter between those two contracts.

It deliberately does not import the sim.  It consumes only the real game frame,
builds the same canonical observation shape used by ``SimGameEnv``, enumerates
backend-agnostic semantic actions, and maps a chosen semantic action to a
CommunicationMod command string.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from env.core import action_space as A
from env.core import actions
from env.core import encode
from env.core import game_actions as ga
from env.core import vocab as V
from env.core import native_actions as C
from schemas.combat_state import _card, normalize_real

LOG = logging.getLogger(__name__)


_SCREEN_PHASE = {
    "EVENT": "event",
    "CARD_REWARD": "combat_reward",
    "COMBAT_REWARD": "combat_reward",
    "GRID": "card_select",
    "MAP": "map",
    "REST": "rest",
    "CHEST": "chest",
    "BOSS_REWARD": "boss_relic",
    "SHOP_ROOM": "shop",
    "SHOP_SCREEN": "shop",
    "GAME_OVER": "none",
}

_ROOM_LABELS = {
    "m": "MONSTER",
    "monster": "MONSTER",
    "enemy": "MONSTER",
    "e": "ELITE",
    "elite": "ELITE",
    "?": "EVENT",
    "event": "EVENT",
    "$": "SHOP",
    "shop": "SHOP",
    "r": "REST",
    "rest": "REST",
    "t": "TREASURE",
    "treasure": "TREASURE",
    "chest": "TREASURE",
    "boss": "BOSS",
}

_BOSS_ALIASES = {
    "The Guardian": "THE_GUARDIAN",
    "Guardian": "THE_GUARDIAN",
    "Slime Boss": "SLIME_BOSS",
    "SlimeBoss": "SLIME_BOSS",
    "Hexaghost": "HEXAGHOST",
    "Bronze Automaton": "AUTOMATON",
    "BronzeAutomaton": "AUTOMATON",
    "The Collector": "COLLECTOR",
    "Collector": "COLLECTOR",
    "The Champ": "CHAMP",
    "Champ": "CHAMP",
    "Awakened One": "AWAKENED_ONE",
    "AwakenedOne": "AWAKENED_ONE",
    "Time Eater": "TIME_EATER",
    "TimeEater": "TIME_EATER",
    "Donu and Deca": "DONU_AND_DECA",
    "Donu & Deca": "DONU_AND_DECA",
    "DonuDeca": "DONU_AND_DECA",
}

_EVENT_ALIASES = {
    "Neow Event": "NEOW",
    "Neow": "NEOW",
    "We Meet Again": "WeMeetAgain",
    "WeMeetAgain": "WeMeetAgain",
    "Bonfire Spirits": "Bonfire Elementals",
}

_CAMPFIRE_ALIASES = {
    "rest": "rest",
    "smith": "smith",
    "recall": "recall",
    "lift": "lift",
    "toke": "toke",
    "dig": "dig",
}

_TARGETED_POTIONS = {
    "Fear Potion",
    "Fire Potion",
    "Poison Potion",
    "Weak Potion",
}
_TARGETED_POTION_KEYS = {"fearpotion", "firepotion", "poisonpotion", "weakpotion"}

_MAP_MAX_Y = 14


@dataclass(frozen=True)
class RealDecision:
    """A command chosen for one CommunicationMod frame."""

    command: str
    reason: str
    action_id: int | None = None
    semantic: tuple | None = None
    legal_ids: tuple[int, ...] = ()
    top: tuple[tuple[int, float, str], ...] = ()


def _norm_label(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _commands(frame: dict[str, Any]) -> set[str]:
    return {str(c).upper() for c in frame.get("available_commands", []) or []}


def _choice_list(gs: dict[str, Any]) -> list[Any]:
    return list(gs.get("choice_list") or [])


def _screen_state(gs: dict[str, Any]) -> dict[str, Any]:
    ss = gs.get("screen_state") or {}
    return ss if isinstance(ss, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _real_card_tuple(card: dict[str, Any]) -> tuple[str, int]:
    cid = card.get("id") or card.get("name") or "EMPTY"
    for key in ("upgrades", "upgrade_count", "up"):
        if key in card:
            up = int(card[key] or 0)
            break
    else:
        raise ValueError(f"card record is missing its upgrade count: {card!r}")
    c = _card(str(cid), up)
    return c["id"], c["up"]


def _real_card_dict(card: dict[str, Any], *, include_cost: bool = False) -> dict[str, Any]:
    cid, up = _real_card_tuple(card)
    out: dict[str, Any] = {"id": cid, "up": up}
    if include_cost:
        out["cost"] = int(card.get("cost", 0) or 0)
    return out


def _relic_id(relic: Any) -> str:
    if isinstance(relic, dict):
        return str(relic.get("id") or relic.get("name") or "PAD")
    return str(relic or "PAD")


_COMBAT_COUNTER_KEYS = {
    "Happy Flower": "happy_flower",
    "Incense Burner": "incense_burner",
    "InkBottle": "ink_bottle",
    "Inserter": "inserter",
    "Nunchaku": "nunchaku",
    "Pen Nib": "pen_nib",
    "Sundial": "sundial",
}


def _potion_id(potion: Any) -> str:
    if isinstance(potion, dict):
        pid = str(potion.get("id") or potion.get("name") or "PAD")
    else:
        pid = str(potion or "PAD")
    return "PAD" if pid in {"Potion Slot", "EMPTY", "None", ""} else pid


def _potion_requires_target(potion: Any, pid: str) -> bool:
    if isinstance(potion, dict) and "requires_target" in potion:
        return bool(potion.get("requires_target"))
    key = re.sub(r"[^a-z0-9]+", "", pid.lower())
    return pid in _TARGETED_POTIONS or key in _TARGETED_POTION_KEYS


def _price(item: Any) -> int:
    if not isinstance(item, dict):
        return -1
    return int(item.get("price", item.get("cost", -1)) or -1)


def _normalize_boss(value: Any) -> str:
    if not value:
        return "PAD"
    text = str(value)
    if text in V.BOSS_VOCAB_INDEX:
        return text
    if text in _BOSS_ALIASES:
        return _BOSS_ALIASES[text]
    enumish = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return enumish if enumish in V.BOSS_VOCAB_INDEX else "PAD"


def _normalize_event(value: Any) -> str:
    if not value:
        return "PAD"
    text = str(value)
    if text in _EVENT_ALIASES:
        return _EVENT_ALIASES[text]
    if text in V.EVENT_VOCAB_INDEX:
        return text
    if text.endswith(" Event"):
        stripped = text[:-6]
        if stripped in _EVENT_ALIASES:
            return _EVENT_ALIASES[stripped]
        if stripped in V.EVENT_VOCAB_INDEX:
            return stripped
    return text if text in V.EVENT_VOCAB_INDEX else "PAD"


def _screen_cards(ss: dict[str, Any]) -> list[tuple[str, int]]:
    cards = ss.get("cards") or ss.get("card_reward") or ss.get("cardRewards") or []
    if not isinstance(cards, list):
        return []
    out = []
    for card in cards:
        if isinstance(card, dict):
            out.append(_real_card_tuple(card))
    return out


def _screen_relics(ss: dict[str, Any]) -> list[str]:
    relics = ss.get("relics") or ss.get("boss_relics") or []
    if not isinstance(relics, list):
        return []
    return [_relic_id(r) for r in relics]


def _screen_potions(ss: dict[str, Any]) -> list[tuple[str, int]]:
    potions = ss.get("potions") or []
    if not isinstance(potions, list):
        return []
    return [(_potion_id(p), _price(p)) for p in potions]


def _structured_event_choices(ss: dict[str, Any], legal_count: int) -> list[dict[str, Any]]:
    """Pass through typed event-button metadata when a bridge supplies it.

    Stock CommunicationMod currently exports localized ``text``/``label`` only.
    Those strings are useful for display but are not stable entity identifiers,
    so this adapter deliberately does not parse card/relic/potion names from
    them.  A richer bridge can add ``kind`` plus typed fields to each option.
    """
    raw_options = ss.get("options") or []
    if not isinstance(raw_options, list):
        return []
    choices = []
    for raw in raw_options:
        if not isinstance(raw, dict) or raw.get("disabled"):
            continue
        index = _int_or_none(raw.get("choice_index"))
        if index is None:
            return []
        if not (0 <= index < min(legal_count, A.EVENT_OPT_W)):
            continue
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            return []
        detail: dict[str, Any] = {"index": index, "kind": kind}
        try:
            if kind in {"lose_card", "gain_card_then_remove"}:
                card = raw.get("card")
                if not isinstance(card, dict):
                    return []
                detail["card"] = _real_card_tuple(card)
            elif kind == "lose_relic":
                detail["relic"] = _relic_id(raw["relic"])
            elif kind == "lose_potion":
                detail["potion"] = _potion_id(raw["potion"])
            elif kind == "lose_gold":
                detail["gold"] = int(raw["gold"])
            elif kind == "lose_hp":
                detail["hp"] = int(raw["hp"])
            elif kind in {"upgrade_card", "upgrade_random_cards", "remove_card",
                          "transform_random_cards", "remove_card_upgrade_random"}:
                detail["gold"] = int(raw["gold"])
                if "count" in raw:
                    detail["count"] = int(raw["count"])
            elif kind != "leave":
                return []
        except (KeyError, TypeError, ValueError):
            return []
        choices.append(detail)
    if {choice["index"] for choice in choices} != set(range(min(legal_count, A.EVENT_OPT_W))):
        return []
    return choices


def _choice_index(labels: list[Any], aliases: set[str], occurrence: int = 0) -> int | None:
    seen = 0
    for idx, label in enumerate(labels):
        if _norm_label(label) in aliases:
            if seen == occurrence:
                return idx
            seen += 1
    return None


def _map_choice(label: Any, fallback_x: int) -> tuple[int, str]:
    text = str(label)
    lower = text.strip().lower()
    if lower == "boss":
        return fallback_x, "BOSS"
    x = fallback_x
    match = re.search(r"x\s*=\s*(-?\d+)", text)
    if match:
        x = int(match.group(1))
    room = "NONE"
    for token, mapped in _ROOM_LABELS.items():
        if token and re.search(rf"(^|[^a-z]){re.escape(token)}([^a-z]|$)", lower):
            room = mapped
            break
    return x, room


def _map_room(value: Any) -> str:
    text = str(value or "NONE")
    return _ROOM_LABELS.get(_norm_label(text), text)


def _node_xy(node: Any) -> tuple[int, int] | None:
    if not isinstance(node, dict):
        return None
    x = _int_or_none(node.get("x"))
    y = _int_or_none(node.get("y"))
    if x is None or y is None:
        return None
    return x, y


def _node_room(node: dict[str, Any]) -> str:
    return _map_room(node.get("room") or node.get("room_type") or node.get("symbol"))


def _node_children(node: dict[str, Any]) -> list[tuple[int, int]]:
    out = []
    for child in node.get("children") or []:
        if isinstance(child, dict):
            xy = _node_xy(child)
            if xy is not None:
                out.append(xy)
        else:
            x = _int_or_none(child)
            y = _int_or_none(node.get("y"))
            if x is not None and y is not None:
                out.append((x, y + 1))
    return out


def _campfire_option(label: Any) -> str | None:
    return _CAMPFIRE_ALIASES.get(_norm_label(label))


class RealGameAdapter:
    """Stateful adapter for one real Slay the Spire run."""

    def __init__(
        self,
        *,
        character: str = "IRONCLAD",
        ascension: int = 0,
        global_map_view: bool = False,
        card_language: bool = False,
        combat_language: bool = False,
        deck_size_observation: bool = False,
        neow_observation: bool = False,
        event_option_observation: bool = False,
        combat_action_contract: str = "legacy",
    ) -> None:
        if combat_action_contract not in ("legacy", *C.NEURAL_CONTRACTS):
            raise ValueError("unsupported bridge combat action contract")
        self.combat_action_contract = combat_action_contract
        self._joint_multi_plan = None
        self.character = character
        self.ascension = int(ascension)
        self.global_map_view = bool(global_map_view)
        self.card_language = bool(card_language)
        self.combat_language = bool(combat_language)
        self.deck_size_observation = bool(deck_size_observation)
        self.neow_observation = bool(neow_observation)
        self.event_option_observation = bool(event_option_observation)

        self.frame: dict[str, Any] = {}
        self.gs: dict[str, Any] = {}
        self.multi_selected: set[int] = set()
        self._multi_signature: tuple[Any, ...] | None = None
        self._leaving_shop = False
        # CommunicationMod's SKIP on CARD_REWARD closes the card sub-screen
        # instead of marking the combat card reward skipped.  Track this case so
        # the returned COMBAT_REWARD frame behaves like the sim's skipRewardCards:
        # the same card reward is masked out and the policy can claim remaining
        # non-card rewards or proceed.
        self._combat_reward_card_open = False
        self._skipped_combat_card_reward = False

    def update(self, frame: dict[str, Any]) -> None:
        self.frame = frame
        self.gs = frame.get("game_state") or {}
        self._reconcile_multi()
        if self.screen_type() == "SHOP_SCREEN":
            self._leaving_shop = False
        if self.screen_type() not in {"COMBAT_REWARD", "CARD_REWARD"}:
            self._combat_reward_card_open = False
            self._skipped_combat_card_reward = False

    def screen_type(self) -> str:
        return str(self.gs.get("screen_type") or "NONE")

    def phase(self) -> str:
        if self.gs.get("combat_state") and str(self.gs.get("room_phase") or "").upper() == "COMBAT":
            return "combat"
        return _SCREEN_PHASE.get(self.screen_type(), "none")

    def in_game(self) -> bool:
        return bool(self.frame.get("in_game"))

    def commands(self) -> set[str]:
        return _commands(self.frame)

    def choices(self) -> list[Any]:
        return _choice_list(self.gs)

    def _refresh_command(self) -> str:
        if "STATE" in self.commands():
            return "STATE"
        return "WAIT 1" if "WAIT" in self.commands() else "STATE"

    def _combat_ready_for_policy(self) -> bool:
        if not self.gs.get("combat_state"):
            return False
        if self._combat_card_choice_screen():
            return True
        # CommunicationMod can expose an actionable hand-selection screen
        # while AbstractDungeon.actionManager is still EXECUTING_ACTIONS.
        # The settled screen and its advertised commands are the command
        # contract; waiting for action_phase to change deadlocks selections
        # opened by effects such as Gambler's Brew.
        if (self.screen_type() == "HAND_SELECT"
                and ({"CHOOSE", "CONFIRM"} & self.commands())):
            return True
        action_phase = str(self.gs.get("action_phase") or "").upper()
        if action_phase and action_phase != "WAITING_ON_USER":
            return False
        if self.screen_type() == "NONE" and bool(self.gs.get("is_screen_up")):
            screen_name = str(self.gs.get("screen_name") or "").upper()
            if screen_name and screen_name != "NONE":
                return False
        return True

    def _combat_card_choice_screen(self) -> bool:
        if str(self.gs.get("room_phase") or "").upper() != "COMBAT":
            return False
        if self.screen_type() not in {"CARD_REWARD", "GRID"}:
            return False
        if "CHOOSE" not in self.commands():
            return False
        ss = _screen_state(self.gs)
        return bool(_screen_cards(ss) or self.choices())

    def auto_command(self, *, auto_enter_shop: bool = True) -> tuple[str, str] | None:
        """Commands that should bypass the policy to match the sim state model."""
        if "error" in self.frame:
            return "STATE", "mod_error_state_refresh"
        if not self.in_game():
            return None
        if not self.frame.get("ready_for_command", True):
            return self._refresh_command(), "not_ready"
        if self.gs.get("combat_state") and not self._combat_ready_for_policy():
            return self._refresh_command(), "combat_not_waiting"
        if self._joint_multi_plan is not None:
            command = self._continue_joint_multi()
            if command is not None:
                return command, "complete_multi_continuation"

        st = self.screen_type()
        ss = _screen_state(self.gs)
        cl = self.choices()

        if st == "EVENT" and _normalize_event(ss.get("event_id")) == "NEOW" and len(cl) == 1:
            return "CHOOSE 0", "neow_intro"

        if auto_enter_shop and st == "SHOP_ROOM":
            if self._leaving_shop:
                self._leaving_shop = False
                return "PROCEED", "leave_shop_room"
            if "CHOOSE" in self.commands() and cl:
                return "CHOOSE 0", "enter_shop_screen"

        # The simulator exposes the actual card candidates before the upper
        # policy chooses a reward card or skips.  CommunicationMod instead
        # exposes a placeholder on COMBAT_REWARD and reveals the candidates
        # only after CHOOSE.  Opening that sub-screen is transport, not a
        # commitment: CARD_REWARD still offers SKIP.  Once no claimable
        # non-card reward remains, enter it automatically so the policy sees
        # the same candidate-bearing state it was trained on.
        if (st == "COMBAT_REWARD" and not self._skipped_combat_card_reward
                and "CHOOSE" in self.commands()):
            normalized = [_norm_label(choice) for choice in cl]
            card_indices = [i for i, label in enumerate(normalized) if label == "card"]
            claimable_other = any(
                label in {"gold", "stolen_gold", "relic"}
                or (label == "potion" and self._has_empty_potion_slot())
                for label in normalized
            )
            if len(card_indices) == 1 and not claimable_other:
                self._combat_reward_card_open = True
                return f"CHOOSE {card_indices[0]}", "open_card_reward"

        return None

    # ---------------------------------------------------------------- obs/mask
    def observation(self, pending: tuple | None = None) -> dict[str, np.ndarray]:
        obs = encode.encode(
            self.obs_canonical(pending=pending),
            global_map_view=self.global_map_view,
            card_language=self.card_language,
            combat_language=self.combat_language,
            deck_size_observation=self.deck_size_observation,
            neow_observation=self.neow_observation,
            event_option_observation=self.event_option_observation,
        )
        if self.combat_action_contract in C.NEURAL_CONTRACTS:
            obs["combat_legal"] = self.action_mask()[C.policy_offset(self.combat_action_contract):]
        return obs

    def obs_canonical(self, pending: tuple | None = None) -> dict[str, Any]:
        canon = self._base_canonical()
        canon["pending_card_slot"] = int(pending[1]) + 1 if pending and pending[0] == "play" else 0
        canon["pending_potion_slot"] = int(pending[1]) + 1 if pending and pending[0] == "potion" else 0
        desc = self.select_descriptor()
        legal = self.legal_semantic_actions()
        if A.is_multi_select(desc):
            canon["arg_step"] = A.PICK_HAND
            sel = dict((canon.get("options") or {}).get("select") or desc)
            sel["hand_selected"] = sorted(self.multi_selected)
            sel["pick_remaining"] = max(
                0, int(sel.get("pick_max", 0)) - len(self.multi_selected)
            )
            canon.setdefault("options", {})["select"] = sel
        else:
            canon["arg_step"] = A.arg_step_name(pending, legal, self.phase())
        return canon

    def action_mask(self, pending: tuple | None = None) -> np.ndarray:
        if self.combat_action_contract in C.NEURAL_CONTRACTS and self.phase() == "combat":
            return C.joint_mask(list(self.joint_semantic_actions()), contract=self.combat_action_contract)
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        desc = self.select_descriptor()
        if A.is_multi_select(desc):
            ids = A.multi_legal_ids(desc, self.multi_selected)
        else:
            ids = A.legal_ids(self.legal_semantic_actions(), pending, self.phase())
        for action_id in ids:
            if 0 <= action_id < A.N_ACTIONS:
                mask[action_id] = True
        return (np.pad(mask, (0, C.policy_size(self.combat_action_contract) - C.JOINT_END)) if self.combat_action_contract in C.NEURAL_CONTRACTS else mask)

    def joint_semantic_actions(self):
        """Complete actions share policy IDs with the simulator; UI indices stay handles."""
        desc = self.select_descriptor()
        if A.is_multi_select(desc):
            count = int(desc["cards_in_hand"])
            if count > A.HAND_SLOTS:
                raise ValueError("multi selection exceeds ten hand slots")
            maximum = min(count, int(desc["pick_max"]))
            if self.combat_action_contract == C.AUTOREGRESSIVE_CONTRACT:
                result = {}
                if len(self.multi_selected) < maximum:
                    first = max(self.multi_selected, default=-1) + 1
                    result.update({C.AUTOREGRESSIVE_MULTI + i: ("select_next", i)
                                   for i in range(first, count)})
                if len(self.multi_selected) >= int(desc.get("pick_min", 0)):
                    result[C.AUTOREGRESSIVE_STOP] = (ga.MULTI_SELECT, tuple(sorted(self.multi_selected)))
                return result
            return {C.JOINT_MULTI + bits: (ga.MULTI_SELECT, tuple(i for i in range(count) if bits >> i & 1))
                    for bits in range(1 << count) if bits.bit_count() <= maximum}
        result = {}
        for semantic in self.legal_semantic_actions():
            if semantic[0] == ga.COMBAT:
                kind = actions.decode(semantic[1])
                aid = C.JOINT_END if kind[0] == actions.KIND_END else (
                    C.JOINT_PLAY + kind[1] * 6 + (5 if kind[2] is None else kind[2]))
            elif semantic[0] == ga.USE_POTION:
                aid = C.JOINT_POTION + semantic[1] * 6 + (5 if semantic[2] is None else semantic[2])
            elif semantic[0] == ga.SELECT:
                aid = C.JOINT_PICK + semantic[1]
            else:
                raise ValueError("unsupported joint combat semantic: " + str(semantic))
            if self.combat_action_contract == C.AUTOREGRESSIVE_CONTRACT:
                aid -= C.JOINT_END
            result[aid] = semantic
        return result

    def command_for_joint_action(self, action_id):
        semantic = self.joint_semantic_actions().get(int(action_id))
        if semantic is None:
            raise ValueError("complete combat action is no longer legal")
        if semantic[0] == "select_next":
            self.multi_selected.add(semantic[1])
            return None  # Internal prefix decision; no UI command until STOP.
        if semantic[0] != ga.MULTI_SELECT:
            return self.command_for_semantic(semantic)
        indices = sorted(semantic[1], reverse=True)
        self._joint_multi_plan = dict(signature=self._joint_screen_identity(),
                                      remaining=indices, previous=None, sent=None, selected=0)
        return self._continue_joint_multi()

    def _joint_screen_identity(self):
        desc = self.select_descriptor() or {}
        return (self.screen_type(), self.gs.get("floor"),
                (self.gs.get("combat_state") or {}).get("turn"), desc.get("context"))

    def _continue_joint_multi(self):
        """Expand one subset into acknowledged UI choices without another model call."""
        plan = self._joint_multi_plan
        if plan["signature"] != self._joint_screen_identity():
            self._joint_multi_plan = None
            return None
        hand = tuple(self._combat_hand_cards())
        count = int(_screen_state(self.gs).get("have_selected", 0))
        if plan["sent"] is not None:
            previous, index = plan["previous"], plan["sent"]
            removed = previous[:index] + previous[index + 1:]
            if hand == previous and count == plan["selected"]:
                return self._refresh_command()  # Await acknowledgement; never select twice.
            if not (hand == removed or (hand == previous and count == plan["selected"] + 1)):
                self._joint_multi_plan = None
                raise ValueError("multi-selection frame changed unexpectedly; discard the pending subset")
        if not plan["remaining"]:
            if "CONFIRM" not in self.commands():
                return self._refresh_command()
            self._joint_multi_plan = None
            return "CONFIRM"
        index = plan["remaining"].pop(0)
        if index >= len(hand) or "CHOOSE" not in self.commands():
            self._joint_multi_plan = None
            raise ValueError("multi-selection candidate is no longer available")
        plan.update(previous=hand, selected=count, sent=index)
        return f"CHOOSE {index}"

    # ---------------------------------------------------------- canonical state
    def _base_canonical(self) -> dict[str, Any]:
        if self.gs.get("combat_state"):
            try:
                canon = normalize_real(self.gs) or {}
            except Exception:
                LOG.exception("normalize_real failed")
                canon = {}
            canon["phase"] = "combat"
            desc = self.select_descriptor()
            if desc is not None:
                canon.setdefault("options", {})["select"] = desc
            canon["global"] = self._global_block()
            counters = canon.setdefault("player", {}).setdefault(
                "combat_counters", {}
            )
            for relic, value in zip(
                canon["global"]["relics"], canon["global"]["relic_counters"]
            ):
                key = _COMBAT_COUNTER_KEYS.get(relic)
                if key is not None:
                    counters[key] = value
            return canon

        phase = self.phase()
        canon = {
            "floor": int(self.gs.get("floor", 0) or 0),
            "hp": int(self.gs.get("current_hp", 0) or 0),
            "max_hp": int(self.gs.get("max_hp", 0) or 0),
            "gold": int(self.gs.get("gold", 0) or 0),
            "deck": sorted(self._deck_cards()),
            "relics": sorted(self._relics()),
            "phase": phase,
            "options": self._options(phase),
            "global": self._global_block(),
        }
        return canon

    def _global_block(self) -> dict[str, Any]:
        from env.core.boss_value import is_boss_room
        ss = _screen_state(self.gs)
        return {
            "character": self.gs.get("class") or self.character,
            "ascension": int(
                self.gs.get("ascension_level", self.gs.get("ascension", self.ascension))
                or self.ascension
            ),
            "act": int(self.gs.get("act", 0) or 0),
            "floor": int(self.gs.get("floor", 0) or 0),
            "combat_is_boss": is_boss_room(self.gs.get("room_type", "")),
            "boss": _normalize_boss(
                self.gs.get("boss")
                or self.gs.get("act_boss")
                or ss.get("boss")
                or ss.get("act_boss")
            ),
            "relics": self._relics(),
            "relic_counters": self._relic_counters(),
            "potions": self._potions(),
            "gold": int(self.gs.get("gold", 0) or 0),
            "hp": int(self.gs.get("current_hp", 0) or 0),
            "max_hp": int(self.gs.get("max_hp", 0) or 0),
            "deck": self._deck_cards(),
            "boss_relics": _screen_relics(ss) if self.phase() == "boss_relic" else [],
        }

    def _deck_cards(self) -> list[tuple[str, int]]:
        deck = self.gs.get("deck") or []
        if not isinstance(deck, list):
            return []
        return [_real_card_tuple(c) for c in deck if isinstance(c, dict)]

    def _relics(self) -> list[str]:
        relics = self.gs.get("relics") or []
        if not isinstance(relics, list):
            return []
        return [_relic_id(r) for r in relics]

    def _relic_counters(self) -> list[int]:
        relics = self.gs.get("relics") or []
        if not isinstance(relics, list):
            return []
        return [
            int(r.get("counter", 0) or 0) if isinstance(r, dict) else 0
            for r in relics
        ]

    def _potions(self) -> list[str]:
        return [_potion_id(p) for p in self._potion_records()]

    def _potion_records(self) -> list[Any]:
        potions = self.gs.get("potions") or []
        if not isinstance(potions, list):
            return []
        return potions

    def _options(self, phase: str) -> dict[str, Any]:
        ss = _screen_state(self.gs)
        st = self.screen_type()
        cl = self.choices()
        opts: dict[str, Any] = {}

        if phase == "combat_reward":
            cards = _screen_cards(ss)
            relics = _screen_relics(ss)
            potions = [pid for pid, _price_ in _screen_potions(ss)]
            has_card = not self._skipped_combat_card_reward
            opts.update({
                "gold": [0] * sum(_norm_label(c) in {"gold", "stolen_gold"} for c in cl),
                "relics": relics,
                "potions": potions,
                "card_count": (len(cards) if cards else int("card" in {_norm_label(c) for c in cl}))
                              if has_card else 0,
                "cards": cards if has_card else [],
            })
        elif phase == "event":
            event = _normalize_event(ss.get("event_id") or ss.get("event_name"))
            opts["event"] = event
            if event == "NEOW" and self.neow_observation:
                opts["neow"] = self._neow_options(ss)
            elif event != "NEOW":
                opts["event_options"] = list(range(min(len(cl), A.EVENT_OPT_W)))
                details = _structured_event_choices(ss, len(cl))
                if details:
                    opts["event_choices"] = details
        elif phase == "card_select":
            cards = _screen_cards(ss)
            opts["cards"] = cards
            opts["select"] = self._select_block(cards=cards, overworld=True)
        elif phase == "map":
            opts["nodes"] = self._map_nodes()
            if self.global_map_view:
                opts["full_map"] = self._full_map_nodes(ss)
        elif phase == "shop":
            opts.update(self._shop_options(ss))
        return opts

    def _neow_options(self, ss: dict[str, Any]) -> list[dict[str, int]]:
        raw = ss.get("options") or ss.get("rewards") or []
        out = []
        if isinstance(raw, list):
            for item in raw[:A.NEOW_W]:
                if isinstance(item, dict):
                    out.append({
                        "bonus_id": int(item.get("bonus_id", 0) or 0),
                        "drawback_id": int(item.get("drawback_id", 0) or 0),
                    })
        while len(out) < A.NEOW_W:
            out.append({"bonus_id": 0, "drawback_id": 0})
        return out

    def _map_nodes(self) -> list[tuple[int, str]]:
        ss = _screen_state(self.gs)
        next_nodes = ss.get("next_nodes") or []
        if isinstance(next_nodes, list) and next_nodes:
            nodes = []
            for node in next_nodes:
                xy = _node_xy(node)
                if xy is None:
                    continue
                x, _y = xy
                if 0 <= x < A.N_COLS:
                    nodes.append((x, _node_room(node)))
            return nodes[:encode.MAP_EDGES]

        if ss.get("boss_available"):
            current = _node_xy(ss.get("current_node"))
            if current is not None:
                return [(current[0], "BOSS")]

        nodes = []
        fallback = int(self.gs.get("current_x", self.gs.get("map_node_x", 0)) or 0)
        for i, choice in enumerate(self.choices()):
            x, room = _map_choice(choice, fallback + i)
            if 0 <= x < A.N_COLS:
                nodes.append((x, room))
        return nodes[:encode.MAP_EDGES]

    def _full_map_nodes(self, ss: dict[str, Any]) -> list[dict[str, Any]]:
        raw = self.gs.get("map") or ss.get("map") or ss.get("nodes") or ss.get("full_map") or []
        if not isinstance(raw, list):
            return []

        raw_nodes: dict[tuple[int, int], dict[str, Any]] = {}
        for node in raw:
            xy = _node_xy(node)
            if xy is None:
                continue
            x, y = xy
            if not (0 <= x < A.N_COLS and 0 <= y <= _MAP_MAX_Y):
                continue
            if isinstance(node, dict):
                raw_nodes[(x, y)] = node

        current = _node_xy(ss.get("current_node"))
        if current is None:
            current = (
                _int_or_none(self.gs.get("current_x")),
                _int_or_none(self.gs.get("current_y")),
            )
            if current[0] is None or current[1] is None:
                current = None

        if current is None and ss.get("first_node_chosen") is False:
            current = (-1, -1)

        choices: set[tuple[int, int]] = set()
        if ss.get("boss_available") and current is not None and current[1] >= _MAP_MAX_Y:
            choices.add(current)
        else:
            next_nodes = ss.get("next_nodes") or []
            if isinstance(next_nodes, list):
                for node in next_nodes:
                    xy = _node_xy(node)
                    if xy is not None:
                        choices.add(xy)

        if not choices:
            for x, _room in self._map_nodes():
                if current is not None and current[1] is not None and current[1] >= 0:
                    choices.add((x, int(current[1]) + 1))
                else:
                    choices.add((x, 0))

        downstream = set(choices)
        frontier = list(choices)
        while frontier:
            xy = frontier.pop()
            node = raw_nodes.get(xy)
            if node is None:
                continue
            for child in _node_children(node):
                if child not in downstream:
                    downstream.add(child)
                    frontier.append(child)

        out = []
        for (x, y), node in sorted(raw_nodes.items(), key=lambda item: (item[0][1], item[0][0])):
            children = []
            if y < _MAP_MAX_Y:
                children = [
                    child_x
                    for child_x, child_y in _node_children(node)
                    if child_y == y + 1 and 0 <= child_x < A.N_COLS
                ]
            out.append({
                "x": x,
                "y": y,
                "room": _node_room(node),
                "current": current == (x, y),
                "choice": (x, y) in choices,
                "reachable": (x, y) in downstream,
                "children": children,
            })
        return out

    def _shop_options(self, ss: dict[str, Any]) -> dict[str, Any]:
        cards = []
        for card in ss.get("cards") or []:
            if isinstance(card, dict):
                cid, up = _real_card_tuple(card)
                cards.append((cid, up, _price(card)))
        relics = [(_relic_id(r), _price(r)) for r in ss.get("relics") or []]
        potions = _screen_potions(ss)
        remove_cost = int(ss.get("purge_cost", ss.get("remove_cost", -1)) or -1)
        return {
            "shop_cards": cards,
            "shop_relics": relics,
            "shop_potions": potions,
            "shop_remove_cost": remove_cost,
        }

    # --------------------------------------------------------------- selectors
    def select_descriptor(self) -> dict[str, Any] | None:
        if not self.gs.get("combat_state"):
            return None
        if self._combat_card_choice_screen():
            return self._select_block(cards=_screen_cards(_screen_state(self.gs)), overworld=False)
        if self.screen_type() != "HAND_SELECT":
            return None
        return self._select_block(cards=self._combat_hand_cards(), overworld=False)

    def _select_block(
        self, *, cards: list[tuple[str, int]], overworld: bool
    ) -> dict[str, Any]:
        ss = _screen_state(self.gs)
        context = self._select_context(ss, cards)
        pick_count = (
            _int_or_none(ss.get("pick_count"))
            or _int_or_none(ss.get("num_cards"))
            or _int_or_none(ss.get("max_cards"))
            or 1
        )
        have_selected = _int_or_none(ss.get("have_selected")) or 0
        can_pick_zero = bool(ss.get("can_pick_zero", ss.get("canPickZero", False)))
        can_pick_any = bool(ss.get("can_pick_any", ss.get("canPickAny", False)))
        multi = bool(ss.get("multi", False))
        if not overworld and (
            context in {"EXHAUST_MANY", "GAMBLE"}
            or can_pick_any
            or ("CONFIRM" in self.commands() and pick_count != 1)
        ):
            multi = True
        if context == "GAMBLE":
            can_pick_any = True
        pick_max = _int_or_none(ss.get("pick_max")) or _int_or_none(ss.get("max_cards"))
        if pick_max is None:
            pick_max = len(cards) if multi and can_pick_any else pick_count
        raw_cards = ss.get("cards") or (self.gs.get("combat_state") or {}).get("hand") or []
        candidates = []
        for i, (cid, up) in enumerate(cards):
            card = {"id": cid, "up": up}
            if i < len(raw_cards) and isinstance(raw_cards[i], dict) and "cost" in raw_cards[i]:
                card["cost"] = raw_cards[i]["cost"]
            candidates.append({"native_index": i, "card": card})
        if not overworld and context == "CODEX" and "SKIP" in self.commands():
            candidates.append({"native_index": len(cards), "card": None})
        if not overworld and context in {"SECRET_WEAPON", "SECRET_TECHNIQUE", "SEEK"}:
            # Static vocabulary preserves native CardId order without importing the sim.
            candidates.sort(key=lambda c: (encode.CARD_INDEX.get(c["card"]["id"], 0),
                                            c["card"]["up"], c["card"].get("cost", 0)))
        return {
            "context": context,
            "pick_remaining": max(0, pick_count - have_selected),
            "can_pick_zero": can_pick_zero or multi,
            "can_pick_any": can_pick_any,
            "selected": [],
            "multi": multi,
            "pick_max": int(pick_max),
            "cards_in_hand": len(cards) if not overworld else 0,
            "zone": "grid" if overworld else (
                "hand" if self.screen_type() == "HAND_SELECT" else
                {"HEADBUTT": "discard", "HOLOGRAM": "discard",
                 "LIQUID_MEMORIES_POTION": "discard", "EXHUME": "exhaust",
                 "SECRET_WEAPON": "draw", "SECRET_TECHNIQUE": "draw",
                 "SEEK": "draw", "CODEX": "generated", "DISCOVERY": "generated"}.get(context, "grid")),
            "candidates": candidates,
        }

    def _select_context(self, ss: dict[str, Any], cards: list[tuple[str, int]]) -> str:
        for key in ("context", "task", "select_context", "screen_name", "screen_type"):
            value = ss.get(key)
            if value:
                enumish = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").upper()
                if enumish in V.SELECT_CONTEXT_VOCAB_INDEX:
                    return enumish
        if ss.get("for_upgrade") or ss.get("forUpgrade"):
            return "UPGRADE"
        if ss.get("for_transform") or ss.get("forTransform"):
            return "TRANSFORM"
        if ss.get("for_purge") or ss.get("forPurge"):
            return "REMOVE"
        if self.screen_type() == "GRID":
            return "REMOVE" if any(_norm_label(c) == "purge" for c in self.choices()) else "PAD"
        if self.screen_type() == "HAND_SELECT":
            names = {cid for cid, _up in cards}
            if "Calculated Gamble" in names:
                return "GAMBLE"
        current_action = str(self.gs.get("current_action") or "")
        if current_action == "DiscoveryAction":
            return "DISCOVERY"
        return "PAD"

    def _select_signature(self) -> tuple[Any, ...] | None:
        desc = self.select_descriptor()
        if not A.is_multi_select(desc):
            return None
        return (
            self.screen_type(),
            self.gs.get("floor"),
            self.gs.get("combat_state", {}).get("turn"),
            desc.get("context"),
            tuple(self._combat_hand_cards()),
        )

    def _reconcile_multi(self) -> None:
        sig = self._select_signature()
        if sig != self._multi_signature:
            self.multi_selected.clear()
            self._multi_signature = sig

    def _combat_hand_cards(self) -> list[tuple[str, int]]:
        cs = self.gs.get("combat_state") or {}
        hand = cs.get("hand") or []
        if not isinstance(hand, list):
            return []
        return [_real_card_tuple(c) for c in hand if isinstance(c, dict)]

    # --------------------------------------------------------------- legal acts
    def legal_semantic_actions(self) -> list[tuple]:
        if not self.in_game() or self.screen_type() == "GAME_OVER":
            return []
        if self.phase() == "combat":
            return self._legal_combat_actions()

        pots = self._legal_potion_actions(combat=False)
        phase = self.phase()
        if phase == "event":
            return self._legal_event_actions() + pots
        if phase == "map":
            return [ga.map_node(x) for x, _room in self._map_nodes()] + pots
        if phase == "combat_reward":
            return self._legal_reward_actions() + pots
        if phase == "card_select":
            return self._legal_card_select_actions() + pots
        if phase == "rest":
            return self._legal_rest_actions() + pots
        if phase == "chest":
            acts = []
            if "CHOOSE" in self.commands():
                acts.append(ga.chest(True))
            if "PROCEED" in self.commands():
                acts.append(ga.chest(False))
            return acts + pots
        if phase == "boss_relic":
            n = len(_screen_relics(_screen_state(self.gs))) or len(self.choices())
            return [ga.boss_relic(i) for i in range(min(n, A.BOSS_RELIC_W))] + pots
        if phase == "shop":
            return self._legal_shop_actions() + pots
        return pots

    def _legal_combat_actions(self) -> list[tuple]:
        if not self._combat_ready_for_policy():
            return []
        if self._combat_card_choice_screen():
            ss = _screen_state(self.gs)
            n = len(_screen_cards(ss)) or len(self.choices())
            desc = self.select_descriptor()
            if desc and desc["candidates"]:
                n = len(desc["candidates"])
            if n > A.G_MAX:
                raise ValueError("combat candidate capacity 96 exceeded")
            return [ga.select(i) for i in range(n)]
        if self.screen_type() == "HAND_SELECT":
            if A.is_multi_select(self.select_descriptor()):
                return []
            n = len(self._combat_hand_cards())
            return [ga.select(i) for i in range(n)]

        acts: list[tuple] = []
        cs = self.gs.get("combat_state") or {}
        if "END" in self.commands():
            acts.append(ga.combat(actions.END_TURN))
        monsters = cs.get("monsters") or []
        targets = [
            i for i, m in enumerate(monsters[:actions.MAX_MONSTERS])
            if isinstance(m, dict) and not m.get("is_gone") and int(m.get("current_hp", 0) or 0) > 0
        ]
        for h, card in enumerate((cs.get("hand") or [])[:actions.HAND_SLOTS]):
            if not isinstance(card, dict) or not card.get("is_playable", True):
                continue
            if card.get("has_target"):
                for target in targets:
                    acts.append(ga.combat(actions.encode_play(h, target)))
            else:
                acts.append(ga.combat(actions.encode_play(h, None)))
        return acts + self._legal_potion_actions(combat=True)

    def _legal_event_actions(self) -> list[tuple]:
        ss = _screen_state(self.gs)
        event = _normalize_event(ss.get("event_id") or ss.get("event_name"))
        n = len(self.choices())
        if event == "NEOW" and n >= A.NEOW_W:
            return [ga.neow(i) for i in range(A.NEOW_W)]
        return [ga.event(i) for i in range(min(n, A.EVENT_OPT_W))]

    def _legal_card_select_actions(self) -> list[tuple]:
        cmds = self.commands()
        if "CONFIRM" in cmds and "CHOOSE" not in cmds:
            return [ga.proceed()]
        if "CHOOSE" not in cmds:
            return [ga.proceed()] if "CONFIRM" in cmds else []
        acts = [ga.select(i) for i, _ in enumerate(_screen_cards(_screen_state(self.gs))[:A.G_MAX])]
        if "CONFIRM" in cmds and self._grid_can_confirm_without_more_picks():
            acts.append(ga.proceed())
        return acts

    def _grid_can_confirm_without_more_picks(self) -> bool:
        ss = _screen_state(self.gs)
        if bool(ss.get("any_number") or ss.get("can_pick_any") or ss.get("canPickAny")):
            return True
        selected = ss.get("selected_cards") or ss.get("selectedCards") or []
        selected_count = len(selected) if isinstance(selected, list) else 0
        required = (
            _int_or_none(ss.get("num_cards"))
            or _int_or_none(ss.get("pick_count"))
            or _int_or_none(ss.get("max_cards"))
            or 1
        )
        return selected_count >= required

    def _legal_reward_actions(self) -> list[tuple]:
        st = self.screen_type()
        ss = _screen_state(self.gs)
        cl = self.choices()
        if st == "CARD_REWARD":
            cards = _screen_cards(ss)
            n = len(cards) or len(cl)
            acts = [ga.reward_card(i) for i in range(min(n, A.REWARD_CARD))]
            if "SKIP" in self.commands():
                acts.append(ga.skip_cards())
            return acts

        acts = []
        counts = {ga.CLAIM_GOLD: 0, ga.CLAIM_RELIC: 0, ga.CLAIM_POTION: 0}
        for label in cl:
            norm = _norm_label(label)
            if norm in {"gold", "stolen_gold"}:
                acts.append(ga.claim(ga.CLAIM_GOLD, counts[ga.CLAIM_GOLD]))
                counts[ga.CLAIM_GOLD] += 1
            elif norm == "relic":
                acts.append(ga.claim(ga.CLAIM_RELIC, counts[ga.CLAIM_RELIC]))
                counts[ga.CLAIM_RELIC] += 1
            elif norm == "potion":
                if self._has_empty_potion_slot():
                    acts.append(ga.claim(ga.CLAIM_POTION, counts[ga.CLAIM_POTION]))
                counts[ga.CLAIM_POTION] += 1
            elif norm == "card" and not self._skipped_combat_card_reward:
                acts.append(ga.reward_card(0))
        if "PROCEED" in self.commands():
            acts.append(ga.proceed())
        return acts

    def _legal_rest_actions(self) -> list[tuple]:
        acts = []
        for label in self.choices():
            opt = _campfire_option(label)
            if opt is not None:
                acts.append(ga.campfire(opt))
        if "PROCEED" in self.commands():
            acts.append(ga.proceed())
        return acts

    def _legal_shop_actions(self) -> list[tuple]:
        if self.screen_type() != "SHOP_SCREEN":
            return [ga.shop_proceed()] if "PROCEED" in self.commands() else []
        acts = [action for action, _idx in self._shop_choice_actions()]
        if "LEAVE" in self.commands() or "PROCEED" in self.commands():
            acts.append(ga.shop_proceed())
        return acts

    def _legal_potion_actions(self, *, combat: bool) -> list[tuple]:
        if "POTION" not in self.commands():
            return []
        if combat and not self.gs.get("combat_state"):
            return []
        acts = []
        for slot, potion in enumerate(self._potion_records()[:A.P_MAX]):
            pid = _potion_id(potion)
            if pid == "PAD":
                continue
            if isinstance(potion, dict) and potion.get("can_use") is False:
                continue
            if not combat and not (isinstance(potion, dict) and potion.get("can_use") is True):
                continue
            if combat and _potion_requires_target(potion, pid):
                for target in self._live_monster_indices():
                    acts.append(ga.use_potion(slot, target))
            else:
                acts.append(ga.use_potion(slot, None))
        return acts

    def _live_monster_indices(self) -> list[int]:
        cs = self.gs.get("combat_state") or {}
        out = []
        for i, m in enumerate((cs.get("monsters") or [])[:actions.MAX_MONSTERS]):
            if isinstance(m, dict) and not m.get("is_gone") and int(m.get("current_hp", 0) or 0) > 0:
                out.append(i)
        return out

    def _has_empty_potion_slot(self) -> bool:
        return any(pid == "PAD" for pid in self._potions()) or len(self._potions()) < A.P_MAX

    # ---------------------------------------------------------- command mapping
    def command_for_semantic(self, action: tuple) -> str:
        verb = action[0]
        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            if kind[0] == actions.KIND_END:
                return "END"
            if kind[0] == actions.KIND_PLAY:
                _, hand_idx, target = kind
                return f"PLAY {hand_idx + 1}" if target is None else f"PLAY {hand_idx + 1} {target}"
            if kind[0] == actions.KIND_SELECT:
                return f"CHOOSE {kind[1]}"
        if verb == ga.NEOW:
            return f"CHOOSE {action[1]}"
        if verb == ga.MAP:
            idx = self._map_command_index(action[1])
            return f"CHOOSE {idx}"
        if verb == ga.CLAIM:
            idx = self._claim_command_index(action[1], action[2])
            return f"CHOOSE {idx}"
        if verb == ga.REWARD_CARD:
            if self.screen_type() == "COMBAT_REWARD":
                idx = _choice_index(self.choices(), {"card"}) or 0
                self._combat_reward_card_open = True
                return f"CHOOSE {idx}"
            self._combat_reward_card_open = False
            self._skipped_combat_card_reward = False
            return f"CHOOSE {action[1]}"
        if verb == ga.SKIP_CARDS:
            if self.screen_type() == "CARD_REWARD" and self._combat_reward_card_open:
                self._combat_reward_card_open = False
                self._skipped_combat_card_reward = True
            return "SKIP"
        if verb == ga.SELECT:
            desc = self.select_descriptor()
            if desc and desc["candidates"][int(action[1])]["card"] is None:
                return "SKIP"
            index = desc["candidates"][int(action[1])]["native_index"] if desc else action[1]
            return f"CHOOSE {index}"
        if verb == ga.MULTI_SELECT:
            return "CONFIRM"
        if verb == ga.CAMPFIRE:
            idx = self._campfire_command_index(action[1])
            return f"CHOOSE {idx}"
        if verb == ga.EVENT:
            return f"CHOOSE {action[1]}"
        if verb == ga.CHEST:
            return "CHOOSE 0" if action[1] else "PROCEED"
        if verb == ga.BOSS_RELIC:
            return f"CHOOSE {action[1]}"
        if verb == ga.PROCEED:
            if self.screen_type() == "GRID" and "CONFIRM" in self.commands():
                return "CONFIRM"
            if self.screen_type() == "COMBAT_REWARD":
                self._combat_reward_card_open = False
                self._skipped_combat_card_reward = False
            return "PROCEED"
        if verb == ga.SHOP_PROCEED:
            if self.screen_type() == "SHOP_SCREEN":
                self._leaving_shop = True
                return "LEAVE"
            return "PROCEED"
        if verb == ga.SHOP_BUY:
            idx = self._shop_command_index(action)
            return f"CHOOSE {idx}"
        if verb == ga.SHOP_REMOVE:
            idx = self._shop_remove_index()
            return f"CHOOSE {idx}"
        if verb == ga.USE_POTION:
            slot, target = action[1], action[2]
            return f"POTION use {slot}" if target is None else f"POTION use {slot} {target}"
        raise ValueError(f"cannot map semantic action to real command: {action!r}")

    def command_for_multi_action(self, action_id: int) -> str:
        if int(action_id) == A._CONFIRM:
            return "CONFIRM"
        seg, off = A.segment_of(action_id)
        if seg != "PICK_CARD":
            raise ValueError(f"expected PICK_CARD/CONFIRM on multi-select, got {action_id}")
        self.multi_selected.add(off)
        return f"CHOOSE {off}"

    def _map_command_index(self, x: int) -> int:
        for idx, (choice_x, _room) in enumerate(self._map_nodes()):
            if choice_x == x:
                return idx
        raise ValueError(f"map x={x} is not in current choices {self.choices()!r}")

    def _claim_command_index(self, kind: str, occurrence: int) -> int:
        aliases = {
            ga.CLAIM_GOLD: {"gold", "stolen_gold"},
            ga.CLAIM_RELIC: {"relic"},
            ga.CLAIM_POTION: {"potion"},
        }[kind]
        idx = _choice_index(self.choices(), aliases, occurrence)
        if idx is None:
            raise ValueError(f"claim {kind} #{occurrence} not found in {self.choices()!r}")
        return idx

    def _campfire_command_index(self, option: str) -> int:
        for idx, label in enumerate(self.choices()):
            if _campfire_option(label) == option:
                return idx
        raise ValueError(f"campfire option {option!r} not found in {self.choices()!r}")

    def _shop_choice_actions(self) -> list[tuple[tuple, int]]:
        cl = self.choices()
        ss = _screen_state(self.gs)
        gold = int(self.gs.get("gold", 0) or 0)

        affordable: list[tuple[str, int]] = []
        for kind, key in (
            (ga.SHOP_CARD, "cards"),
            (ga.SHOP_RELIC, "relics"),
            (ga.SHOP_POTION, "potions"),
        ):
            for i, item in enumerate(ss.get(key) or []):
                price = _price(item)
                if 0 <= price <= gold:
                    affordable.append((kind, i))

        out: list[tuple[tuple, int]] = []
        non_purge_pos = 0
        for choice_idx, label in enumerate(cl):
            norm = _norm_label(label)
            if norm in {"purge", "remove"}:
                out.append((ga.shop_remove(), choice_idx))
                continue
            if non_purge_pos < len(affordable):
                kind, shelf_idx = affordable[non_purge_pos]
                out.append((ga.shop_buy(kind, shelf_idx), choice_idx))
                non_purge_pos += 1
        if not out and affordable:
            out = [(ga.shop_buy(kind, shelf_idx), i) for i, (kind, shelf_idx) in enumerate(affordable)]
        return out

    def _shop_command_index(self, action: tuple) -> int:
        for candidate, idx in self._shop_choice_actions():
            if candidate == action:
                return idx
        raise ValueError(f"shop action {action!r} not found in choices {self.choices()!r}")

    def _shop_remove_index(self) -> int:
        for candidate, idx in self._shop_choice_actions():
            if candidate[0] == ga.SHOP_REMOVE:
                return idx
        idx = _choice_index(self.choices(), {"purge", "remove"})
        if idx is None:
            raise ValueError(f"shop remove not found in choices {self.choices()!r}")
        return idx

    # --------------------------------------------------------------- debugging
    def describe_action_id(self, action_id: int, pending: tuple | None = None) -> str:
        if self.combat_action_contract in C.NEURAL_CONTRACTS and self.phase() == "combat":
            return str(self.joint_semantic_actions().get(int(action_id), ("invalid", action_id)))
        desc = self.select_descriptor()
        if A.is_multi_select(desc):
            if int(action_id) == A._CONFIRM:
                return f"confirm multi-select {sorted(self.multi_selected)}"
            seg, off = A.segment_of(action_id)
            return f"{seg}[{off}]"
        try:
            semantic, pend = A.resolve(
                int(action_id), self.legal_semantic_actions(), pending, self.phase()
            )
        except Exception:
            try:
                seg, off = A.segment_of(action_id)
                return f"{seg}[{off}]"
            except Exception:
                return str(action_id)
        if semantic is None:
            return f"{pend[0]} #{pend[1]} ..."
        return ga.describe(semantic, self.obs_canonical(pending=pending))

    def fallback_command(self) -> str:
        cmds = self.commands()
        if "PROCEED" in cmds:
            return "PROCEED"
        if "LEAVE" in cmds:
            self._leaving_shop = True
            return "LEAVE"
        if "SKIP" in cmds:
            return "SKIP"
        if "CONFIRM" in cmds:
            return "CONFIRM"
        if "CHOOSE" in cmds and self.choices():
            return "CHOOSE 0"
        if "END" in cmds:
            return "END"
        if "STATE" in cmds:
            return "STATE"
        if "WAIT" in cmds:
            return "WAIT 1"
        return "STATE"

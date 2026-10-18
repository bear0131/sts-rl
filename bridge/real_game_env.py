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
from collections import Counter
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
from bridge import event_contract
from bridge.schemas.entity_ids import POTION_ALIASES

LOG = logging.getLogger(__name__)


def _neow_option_from_text(text: str) -> dict[str, int]:
    """Decode stock English/zh-CN Neow labels into the native enum contract.

    CommunicationMod exports text, not the simulator's bonus/drawback IDs.
    Unknown translations must fail rather than become a valid all-zero offer.
    """
    s = re.sub(r'\s+', '', re.sub(r'#[a-zA-Z]', '', text)).lower()
    if any(x in s for x in ('loseyourstartingrelic', 'loseyourstarterrelic', '失去你的初始遗物')):
        drawback = 6
    elif 'loseallgold' in s or '失去所有金币' in s:
        drawback = 3
    elif 'obtainacurse' in s or '得到一张诅咒' in s:
        drawback = 4
    elif re.search(r'lose\d+maxhp|失去\d+最大生命', s):
        drawback = 2
    elif re.search(r'take\d+damage|受到\d+点伤害', s):
        drawback = 5
    else:
        drawback = 1
    patterns = (
        (18, r'bossrelic|boss遗物'),
        (11, r'rarecolorlesscard|稀有无色牌'),
        (5, r'colorlesscard|无色牌'),
        (12, r'remove(?:2|two)cards|去除2张牌'),
        (16, r'transform(?:2|two)cards|变化2张牌'),
        (14, r'chooseararecard|选择并获得一张稀有牌'),
        (1, r'randomrarecard|随机获得一张稀有牌'),
        (13, r'rarerelic|稀有遗物'),
        (7, r'commonrelic|普通遗物'),
        (6, r'(?:3|three)(?:random)?potions|3瓶药水'),
        (9, r'nextthreecombats|neow.s lament|接下来3场战斗'),
        (15, r'(?:obtain|gain)250gold|获得250金币'),
        (10, r'(?:obtain|gain)100gold|获得100金币'),
        (2, r'removeacard|从牌组中移除一张牌'),
        (3, r'upgradeacard|升级一张牌'),
        (4, r'transformacard|变化一张牌'),
        (0, r'chooseacard|选择并获得一张牌'),
    )
    bonus = next((value for value, pattern in patterns if re.search(pattern, s)), None)
    if bonus is None and re.search(r'maxhp\+|最大生命(?:值)?\+', s):
        bonus = 8 if drawback == 1 else 17
    if bonus is None or (bonus == 18) != (drawback == 6):
        raise ValueError(f'unsupported Neow option text: {text!r}')
    return {'bonus_id': bonus, 'drawback_id': drawback}


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
    if pid in {"Potion Slot", "EMPTY", "None", "", "PAD"}:
        return "PAD"
    canonical = POTION_ALIASES.get(pid, pid)
    if canonical not in V.POTION_VOCAB_INDEX:
        raise ValueError(f"unknown real potion ID: {pid}")
    return canonical


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


def _reward_records(ss: dict[str, Any]) -> list[dict[str, Any]]:
    rewards = ss.get("rewards") or []
    if not isinstance(rewards, list):
        return []
    return [reward for reward in rewards if isinstance(reward, dict)]


def _reward_type(reward: dict[str, Any]) -> str:
    return str(reward.get("reward_type") or reward.get("type") or "").upper()


def _screen_reward_gold(ss: dict[str, Any]) -> list[int]:
    gold = []
    for reward in _reward_records(ss):
        if _reward_type(reward) not in {"GOLD", "STOLEN_GOLD"}:
            continue
        amount = _int_or_none(reward.get("gold"))
        if amount is not None:
            gold.append(amount)
    return gold


def _screen_relics(ss: dict[str, Any]) -> list[str]:
    relics = ss.get("relics") or ss.get("boss_relics") or []
    if not isinstance(relics, list):
        return []
    if not relics:
        relics = [
            reward["relic"]
            for reward in _reward_records(ss)
            if _reward_type(reward) == "RELIC"
            and isinstance(reward.get("relic"), dict)
        ]
    return [_relic_id(r) for r in relics]


def _screen_potions(ss: dict[str, Any]) -> list[tuple[str, int]]:
    potions = ss.get("potions") or []
    if not isinstance(potions, list):
        return []
    if not potions:
        potions = [
            reward["potion"]
            for reward in _reward_records(ss)
            if _reward_type(reward) == "POTION"
            and isinstance(reward.get("potion"), dict)
        ]
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
        self._combat_reward_outer_choices: tuple[Any, ...] = ()
        self._combat_reward_outer_commands: frozenset[str] = frozenset()
        self._combat_reward_outer_options: dict[str, list[Any]] = {}
        self._deferred_combat_reward_action: tuple | None = None
        # CommunicationMod publishes the first post-GRID frame before some
        # vanilla relic callbacks have consumed ``selectedCards``.  Sending the
        # next event/map command in that frame can clear the selection before
        # the callback runs.  Astrolabe then removes three cards without ever
        # adding their transformed replacements.  Hold one real render interval
        # after every overworld grid closes, and keep the exact Astrolabe deck
        # cardinality invariant until the game's own callback has settled.
        self._post_overworld_grid_wait = False
        self._astrolabe_expected_deck_size: int | None = None
        self._last_relic_ids: set[str] = set()
        self._seen_game_frame = False
        self._combat_identity = None
        self._combat_initialized = False

    def update(self, frame: dict[str, Any]) -> None:
        previous_overworld_grid = (
            self.screen_type() == "GRID" and not bool(self.gs.get("combat_state"))
        )
        previous_relic_ids = self._last_relic_ids
        self.frame = frame
        self.gs = frame.get("game_state") or {}
        if "error" not in frame:
            combat = self.gs.get("combat_state")
            identity = ((self.gs.get("seed"), self.gs.get("floor"))
                        if combat else None)
            if identity != self._combat_identity or not combat:
                self._combat_initialized = False
            self._combat_identity = identity
            if combat and (combat.get("hand")
                           or combat.get("player", {}).get("energy", 0) > 0
                           or combat.get("turn", 0) > 1):
                self._combat_initialized = True
        current_relic_ids = {
            str(relic.get("id"))
            for relic in self.gs.get("relics") or []
            if isinstance(relic, dict) and relic.get("id")
        }
        current_overworld_grid = (
            self.screen_type() == "GRID" and not bool(self.gs.get("combat_state"))
        )
        if current_overworld_grid and "Astrolabe" in current_relic_ids:
            astrolabe_just_appeared = (
                self._seen_game_frame and "Astrolabe" not in previous_relic_ids
            )
            attaching_to_opening_astrolabe = (
                not self._seen_game_frame
                and _int_or_none(self.gs.get("floor")) == 0
            )
            if astrolabe_just_appeared or attaching_to_opening_astrolabe:
                self._astrolabe_expected_deck_size = len(self.gs.get("deck") or [])
        if previous_overworld_grid and not current_overworld_grid:
            self._post_overworld_grid_wait = True
        self._last_relic_ids = current_relic_ids
        self._seen_game_frame = self._seen_game_frame or bool(frame.get("in_game"))
        self._reconcile_multi()
        if self.screen_type() == "SHOP_SCREEN":
            self._leaving_shop = False
        if self.screen_type() == "COMBAT_REWARD":
            self._combat_reward_outer_choices = tuple(self.choices())
            self._combat_reward_outer_commands = frozenset(self.commands())
            self._combat_reward_outer_options = self._current_outer_reward_options()
        elif self.screen_type() not in {"COMBAT_REWARD", "CARD_REWARD"}:
            self._combat_reward_card_open = False
            self._skipped_combat_card_reward = False
            self._combat_reward_outer_choices = ()
            self._combat_reward_outer_commands = frozenset()
            self._combat_reward_outer_options = {}
            self._deferred_combat_reward_action = None

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
        if self.screen_type() == "NONE":
            if self.gs.get("current_action"):
                return False
            combat = self.gs["combat_state"]
            # The mod can advertise END before opening draw/energy actions.
            # An empty, unenergized opening is not a policy decision point.
            if (not self._combat_initialized
                    and combat.get("turn") == 1 and not combat.get("hand")
                    and combat.get("player", {}).get("energy") == 0
                    and combat.get("draw_pile")):
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
        # The in-combat map is an inspect-only overlay.  CommunicationMod still
        # publishes the underlying combat state and room phase, but replaces
        # PLAY/END with CHOOSE/RETURN while the overlay is open.  Sending that
        # frame into native combat search makes the real and simulated legal
        # masks disagree; close the overlay before making another decision.
        if (self.gs.get("combat_state")
                and str(self.gs.get("room_phase") or "").upper() == "COMBAT"
                and self.screen_type() == "MAP"):
            if "RETURN" in self.commands():
                return "RETURN", "close_combat_map_overlay"
            return self._refresh_command(), "await_combat_map_overlay_close"
        if self.gs.get("combat_state") and not self._combat_ready_for_policy():
            return self._refresh_command(), "combat_not_waiting"
        if self._joint_multi_plan is not None:
            command = self._continue_joint_multi()
            if command is not None:
                return command, "complete_multi_continuation"

        st = self.screen_type()
        ss = _screen_state(self.gs)
        cl = self.choices()

        if self._deferred_combat_reward_action is not None:
            if st == "COMBAT_REWARD":
                action = self._deferred_combat_reward_action
                self._deferred_combat_reward_action = None
                return self.command_for_semantic(action), "complete_deferred_reward_action"
            return self._refresh_command(), "await_combat_reward_return"

        if self._post_overworld_grid_wait:
            self._post_overworld_grid_wait = False
            return (
                "WAIT 1" if "WAIT" in self.commands() else self._refresh_command(),
                "settle_overworld_grid",
            )

        if self._astrolabe_expected_deck_size is not None and st != "GRID":
            current_deck_size = len(self.gs.get("deck") or [])
            if current_deck_size != self._astrolabe_expected_deck_size:
                return (
                    "WAIT 1" if "WAIT" in self.commands() else self._refresh_command(),
                    "await_astrolabe_transform",
                )
            self._astrolabe_expected_deck_size = None

        # CommunicationMod moves a chosen combat card from ``hand`` to
        # ``selected`` and then advertises only CONFIRM.  That is an
        # acknowledgement step, not a fresh policy decision over the shortened
        # hand.  Confirm it before constructing another action mask.
        if (st == "HAND_SELECT" and "CONFIRM" in self.commands()
                and "CHOOSE" not in self.commands()):
            return "CONFIRM", "complete_hand_selection"

        # Simulator overworld grids settle immediately after the required last
        # pick.  The real UI exposes an extra CONFIRM transport step; execute it
        # automatically so the upper policy sees the same sequence of states.
        if (st == "GRID" and "CONFIRM" in self.commands()
                and self._grid_required_selection_complete()):
            return "CONFIRM", "complete_grid_selection"

        if st == "EVENT" and _normalize_event(ss.get("event_id")) == "NEOW" and len(cl) == 1:
            return "CHOOSE 0", "neow_intro"

        if st == "EVENT":
            dialogue = event_contract.transport_only(self.gs)
            if dialogue is not None:
                return f"CHOOSE {dialogue}", "event_dialogue_transport"

        if auto_enter_shop and st == "SHOP_ROOM":
            if self._leaving_shop:
                self._leaving_shop = False
                return "PROCEED", "leave_shop_room"
            if "CHOOSE" in self.commands() and cl:
                return "CHOOSE 0", "enter_shop_screen"

        # The simulator exposes the actual card candidates before the upper
        # policy chooses any reward action.  CommunicationMod instead exposes a
        # placeholder on COMBAT_REWARD and reveals candidates only after CHOOSE.
        # Open that reversible sub-screen before every policy decision.  While
        # it is open, the adapter merges these cards with the cached outer
        # rewards and defers outer commands through a transport-only SKIP.
        if (st == "COMBAT_REWARD" and not self._skipped_combat_card_reward
                and "CHOOSE" in self.commands()):
            normalized = [_norm_label(choice) for choice in cl]
            card_indices = [i for i, label in enumerate(normalized) if label == "card"]
            if len(card_indices) == 1:
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
        screen_state = _screen_state(self.gs)
        selected = (
            screen_state.get("selected_cards") or screen_state.get("selectedCards")
            or screen_state.get("selected") or []
        )
        count = int(screen_state.get(
            "have_selected", len(selected) if isinstance(selected, list) else 0
        ))
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
                real_monsters = canon.get("monsters") or []
                if len(real_monsters) > actions.MAX_MONSTERS:
                    canon["monsters"] = [
                        real_monsters[real_index]
                        for real_index in self.combat_target_layout()
                    ]
            except Exception:
                LOG.exception("normalize_real failed")
                raise
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
            has_card = not self._skipped_combat_card_reward
            outer = (
                self._combat_reward_outer_options
                if st == "CARD_REWARD" and self._combat_reward_card_open
                else self._current_outer_reward_options()
            )
            opts.update({
                "gold": list(outer.get("gold", [])),
                "relics": list(outer.get("relics", [])),
                "potions": list(outer.get("potions", [])),
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
                mapping = event_contract.options(self.gs)
                opts["event_options"] = sorted(mapping)
                details = _structured_event_choices(ss, len(cl))
                if details:
                    reverse = {ui: native for native, (ui, _) in mapping.items()}
                    opts["event_choices"] = [dict(d, index=reverse[d["index"]]) for d in details]
        elif phase == "card_select":
            cards = _screen_cards(ss)
            select = self._select_block(cards=cards, overworld=True)
            # The simulator removes already-selected grid cards from its public
            # candidate list.  BaseMod keeps every card in ``cards`` and reports
            # progress separately in ``selected_cards``.  Encode the real frame
            # in the simulator's public shape while preserving each remaining
            # card's original UI index in the descriptor below.
            opts["cards"] = [
                (candidate["card"]["id"], candidate["card"]["up"])
                for candidate in select["candidates"]
                if candidate["card"] is not None
            ]
            opts["select"] = select
        elif phase == "map":
            opts["nodes"] = self._map_nodes()
            if self.global_map_view:
                opts["full_map"] = self._full_map_nodes(ss)
        elif phase == "shop":
            opts.update(self._shop_options(ss))
        return opts

    def _current_outer_reward_options(self) -> dict[str, list[Any]]:
        ss = _screen_state(self.gs)
        labels = [_norm_label(choice) for choice in self.choices()]
        expected = {
            "gold": sum(label in {"gold", "stolen_gold"} for label in labels),
            "relics": labels.count("relic"),
            "potions": labels.count("potion"),
        }
        actual: dict[str, list[Any]] = {
            "gold": _screen_reward_gold(ss),
            "relics": _screen_relics(ss),
            "potions": [pid for pid, _price_ in _screen_potions(ss)],
        }
        for kind, count in expected.items():
            if len(actual[kind]) != count:
                raise ValueError(
                    "CommunicationMod reward details do not match choices: "
                    f"kind={kind} expected={count} actual={actual[kind]!r} "
                    f"choices={self.choices()!r}"
                )
        return actual

    def _neow_options(self, ss: dict[str, Any]) -> list[dict[str, int]]:
        raw = ss.get("options") or ss.get("rewards") or []
        out = []
        if isinstance(raw, list):
            for item in raw[:A.NEOW_W]:
                if isinstance(item, dict):
                    if "bonus_id" in item and "drawback_id" in item:
                        out.append({"bonus_id": int(item["bonus_id"]),
                                    "drawback_id": int(item["drawback_id"])})
                    else:
                        out.append(_neow_option_from_text(str(item.get("text") or item.get("label") or "")))
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
        if self.screen_type() == "GRID":
            return self._select_block(
                cards=_screen_cards(_screen_state(self.gs)),
                overworld=not bool(self.gs.get("combat_state")),
            )
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
        selected_raw = (
            ss.get("selected_cards") or ss.get("selectedCards")
            or ss.get("selected") or []
        )
        if not isinstance(selected_raw, list):
            selected_raw = []
        selected_cards = [
            _real_card_tuple(card) for card in selected_raw if isinstance(card, dict)
        ]
        explicit_have_selected = _int_or_none(ss.get("have_selected"))
        have_selected = (
            explicit_have_selected
            if explicit_have_selected is not None
            else len(selected_cards)
        )
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
        # BaseMod uses 99 as the unbounded hand-selection sentinel (notably
        # Gambler's Brew).  Normalize that transport value to the actual
        # selectable hand so it matches the simulator's public contract.
        if multi and not overworld and pick_max is not None and pick_max >= 99:
            can_pick_any = True
            pick_count = len(cards)
            pick_max = len(cards)
        if pick_max is None:
            pick_max = len(cards) if multi and can_pick_any else pick_count
        raw_cards = ss.get("cards") or (self.gs.get("combat_state") or {}).get("hand") or []
        selected_uuids = {
            str(card["uuid"])
            for card in selected_raw
            if isinstance(card, dict) and card.get("uuid")
        }
        selected_without_uuid = Counter(
            _real_card_tuple(card)
            for card in selected_raw
            if isinstance(card, dict) and not card.get("uuid")
        )
        candidates = []
        for i, (cid, up) in enumerate(cards):
            raw_card = raw_cards[i] if i < len(raw_cards) and isinstance(raw_cards[i], dict) else {}
            if overworld:
                raw_uuid = str(raw_card.get("uuid") or "")
                selected = bool(raw_uuid and raw_uuid in selected_uuids)
                if not raw_uuid and selected_without_uuid[(cid, up)] > 0:
                    selected_without_uuid[(cid, up)] -= 1
                    selected = True
                if selected:
                    continue
            card = {"id": cid, "up": up}
            if "cost" in raw_card:
                card["cost"] = raw_card["cost"]
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
            "selected": selected_cards if overworld else [],
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
        # Grid actions carry their source on game_state, not screen_state.
        # Identify draw-pile searches before the generic GRID/PAD fallback so
        # candidates use native CardId order while retaining UI handles.
        action_context = {
            "SkillFromDeckToHandAction": "SECRET_TECHNIQUE",
            "AttackFromDeckToHandAction": "SECRET_WEAPON",
            "SeekAction": "SEEK",
        }.get(str(self.gs.get("current_action") or ""))
        if action_context is not None and self.gs.get("combat_state"):
            return action_context
        if self.screen_type() == "GRID":
            return "REMOVE" if any(_norm_label(c) == "purge" for c in self.choices()) else "PAD"
        if self.screen_type() == "HAND_SELECT":
            names = {cid for cid, _up in cards}
            if "Calculated Gamble" in names:
                return "GAMBLE"
        current_action = str(self.gs.get("current_action") or "")
        generated_context = {
            "CodexAction": "CODEX",
            "DiscoveryAction": "DISCOVERY",
        }.get(current_action)
        if generated_context is not None:
            return generated_context
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
            if "CHOOSE" not in self.commands():
                return []
            n = len(self._combat_hand_cards())
            return [ga.select(i) for i in range(n)]

        acts: list[tuple] = []
        cs = self.gs.get("combat_state") or {}
        if "END" in self.commands():
            acts.append(ga.combat(actions.END_TURN))
        monsters = cs.get("monsters") or []
        targets = [
            policy_index
            for policy_index, real_index in enumerate(self.combat_target_layout())
            for m in (monsters[real_index],)
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
        return [ga.event(i) for i in sorted(event_contract.options(self.gs))]

    def _legal_card_select_actions(self) -> list[tuple]:
        cmds = self.commands()
        if "CONFIRM" in cmds and "CHOOSE" not in cmds:
            return [ga.proceed()]
        if "CHOOSE" not in cmds:
            return [ga.proceed()] if "CONFIRM" in cmds else []
        desc = self.select_descriptor() or {}
        acts = [ga.select(i) for i, _ in enumerate((desc.get("candidates") or [])[:A.G_MAX])]
        if "CONFIRM" in cmds and self._grid_can_confirm_without_more_picks():
            acts.append(ga.proceed())
        return acts

    def _grid_can_confirm_without_more_picks(self) -> bool:
        ss = _screen_state(self.gs)
        if bool(ss.get("any_number") or ss.get("can_pick_any") or ss.get("canPickAny")):
            return True
        selected = (
            ss.get("selected_cards") or ss.get("selectedCards")
            or ss.get("selected") or []
        )
        selected_count = len(selected) if isinstance(selected, list) else 0
        required = (
            _int_or_none(ss.get("num_cards"))
            or _int_or_none(ss.get("pick_count"))
            or _int_or_none(ss.get("max_cards"))
            or 1
        )
        return selected_count >= required

    def _grid_required_selection_complete(self) -> bool:
        ss = _screen_state(self.gs)
        selected = (
            ss.get("selected_cards") or ss.get("selectedCards")
            or ss.get("selected") or []
        )
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
            acts = []
            if self._combat_reward_card_open:
                acts.extend(self._outer_reward_claim_actions())
            acts.extend(ga.reward_card(i) for i in range(min(n, A.REWARD_CARD)))
            if "SKIP" in self.commands():
                acts.append(ga.skip_cards())
            if (self._combat_reward_card_open
                    and "PROCEED" in self._combat_reward_outer_commands):
                acts.append(ga.proceed())
            return acts

        acts = self._outer_reward_claim_actions()
        if not self._skipped_combat_card_reward and any(
                _norm_label(label) == "card" for label in cl):
            acts.append(ga.reward_card(0))
        if "PROCEED" in self.commands():
            acts.append(ga.proceed())
        return acts

    def _outer_reward_claim_actions(self) -> list[tuple]:
        labels = (
            self._combat_reward_outer_choices
            if self.screen_type() == "CARD_REWARD" and self._combat_reward_card_open
            else tuple(self.choices())
        )
        acts = []
        counts = {ga.CLAIM_GOLD: 0, ga.CLAIM_RELIC: 0, ga.CLAIM_POTION: 0}
        for label in labels:
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
        commands = self.commands()
        if (not combat and self.screen_type() == "CARD_REWARD"
                and self._combat_reward_card_open):
            commands |= set(self._combat_reward_outer_commands)
        if "POTION" not in commands:
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
        monsters = cs.get("monsters") or []
        out: list[int] = []
        for policy_index, real_index in enumerate(self.combat_target_layout()):
            m = monsters[real_index]
            if isinstance(m, dict) and not m.get("is_gone") and int(m.get("current_hp", 0) or 0) > 0:
                out.append(policy_index)
        return out

    def combat_target_layout(self) -> tuple[int, ...]:
        """Map bounded policy target slots to CommunicationMod monster indices.

        CommunicationMod normally preserves the encounter's positional monster
        slots, which are also the policy's five target slots.  Gremlin Leader,
        however, retains every dead summon in its Java list and inserts later
        summons before the leader.  That transport list can therefore grow past
        five even though at most five monsters are alive.  Once it overflows,
        compact the authoritative live entities into the bounded policy slots;
        commands translate those slots back to the current Java indices.
        """
        cs = self.gs.get("combat_state") or {}
        monsters = cs.get("monsters") or []
        if not isinstance(monsters, list):
            return ()
        if len(monsters) <= actions.MAX_MONSTERS:
            return tuple(range(len(monsters)))
        live = tuple(
            index
            for index, monster in enumerate(monsters)
            if isinstance(monster, dict)
            and not monster.get("is_gone")
            and int(monster.get("current_hp", 0) or 0) > 0
        )
        if len(live) > actions.MAX_MONSTERS:
            raise ValueError(
                "real combat exposes more live monsters than the policy target capacity: "
                f"{len(live)} > {actions.MAX_MONSTERS}"
            )
        return live

    def real_target_index(self, policy_target: int) -> int:
        layout = self.combat_target_layout()
        target = int(policy_target)
        if not 0 <= target < len(layout):
            raise ValueError(
                f"policy target slot {target} is absent from real layout {layout!r}"
            )
        return int(layout[target])

    def policy_target_index(self, real_target: int) -> int:
        layout = self.combat_target_layout()
        target = int(real_target)
        try:
            return layout.index(target)
        except ValueError as exc:
            raise ValueError(
                f"real target index {target} is absent from policy layout {layout!r}"
            ) from exc

    def _has_empty_potion_slot(self) -> bool:
        # CommunicationMod includes every actual slot, including empty slots.
        # The policy reserves five indices; it does not grant five game slots.
        return any(pid == "PAD" for pid in self._potions())

    # ---------------------------------------------------------- command mapping
    def command_for_semantic(self, action: tuple) -> str:
        verb = action[0]
        if (self.screen_type() == "CARD_REWARD"
                and self._combat_reward_card_open
                and verb in {ga.CLAIM, ga.PROCEED, ga.USE_POTION}):
            if self._deferred_combat_reward_action is not None:
                raise RuntimeError("combat reward action is already deferred")
            self._deferred_combat_reward_action = action
            self._combat_reward_card_open = False
            return "SKIP"
        if verb == ga.COMBAT:
            kind = actions.decode(action[1])
            if kind[0] == actions.KIND_END:
                return "END"
            if kind[0] == actions.KIND_PLAY:
                _, hand_idx, target = kind
                return (
                    f"PLAY {hand_idx + 1}"
                    if target is None
                    else f"PLAY {hand_idx + 1} {self.real_target_index(target)}"
                )
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
            return f"CHOOSE {event_contract.options(self.gs)[action[1]][0]}"
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
            return (
                f"POTION use {slot}"
                if target is None
                else f"POTION use {slot} {self.real_target_index(target)}"
            )
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
        can_buy_potion = (
            "Sozu" not in self._relics()
            and self._has_empty_potion_slot()
        )

        affordable = []
        for kind, key in (
            (ga.SHOP_CARD, "cards"),
            (ga.SHOP_RELIC, "relics"),
            (ga.SHOP_POTION, "potions"),
        ):
            if kind == ga.SHOP_POTION and not can_buy_potion:
                continue
            for i, item in enumerate(ss.get(key) or []):
                price = _price(item)
                if 0 <= price <= gold:
                    affordable.append((kind, i, {_norm_label(item.get("name", "")),
                                                  _norm_label(item.get("id", ""))} - {""}))

        out: list[tuple[tuple, int]] = []
        used = set()
        for choice_idx, label in enumerate(cl):
            norm = _norm_label(label)
            if norm in {"purge", "remove"}:
                out.append((ga.shop_remove(), choice_idx))
                continue
            matches = [(kind, i) for kind, i, names in affordable
                       if norm in names and (kind, i) not in used]
            if len(matches) > 1:
                # Stock shops may offer the same potion twice at different
                # prices. CommunicationMod preserves shelf order for these
                # indistinguishable labels; never use this for different IDs.
                keys = {ga.SHOP_CARD: "cards", ga.SHOP_RELIC: "relics", ga.SHOP_POTION: "potions"}
                identities = {(kind, ss[keys[kind]][i].get("id")) for kind, i in matches}
                if len(identities) == 1:
                    matches = matches[:1]
            if len(matches) != 1:
                raise ValueError(f"shop choice identity mismatch/ambiguity: {label!r}")
            kind, shelf_idx = matches[0]
            used.add((kind, shelf_idx))
            out.append((ga.shop_buy(kind, shelf_idx), choice_idx))
        if len(used) != len(affordable):
            raise ValueError("shop affordable inventory/choice coverage mismatch")
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

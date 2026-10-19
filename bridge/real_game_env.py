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
import time
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
from bridge.schemas.card_state import real_card_state, real_master_deck_state
from bridge.schemas.entity_ids import (
    BOSS_ALIASES,
    BOSS_CANONICAL_IDS,
    COMBAT_SELECTION_SOURCES,
    DYNAMIC_COMBAT_SELECTION_SOURCES,
    DYNAMIC_EVENT_OPTION_KINDS,
    DYNAMIC_EVENT_OPTION_KIND_ALIASES,
    EMPTY_REAL_POTION_IDS,
    EVENT_ALIASES,
    OVERWORLD_SELECTION_EVENTS,
    OVERWORLD_SELECTION_FLAGS,
    OVERWORLD_SELECTION_RELICS,
    POTION_ALIASES,
    REAL_REWARD_TYPE_IDS,
    REAL_COMMAND_IDS,
    REAL_CHARACTER_IDS,
    REAL_ROOM_PHASES,
    REAL_ACTION_PHASES,
    REAL_ANONYMOUS_ACTION_CONTRACTS,
    REWARD_CHOICE_TYPES,
    ROOM_ALIASES,
    RUN_KEY_FIELDS,
    SCREEN_PHASES,
    SIMULATOR_PARITY_EXPORT_VERSION,
    native_relic_counter,
    native_combat_relic_counter,
    neow_option_from_types,
    NEOW_SCREEN_CONTRACTS,
    UNSUPPORTED_REAL_RELIC_IDS,
    UNSUPPORTED_SELECTION_SOURCES,
)

LOG = logging.getLogger(__name__)


_SCREEN_PHASE = SCREEN_PHASES
_ROOM_LABELS = ROOM_ALIASES
_BOSS_ALIASES = BOSS_ALIASES
_EVENT_ALIASES = EVENT_ALIASES

_CAMPFIRE_ALIASES = {
    "rest": "rest",
    "smith": "smith",
    "recall": "recall",
    "lift": "lift",
    "toke": "toke",
    "dig": "dig",
}

_MAP_MAX_Y = 14


def _is_terminal_rest_node(node: dict[str, Any]) -> bool:
    """Java draws the mandatory pre-boss rest at (3, 16).

    The simulator/global-map contract contains only the 15 generated rows
    0..14; that fixed rest is represented by the current local map choices, not
    by a fabricated/clamped row in the global grid.
    """
    return _node_xy(node) == (3, 16) and _node_room(node) == "REST"


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
    raw = frame.get("available_commands")
    if raw is None:
        raise ValueError("frame is missing its explicit available_commands list")
    if not isinstance(raw, list):
        raise ValueError(f"available_commands must be a list, got {raw!r}")
    commands = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"available_commands[{index}] must be a nonempty string, got {value!r}"
            )
        command = value.strip().upper()
        if command not in REAL_COMMAND_IDS:
            raise ValueError(f"unknown CommunicationMod command keyword: {value!r}")
        commands.add(command)
    return commands


def _choice_list(gs: dict[str, Any]) -> list[Any]:
    if not gs:  # No game state on a genuine main-menu/error response.
        return []
    raw = gs.get("choice_list")
    if not isinstance(raw, list):
        raise ValueError(f"choice_list must be a list, got {raw!r}")
    return list(raw)


def _screen_state(gs: dict[str, Any]) -> dict[str, Any]:
    if not gs:  # This is not an in-game missing-payload fallback.
        return {}
    ss = gs.get("screen_state")
    if not isinstance(ss, dict):
        raise ValueError(f"screen_state must be an object, got {ss!r}")
    return ss


def validate_transport_frame(frame: dict[str, Any]) -> None:
    """Check presence before ACKs/menu handling can turn missing data into actions.

    Error replies deliberately have no in_game/game_state. Normal in-game
    frames have explicit collections and an explicitly nullable current action.
    Detailed identity/value checks remain in their existing phase codecs.
    """
    if not isinstance(frame, dict):
        raise ValueError("CommunicationMod frame must be an object")
    _required_bool(frame, "ready_for_command", context="frame")
    if "error" in frame:
        if not isinstance(frame["error"], str) or not frame["error"]:
            raise ValueError("error response requires a nonempty message")
        if "in_game" in frame or "game_state" in frame:
            raise ValueError("error response must not impersonate a normal game/menu frame")
        if "available_commands" in frame:
            _commands(frame)
        return
    _commands(frame)
    if not _required_bool(frame, "in_game", context="frame"):
        if "game_state" in frame:
            raise ValueError("main-menu response must not carry a game_state")
        return
    gs = _required(frame, "game_state", context="in-game frame")
    if not isinstance(gs, dict) or not gs:
        raise ValueError("in-game frame requires a nonempty game_state object")
    if gs.get("simulator_parity_export") != SIMULATOR_PARITY_EXPORT_VERSION:
        raise ValueError("in-game frame requires the current parity wire protocol")
    _required_bool(gs, "is_screen_up", context="game_state")
    action_phase = _required(gs, "action_phase", context="game_state")
    if action_phase not in REAL_ACTION_PHASES:
        raise ValueError(f"unknown real-game action_phase keyword: {action_phase!r}")
    _screen_state(gs)
    _choice_list(gs)
    for key in ("deck", "relics", "potions", "map"):
        _list_value(gs, key, context="game_state")
    if "current_action" not in gs:
        raise ValueError("game_state requires an explicit nullable current_action")
    action = gs["current_action"]
    if action is not None and (not isinstance(action, str) or not action.strip()):
        raise ValueError("current_action must be null or a nonempty action class name")
    anonymous = _required_bool(gs, "current_action_is_anonymous", context="game_state")
    contract = REAL_ANONYMOUS_ACTION_CONTRACTS.get(action)
    if anonymous:
        if contract is None:
            raise ValueError(f"unmapped anonymous action identity: {action!r}")
        if contract['transport'] != 'settle':
            raise ValueError(
                f"unsupported anonymous action {action!r}: {contract['native_audit']}"
            )
        # Registered stock wrappers never own a selection UI. Do not release
        # an ordinary policy decision or interpret an anonymous wrapper as a
        # generic HAND_SELECT/GRID source, even if the mod advertises CHOOSE.
        if gs.get('screen_type') in {'HAND_SELECT', 'GRID', 'CARD_REWARD'}:
            raise ValueError(f"anonymous wrapper cannot own this selection: {action!r}")
    elif contract is not None:
        raise ValueError(f"anonymous action class incorrectly marked as named: {action!r}")
    room_phase = _required(gs, "room_phase", context="game_state")
    if room_phase not in REAL_ROOM_PHASES:
        raise ValueError(f"unknown real-game room_phase keyword: {room_phase!r}")
    if room_phase == "COMBAT":
        combat = _required(gs, "combat_state", context="combat room")
        if not isinstance(combat, dict) or not combat:
            raise ValueError("combat room requires an explicit combat_state object")
    elif "combat_state" in gs:
        raise ValueError("noncombat room unexpectedly carries combat_state")


def _int_or_none(value: Any) -> int | None:
    # JSON integer fields arrive as actual Python ints.  Coercing strings,
    # floats, or booleans lets malformed transport data impersonate an index.
    return value if type(value) is int else None


def _required(mapping: dict[str, Any], *keys: str, context: str) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is None:
                continue
            return value
    raise ValueError(f"{context} is missing required field {keys!r}")


def _required_bool(mapping: dict[str, Any], key: str, *, context: str) -> bool:
    value = _required(mapping, key, context=context)
    if type(value) is not bool:
        raise ValueError(f"{context}.{key} must be a boolean, got {value!r}")
    return value


def _required_int(mapping: dict[str, Any], *keys: str, context: str) -> int:
    value = _required(mapping, *keys, context=context)
    if type(value) is not int:
        raise ValueError(f"{context}.{keys!r} must be an integer, got {value!r}")
    return value


def _list_value(
    mapping: dict[str, Any],
    *keys: str,
    context: str,
    required: bool = True,
) -> list[Any]:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if not isinstance(value, list):
                raise ValueError(f"{context}.{key} must be a list, got {value!r}")
            return value
    if required:
        raise ValueError(f"{context} is missing required list {keys!r}")
    return []


def _list_alias_value(
    mapping: dict[str, Any],
    *keys: str,
    context: str,
    required: bool = True,
) -> list[Any]:
    """Multiple aliases must agree; an explicit empty list is still data."""
    present = []
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if not isinstance(value, list):
            raise ValueError(f"{context}.{key} must be a list, got {value!r}")
        present.append((key, value))
    if not present:
        if required:
            raise ValueError(f"{context} is missing required list {keys!r}")
        return []
    first = present[0][1]
    if any(value != first for _key, value in present[1:]):
        raise ValueError(
            f"{context} contains conflicting list aliases: "
            f"{[key for key, _value in present]!r}"
        )
    return first


def _dict_records(values: list[Any], *, context: str) -> list[dict[str, Any]]:
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{context}[{index}] must be an object, got {value!r}")
    return values


def _real_card_tuple(card: dict[str, Any]) -> tuple[str, int]:
    cid = _required(card, "id", context="real card record")
    up = _required_int(card, "upgrades", context="real card record")
    if up < 0:
        raise ValueError(f"real card upgrade count must be nonnegative: {card!r}")
    c = _card(str(cid), up)
    return c["id"], c["up"]


def _real_card_dict(card: dict[str, Any], *, include_cost: bool = False) -> dict[str, Any]:
    cid, up = _real_card_tuple(card)
    out: dict[str, Any] = {"id": cid, "up": up}
    if include_cost:
        if "cost" not in card:
            raise ValueError(f"real hand card is missing its current cost: {card!r}")
        out["cost"] = int(card["cost"])
    return out


def _relic_id(relic: Any) -> str:
    if isinstance(relic, dict):
        value = _required(relic, "id", context="real relic record")
    else:
        value = relic
    if not value:
        raise ValueError(f"real relic record is missing its identity: {relic!r}")
    rid = str(value)
    if rid in UNSUPPORTED_REAL_RELIC_IDS:
        raise ValueError(
            f"unsupported real relic ID {rid!r}: "
            f"{UNSUPPORTED_REAL_RELIC_IDS[rid]}"
        )
    if rid == "PAD" or rid not in V.RELIC_VOCAB_INDEX:
        raise ValueError(f"unknown real relic ID: {rid}")
    return rid


_COMBAT_COUNTER_KEYS = {
    "Happy Flower": "happy_flower",
    "Incense Burner": "incense_burner",
    "InkBottle": "ink_bottle",
    "Inserter": "inserter",
    "Nunchaku": "nunchaku",
    "Pen Nib": "pen_nib",
    "Sundial": "sundial",
    "NeowsBlessing": "neow_lament",
}


def _potion_id(potion: Any) -> str:
    if isinstance(potion, dict):
        value = _required(potion, "id", context="real potion record")
    else:
        value = potion
    if not value:
        raise ValueError(f"real potion record is missing its identity: {potion!r}")
    pid = str(value)
    if pid in EMPTY_REAL_POTION_IDS:
        return "PAD"
    if pid == "PAD":
        raise ValueError("real potion data cannot use the bridge PAD sentinel")
    canonical = POTION_ALIASES.get(pid, pid)
    if canonical not in V.POTION_VOCAB_INDEX:
        raise ValueError(f"unknown real potion ID: {pid}")
    return canonical


def _potion_requires_target(potion: Any, pid: str) -> bool:
    if not isinstance(potion, dict):
        raise ValueError(f"real potion record must be an object: {potion!r}")
    return _required_bool(
        potion, "requires_target", context=f"real potion {pid!r}"
    )


def _monster_is_live(monster: Any, *, context: str) -> bool:
    if not isinstance(monster, dict):
        raise ValueError(f"{context} must be an object, got {monster!r}")
    gone = _required_bool(monster, "is_gone", context=context)
    hp = int(_required(monster, "current_hp", context=context))
    return not gone and hp > 0


def _price(item: Any, *, required: bool = True) -> int | None:
    if not isinstance(item, dict):
        raise ValueError(f"priced item must be an object, got {item!r}")
    for key in ("price", "cost"):
        if key in item and item[key] is not None:
            return int(item[key])
    if required:
        raise ValueError(f"item is missing its price: {item!r}")
    return None


def _normalize_boss(value: Any) -> str:
    if not value:
        raise ValueError("real game state is missing the act boss identity")
    text = str(value)
    if text == "PAD":
        raise ValueError("real boss data cannot use the bridge PAD sentinel")
    if text in BOSS_CANONICAL_IDS:
        return text
    if text in _BOSS_ALIASES:
        return _BOSS_ALIASES[text]
    raise ValueError(f"unknown real boss identity: {value!r}")


def _normalize_event(value: Any) -> str:
    if not value:
        raise ValueError("real event screen is missing its event identity")
    text = str(value)
    if text == "PAD":
        raise ValueError("real event data cannot use the bridge PAD sentinel")
    if text in _EVENT_ALIASES:
        return _EVENT_ALIASES[text]
    if text in V.EVENT_VOCAB_INDEX:
        return text
    raise ValueError(f"unknown real event identity: {value!r}")


def _screen_cards(ss: dict[str, Any]) -> list[tuple[str, int]]:
    cards = _list_alias_value(
        ss, "cards", "card_reward", "cardRewards",
        context="screen_state",
    )
    return [_real_card_tuple(card) for card in _dict_records(cards, context="screen cards")]


def _reward_records(ss: dict[str, Any]) -> list[dict[str, Any]]:
    rewards = _list_value(ss, "rewards", context="screen_state")
    return _dict_records(rewards, context="screen rewards")


def _reward_type(reward: dict[str, Any]) -> str:
    value = str(_required(
        reward, "reward_type", context="combat reward record"
    )).upper()
    if value not in REAL_REWARD_TYPE_IDS:
        raise ValueError(f"unknown real reward type: {value!r}")
    return value


def _reward_choice_type(choice: Any) -> str:
    label = _norm_label(choice)
    try:
        return REWARD_CHOICE_TYPES[label]
    except KeyError as exc:
        raise ValueError(f"unknown combat reward choice keyword: {choice!r}") from exc


def _screen_reward_gold(ss: dict[str, Any]) -> list[int]:
    gold = []
    for reward in _reward_records(ss):
        if _reward_type(reward) not in {"GOLD", "STOLEN_GOLD"}:
            continue
        amount = int(_required(reward, "gold", context="gold reward record"))
        if amount < 0:
            raise ValueError(f"gold reward amount must be nonnegative: {amount}")
        gold.append(amount)
    return gold


def _screen_reward_keys(ss: dict[str, Any]) -> list[str]:
    keys = [
        reward_type
        for reward in _reward_records(ss)
        if (reward_type := _reward_type(reward))
        in {"EMERALD_KEY", "SAPPHIRE_KEY"}
    ]
    if len(keys) > 1:
        raise ValueError(f"multiple simultaneous key rewards are unsupported: {keys!r}")
    return keys


def _screen_relics(ss: dict[str, Any]) -> list[str]:
    direct = any(key in ss for key in ("relics", "boss_relics"))
    relics = _list_alias_value(
        ss, "relics", "boss_relics", context="screen_state", required=False
    )
    if "rewards" in ss:
        reward_relics = [
            reward["relic"]
            for reward in _reward_records(ss)
            if _reward_type(reward) == "RELIC"
        ]
        if direct and relics != reward_relics:
            raise ValueError("explicit relic list conflicts with reward records")
        relics = reward_relics
    elif not direct:
        raise ValueError("screen has neither an explicit relic list nor reward records")
    return [_relic_id(r) for r in relics]


def _screen_potions(
    ss: dict[str, Any], *, require_price: bool = False
) -> list[tuple[str, int | None]]:
    potions = _list_value(ss, "potions", context="screen_state", required=False)
    if require_price and "potions" not in ss:
        raise ValueError("shop screen is missing its explicit potion inventory")
    if "rewards" in ss:
        reward_potions = [
            reward["potion"]
            for reward in _reward_records(ss)
            if _reward_type(reward) == "POTION"
        ]
        if "potions" in ss and potions != reward_potions:
            raise ValueError("explicit potion list conflicts with reward records")
        potions = reward_potions
    elif "potions" not in ss:
        raise ValueError("screen has neither an explicit potion list nor reward records")
    return [(_potion_id(p), _price(p, required=require_price)) for p in potions]


def _structured_event_choices(ss: dict[str, Any], legal_count: int) -> list[dict[str, Any]]:
    """Pass through typed event-button metadata when a bridge supplies it.

    Stock CommunicationMod currently exports localized ``text``/``label`` only.
    Those strings are useful for display but are not stable entity identifiers,
    so this adapter deliberately does not parse card/relic/potion names from
    them.  A richer bridge can add ``kind`` plus typed fields to each option.
    """
    raw_options = _list_value(ss, "options", context="event screen", required=False)
    # ``choice_index`` is stock transport metadata, not a semantic type.  Only
    # an explicit ``kind`` opts a richer bridge into typed option decoding.
    # Treating every stock option as typed would either parse localized labels
    # as identities or reject all ordinary event screens.
    typed = any(
        isinstance(raw, dict) and "kind" in raw
        for raw in raw_options
    )
    if not typed:
        return []
    _dict_records(raw_options, context="event options")
    choices = []
    for raw_index, raw in enumerate(raw_options):
        if raw.get("disabled"):
            continue
        index = _int_or_none(raw.get("choice_index"))
        if index is None:
            raise ValueError(f"typed event option {raw_index} has no choice_index")
        if not (0 <= index < min(legal_count, A.EVENT_OPT_W)):
            raise ValueError(f"typed event option index is out of range: {index}")
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"typed event option {index} has no kind")
        aliases = DYNAMIC_EVENT_OPTION_KIND_ALIASES.get(ss.get("event_id"), {})
        kind = aliases.get(kind, kind)
        detail: dict[str, Any] = {"index": index, "kind": kind}
        try:
            if kind in {"lose_card", "gain_card_then_remove"}:
                card = raw.get("card")
                if not isinstance(card, dict):
                    raise ValueError(f"typed event option {index} has no card record")
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
                          "transform_cards", "transform_random_cards", "remove_card_upgrade_random"}:
                detail["gold"] = int(raw["gold"])
                if "count" in raw:
                    detail["count"] = int(raw["count"])
            elif kind != "leave":
                raise ValueError(f"unsupported typed event option kind: {kind!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed typed event option {index}: {raw!r}") from exc
        choices.append(detail)
    if {choice["index"] for choice in choices} != set(range(min(legal_count, A.EVENT_OPT_W))):
        raise ValueError("typed event options do not cover every legal event choice")
    return choices


def _choice_index(labels: list[Any], aliases: set[str], occurrence: int = 0) -> int | None:
    seen = 0
    for idx, label in enumerate(labels):
        if _norm_label(label) in aliases:
            if seen == occurrence:
                return idx
            seen += 1
    return None


def _map_room(value: Any) -> str:
    if value is None or value == "":
        raise ValueError("map node is missing its room identity")
    text = str(value)
    room = _ROOM_LABELS.get(_norm_label(text), text)
    if room not in V.ROOM_VOCAB_INDEX or room in {"NONE", "INVALID"}:
        raise ValueError(f"unknown/non-actionable map room identity: {value!r}")
    return room


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
    children = _list_value(node, "children", context="map node")
    for child_index, child in enumerate(children):
        if isinstance(child, dict):
            xy = _node_xy(child)
            if xy is None:
                raise ValueError(f"map child {child_index} is missing x/y: {child!r}")
            out.append(xy)
        else:
            x = _int_or_none(child)
            y = _int_or_none(node.get("y"))
            if x is None or y is None:
                raise ValueError(f"malformed map child {child_index}: {child!r}")
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
        # Refuse to start a live bridge if any identity/source vocabulary has
        # drifted. Validation expands every canonical identity and alias; PAD is
        # reserved solely for structurally absent fixed-width slots.
        from bridge.schemas.codec_contract import validate_codec_contract
        validate_codec_contract()
        if combat_action_contract not in ("legacy", *C.NEURAL_CONTRACTS):
            raise ValueError("unsupported bridge combat action contract")
        self.combat_action_contract = combat_action_contract
        self._joint_multi_plan = None
        self._combat_grid_receipt = None
        self._paused_wait_seconds = 0.0
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
        self._shop_proceed_sent = False
        # CommunicationMod's SKIP on CARD_REWARD closes the card sub-screen
        # instead of marking the combat card reward skipped.  Track this case so
        # the returned COMBAT_REWARD frame behaves like the sim's skipRewardCards:
        # the same card reward is masked out and the policy can claim remaining
        # non-card rewards or proceed.
        self._combat_reward_card_open = False
        self._skipped_combat_card_reward = False
        self._skipped_reward_card_count = 0
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

    def exclude_wait_time(self, seconds: float) -> None:
        """Share only the caller's explicitly observed settings-pause duration."""
        if seconds < 0:
            raise ValueError('paused wait duration cannot be negative')
        self._paused_wait_seconds += seconds

    def _wait_clock(self) -> float:
        return time.monotonic() - self._paused_wait_seconds

    def update(self, frame: dict[str, Any]) -> None:
        validate_transport_frame(frame)
        previous_overworld_grid = (
            self.screen_type() == "GRID" and not bool(self.gs.get("combat_state"))
        )
        previous_relic_ids = self._last_relic_ids
        self.frame = frame
        self.gs = frame["game_state"] if "game_state" in frame else {}
        for key, table in (("room_phase", REAL_ROOM_PHASES),
                           ("action_phase", REAL_ACTION_PHASES)):
            if key in self.gs and self.gs[key] not in table:
                raise ValueError(f"unknown real-game {key} keyword: {self.gs[key]!r}")
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
        if self.screen_type() == "MAP" or self.gs.get("room_type") != "ShopRoom":
            self._leaving_shop = False
            self._shop_proceed_sent = False
        if self.screen_type() == "COMBAT_REWARD":
            self._combat_reward_outer_choices = tuple(self.choices())
            self._combat_reward_outer_commands = frozenset(self.commands())
            self._combat_reward_outer_options = self._current_outer_reward_options()
        elif self.screen_type() not in {"COMBAT_REWARD", "CARD_REWARD"}:
            self._combat_reward_card_open = False
            self._skipped_combat_card_reward = False
            self._skipped_reward_card_count = 0
            self._combat_reward_outer_choices = ()
            self._combat_reward_outer_commands = frozenset()
            self._combat_reward_outer_options = {}
            self._deferred_combat_reward_action = None

    def screen_type(self) -> str:
        if "screen_type" not in self.gs or self.gs["screen_type"] is None:
            if not self.in_game() or "error" in self.frame:
                return "NONE"
            raise ValueError("in-game frame is missing required screen_type")
        return str(self.gs["screen_type"])

    def phase(self) -> str:
        if self.gs.get("combat_state") and str(self.gs.get("room_phase") or "").upper() == "COMBAT":
            return "combat"
        screen = self.screen_type()
        try:
            return _SCREEN_PHASE[screen]
        except KeyError as exc:
            raise ValueError(f"unknown real-game screen type: {screen!r}") from exc

    def in_game(self) -> bool:
        return bool(self.frame.get("in_game"))

    def commands(self) -> set[str]:
        return _commands(self.frame)

    def choices(self) -> list[Any]:
        return _choice_list(self.gs)

    def _refresh_command(self) -> str:
        if "STATE" in self.commands():
            return "STATE"
        if "WAIT" in self.commands():
            return "WAIT 1"
        raise RuntimeError(
            "frame requires refresh but advertises neither STATE nor WAIT"
        )

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
        if (self.screen_type() in {"HAND_SELECT", "GRID"}
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
        return bool(_screen_cards(ss))

    def auto_command(self, *, auto_enter_shop: bool = True) -> tuple[str, str] | None:
        """Commands that should bypass the policy to match the sim state model."""
        if self._combat_grid_receipt is not None:
            if self._wait_clock() - self._combat_grid_receipt['started'] > 10:
                raise RuntimeError('Liquid Memories GRID acknowledgement timed out')
        if "error" in self.frame:
            return "STATE", "mod_error_state_refresh"
        if not self.in_game():
            return None
        if not self.frame.get("ready_for_command", False):
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
        if self._combat_grid_receipt is not None:
            command = self._continue_combat_grid()
            if command is not None:
                return command, "acknowledge_combat_grid_selection"
        if self._joint_multi_plan is not None:
            command = self._continue_joint_multi()
            if command is not None:
                return command, "complete_multi_continuation"

        st = self.screen_type()
        ss = _screen_state(self.gs)
        cl = self.choices()

        if st == "EVENT" and not ss.get('options') and not cl:
            identity = (self.gs.get('seed'), self.gs.get('floor'), ss.get('event_id'))
            pending = getattr(self, '_empty_event_wait', None)
            if pending is None or pending[0] != identity:
                self._empty_event_wait = (identity, self._wait_clock())
            elif self._wait_clock() - pending[1] > 10:
                raise RuntimeError('event options did not load within 10 seconds')
            return self._refresh_command(), 'await_event_options'
        self._empty_event_wait = None

        # Completion animation is a transport page, not a policy choice or a
        # GAME_OVER receipt. Wait for the actual terminal frame to record it.
        if st == "COMPLETE" and not self.gs.get("combat_state") and not cl:
            if "PROCEED" in self.commands():
                return "PROCEED", "complete_run_transition"
            return self._refresh_command(), "await_complete_run_transition"

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

        if st == "EVENT":
            dialogue = event_contract.transport_only(self.gs)
            if dialogue is not None:
                return f"CHOOSE {dialogue}", "event_dialogue_transport"
            if _normalize_event(ss.get("event_id")) == "NEOW":
                stage = _required(self.gs, "simulator_neow_screen", context="Neow screen")
                if type(stage) is not int or stage not in NEOW_SCREEN_CONTRACTS:
                    raise ValueError(f"unknown Neow screen keyword: {stage!r}")
                role = NEOW_SCREEN_CONTRACTS[stage]
                if role.startswith("unsupported_"):
                    raise ValueError(f"Neow screen is explicitly unsupported: {stage}/{role}")
                if role.endswith("_transport"):
                    if len(cl) != 1:
                        raise ValueError("Neow dialogue transport requires exactly one choice")
                    return "CHOOSE 0", "neow_intro"

        if auto_enter_shop and self._leaving_shop:
            if st == "SHOP_SCREEN":
                return self._refresh_command(), "await_shop_screen_leave"
            if st == "SHOP_ROOM":
                if self._shop_proceed_sent or "PROCEED" not in self.commands():
                    return self._refresh_command(), "await_shop_room_exit"
                self._shop_proceed_sent = True
                return "PROCEED", "leave_shop_room"
        if auto_enter_shop and st == "SHOP_ROOM":
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
            reward_types = [_reward_choice_type(choice) for choice in cl]
            card_indices = [
                i for i, reward_type in enumerate(reward_types)
                if reward_type == "CARD"
            ]
            if len(card_indices) > self._skipped_reward_card_count:
                self._combat_reward_card_open = True
                # Native consumes the last remaining reward, not the first.
                index = card_indices[-1 - self._skipped_reward_card_count]
                return f"CHOOSE {index}", "open_card_reward"

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
                0, int(sel["pick_max"]) - len(self.multi_selected)
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
        if self.combat_action_contract in C.NEURAL_CONTRACTS:
            policy_size = C.policy_size(self.combat_action_contract)
            if policy_size < len(mask):
                raise ValueError(
                    f"policy action contract is narrower than overworld actions: "
                    f"{policy_size} < {len(mask)}"
                )
            return np.pad(mask, (0, policy_size - len(mask)))
        return mask

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
                if len(self.multi_selected) >= int(desc["pick_min"]):
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
        desc = self.select_descriptor()
        # Preserve native transition order. Forethought inserts each selected
        # card at the bottom in ascending source order; exhaust/gamble use
        # descending order. UUID lookup below handles shifted Java indices.
        indices = sorted(semantic[1], reverse=desc['context'] != 'FORETHOUGHT')
        hand = tuple(real_card_state(card) for card in _list_value(
            self.gs['combat_state'], 'hand', context='multi-selection hand'))
        if _list_value(_screen_state(self.gs), 'selected', context='HAND_SELECT selection'):
            raise ValueError('cannot start a multi-selection plan with unacknowledged selected cards')
        self._joint_multi_plan = dict(signature=self._joint_screen_identity(),
                                      remaining=[hand[index] for index in indices],
                                      previous=None, sent=None, selected=())
        return self._continue_joint_multi()

    def _joint_screen_identity(self):
        desc = self.select_descriptor()
        if desc is None:
            if self.screen_type() not in {"HAND_SELECT", "GRID"}:
                return None
            raise ValueError("multi-selection plan is missing its descriptor")
        return (self.screen_type(), self.gs.get("seed"), self.gs.get("floor"),
                (self.gs.get("combat_state") or {}).get("turn"), desc.get("context"),
                _screen_state(self.gs).get("selection_source"))

    def _continue_joint_multi(self):
        """Expand one subset into acknowledged UI choices without another model call."""
        plan = self._joint_multi_plan
        if plan["signature"] != self._joint_screen_identity():
            self._joint_multi_plan = None
            return None
        hand = tuple(real_card_state(card) for card in _list_value(
            self.gs['combat_state'], 'hand', context='multi-selection hand'))
        screen_state = _screen_state(self.gs)
        selected = tuple(real_card_state(card) for card in _list_value(
            screen_state, "selected", context="HAND_SELECT selection"
        ))
        identities = [card['uuid'] for card in hand + selected]
        if len(set(identities)) != len(identities):
            self._joint_multi_plan = None
            raise ValueError('multi-selection repeats a physical card UUID')
        if plan["sent"] is not None:
            previous, index = plan["previous"], plan["sent"]
            removed = previous[:index] + previous[index + 1:]
            if hand == previous and selected == plan["selected"]:
                return self._refresh_command()  # Await acknowledgement; never select twice.
            # HandCardSelectScreen removes the actual object from hand and
            # appends it to selectedCards; both sides of that move must agree.
            if hand != removed or selected != plan["selected"] + (previous[index],):
                self._joint_multi_plan = None
                raise ValueError("multi-selection frame changed unexpectedly; discard the pending subset")
        if not plan["remaining"]:
            if "CONFIRM" not in self.commands():
                return self._refresh_command()
            self._joint_multi_plan = None
            return "CONFIRM"
        card = plan["remaining"][0]
        indices = [i for i, current in enumerate(hand) if current['uuid'] == card['uuid']]
        if len(indices) != 1 or hand[indices[0]] != card or "CHOOSE" not in self.commands():
            self._joint_multi_plan = None
            raise ValueError("multi-selection candidate is no longer available")
        index = indices[0]
        plan["remaining"].pop(0)
        plan.update(previous=hand, selected=selected, sent=index)
        return f"CHOOSE {index}"

    def _liquid_memory_piles(self):
        combat = _required(self.gs, 'combat_state', context='Liquid Memories receipt')
        return {
            name: tuple(real_card_state(card) for card in _list_value(
                combat, name, context='Liquid Memories receipt'))
            for name in ('hand', 'draw_pile', 'discard_pile', 'exhaust_pile')
        }

    def _begin_combat_grid_receipt(self, desc, instance):
        if self._combat_grid_receipt is not None:
            raise RuntimeError('cannot choose another GRID card before acknowledgement')
        previous = tuple(desc['selected_instances'])
        chosen = previous + (instance,)
        if len({card['uuid'] for card in chosen}) != len(chosen):
            raise ValueError('combat GRID attempted to select an instance twice')
        piles = self._liquid_memory_piles()
        expected = {name: [dict(card) for card in cards] for name, cards in piles.items()}
        hand, discard = expected['hand'], expected['discard_pile']
        if len(hand) > actions.HAND_SLOTS:
            raise ValueError('Liquid Memories receipt exceeds hand capacity')
        for card in chosen:
            matches = [i for i, source in enumerate(discard) if source == card]
            if len(matches) != 1:
                raise ValueError('Liquid Memories selected instance is not in discard')
            if len(hand) < actions.HAND_SLOTS:
                moved = discard.pop(matches[0])
                if moved['cost'] >= 0:
                    moved['cost'] = 0
                hand.append(moved)
        # Java calls applyPowers on every selected card (even when full) and
        # finally on the whole hand. These three overrides change base fields;
        # PerfectedStrike restores its temporary baseDamage before returning.
        # Account for the exact refresh instead of ignoring those wire fields.
        refreshed = {card['uuid'] for card in (*chosen, *hand)}
        combat = self.gs['combat_state']
        block = _required(_required(combat, 'player', context='Liquid Memories receipt'),
                          'block', context='Liquid Memories receipt player')
        if type(block) is not int or block < 0:
            raise ValueError('Liquid Memories receipt requires explicit nonnegative block')
        for cards in expected.values():
            for card in cards:
                if card['uuid'] not in refreshed:
                    continue
                if card['id'] == 'Body Slam':
                    card['base_damage'] = block
                elif card['id'] == 'Mind Blast':
                    card['base_damage'] = len(expected['draw_pile'])
                elif card['id'] == 'Ritual Dagger':
                    card['base_block'] = card['misc']
        expected = {name: tuple(cards) for name, cards in expected.items()}
        self._combat_grid_receipt = dict(
            signature=self._joint_screen_identity(), previous=previous, chosen=chosen,
            remaining=int(desc['pick_remaining']) - 1, piles=piles, expected=expected,
            confirmed=False, started=self._wait_clock())

    def _continue_combat_grid(self):
        receipt = self._combat_grid_receipt
        if self._wait_clock() - receipt['started'] > 10:
            raise RuntimeError('Liquid Memories GRID acknowledgement timed out')
        identity = (self.gs.get('seed'), self.gs.get('floor'),
                    (self.gs.get('combat_state') or {}).get('turn'))
        if identity != receipt['signature'][1:4]:
            raise RuntimeError('combat identity changed before Liquid Memories acknowledgement')
        if self.screen_type() == 'GRID':
            if self._joint_screen_identity() != receipt['signature']:
                raise RuntimeError('combat GRID source changed before acknowledgement')
            desc = self.select_descriptor()
            selected = tuple(desc['selected_instances'])
            if self._liquid_memory_piles() != receipt['piles']:
                raise RuntimeError('Liquid Memories changed piles before GRID completion')
            if selected == receipt['previous']:
                return self._refresh_command()  # Old frame: never re-issue CHOOSE.
            if selected != receipt['chosen'] or desc['pick_remaining'] != receipt['remaining']:
                raise RuntimeError('Liquid Memories acknowledged a different selection')
            if receipt['remaining']:
                self._combat_grid_receipt = None
                return None
            if 'CONFIRM' in self.commands() and not receipt['confirmed']:
                receipt['confirmed'] = True
                return 'CONFIRM'
            return self._refresh_command()
        if receipt['remaining']:
            raise RuntimeError('Liquid Memories closed before all required cards were chosen')
        if self._liquid_memory_piles() != receipt['expected']:
            raise RuntimeError('Liquid Memories settled pile/instance/cost receipt differs')
        self._combat_grid_receipt = None
        return None

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
            canon["strict_complete"] = True
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
                    if int(counters[key]) != native_combat_relic_counter(relic, value):
                        raise ValueError(
                            "combat counter/relic export mismatch: "
                            f"{relic!r} counter={value} combat[{key}]={counters[key]}"
                        )
            return canon

        phase = self.phase()
        canon = {
            "strict_complete": True,
            "floor": int(_required(self.gs, "floor", context="game_state")),
            "hp": int(_required(self.gs, "current_hp", context="game_state")),
            "max_hp": int(_required(self.gs, "max_hp", context="game_state")),
            "gold": int(_required(self.gs, "gold", context="game_state")),
            "deck": sorted(self._deck_cards()),
            "relics": sorted(self._relics()),
            "keys": dict(self._run_keys()),
            "phase": phase,
            "options": self._options(phase),
            "global": self._global_block(),
        }
        return canon

    def _global_block(self) -> dict[str, Any]:
        from env.core.boss_value import is_boss_room
        ss = _screen_state(self.gs)
        parity_version = _required(
            self.gs, "simulator_parity_export", context="game_state"
        )
        if parity_version != SIMULATOR_PARITY_EXPORT_VERSION:
            raise ValueError(
                "real game state was not produced by the required parity exporter: "
                f"{parity_version!r}"
            )
        raw_character = _required(self.gs, "class", context="game_state")
        if raw_character not in REAL_CHARACTER_IDS:
            raise ValueError(f"unknown real-game character: {raw_character!r}")
        character = REAL_CHARACTER_IDS[raw_character]
        ascension = _required_int(
            self.gs, "ascension_level", "ascension", context="game_state"
        )
        room_type = _required(self.gs, "room_type", context="game_state")
        return {
            "character": character,
            "ascension": ascension,
            "act": _required_int(self.gs, "act", context="game_state"),
            "floor": _required_int(self.gs, "floor", context="game_state"),
            "combat_is_boss": is_boss_room(room_type),
            "boss": _normalize_boss(
                _required(
                    self.gs, "boss", "act_boss", context="game_state"
                )
            ),
            "relics": self._relics(),
            "relic_counters": self._relic_counters(),
            "potions": self._potions(),
            "keys": dict(self._run_keys()),
            "gold": _required_int(self.gs, "gold", context="game_state"),
            "hp": _required_int(self.gs, "current_hp", context="game_state"),
            "max_hp": _required_int(self.gs, "max_hp", context="game_state"),
            "deck": self._deck_cards(),
            "deck_state": real_master_deck_state(_list_value(self.gs, 'deck', context='game_state')),
            "boss_relics": _screen_relics(ss) if self.phase() == "boss_relic" else [],
        }

    def _run_keys(self) -> dict[str, bool]:
        return {
            semantic: _required_bool(
                self.gs, fields["real"], context="game_state"
            )
            for semantic, fields in RUN_KEY_FIELDS.items()
        }

    def _deck_cards(self) -> list[tuple[str, int]]:
        deck = _list_value(self.gs, "deck", context="game_state")
        return [
            _real_card_tuple(card)
            for card in _dict_records(deck, context="game_state.deck")
        ]

    def _relics(self) -> list[str]:
        relics = _list_value(self.gs, "relics", context="game_state")
        return [_relic_id(r) for r in relics]

    def _relic_counters(self) -> list[int]:
        relics = _dict_records(
            _list_value(self.gs, "relics", context="game_state"),
            context="game_state.relics",
        )
        counters = []
        for relic in relics:
            counters.append(native_relic_counter(
                _relic_id(relic), _required(relic, "counter", context="relic record")
            ))
        return counters

    def _potions(self) -> list[str]:
        return [_potion_id(p) for p in self._potion_records()]

    def _potion_records(self) -> list[Any]:
        return _list_value(self.gs, "potions", context="game_state")

    def _options(self, phase: str) -> dict[str, Any]:
        ss = _screen_state(self.gs)
        st = self.screen_type()
        cl = self.choices()
        opts: dict[str, Any] = {}

        if phase == "combat_reward":
            cards = _screen_cards(ss) if st == "CARD_REWARD" else []
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
                "keys": list(outer.get("keys", [])),
                # card_count is the number of reward groups, not the number of
                # cards inside the currently-open group.
                "card_count": max(
                    0,
                    sum(
                        _reward_choice_type(choice) == "CARD"
                        for choice in (
                            self._combat_reward_outer_choices or tuple(cl)
                        )
                    ) - self._skipped_reward_card_count,
                ) if has_card else 0,
                "cards": cards if has_card else [],
            })
        elif phase == "event":
            event = _normalize_event(_required(
                ss, "event_id", context="event screen"
            ))
            if event == "PAD":
                raise ValueError(f"unknown event identity: {ss.get('event_id')!r}")
            opts["event"] = event
            if event == "NEOW" and self.neow_observation:
                opts["neow"] = self._neow_options(ss)
            elif event != "NEOW":
                details = _structured_event_choices(ss, len(cl))
                if ss.get('event_id') == 'Falling':
                    mapping = event_contract.options(self.gs)
                    opts["event_options"] = sorted(mapping)
                    opts['event_choices'] = [dict(index=d['index'], kind='lose_card', card=_real_card_tuple(d['card']))
                                             for d in event_contract.falling_choices(self.gs)]
                    details = []
                elif ss.get('event_id') == "N'loth":
                    mapping = event_contract.options(self.gs)
                    opts["event_options"] = sorted(mapping)
                    opts['event_choices'] = [
                        dict(index=d['index'], kind=d['kind'], **({'relic': _relic_id(d['relic'])}
                             if d['kind'] == 'lose_relic' else {}))
                        for d in event_contract.nloth_choices(self.gs)]
                    details = []
                elif details and event in DYNAMIC_EVENT_OPTION_KINDS:
                    kind_to_semantic = DYNAMIC_EVENT_OPTION_KINDS[event]
                    unknown = sorted({d["kind"] for d in details} - set(kind_to_semantic))
                    if unknown:
                        raise ValueError(
                            f"unknown typed event option kinds for {event}: {unknown!r}"
                        )
                    semantic_details = [
                        dict(d, index=kind_to_semantic[d["kind"]]) for d in details
                    ]
                    semantic_indices = [d["index"] for d in semantic_details]
                    if len(semantic_indices) != len(set(semantic_indices)):
                        raise ValueError(f"duplicate typed event semantics for {event}")
                    opts["event_options"] = sorted(semantic_indices)
                    opts["event_choices"] = semantic_details
                    details = []
                else:
                    mapping = event_contract.options(self.gs)
                    opts["event_options"] = sorted(mapping)
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
            if self.gs.get('act') == 4:
                opts['terminal_map_nodes'] = [dict(n) for n in self._map_leaf_nodes()
                                             if n['room_class'] == 'TrueVictoryRoom']
        elif phase == "shop":
            opts.update(self._shop_options(ss))
        return opts

    def _event_option_mapping(self) -> dict[int, tuple[int, str]]:
        """Map frozen native EVENT_OPT IDs to current CommunicationMod ordinals."""
        ss = _screen_state(self.gs)
        event = _normalize_event(_required(ss, "event_id", context="event screen"))
        details = _structured_event_choices(ss, len(self.choices()))
        kind_to_semantic = DYNAMIC_EVENT_OPTION_KINDS.get(event)
        if details and kind_to_semantic is not None:
            result: dict[int, tuple[int, str]] = {}
            for detail in details:
                kind = detail["kind"]
                if kind not in kind_to_semantic:
                    raise ValueError(
                        f"unknown typed event option kind for {event}: {kind!r}"
                    )
                semantic = kind_to_semantic[kind]
                if semantic in result:
                    raise ValueError(f"duplicate typed event semantic for {event}: {semantic}")
                result[semantic] = (detail["index"], kind)
            return result
        return event_contract.options(self.gs)

    def _current_outer_reward_options(self) -> dict[str, list[Any]]:
        ss = _screen_state(self.gs)
        labels = [_reward_choice_type(choice) for choice in self.choices()]
        record_types = [_reward_type(record) for record in _reward_records(ss)]
        if record_types != labels:
            raise ValueError(
                "CommunicationMod reward records and choice keywords disagree: "
                f"records={record_types!r} choices={labels!r}"
            )
        expected = {
            "gold": sum(label in {"GOLD", "STOLEN_GOLD"} for label in labels),
            "relics": labels.count("RELIC"),
            "potions": labels.count("POTION"),
            "keys": sum(
                label in {"EMERALD_KEY", "SAPPHIRE_KEY"} for label in labels
            ),
        }
        actual: dict[str, list[Any]] = {
            "gold": _screen_reward_gold(ss),
            "relics": _screen_relics(ss),
            "potions": [pid for pid, _price_ in _screen_potions(ss)],
            "keys": _screen_reward_keys(ss),
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
        if _required(self.gs, "simulator_neow_screen", context="Neow screen") != 3:
            raise ValueError("Neow policy offers requested outside the blessing screen")
        raw = _list_value(self.gs, "simulator_neow_options", context="Neow screen")
        if len(raw) not in (2, A.NEOW_W) or len(raw) != len(self.choices()):
            raise ValueError("Neow typed offers do not exactly cover the two/four UI choices")
        out = []
        for index, item in enumerate(_dict_records(raw, context="Neow offers")):
            if type(item.get("choice_index")) is not int or item["choice_index"] != index:
                raise ValueError("Neow typed choice index is missing, reordered or duplicated")
            option = neow_option_from_types(
                _required(item, "bonus_type", context="Neow offer"),
                _required(item, "drawback_type", context="Neow offer"),
            )
            unit = _required(item, "hp_unit", context="Neow offer")
            maximum = _required(self.gs, "max_hp", context="game_state")
            if type(unit) is not int or type(maximum) is not int or unit != maximum // 10:
                raise ValueError("Neow cached HP unit differs from the supported current-maxHP contract")
            option["hp_unit"] = unit
            out.append(option)
        return out

    def _map_nodes(self) -> list[tuple[int, str]]:
        ss = _screen_state(self.gs)
        if ss.get("boss_available"):
            current = _node_xy(ss.get("current_node"))
            if current is None:
                raise ValueError("boss map choice is missing current_node coordinates")
            return [(current[0], "BOSS")]

        next_nodes = _dict_records(
            _list_value(ss, "next_nodes", context="map screen"),
            context="map next_nodes",
        )
        if not next_nodes:
            raise ValueError("actionable map screen has no structured next_nodes")
        if len(next_nodes) > A.N_COLS:
            raise ValueError(
                f"map column capacity {A.N_COLS} exceeded: {len(next_nodes)}"
            )
        nodes = []
        rows = set()
        for node in next_nodes:
            xy = _node_xy(node)
            if xy is None:
                raise ValueError(f"map choice is missing x/y coordinates: {node!r}")
            x, y = xy
            if not (0 <= x < A.N_COLS and 0 <= y <= _MAP_MAX_Y):
                raise ValueError(f"map choice is out of range: {(x, y)!r}")
            rows.add(y)
            nodes.append((x, _node_room(node)))
        if len(rows) != 1 or len({x for x, _room in nodes}) != len(nodes):
            raise ValueError(f"map choices have mixed rows or duplicate columns: {nodes!r}")
        return nodes

    def _map_leaf_nodes(self):
        if self.gs.get('simulator_parity_export') != SIMULATOR_PARITY_EXPORT_VERSION:
            raise ValueError('complete map leaf data requires the current parity exporter')
        leaves = _dict_records(_list_value(self.gs, 'simulator_map_leaf_nodes',
                               context='complete map'), context='map leaf nodes')
        seen = set()
        for node in leaves:
            xy = _node_xy(node)
            if xy is None or xy in seen:
                raise ValueError('map leaf identity is absent or duplicated')
            seen.add(xy)
            if _node_children(node):
                raise ValueError('map leaf export contains an outgoing edge')
            room_class = _required(node, 'room_class', context='map leaf')
            if self.gs.get('act') != 4 or (xy, room_class) not in {
                ((3, 3), 'MonsterRoomBoss'), ((3, 4), 'TrueVictoryRoom')
            }:
                raise ValueError(f'unsupported actual map leaf: {xy}/{room_class}')
            if room_class == 'MonsterRoomBoss' and _node_room(node) != 'BOSS':
                raise ValueError('map Boss leaf class and symbol disagree')
        if self.gs.get('act') == 4 and seen != {(3, 3), (3, 4)}:
            raise ValueError('TheEnding map is missing actual Boss/victory leaf nodes')
        return leaves

    def _full_map_nodes(self, ss: dict[str, Any]) -> list[dict[str, Any]]:
        raw = _list_value(
            self.gs, "map", context="game_state", required=False
        ) or _list_value(
            ss, "map", "nodes", "full_map", context="map screen", required=False
        )
        if not raw:
            raise ValueError("global map observation requires the structured full map")
        if self.gs.get('act') == 4:
            raw = list(raw) + [n for n in self._map_leaf_nodes()
                              if n['room_class'] == 'MonsterRoomBoss']

        raw_nodes: dict[tuple[int, int], dict[str, Any]] = {}
        for node in _dict_records(raw, context="full map"):
            xy = _node_xy(node)
            if xy is None:
                raise ValueError(f"full-map node is missing x/y coordinates: {node!r}")
            x, y = xy
            if _is_terminal_rest_node(node):
                continue
            if not (0 <= x < A.N_COLS and 0 <= y <= _MAP_MAX_Y):
                raise ValueError(f"full-map node coordinate is out of range: {(x, y)!r}")
            if (x, y) in raw_nodes:
                raise ValueError(f"duplicate full-map node coordinate: {(x, y)!r}")
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
        if ss.get("boss_available"):
            if self.gs.get('act') == 4 and current == (3, 2):
                choices.add((3, 3))
            elif current is not None and current[1] == _MAP_MAX_Y:
                choices.add(current)
            else:
                raise ValueError('boss map choice has an unsupported current position')
        else:
            for node in _dict_records(
                _list_value(ss, "next_nodes", context="map screen"),
                context="map next_nodes",
            ):
                xy = _node_xy(node)
                if xy is None:
                    raise ValueError(f"map choice is missing x/y coordinates: {node!r}")
                choices.add(xy)

        if not choices:
            raise ValueError("structured map screen exposes no reachable choices")

        downstream = set(choices)
        frontier = list(choices)
        while frontier:
            xy = frontier.pop()
            node = raw_nodes.get(xy)
            if node is None:
                if xy == (3, 16):
                    continue
                raise ValueError(f"reachable map coordinate is absent from full map: {xy!r}")
            for child in _node_children(node):
                if child not in raw_nodes:
                    if child == (3, 16):
                        continue
                    raise ValueError(f"map edge points to absent node: {xy!r} -> {child!r}")
                if child not in downstream:
                    downstream.add(child)
                    frontier.append(child)

        out = []
        for (x, y), node in sorted(raw_nodes.items(), key=lambda item: (item[0][1], item[0][0])):
            children = []
            if y < _MAP_MAX_Y:
                for child_x, child_y in _node_children(node):
                    if child_y != y + 1 or not 0 <= child_x < A.N_COLS:
                        raise ValueError(f"map edge does not reach the next row: {(x, y)} -> {(child_x, child_y)}")
                    children.append(child_x)
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
        card_records = _dict_records(
            _list_value(ss, "cards", context="shop screen"), context="shop cards"
        )
        cards = []
        for card in card_records:
            cid, up = _real_card_tuple(card)
            cards.append((cid, up, int(_price(card))))
        relics = [
            (_relic_id(relic), int(_price(relic)))
            for relic in _dict_records(
                _list_value(ss, "relics", context="shop screen"),
                context="shop relics",
            )
        ]
        potions = _screen_potions(ss, require_price=True)
        remove_cost = int(_required(
            ss, "purge_cost", "remove_cost", context="shop screen"
        ))
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
        if ss.get("simulator_parity_export") != SIMULATOR_PARITY_EXPORT_VERSION:
            raise ValueError(
                "actionable selection was not produced by the required parity "
                f"exporter: {ss.get('simulator_parity_export')!r}"
            )
        context = self._select_context(ss, cards, overworld=overworld)
        grid_screen = self.screen_type() == "GRID"
        hand_screen = self.screen_type() == "HAND_SELECT"
        card_reward_screen = self.screen_type() == "CARD_REWARD"
        if grid_screen:
            pick_count = int(_required(ss, "num_cards", context="GRID selection"))
            selected_raw = _list_value(
                ss, "selected_cards", context="GRID selection"
            )
            can_pick_any = _required_bool(
                ss, "any_number", context="GRID selection"
            )
            for flag in OVERWORLD_SELECTION_FLAGS:
                _required(ss, flag, context="GRID selection")
            can_pick_zero = can_pick_any
        elif hand_screen:
            pick_count = int(_required(
                ss, "max_cards", context="HAND_SELECT selection"
            ))
            selected_raw = _list_value(
                ss, "selected", context="HAND_SELECT selection"
            )
            can_pick_zero = _required_bool(
                ss, "can_pick_zero", context="HAND_SELECT selection"
            )
            can_pick_any = _required_bool(
                ss, "any_number", context="HAND_SELECT selection"
            )
        elif card_reward_screen:
            # Combat Discovery/Codex screens select exactly one generated card.
            # Codex exposes SKIP as an explicit extra candidate; that is not
            # the same contract as confirming an empty selection.
            pick_count = 1
            selected_raw = []
            can_pick_zero = False
            can_pick_any = False
        else:
            raise ValueError(
                f"unsupported actionable selection screen {self.screen_type()!r}"
            )
        if pick_count < 0:
            raise ValueError(f"selection count must be nonnegative, got {pick_count}")
        selected_cards = [
            _real_card_tuple(card)
            for card in _dict_records(selected_raw, context="selected cards")
        ]
        have_selected = len(selected_cards)
        multi = bool(hand_screen and (
            context in {"EXHAUST_MANY", "GAMBLE"}
            or can_pick_any
            or pick_count != 1
        ))
        pick_max = pick_count
        # BaseMod uses 99 as the unbounded hand-selection sentinel (notably
        # Gambler's Brew).  Normalize that transport value to the actual
        # selectable hand so it matches the simulator's public contract.
        if multi and hand_screen and pick_max >= 99:
            pick_count = len(cards)
            pick_max = len(cards)
        if pick_max < 0:
            raise ValueError(f"selection maximum must be nonnegative, got {pick_max}")
        raw_cards = (
            _required(ss, "cards", context=f"{self.screen_type()} selection")
            if not hand_screen
            else _required(
                self.gs.get("combat_state") or {}, "hand",
                context="HAND_SELECT combat_state",
            )
        )
        if not isinstance(raw_cards, list):
            raise ValueError(f"selection source cards must be a list, got {raw_cards!r}")
        _dict_records(raw_cards, context="selection source cards")
        if len(raw_cards) != len(cards):
            raise ValueError(
                "selection card records/canonical identities have different lengths: "
                f"{len(raw_cards)} != {len(cards)}"
            )
        selected_instances = []
        if grid_screen:
            from bridge.schemas.card_state import real_selection_card_states
            grid_instances, selected_instances = real_selection_card_states(raw_cards, selected_raw)
            if have_selected > pick_count:
                raise ValueError('GRID has more selected cards than requested')
        selected_uuids = {card['uuid'] for card in selected_instances}
        candidates = []
        for i, (cid, up) in enumerate(cards):
            raw_card = raw_cards[i]
            instance = grid_instances[i] if grid_screen else real_card_state(raw_card)
            if grid_screen and instance['uuid'] in selected_uuids:
                continue
            card = {"id": cid, "up": up}
            if not overworld:
                card["cost"] = 0 if instance["free_to_play_once"] else instance["cost"]
            else:
                card["cost"] = instance["cost"]
            candidate = {"native_index": i, "card": card, "instance": instance}
            candidates.append(candidate)
        if not overworld and context == "CODEX" and "SKIP" in self.commands():
            candidates.append({"native_index": len(cards), "card": None, "instance": None})
        if len(candidates) > A.G_MAX:
            raise ValueError(
                f"selection candidate capacity {A.G_MAX} exceeded: {len(candidates)}"
            )
        if not overworld and context in {"SECRET_WEAPON", "SECRET_TECHNIQUE", "SEEK"}:
            # Static vocabulary preserves native CardId order without importing the sim.
            missing_cost = [
                candidate["native_index"]
                for candidate in candidates
                if candidate["card"] is not None and "cost" not in candidate["card"]
            ]
            if missing_cost:
                raise ValueError(
                    "draw-pile selection is missing current card costs at indices "
                    f"{missing_cost!r}"
                )
            candidates.sort(key=lambda c: (encode.CARD_INDEX[c["card"]["id"]],
                                            c["card"]["up"], c["card"]["cost"]))
        return {
            "context": context,
            "pick_remaining": max(0, pick_count - have_selected),
            "can_pick_zero": can_pick_zero,
            "can_pick_any": can_pick_any,
            "selected": selected_cards if (overworld or grid_screen) else [],
            "selected_instances": selected_instances,
            "multi": multi,
            "pick_min": 0 if (can_pick_zero or can_pick_any) else int(pick_count),
            "pick_max": int(pick_max),
            "cards_in_hand": len(cards) if not overworld else 0,
            "zone": "grid" if overworld else {
                "ARMAMENTS": "hand", "DUAL_WIELD": "hand",
                "FORETHOUGHT": "hand", "GAMBLE": "hand",
                "NIGHTMARE": "hand", "RECYCLE": "hand", "SETUP": "hand",
                "WARCRY": "hand", "EXHAUST_ONE": "hand",
                "EXHAUST_MANY": "hand", "HEADBUTT": "discard",
                "HOLOGRAM": "discard", "LIQUID_MEMORIES_POTION": "discard",
                "MEDITATE": "discard", "EXHUME": "exhaust",
                "SECRET_WEAPON": "draw", "SECRET_TECHNIQUE": "draw",
                "SEEK": "draw", "CODEX": "generated",
                "DISCOVERY": "generated",
            }[context],
            "candidates": candidates,
        }

    def _select_context(
        self, ss: dict[str, Any], cards: list[tuple[str, int]], *, overworld: bool
    ) -> str:
        context = None
        for key in ("selection_context", "context", "task", "select_context"):
            value = ss.get(key)
            if value:
                enumish = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").upper()
                if enumish in V.SELECT_CONTEXT_VOCAB_INDEX and enumish != "PAD":
                    context = enumish
                    break
        if context is None:
            raise ValueError(
                "actionable card selection is missing a recognized exported context: "
                f"screen={self.screen_type()!r}, "
                f"current_action={self.gs.get('current_action')!r}, "
                f"selection_source={ss.get('selection_source')!r}, "
                f"screen_state={ss!r}"
            )
        source = str(_required(ss, "selection_source", context="selection screen"))
        if source in {"", "NONE"}:
            raise ValueError("actionable selection is missing its source class")
        source_class = source.rsplit(".", 1)[-1]
        if not overworld:
            if source_class in UNSUPPORTED_SELECTION_SOURCES:
                raise ValueError(
                    f"unsupported combat selection source {source_class}: "
                    f"{UNSUPPORTED_SELECTION_SOURCES[source_class]}"
                )
            expected = COMBAT_SELECTION_SOURCES.get(source_class)
            if expected is None and source_class in DYNAMIC_COMBAT_SELECTION_SOURCES:
                allowed = set(DYNAMIC_COMBAT_SELECTION_SOURCES[source_class].values())
                if context not in allowed:
                    raise ValueError(
                        f"combat selection source/context mismatch: "
                        f"{source_class!r} cannot produce {context!r}"
                    )
            elif expected is None:
                raise ValueError(
                    f"unknown combat selection source class: {source_class!r}"
                )
            elif context != expected:
                raise ValueError(
                    f"combat selection source/context mismatch: "
                    f"{source_class!r} expected {expected!r}, got {context!r}"
                )
        else:
            possible = {
                mapped
                for flag, mapped in OVERWORLD_SELECTION_FLAGS.items()
                if _required_bool(ss, flag, context="GRID selection")
            }
            possible.update(
                mapped
                for relic, mapped in OVERWORLD_SELECTION_RELICS.items()
                if relic in self._relics()
            )
            event_context = OVERWORLD_SELECTION_EVENTS.get(source_class)
            if event_context is not None:
                possible.add(event_context)
            if context not in possible:
                raise ValueError(
                    "overworld selection source/context mismatch: "
                    f"source={source_class!r}, context={context!r}, "
                    f"derivable={sorted(possible)!r}"
                )
        return context

    def _select_signature(self) -> tuple[Any, ...] | None:
        desc = self.select_descriptor()
        if not A.is_multi_select(desc):
            return None
        candidates = tuple(
            (
                int(candidate["native_index"]),
                None if candidate["card"] is None else (
                    candidate["card"]["id"],
                    int(candidate["card"]["up"]),
                    candidate["card"].get("cost"),
                ),
            )
            for candidate in desc["candidates"]
        )
        return (
            self.screen_type(),
            _required(self.gs, "seed", context="game_state"),
            _required(self.gs, "floor", context="game_state"),
            (_required(self.gs["combat_state"], "turn", context="combat_state")
             if self.phase() == "combat" else None),
            desc.get("context"),
            candidates,
            tuple(desc.get("selected", ())),
        )

    def _reconcile_multi(self) -> None:
        sig = self._select_signature()
        if sig != self._multi_signature:
            self.multi_selected.clear()
            self._multi_signature = sig

    def _combat_hand_cards(self) -> list[tuple[str, int]]:
        cs = self.gs.get("combat_state")
        if not isinstance(cs, dict):
            raise ValueError("combat hand requested without combat_state")
        hand = _list_value(cs, "hand", context="combat_state")
        return [
            _real_card_tuple(card)
            for card in _dict_records(hand, context="combat hand")
        ]

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
            n = len(_screen_relics(_screen_state(self.gs)))
            if n != len(self.choices()):
                raise ValueError("boss relic identities and choice handles disagree")
            if n > A.BOSS_RELIC_W:
                raise ValueError(
                    f"boss relic capacity {A.BOSS_RELIC_W} exceeded: {n}"
                )
            return [ga.boss_relic(i) for i in range(n)] + pots
        if phase == "shop":
            return self._legal_shop_actions() + pots
        return pots

    def _legal_combat_actions(self) -> list[tuple]:
        if not self._combat_ready_for_policy():
            return []
        if self._combat_card_choice_screen():
            desc = self.select_descriptor()
            if not desc or not desc["candidates"]:
                raise ValueError("combat card choice requires explicit selection candidates")
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
        monsters = _list_value(cs, "monsters", context="combat_state")
        targets = [
            policy_index
            for policy_index, real_index in enumerate(self.combat_target_layout())
            for m in (monsters[real_index],)
            if _monster_is_live(m, context=f"combat monster[{real_index}]")
        ]
        hand = _dict_records(
            _list_value(cs, "hand", context="combat_state"), context="combat hand"
        )
        if len(hand) > actions.HAND_SLOTS:
            raise ValueError(
                f"combat hand capacity {actions.HAND_SLOTS} exceeded: {len(hand)}"
            )
        for h, card in enumerate(hand):
            if not _required_bool(card, "is_playable", context=f"combat hand[{h}]"):
                continue
            if _required_bool(card, "has_target", context=f"combat hand[{h}]"):
                for target in targets:
                    acts.append(ga.combat(actions.encode_play(h, target)))
            else:
                acts.append(ga.combat(actions.encode_play(h, None)))
        return acts + self._legal_potion_actions(combat=True)

    def _legal_event_actions(self) -> list[tuple]:
        ss = _screen_state(self.gs)
        event = _normalize_event(_required(
            ss, "event_id", context="event screen"
        ))
        n = len(self.choices())
        if event == "NEOW":
            return [ga.neow(i) for i in range(len(self._neow_options(ss)))]
        return [ga.event(i) for i in sorted(self._event_option_mapping())]

    def _legal_card_select_actions(self) -> list[tuple]:
        cmds = self.commands()
        if "CONFIRM" in cmds and "CHOOSE" not in cmds:
            return [ga.proceed()]
        if "CHOOSE" not in cmds:
            return [ga.proceed()] if "CONFIRM" in cmds else []
        desc = self.select_descriptor()
        if desc is None:
            raise ValueError("card selection is missing its descriptor")
        candidates = desc["candidates"]
        if len(candidates) > A.G_MAX:
            raise ValueError(
                f"selection candidate capacity {A.G_MAX} exceeded: {len(candidates)}"
            )
        acts = [ga.select(i) for i, _ in enumerate(candidates)]
        if "CONFIRM" in cmds and self._grid_can_confirm_without_more_picks():
            acts.append(ga.proceed())
        return acts

    def _grid_can_confirm_without_more_picks(self) -> bool:
        desc = self.select_descriptor()
        if desc is None:
            raise ValueError("GRID confirm is missing its selection descriptor")
        return bool(
            desc["can_pick_zero"]
            or desc["can_pick_any"]
            or int(desc["pick_remaining"]) == 0
        )

    def _grid_required_selection_complete(self) -> bool:
        desc = self.select_descriptor()
        if desc is None:
            raise ValueError("GRID acknowledgement is missing its selection descriptor")
        return int(desc["pick_remaining"]) == 0

    def _legal_reward_actions(self) -> list[tuple]:
        st = self.screen_type()
        ss = _screen_state(self.gs)
        cl = self.choices()
        if st == "CARD_REWARD":
            cards = _screen_cards(ss)
            if not cards:
                raise ValueError("CARD_REWARD is missing structured card records")
            n = len(cards)
            acts = []
            if self._combat_reward_card_open:
                acts.extend(self._outer_reward_claim_actions())
            if n > A.REWARD_CARD:
                raise ValueError(
                    f"card reward capacity {A.REWARD_CARD} exceeded: {n}"
                )
            acts.extend(ga.reward_card(i) for i in range(n))
            if "SKIP" in self.commands():
                acts.append(ga.skip_cards())
            if (self._combat_reward_card_open
                    and "PROCEED" in self._combat_reward_outer_commands):
                acts.append(ga.proceed())
            return acts

        acts = self._outer_reward_claim_actions()
        if not self._skipped_combat_card_reward and any(
                _reward_choice_type(label) == "CARD" for label in cl):
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
        counts = {
            ga.CLAIM_GOLD: 0,
            ga.CLAIM_RELIC: 0,
            ga.CLAIM_POTION: 0,
            ga.CLAIM_KEY: 0,
        }
        for label in labels:
            reward_type = _reward_choice_type(label)
            if reward_type in {"GOLD", "STOLEN_GOLD"}:
                acts.append(ga.claim(ga.CLAIM_GOLD, counts[ga.CLAIM_GOLD]))
                counts[ga.CLAIM_GOLD] += 1
            elif reward_type == "RELIC":
                acts.append(ga.claim(ga.CLAIM_RELIC, counts[ga.CLAIM_RELIC]))
                counts[ga.CLAIM_RELIC] += 1
            elif reward_type == "POTION":
                if "Sozu" not in self._relics() and self._has_empty_potion_slot():
                    acts.append(ga.claim(ga.CLAIM_POTION, counts[ga.CLAIM_POTION]))
                counts[ga.CLAIM_POTION] += 1
            elif reward_type in {"EMERALD_KEY", "SAPPHIRE_KEY"}:
                if counts[ga.CLAIM_KEY]:
                    raise ValueError("multiple simultaneous key reward choices")
                acts.append(ga.claim(ga.CLAIM_KEY, 0))
                counts[ga.CLAIM_KEY] += 1
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
        potions = self._potion_records()
        if len(potions) > A.P_MAX:
            raise ValueError(
                f"potion slot capacity {A.P_MAX} exceeded: {len(potions)}"
            )
        for slot, potion in enumerate(potions):
            pid = _potion_id(potion)
            if pid == "PAD":
                continue
            if not isinstance(potion, dict):
                raise ValueError(f"real potion record must be an object: {potion!r}")
            if not _required_bool(potion, "can_use", context=f"potion slot {slot}"):
                continue
            if combat and _potion_requires_target(potion, pid):
                for target in self._live_monster_indices():
                    acts.append(ga.use_potion(slot, target))
            else:
                acts.append(ga.use_potion(slot, None))
        return acts

    def _live_monster_indices(self) -> list[int]:
        cs = self.gs.get("combat_state")
        if not isinstance(cs, dict):
            raise ValueError("monster targets requested without combat_state")
        monsters = _list_value(cs, "monsters", context="combat_state")
        out: list[int] = []
        for policy_index, real_index in enumerate(self.combat_target_layout()):
            m = monsters[real_index]
            if _monster_is_live(m, context=f"combat monster[{real_index}]"):
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
        monsters = _list_value(cs, "monsters", context="combat_state")
        if len(monsters) <= actions.MAX_MONSTERS:
            return tuple(range(len(monsters)))
        live = tuple(
            index
            for index, monster in enumerate(monsters)
            if _monster_is_live(monster, context=f"combat monster[{index}]")
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
            if self.screen_type() != "COMBAT_REWARD" or action not in self._legal_reward_actions():
                raise ValueError(f"claim is unavailable on the current reward screen: {action!r}")
            idx = self._claim_command_index(action[1], action[2])
            return f"CHOOSE {idx}"
        if verb == ga.REWARD_CARD:
            if self.screen_type() == "COMBAT_REWARD":
                idx = next(
                    (
                        i for i, choice in enumerate(self.choices())
                        if _reward_choice_type(choice) == "CARD"
                    ),
                    None,
                )
                if idx is None:
                    raise ValueError(
                        "combat reward card placeholder is absent from choices"
                    )
                self._combat_reward_card_open = True
                return f"CHOOSE {idx}"
            self._combat_reward_card_open = False
            self._skipped_combat_card_reward = False
            return f"CHOOSE {action[1]}"
        if verb == ga.SKIP_CARDS:
            if self.screen_type() == "CARD_REWARD" and self._combat_reward_card_open:
                self._combat_reward_card_open = False
                bowl = _choice_index(self.choices(), {"bowl"})
                if bowl is not None:
                    return f"CHOOSE {bowl}"
                self._skipped_reward_card_count += 1
                total = sum(
                    _reward_choice_type(choice) == "CARD"
                    for choice in self._combat_reward_outer_choices
                )
                self._skipped_combat_card_reward = self._skipped_reward_card_count >= total
            return "SKIP"
        if verb == ga.SELECT:
            desc = self.select_descriptor()
            if desc is None:
                raise ValueError("selection action is missing its descriptor")
            candidate_index = int(action[1])
            if not 0 <= candidate_index < len(desc["candidates"]):
                raise ValueError(
                    f"selection index {candidate_index} is outside current candidates"
                )
            if desc["candidates"][candidate_index]["card"] is None:
                return "SKIP"
            if (self.gs.get('combat_state') and self.screen_type() == 'GRID'
                    and desc['context'] == 'LIQUID_MEMORIES_POTION'):
                self._begin_combat_grid_receipt(
                    desc, desc['candidates'][candidate_index]['instance'])
            index = desc["candidates"][candidate_index]["native_index"]
            return f"CHOOSE {index}"
        if verb == ga.MULTI_SELECT:
            return "CONFIRM"
        if verb == ga.CAMPFIRE:
            idx = self._campfire_command_index(action[1])
            return f"CHOOSE {idx}"
        if verb == ga.EVENT:
            return f"CHOOSE {self._event_option_mapping()[action[1]][0]}"
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
                self._shop_proceed_sent = False
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

    @staticmethod
    def map_flight_counter(gs):
        relics = _dict_records(_list_value(gs, 'relics', context='map receipt'),
                              context='map receipt relics')
        boots = [r for r in relics if _relic_id(r) == 'WingedGreaves']
        if len(boots) > 1:
            raise ValueError('map receipt has duplicate Wing Boots')
        if not boots:
            return None  # Explicit absence, not a guessed counter for a live relic.
        return native_relic_counter('WingedGreaves',
            _required(boots[0], 'counter', context='Wing Boots map receipt'))

    def map_move_expectation(self, column):
        """Validate the chosen connection and record its visible charge effect."""
        if self.gs.get('simulator_parity_export') != SIMULATOR_PARITY_EXPORT_VERSION:
            raise ValueError('map move acknowledgement requires the current parity exporter')
        ss = _screen_state(self.gs)
        index = self._map_command_index(column)
        counter = self.map_flight_counter(self.gs)
        first = _required(ss, 'first_node_chosen', context='map receipt')
        boss = _required(ss, 'boss_available', context='map receipt')
        if type(first) is not bool or type(boss) is not bool:
            raise ValueError('map receipt needs boolean first-node/boss flags')
        current = _required(ss, 'current_node', context='map receipt')
        current_xy = _node_xy(current)
        if current_xy is None:
            raise ValueError('map receipt needs the explicit current coordinate')
        if not boss:
            nodes = _dict_records(_list_value(ss, 'next_nodes', context='map receipt'),
                                  context='map receipt next nodes')
            target = _node_xy(nodes[index])
            if target is None or target[1] != (current_xy[1] + 1 if first else 0):
                raise ValueError('map destination is not on the next row')
            if first:
                # CommunicationMod current_node/next_nodes only carry x/y/symbol.
                # Edges are exported separately in game_state.map, not there.
                graph = self._full_map_nodes(ss)
                sources = [n for n in graph if (n['x'], n['y']) == current_xy]
                if len(sources) != 1:
                    raise ValueError('map receipt current node is absent from the full map')
                edges = [(x, current_xy[1] + 1) for x in sources[0]['children']]
                if target not in edges:
                    if counter is None or counter <= 0 or not any(y == target[1] for _, y in edges):
                        raise ValueError('disconnected map destination requires a Wing Boots charge')
                    counter -= 1
        expectation = {}
        for field in ('seed', 'act', 'floor'):
            value = _required(self.gs, field, context='map receipt')
            if type(value) is not int:
                raise ValueError(f'map receipt requires integer {field}')
            expectation[field] = value
        expectation['floor'] += 1
        expectation['flight_counter'] = counter
        expectation['position'] = (-1, 15) if boss else target
        return expectation

    def _claim_command_index(self, kind: str, occurrence: int) -> int:
        aliases = {
            ga.CLAIM_GOLD: {"gold", "stolen_gold"},
            ga.CLAIM_RELIC: {"relic"},
            ga.CLAIM_POTION: {"potion"},
            ga.CLAIM_KEY: {"emerald_key", "sapphire_key"},
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
        gold = int(_required(self.gs, "gold", context="game_state"))
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
            items = _dict_records(
                _list_value(ss, key, context="shop screen"),
                context=f"shop {key}",
            )
            for i, item in enumerate(items):
                price = int(_price(item))
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
        raise RuntimeError(
            "policy fallback commands are forbidden; refresh or fail the exact contract"
        )

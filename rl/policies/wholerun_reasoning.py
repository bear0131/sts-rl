"""Anonymous, interactive reasoning protocol for whole-run training.

This module intentionally does not import or initialize the standalone rule
model.  It defines a new token contract for joint action/reasoning training:

* the policy commits an action and the simulator executes it first;
* the decoder then replays the transition autoregressively;
* stochastic values are requested by the decoder and injected by an oracle;
* injected tokens condition later predictions but never receive LM loss.

Canonical card ids are used only by the environment-side compiler to look up a
card program.  They are never serialized into model-visible tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Sequence

from env.core.card_language import (
    LANGUAGE_CARDS,
    TOKEN_VOCAB as CARD_TOKEN_VOCAB,
    card_program,
)


IGNORE_INDEX = -100
# Numeric values are always represented compositionally. Keeping one token per
# literal integer creates a frequency-heavy vocabulary and encourages the model
# to memorize common values instead of learning arithmetic structure.
COMPACT_INTEGER_MIN = -32
COMPACT_INTEGER_MAX = 255
COMPACT_INTEGER_TOKENS: tuple[str, ...] = ()


class TokenOwner(IntEnum):
    """Who supplies a transcript token."""

    ENVIRONMENT = 0
    MODEL = 1


SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "TRACE_BEGIN",
    "TRACE_END",
    "ACTION_BEGIN",
    "ACTION_END",
    "ACTION_SEGMENT",
    "ACTION_OFFSET",
    "CARD_INSTANCE_BEGIN",
    "CARD_INSTANCE_END",
    "CARD_REF",
    "RULE_BEGIN",
    "RULE_END",
    "RUNTIME_COST",
    "CARD_RULE_UNAVAILABLE",
    "EVENT_BEGIN",
    "EVENT_END",
    "FIELD_NAME",
    "FIELD_OWNER",
    "FIELD_STATUS",
    "FIELD_COUNTER",
    "FIELD_SLOT",
    "FIELD_SOURCE",
    "FIELD_DESTINATION",
    "FIELD_CARD",
    "VALUE_AFTER",
    "VALUE_TRUE",
    "VALUE_FALSE",
    "VALUE_NONE",
    "CALC_BEGIN",
    "CALC_END",
    "CALC_INPUT_BEFORE",
    "CALC_INPUT_DELTA",
    "CALC_OP_ADD",
    "CALC_RESULT",
    "QUERY_RANDOM_TARGET",
    "QUERY_DRAW_CARD",
    "QUERY_RANDOM_CARD",
    "QUERY_CREATED_CARD",
    "OBSERVATION_BEGIN",
    "OBSERVATION_END",
    "OBS_RANDOM_TARGET",
    "OBS_DRAW_CARD",
    "OBS_RANDOM_CARD",
    "OBS_CREATED_CARD",
    "TEXT_BEGIN",
    "TEXT_END",
    "NUM_BEGIN",
    "NUM_END",
    "SIGN_NEG",
    *(f"DIGIT_{digit}" for digit in range(10)),
    *(f"BYTE_{value:02X}" for value in range(128)),
)


EVENT_TYPES = (
    "PLAYER_VALUE_CHANGE",
    "MONSTER_VALUE_CHANGE",
    "STATUS_CHANGE",
    "COUNTER_CHANGE",
    "MONSTER_MOVE_CHANGE",
    "MONSTER_GONE_CHANGE",
    "MONSTER_COUNT_CHANGE",
    "CARD_MOVE",
    "CARD_REMOVE",
    "CARD_ADD",
    "TURN_CHANGE",
    "ACTION_COMPLETE",
)


class ReasoningVocab:
    """Closed vocabulary with an ASCII fallback for runtime atoms."""

    def __init__(self) -> None:
        tokens = list(SPECIAL_TOKENS)
        tokens.extend(f"EVENT_{name}" for name in EVENT_TYPES)
        tokens.extend(token for token in CARD_TOKEN_VOCAB if token not in tokens)
        self.tokens = tuple(tokens)
        self.index = {token: idx for idx, token in enumerate(self.tokens)}
        if len(self.index) != len(self.tokens):
            raise ValueError("duplicate interactive-reasoning vocabulary token")

    @property
    def pad_id(self) -> int:
        return self.index["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.index["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.index["<EOS>"]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.index[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.tokens[int(token_id)] for token_id in ids]


VOCAB = ReasoningVocab()


def number_tokens(value: int) -> list[str]:
    """Encode an integer as sign plus decimal digits for every magnitude."""

    value = int(value)
    sign = ["SIGN_NEG"] if value < 0 else []
    return [
        "NUM_BEGIN",
        *sign,
        *(f"DIGIT_{digit}" for digit in str(abs(value))),
        "NUM_END",
    ]


def compact_program_numbers(tokens: Sequence[str]) -> tuple[str, ...]:
    """Compatibility wrapper retaining the compositional numeric program."""

    return tuple(tokens)


def text_tokens(value: Any) -> list[str]:
    """Encode arbitrary runtime labels without extending the vocabulary."""

    raw = str(value).encode("ascii", errors="replace")
    return [
        "TEXT_BEGIN",
        *(f"BYTE_{byte:02X}" for byte in raw),
        "TEXT_END",
    ]


def anonymous_card_tokens(
    card: dict[str, Any] | tuple[str, int] | None,
    *,
    reference: int = 0,
    runtime_cost: int | None = None,
) -> list[str]:
    """Compile a card instance without emitting its canonical id.

    ``reference`` is transition-local.  Callers should freely permute it across
    samples; it is a pointer, not a semantic category.
    """

    if card is None:
        return []
    if isinstance(card, dict):
        card_id = str(card["id"])
        upgrades = int(card.get("up", card.get("upgrades", 0)))
        if runtime_cost is None and "cost" in card:
            runtime_cost = int(card["cost"])
    else:
        card_id, upgrades = str(card[0]), int(card[1])

    out = ["CARD_INSTANCE_BEGIN", "CARD_REF", *number_tokens(reference)]
    out.append("RULE_BEGIN")
    if card_id in LANGUAGE_CARDS:
        out.extend(compact_program_numbers(card_program(card_id, bool(upgrades))))
    else:
        # Missing DSL coverage must not silently reintroduce the card name.
        out.append("CARD_RULE_UNAVAILABLE")
    if runtime_cost is not None:
        out.extend(("RUNTIME_COST", *number_tokens(runtime_cost)))
    out.extend(("RULE_END", "CARD_INSTANCE_END"))
    return out


@dataclass(frozen=True)
class OracleExchange:
    query: str
    response: tuple[str, ...]


@dataclass(frozen=True)
class InteractiveTranscript:
    tokens: tuple[str, ...]
    owners: tuple[TokenOwner, ...]
    exchanges: tuple[OracleExchange, ...] = ()

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.owners):
            raise ValueError("transcript tokens/owners length mismatch")
        if not self.tokens or self.tokens[0] != "<BOS>":
            raise ValueError("interactive transcript must start with <BOS>")
        if self.tokens[-1] != "<EOS>":
            raise ValueError("interactive transcript must end with <EOS>")

    def token_ids(self, vocab: ReasoningVocab = VOCAB) -> list[int]:
        return vocab.encode(self.tokens)

    def teacher_labels(
        self, vocab: ReasoningVocab = VOCAB, *, ignore_index: int = IGNORE_INDEX
    ) -> tuple[list[int], list[int]]:
        """Return decoder inputs and next-token labels.

        Environment-owned next tokens are ignored as targets.  They remain in
        ``inputs`` and therefore condition every subsequent model prediction.
        """

        ids = self.token_ids(vocab)
        inputs = ids[:-1]
        labels = [
            ids[index + 1]
            if self.owners[index + 1] == TokenOwner.MODEL
            else ignore_index
            for index in range(len(ids) - 1)
        ]
        return inputs, labels


class TranscriptBuilder:
    def __init__(self) -> None:
        self.tokens: list[str] = ["<BOS>"]
        self.owners: list[TokenOwner] = [TokenOwner.ENVIRONMENT]
        self.exchanges: list[OracleExchange] = []

    def model(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.owners.extend([TokenOwner.MODEL] * len(tokens))

    def environment(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.owners.extend([TokenOwner.ENVIRONMENT] * len(tokens))

    def exchange(self, query: str, response: Sequence[str]) -> None:
        if not query.startswith("QUERY_"):
            raise ValueError(f"not an oracle query: {query}")
        self.model(query)
        response = tuple(response)
        self.environment(*response)
        self.exchanges.append(OracleExchange(query, response))

    def finish(self) -> InteractiveTranscript:
        self.model("<EOS>")
        return InteractiveTranscript(
            tuple(self.tokens), tuple(self.owners), tuple(self.exchanges)
        )


class InteractiveOracle:
    """Ordered replay oracle used during post-action generation."""

    def __init__(self, exchanges: Sequence[OracleExchange]) -> None:
        self._exchanges = tuple(exchanges)
        self._cursor = 0

    @property
    def complete(self) -> bool:
        return self._cursor == len(self._exchanges)

    def respond(self, query: str) -> tuple[str, ...]:
        if self._cursor >= len(self._exchanges):
            raise RuntimeError(f"unexpected oracle query after trace end: {query}")
        exchange = self._exchanges[self._cursor]
        if query != exchange.query:
            raise RuntimeError(
                f"oracle expected {exchange.query}, model requested {query}"
            )
        self._cursor += 1
        return exchange.response


def _card_key(card: dict[str, Any]) -> tuple[str, int]:
    return str(card["id"]), int(card.get("up", 0))


def _zone_events(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    zones = ("hand", "draw", "discard", "exhaust")
    removed: list[tuple[str, tuple[str, int]]] = []
    added: list[tuple[str, tuple[str, int]]] = []
    for zone in zones:
        left = Counter(map(_card_key, before.get(zone, ())))
        right = Counter(map(_card_key, after.get(zone, ())))
        for card, count in (left - right).items():
            removed.extend((zone, card) for _ in range(count))
        for card, count in (right - left).items():
            added.extend((zone, card) for _ in range(count))

    events: list[dict[str, Any]] = []
    used: set[int] = set()
    for source, card in removed:
        match = next(
            (
                index
                for index, (destination, candidate) in enumerate(added)
                if index not in used and candidate == card
            ),
            None,
        )
        if match is None:
            events.append({
                "type": "CARD_REMOVE",
                "source": source,
                "card": card[0],
                "up": card[1],
            })
            continue
        used.add(match)
        events.append({
            "type": "CARD_MOVE",
            "source": source,
            "destination": added[match][0],
            "card": card[0],
            "up": card[1],
        })
    for index, (destination, card) in enumerate(added):
        if index not in used:
            events.append({
                "type": "CARD_ADD",
                "destination": destination,
                "card": card[0],
                "up": card[1],
            })
    return events


def _status_events(owner: str, before: dict, after: dict) -> list[dict[str, Any]]:
    events = []
    for status in sorted(set(before) | set(after)):
        old, new = int(before.get(status, 0)), int(after.get(status, 0))
        if old != new:
            events.append({
                "type": "STATUS_CHANGE",
                "owner": owner,
                "status": status,
                "before": old,
                "delta": new - old,
                "after": new,
            })
    return events


def diff_combat_states(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one atomic simulator snapshot pair into model events."""

    events: list[dict[str, Any]] = []
    before_player, after_player = before["player"], after["player"]
    for field in ("hp", "max_hp", "block", "energy", "gold"):
        old = int(before_player.get(field, 0))
        new = int(after_player.get(field, 0))
        if old != new:
            events.append({
                "type": "PLAYER_VALUE_CHANGE",
                "field": field,
                "before": old,
                "delta": new - old,
                "after": new,
            })
    events.extend(_status_events(
        "player",
        before_player.get("powers", {}),
        after_player.get("powers", {}),
    ))
    before_counters = before_player.get("combat_counters", {})
    after_counters = after_player.get("combat_counters", {})
    for counter in sorted(set(before_counters) | set(after_counters)):
        old = int(before_counters.get(counter, 0))
        new = int(after_counters.get(counter, 0))
        if old != new:
            events.append({
                "type": "COUNTER_CHANGE",
                "counter": counter,
                "before": old,
                "delta": new - old,
                "after": new,
            })

    before_monsters = before.get("monsters", ())
    after_monsters = after.get("monsters", ())
    for slot, (old_monster, new_monster) in enumerate(
        zip(before_monsters, after_monsters)
    ):
        for field in ("hp", "max_hp", "block"):
            old = int(old_monster.get(field, 0))
            new = int(new_monster.get(field, 0))
            if old != new:
                events.append({
                    "type": "MONSTER_VALUE_CHANGE",
                    "slot": slot,
                    "field": field,
                    "before": old,
                    "delta": new - old,
                    "after": new,
                })
        events.extend(_status_events(
            f"monster_{slot}",
            old_monster.get("powers", {}),
            new_monster.get("powers", {}),
        ))
        if old_monster.get("move") != new_monster.get("move"):
            events.append({
                "type": "MONSTER_MOVE_CHANGE",
                "slot": slot,
                "before_text": old_monster.get("move", "INVALID"),
                "after_text": new_monster.get("move", "INVALID"),
            })
        if bool(old_monster.get("gone")) != bool(new_monster.get("gone")):
            events.append({
                "type": "MONSTER_GONE_CHANGE",
                "slot": slot,
                "after_bool": bool(new_monster.get("gone")),
            })
    if len(before_monsters) != len(after_monsters):
        events.append({
            "type": "MONSTER_COUNT_CHANGE",
            "before": len(before_monsters),
            "delta": len(after_monsters) - len(before_monsters),
            "after": len(after_monsters),
        })
    events.extend(_zone_events(before, after))
    if int(before.get("turn", 0)) != int(after.get("turn", 0)):
        old, new = int(before.get("turn", 0)), int(after.get("turn", 0))
        events.append({
            "type": "TURN_CHANGE", "before": old, "delta": new - old, "after": new,
        })
    return events


def events_from_snapshots(snapshots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not snapshots:
        return [{"type": "ACTION_COMPLETE"}]
    previous = snapshots[0]
    for current in snapshots[1:]:
        events.extend(diff_combat_states(previous, current))
        previous = current
    events.append({"type": "ACTION_COMPLETE"})
    return events


def _numeric_event_tokens(event: dict[str, Any]) -> list[str]:
    """Put the local calculation before the externally visible result."""

    before = int(event["before"])
    delta = int(event["delta"])
    after = int(event["after"])
    return [
        "CALC_BEGIN",
        "CALC_INPUT_BEFORE",
        *number_tokens(before),
        "CALC_INPUT_DELTA",
        *number_tokens(delta),
        "CALC_OP_ADD",
        "CALC_RESULT",
        *number_tokens(after),
        "CALC_END",
        "VALUE_AFTER",
        *number_tokens(after),
    ]


def _event_tokens(event: dict[str, Any]) -> list[str]:
    event_type = str(event["type"])
    out = ["EVENT_BEGIN", f"EVENT_{event_type}"]
    if "field" in event:
        out.extend(("FIELD_NAME", *text_tokens(event["field"])))
    if "owner" in event:
        out.extend(("FIELD_OWNER", *text_tokens(event["owner"])))
    if "status" in event:
        out.extend(("FIELD_STATUS", *text_tokens(event["status"])))
    if "counter" in event:
        out.extend(("FIELD_COUNTER", *text_tokens(event["counter"])))
    if "slot" in event:
        out.extend(("FIELD_SLOT", *number_tokens(event["slot"])))
    if "source" in event:
        out.extend(("FIELD_SOURCE", *text_tokens(event["source"])))
    if "destination" in event:
        out.extend(("FIELD_DESTINATION", *text_tokens(event["destination"])))
    if "card" in event:
        # The event refers to an anonymous runtime card.  Its surface is supplied
        # in an oracle response for stochastic draws/creation, or in the action
        # context for a played card; never serialize the canonical name here.
        out.extend(("FIELD_CARD", "CARD_REF", *number_tokens(event.get("reference", 0))))
    if all(key in event for key in ("before", "delta", "after")):
        out.extend(_numeric_event_tokens(event))
    elif "after_bool" in event:
        out.extend(("VALUE_TRUE" if event["after_bool"] else "VALUE_FALSE",))
    elif "after_text" in event:
        out.extend(("VALUE_AFTER", *text_tokens(event["after_text"])))
    out.append("EVENT_END")
    return out


def _observation(
    kind: str,
    *,
    slot: int | None = None,
    card: dict[str, Any] | tuple[str, int] | None = None,
    reference: int = 0,
) -> list[str]:
    out = ["OBSERVATION_BEGIN", kind]
    if slot is not None:
        out.extend(("FIELD_SLOT", *number_tokens(slot)))
    if card is not None:
        out.extend(anonymous_card_tokens(card, reference=reference))
    out.append("OBSERVATION_END")
    return out


def build_interactive_transcript(
    *,
    events: Sequence[dict[str, Any]],
    action_segment: str,
    action_offset: int = 0,
    action_card: dict[str, Any] | tuple[str, int] | None = None,
    random_target: bool = False,
    random_card_choice: bool = False,
) -> InteractiveTranscript:
    """Build the gold post-action dialogue for one executed transition."""

    builder = TranscriptBuilder()
    builder.model(
        "TRACE_BEGIN",
        "ACTION_BEGIN",
        "ACTION_SEGMENT",
        *text_tokens(action_segment),
        "ACTION_OFFSET",
        *number_tokens(action_offset),
    )
    if action_card is not None:
        builder.model(*anonymous_card_tokens(action_card, reference=0))
    builder.model("ACTION_END")

    next_reference = 1
    for original in events:
        event = dict(original)
        event_type = event.get("type")
        is_draw = (
            event_type == "CARD_MOVE"
            and event.get("source") == "draw"
            and event.get("destination") == "hand"
        )
        is_created = event_type == "CARD_ADD" and "card" in event
        is_random_card = random_card_choice and event_type in ("CARD_MOVE", "CARD_REMOVE")
        is_random_hit = (
            random_target
            and event_type == "MONSTER_VALUE_CHANGE"
            and event.get("field") in ("hp", "block")
        )

        if is_random_hit:
            builder.exchange(
                "QUERY_RANDOM_TARGET",
                _observation("OBS_RANDOM_TARGET", slot=int(event["slot"])),
            )
        if is_draw:
            card = (str(event["card"]), int(event.get("up", 0)))
            event["reference"] = next_reference
            builder.exchange(
                "QUERY_DRAW_CARD",
                _observation(
                    "OBS_DRAW_CARD", card=card, reference=next_reference
                ),
            )
            next_reference += 1
        elif is_created:
            card = (str(event["card"]), int(event.get("up", 0)))
            event["reference"] = next_reference
            builder.exchange(
                "QUERY_CREATED_CARD",
                _observation(
                    "OBS_CREATED_CARD", card=card, reference=next_reference
                ),
            )
            next_reference += 1
        elif is_random_card:
            card = (str(event["card"]), int(event.get("up", 0)))
            event["reference"] = next_reference
            builder.exchange(
                "QUERY_RANDOM_CARD",
                _observation(
                    "OBS_RANDOM_CARD", card=card, reference=next_reference
                ),
            )
            next_reference += 1

        builder.model(*_event_tokens(event))

    builder.model("TRACE_END")
    return builder.finish()


def card_stochastic_flags(card: dict[str, Any] | tuple[str, int] | None) -> dict[str, bool]:
    if card is None:
        return {"random_target": False, "random_card_choice": False}
    if isinstance(card, dict):
        card_id = str(card["id"])
        upgrades = int(card.get("up", card.get("upgrades", 0)))
    else:
        card_id, upgrades = str(card[0]), int(card[1])
    if card_id not in LANGUAGE_CARDS:
        return {"random_target": False, "random_card_choice": False}
    program = set(card_program(card_id, bool(upgrades)))
    return {
        "random_target": "RANDOM_ENEMY" in program,
        "random_card_choice": "RANDOM" in program,
    }


__all__ = [
    "IGNORE_INDEX",
    "COMPACT_INTEGER_MAX",
    "COMPACT_INTEGER_MIN",
    "COMPACT_INTEGER_TOKENS",
    "InteractiveOracle",
    "InteractiveTranscript",
    "OracleExchange",
    "ReasoningVocab",
    "TokenOwner",
    "VOCAB",
    "anonymous_card_tokens",
    "build_interactive_transcript",
    "card_stochastic_flags",
    "diff_combat_states",
    "events_from_snapshots",
    "compact_program_numbers",
    "number_tokens",
    "text_tokens",
]

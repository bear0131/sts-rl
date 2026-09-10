"""Independent v6 data contract and dataset/collation helpers.

The v6 trainer intentionally does not read the legacy canonical rule JSONL
format.  A v6 record is already anonymized at the data boundary: state groups
are fixed numeric vectors, cards/relics are rule-token programs, and the
realized event stream carries its continuous world deltas.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from rl.rule_vector_model import (
    StateVectorConfig,
    SerializedVectorEvents,
    VectorEvent,
    VectorizedState,
    VECTOR_TOKEN_INDEX,
    serialize_vector_events,
)


VECTOR_DATA_SCHEMA_VERSION = 5
SparseDelta = tuple[tuple[int, float], ...]


def _read_json_lines(path: str | Path) -> Iterable[Mapping[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"v6 record at {path}:{line_number} must be an object")
            yield value


def _vector(value: Any, width: int, name: str) -> torch.Tensor:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric list of width {width}")
    if len(value) != width:
        raise ValueError(f"{name} has width {len(value)}; expected {width}")
    try:
        return torch.tensor([float(item) for item in value], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc


def _program(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a token-id list")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} contains a negative token id")
    return result


def _sparse_delta(value: Any, width: int, name: str) -> SparseDelta:
    vector = _vector(value, width, name)
    return tuple(
        (index, float(item))
        for index, item in enumerate(vector.tolist())
        if abs(float(item)) >= 1e-8
    )


def _programs_to_padded(
    programs: Sequence[Sequence[int]], pad_id: int = 0,
) -> torch.Tensor:
    width = max((len(program) for program in programs), default=1)
    result = torch.full((len(programs), width), pad_id, dtype=torch.long)
    for row, program in enumerate(programs):
        if program:
            result[row, :len(program)] = torch.as_tensor(program, dtype=torch.long)
    return result


class VectorRuleV6Example:
    """One validated, anonymized v6 training record."""

    __slots__ = (
        "state", "action_program", "action_upgrade", "hand_programs",
        "hand_upgrades", "relic_programs", "relic_counters", "target_ids",
        "target_delta",
    )

    def __init__(
        self,
        state: VectorizedState,
        action_program: tuple[int, ...],
        action_upgrade: float,
        hand_programs: tuple[tuple[int, ...], ...],
        hand_upgrades: tuple[float, ...],
        relic_programs: tuple[tuple[int, ...], ...],
        relic_counters: tuple[float, ...],
        target_ids: tuple[int, ...],
        target_delta: tuple[SparseDelta | None, ...],
    ) -> None:
        self.state = state
        self.action_program = action_program
        self.action_upgrade = action_upgrade
        self.hand_programs = hand_programs
        self.hand_upgrades = hand_upgrades
        self.relic_programs = relic_programs
        self.relic_counters = relic_counters
        self.target_ids = target_ids
        self.target_delta = target_delta


def example_from_v6_record(
    record: Mapping[str, Any],
    config: StateVectorConfig | None = None,
    *,
    card_vocab_size: int | None = None,
    relic_vocab_size: int | None = None,
    max_cards: int = 10,
    max_relics: int = 32,
    max_tokens: int = 2048,
) -> VectorRuleV6Example:
    """Validate and materialize one independent v6/schema-v5 record.

    Required shape::

        {
          "schema_version": 5,
          "state": {"player": [...], "enemies": [[...]], "piles": [...]},
          "action": {"card_rule": [token ids], "upgrade": 0},
          "hand": [{"card_rule": [...], "upgrade": 0}],
          "relics": [{"rule": [token ids], "counter": 0}],
          "events": [{"kind": ..., "entity_ref": ..., "calculations": [...],
                      "delta": [...]}]
        }

    No card/relic name or pile contents are accepted by this contract.
    """

    config = config or StateVectorConfig()
    if int(record.get("schema_version", -1)) != VECTOR_DATA_SCHEMA_VERSION:
        raise ValueError("record is not a v6 vector-rule record")
    raw_state = record.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("v6 record state must be an object")
    raw_enemies = raw_state.get("enemies")
    if not isinstance(raw_enemies, Sequence) or len(raw_enemies) != config.max_enemies:
        raise ValueError(f"state.enemies must contain {config.max_enemies} fixed slots")
    enemies = torch.stack([
        _vector(row, config.enemy_dim, f"state.enemies[{index}]")
        for index, row in enumerate(raw_enemies)
    ])
    state = VectorizedState(
        player=_vector(raw_state.get("player"), config.player_dim, "state.player"),
        enemies=enemies,
        piles=_vector(raw_state.get("piles"), config.pile_dim, "state.piles"),
    )
    raw_relic_counters = raw_state.get("relic_counters", ())
    if not isinstance(raw_relic_counters, Sequence):
        raise ValueError("state.relic_counters must be a list")

    action = record.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("v6 record action must be an object")
    action_program = _program(action.get("card_rule"), "action.card_rule")
    if card_vocab_size is not None and any(item >= card_vocab_size for item in action_program):
        raise ValueError("action.card_rule contains an out-of-range card token")

    raw_hand = record.get("hand", ())
    if not isinstance(raw_hand, Sequence) or len(raw_hand) > max_cards:
        raise ValueError(f"hand must contain at most {max_cards} rule programs")
    hand_programs: list[tuple[int, ...]] = []
    hand_upgrades: list[float] = []
    for index, item in enumerate(raw_hand):
        if not isinstance(item, Mapping):
            raise ValueError(f"hand[{index}] must be an object")
        program = _program(item.get("card_rule"), f"hand[{index}].card_rule")
        if card_vocab_size is not None and any(token >= card_vocab_size for token in program):
            raise ValueError(f"hand[{index}] contains an out-of-range card token")
        hand_programs.append(program)
        hand_upgrades.append(float(item.get("upgrade", 0)))

    raw_relics = record.get("relics", ())
    if not isinstance(raw_relics, Sequence) or len(raw_relics) > max_relics:
        raise ValueError(f"relics must contain at most {max_relics} rule programs")
    relic_programs: list[tuple[int, ...]] = []
    relic_counters: list[float] = []
    for index, item in enumerate(raw_relics):
        if not isinstance(item, Mapping):
            raise ValueError(f"relics[{index}] must be an object")
        program = _program(item.get("rule"), f"relics[{index}].rule")
        if relic_vocab_size is not None and any(token >= relic_vocab_size for token in program):
            raise ValueError(f"relics[{index}] contains an out-of-range relic token")
        relic_programs.append(program)
        relic_counters.append(float(item.get("counter", 0)))
    if len(raw_relic_counters) not in (0, len(relic_programs)):
        raise ValueError("state.relic_counters must align with relics when supplied")
    if raw_relic_counters:
        relic_counters = [float(value) for value in raw_relic_counters]
    state.relic_counters = torch.tensor(relic_counters, dtype=torch.float32)

    raw_events = record.get("events")
    if not isinstance(raw_events, Sequence) or not raw_events:
        raise ValueError("v6 record events must be a non-empty list")
    events: list[VectorEvent] = []
    sparse_targets: list[SparseDelta | None] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        kind = str(raw_event.get("kind", ""))
        entity_ref = str(raw_event.get("entity_ref", ""))
        delta_value = raw_event.get("delta")
        delta = None if delta_value is None else _vector(
            delta_value, config.world_dim, f"events[{index}].delta"
        )
        raw_calculations = raw_event.get("calculations", ())
        if not isinstance(raw_calculations, (list, tuple)):
            raise ValueError(f"events[{index}].calculations must be a list")
        calculations: list[dict[str, Any]] = []
        for calc_index, calculation in enumerate(raw_calculations):
            if not isinstance(calculation, Mapping):
                raise ValueError(
                    f"events[{index}].calculations[{calc_index}] must be an object"
                )
            calculations.append(dict(calculation))
        events.append(VectorEvent(
            kind, entity_ref,
            delta=None if delta is None else tuple(float(value) for value in delta.tolist()),
            repeat_count=raw_event.get("repeat_count"),
            calculations=tuple(calculations),
        ))
        sparse_targets.append(None if delta is None else tuple(
            (position, float(value))
            for position, value in enumerate(delta.tolist())
            if abs(float(value)) >= 1e-8
        ))
    serialized = serialize_vector_events(events)
    if len(serialized.tokens) > max_tokens:
        raise ValueError(f"event stream has {len(serialized.tokens)} tokens; max is {max_tokens}")
    try:
        target_ids = tuple(VECTOR_TOKEN_INDEX[token] for token in serialized.tokens)
    except KeyError as exc:
        raise ValueError(f"event stream contains an unknown v6 token: {exc.args[0]}") from exc
    # serialize_vector_events may omit a delta for a structural event; use the
    # serializer's alignment rather than assuming one target per event.
    token_deltas: list[SparseDelta | None] = []
    for delta in serialized.delta_targets:
        token_deltas.append(None if delta is None else tuple(
            (position, float(value))
            for position, value in enumerate(delta)
            if abs(float(value)) >= 1e-8
        ))
    return VectorRuleV6Example(
        state=state,
        action_program=action_program,
        action_upgrade=float(action.get("upgrade", 0)),
        hand_programs=tuple(hand_programs),
        hand_upgrades=tuple(hand_upgrades),
        relic_programs=tuple(relic_programs),
        relic_counters=tuple(relic_counters),
        target_ids=target_ids,
        target_delta=tuple(token_deltas),
    )


class VectorRuleV6Dataset(Dataset[VectorRuleV6Example]):
    """Strict dataset for schema-v5 vector-rule records."""

    def __init__(
        self,
        paths_or_records: Iterable[str | Path | Mapping[str, Any] | VectorRuleV6Example],
        config: StateVectorConfig | None = None,
        *,
        card_vocab_size: int | None = None,
        relic_vocab_size: int | None = None,
        max_cards: int = 10,
        max_relics: int = 32,
        max_tokens: int = 2048,
        max_examples: int = 0,
    ) -> None:
        self.examples: list[VectorRuleV6Example] = []
        records: Iterable[Mapping[str, Any]]
        paths = list(paths_or_records)
        if paths and all(isinstance(item, VectorRuleV6Example) for item in paths):
            self.examples = list(paths)
            if max_examples:
                self.examples = self.examples[:max_examples]
            return
        if paths and all(isinstance(item, (str, Path)) for item in paths):
            def read_all():
                for path in paths:
                    yield from _read_json_lines(path)
            records = read_all()
        else:
            records = [item for item in paths if isinstance(item, Mapping)]
        for record in records:
            if max_examples and len(self.examples) >= max_examples:
                break
            self.examples.append(example_from_v6_record(
                record, config,
                card_vocab_size=card_vocab_size,
                relic_vocab_size=relic_vocab_size,
                max_cards=max_cards,
                max_relics=max_relics,
                max_tokens=max_tokens,
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> VectorRuleV6Example:
        return self.examples[index]


def collate_vector_rule_v6_batch(
    batch: Sequence[VectorRuleV6Example],
    *,
    pad_id: int = 0,
) -> dict[str, Any]:
    """Pad programs, hand/relic sets, targets and sparse delta supervision."""

    if not batch:
        raise ValueError("cannot collate an empty v6 vector-rule batch")
    batch_size = len(batch)
    target_width = max(len(example.target_ids) for example in batch) - 1
    world_width = batch[0].state.flat.numel()
    state = VectorizedState(
        player=torch.stack([example.state.player for example in batch]),
        enemies=torch.stack([example.state.enemies for example in batch]),
        piles=torch.stack([example.state.piles for example in batch]),
    )
    target_input = torch.full((batch_size, target_width), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, target_width), pad_id, dtype=torch.long)
    delta_targets = torch.zeros(batch_size, target_width, world_width)
    delta_mask = torch.zeros(batch_size, target_width, dtype=torch.bool)
    for row, example in enumerate(batch):
        ids = torch.as_tensor(example.target_ids, dtype=torch.long)
        target_input[row, :len(ids) - 1] = ids[:-1]
        labels[row, :len(ids) - 1] = ids[1:]
        for position, sparse_delta in enumerate(example.target_delta[1:]):
            if sparse_delta is None:
                continue
            if sparse_delta:
                indices, values = zip(*sparse_delta)
                delta_targets[row, position, list(indices)] = torch.as_tensor(values)
            delta_mask[row, position] = True

    action_programs = _programs_to_padded([example.action_program for example in batch], pad_id)
    action_upgrades = torch.tensor([example.action_upgrade for example in batch], dtype=torch.float32)
    max_hand = max((len(example.hand_programs) for example in batch), default=0)
    max_hand_tokens = max((len(program) for example in batch for program in example.hand_programs), default=1)
    hand_programs = torch.full((batch_size, max_hand, max_hand_tokens), pad_id, dtype=torch.long)
    hand_upgrades = torch.zeros(batch_size, max_hand, dtype=torch.float32)
    hand_mask = torch.zeros(batch_size, max_hand, dtype=torch.bool)
    for row, example in enumerate(batch):
        for slot, program in enumerate(example.hand_programs):
            hand_programs[row, slot, :len(program)] = torch.as_tensor(program)
            hand_upgrades[row, slot] = example.hand_upgrades[slot]
            hand_mask[row, slot] = True

    max_relics = max((len(example.relic_programs) for example in batch), default=0)
    max_relic_tokens = max((len(program) for example in batch for program in example.relic_programs), default=1)
    relic_programs = torch.full((batch_size, max_relics, max_relic_tokens), pad_id, dtype=torch.long)
    relic_counters = torch.zeros(batch_size, max_relics, dtype=torch.float32)
    relic_mask = torch.zeros(batch_size, max_relics, dtype=torch.bool)
    for row, example in enumerate(batch):
        for slot, program in enumerate(example.relic_programs):
            relic_programs[row, slot, :len(program)] = torch.as_tensor(program)
            relic_counters[row, slot] = example.relic_counters[slot]
            relic_mask[row, slot] = True
    state.relic_counters = relic_counters
    return {
        "state": state,
        "target_input_ids": target_input,
        "labels": labels,
        "delta_targets": delta_targets,
        "delta_mask": delta_mask,
        "action_programs": action_programs,
        "action_upgrades": action_upgrades,
        "hand_programs": hand_programs,
        "hand_upgrades": hand_upgrades,
        "hand_mask": hand_mask,
        "relic_programs": relic_programs,
        "relic_counters": relic_counters,
        "relic_mask": relic_mask,
    }


__all__ = [
    "VECTOR_DATA_SCHEMA_VERSION", "VectorRuleV6Example", "VectorRuleV6Dataset",
    "example_from_v6_record", "collate_vector_rule_v6_batch",
]

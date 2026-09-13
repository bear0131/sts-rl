"""Schema-v7 effect-tree records and collation helpers.

The v7 producer deliberately keeps the readable tree tokens as the language
surface. Executor contract 10 assigns every local token to either the model or
the environment. The model predicts read requests and symbolic operations;
after any already-valued register is referenced as an operand, the environment
inserts its exact value inline as a masked ``CONST_VALUE`` expansion. A
continuous delta is emitted at each ``DELTA_SLOT``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from rl.rules.rule_vector_model import (
    StateVectorConfig,
    VectorizedState,
    VECTOR_ENTITY_TOKENS,
    VECTOR_RELIC_ENTITY_TOKENS,
    VECTOR_RUNTIME_ENTITY_VALUE_TOKENS,
    VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS,
    VECTOR_EXECUTOR_VALUE_TOKENS,
    VECTOR_V7_PROTOCOL_TOKENS,
    VECTOR_TOKEN_VOCAB,
    VECTOR_CALCULATION_VALUE_TOKENS,
)
from rl.rules.rule_vector_model import _calculation_atom, _calculation_number


V7_DATA_SCHEMA_VERSION = 7
V7_EXECUTOR_CONTRACT_VERSION = 10


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
                raise ValueError(f"v7 record at {path}:{line_number} must be an object")
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


class V7TokenCodec:
    """A run-local exact-token vocabulary for the v7 readable language.

    The tree serializer has a bounded grammar but carries human-readable
    numeric attributes in a single token.  Building the vocabulary from a
    small bootstrap stream preserves those attributes without forcing the
    outer model to reread card rules.  Tokens not seen during bootstrap map to
    ``<UNK>`` and are counted by the trainer.
    """

    pad_token = "<PAD>"
    unknown_token = "<UNK>"

    def __init__(self, tokens: Sequence[str]) -> None:
        ordered: list[str] = [self.pad_token, self.unknown_token]
        seen = set(ordered)
        for token in tokens:
            token = str(token)
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self.tokens = tuple(ordered)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.unknown_id = self.token_to_id[self.unknown_token]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "V7TokenCodec":
        tokens: list[str] = []
        for record in records:
            if int(record.get("schema_version", -1)) != V7_DATA_SCHEMA_VERSION:
                raise ValueError("vocabulary source contains a non-v7 record")
            if int(record.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
                raise ValueError("vocabulary source uses an incompatible executor contract")
            raw_tokens = record.get("tokens")
            if not isinstance(raw_tokens, Sequence):
                raise ValueError("v7 record tokens must be a list")
            tokens.extend(str(token) for token in raw_tokens)
            local_rows = record.get("node_local_sequences", ())
            if not isinstance(local_rows, Sequence):
                raise ValueError("v7 node_local_sequences must be a list")
            for row in local_rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("tokens"), Sequence):
                    raise ValueError("v7 local sequence must contain a token list")
                tokens.extend(str(token) for token in row["tokens"])
        if not tokens:
            raise ValueError("cannot build a v7 vocabulary from no tokens")
        # Calculation field values are a closed schema vocabulary.  Include
        # every player/enemy/pile field even when a rare slot is absent from
        # the finite bootstrap sample.
        tokens.extend(VECTOR_CALCULATION_VALUE_TOKENS)
        # Entity and runtime binding slots are also closed-schema values. A
        # bootstrap sample must not make a legal enemy/relic slot an <UNK>.
        tokens.extend(VECTOR_ENTITY_TOKENS)
        tokens.extend(VECTOR_RELIC_ENTITY_TOKENS)
        tokens.extend(VECTOR_RUNTIME_ENTITY_VALUE_TOKENS)
        tokens.extend(VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS)
        tokens.extend(VECTOR_EXECUTOR_VALUE_TOKENS)
        tokens.extend(VECTOR_V7_PROTOCOL_TOKENS)
        tokens.extend(VECTOR_TOKEN_VOCAB)
        return cls(tokens)

    @classmethod
    def from_jsonl(cls, paths: Iterable[str | Path]) -> "V7TokenCodec":
        records = (record for path in paths for record in _read_json_lines(path))
        return cls.from_records(records)

    def encode(self, tokens: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.token_to_id.get(str(token), self.unknown_id) for token in tokens)

    def decode(self, ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.tokens[int(index)] for index in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({
                "schema_version": V7_DATA_SCHEMA_VERSION,
                "executor_contract_version": V7_EXECUTOR_CONTRACT_VERSION,
                "tokens": self.tokens,
            },
                       indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "V7TokenCodec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != V7_DATA_SCHEMA_VERSION:
            raise ValueError("tokenizer is not a schema-v7 tokenizer")
        if int(payload.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
            raise ValueError("tokenizer uses an incompatible executor contract")
        # Merge the closed protocol vocabulary on load as well as on bootstrap
        # construction, so an older tokenizer cannot turn a legal
        # dynamic slot or executor operation into <UNK>.
        return cls((*payload.get("tokens", ()), *VECTOR_TOKEN_VOCAB,
                    *VECTOR_V7_PROTOCOL_TOKENS,
                    *VECTOR_RUNTIME_ENTITY_VALUE_TOKENS,
                    *VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS))


@dataclass(frozen=True)
class VectorRuleV7Example:
    state: VectorizedState
    action_program: tuple[int, ...]
    action_upgrade: float
    hand_programs: tuple[tuple[int, ...], ...]
    hand_upgrades: tuple[float, ...]
    relic_programs: tuple[tuple[int, ...], ...]
    relic_counters: tuple[float, ...]
    target_ids: tuple[int, ...]
    target_delta: tuple[tuple[float, ...] | None, ...]
    unknown_tokens: int = 0
    scope_kind: str = "FULL_TREE"
    target_model_mask: tuple[bool, ...] = ()
    # Number of leading tokens supplied as immutable decoder context.  The
    # first supervised label is the token immediately after this prefix.
    condition_length: int = 1


def example_from_v7_record(
    record: Mapping[str, Any],
    codec: V7TokenCodec,
    config: StateVectorConfig | None = None,
    *,
    card_vocab_size: int | None = None,
    relic_vocab_size: int | None = None,
    max_cards: int = 10,
    max_relics: int = 32,
    max_tokens: int | None = 2048,
) -> VectorRuleV7Example:
    config = config or StateVectorConfig()
    if int(record.get("schema_version", -1)) != V7_DATA_SCHEMA_VERSION:
        raise ValueError("record is not a schema-v7 vector-rule record")
    if int(record.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
        raise ValueError("record uses an incompatible executor contract")
    raw_state = record.get("state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("v7 record state must be an object")
    raw_presence = raw_state.get("relic_presence")
    if raw_presence is not None:
        if not isinstance(raw_presence, Sequence) or len(raw_presence) != len(config.relic_presence_names):
            raise ValueError("state.relic_presence must match the stable relic vocabulary")
    raw_enemies = raw_state.get("enemies")
    if not isinstance(raw_enemies, Sequence) or len(raw_enemies) != config.max_enemies:
        raise ValueError(f"state.enemies must contain {config.max_enemies} fixed slots")
    state = VectorizedState(
        player=_vector(raw_state.get("player"), config.player_dim, "state.player"),
        enemies=torch.stack([
            _vector(row, config.enemy_dim, f"state.enemies[{index}]")
            for index, row in enumerate(raw_enemies)
        ]),
        piles=_vector(raw_state.get("piles"), config.pile_dim, "state.piles"),
    )
    raw_relics = record.get("relics", ())
    if not isinstance(raw_relics, Sequence) or len(raw_relics) > max_relics:
        raise ValueError(f"relics must contain at most {max_relics} presence slots")
    relic_programs: list[tuple[int, ...]] = []
    relic_counters: list[float] = []
    for index, item in enumerate(raw_relics):
        if not isinstance(item, Mapping):
            raise ValueError(f"relics[{index}] must be an object")
        # v7 relics are locked presence features.  No relic rule program or
        # exact counter is exposed to the model; counter values are requested
        # by a RELIC_COUNTER node from the environment when needed.
        if "rule" in item and item.get("rule") is not None:
            # Accept old records for inspection, but do not feed their rule
            # tokens into the v7 model.
            program = _program(item.get("rule"), f"relics[{index}].rule")
            if relic_vocab_size is not None and any(token >= relic_vocab_size for token in program):
                raise ValueError(f"relics[{index}] contains an out-of-range relic token")
    raw_counters = raw_state.get("relic_counters", ())
    if raw_counters and not isinstance(raw_counters, Sequence):
        raise ValueError("state.relic_counters must be a list")
    # ``relic_presence`` is the v7 spelling.  ``relic_counters`` is accepted
    # only for backward-compatible records and is never used as a counter input.
    state.relic_counters = torch.empty(0, dtype=torch.float32)

    action = record.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("v7 record action must be an object")
    action_program = _program(action.get("card_rule"), "action.card_rule")
    if card_vocab_size is not None and any(token >= card_vocab_size for token in action_program):
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

    raw_tokens = record.get("tokens")
    if not isinstance(raw_tokens, Sequence) or not raw_tokens:
        raise ValueError("v7 record tokens must be a non-empty list")
    tokens = tuple(str(token) for token in raw_tokens)
    if max_tokens is not None and len(tokens) > max_tokens:
        raise ValueError(f"v7 tree has {len(tokens)} tokens; max is {max_tokens}")
    target_delta: list[tuple[float, ...] | None] = []
    unknown_tokens = 0
    for token in tokens:
        target_delta.append(None)
        if token not in codec.token_to_id:
            unknown_tokens += 1
    target_ids = codec.encode(tokens)
    return VectorRuleV7Example(
        state=state,
        action_program=action_program,
        action_upgrade=float(action.get("upgrade", 0)),
        hand_programs=tuple(hand_programs),
        hand_upgrades=tuple(hand_upgrades),
        relic_programs=tuple(relic_programs),
        relic_counters=tuple(relic_counters),
        target_ids=target_ids,
        target_delta=tuple(target_delta),
        unknown_tokens=unknown_tokens,
        target_model_mask=tuple(True for _ in tokens),
    )


def _header_tokens(node: Mapping[str, Any]) -> list[str]:
    tokens = ["<REGION_BEGIN>", "REGION_KIND", f"REGION_{node['kind']}"]
    entity_ref = node.get("entity_ref")
    if entity_ref is not None:
        tokens.extend(("ENTITY_REF", str(entity_ref)))
    params = node.get("header_params", {})
    if not isinstance(params, Mapping):
        raise ValueError("effect_tree header_params must be an object")
    for key, value in params.items():
        tokens.append(str(key))
        if isinstance(value, bool):
            tokens.append(f"BOOL_{int(value)}")
        elif isinstance(value, (int, float)):
            tokens.extend(_calculation_number(value))
        else:
            tokens.append(f"VALUE_{_calculation_atom(value)}")
    return tokens


def _outer_tree_tokens(node: Mapping[str, Any]) -> list[str]:
    """Serialize only headers and region nesting for the outer decoder."""

    children = node.get("children", ())
    if not isinstance(children, Sequence):
        raise ValueError("effect_tree children must be a list")
    tokens = _header_tokens(node)
    for child in children:
        if not isinstance(child, Mapping):
            raise ValueError("effect_tree child must be an object")
        tokens.extend(_outer_tree_tokens(child))
    tokens.append("<REGION_END>")
    return tokens


def _state_from_flat(flat: torch.Tensor, template: VectorizedState) -> VectorizedState:
    player_width = template.player.numel()
    enemy_width = template.enemies.numel()
    player = flat[:player_width].clone()
    enemies = flat[player_width:player_width + enemy_width].reshape_as(template.enemies).clone()
    piles = flat[player_width + enemy_width:].clone()
    counters = template.relic_counters.clone() if template.relic_counters is not None else None
    return VectorizedState(player=player, enemies=enemies, piles=piles, relic_counters=counters)


def _walk_with_snapshots(
    node: Mapping[str, Any],
    state: VectorizedState,
    path: str = "root",
) -> Iterable[tuple[str, Mapping[str, Any], VectorizedState]]:
    """Walk execution order and expose a pre-node world snapshot.

    A repeat body is represented once in v7, so its static delta is applied
    once while constructing local examples.  Runtime recursive invocations
    still refresh their world vectors in the executor.
    """

    yield path, node, state
    current = state.flat.clone()
    delta = node.get("delta")
    if delta is not None:
        values = torch.as_tensor([float(item) for item in delta], dtype=current.dtype)
        if values.numel() != current.numel():
            raise ValueError("effect_tree delta width does not match world state")
        current = current + values
    child_state = _state_from_flat(current, state)
    children = node.get("children", ())
    if not isinstance(children, Sequence):
        raise ValueError("effect_tree children must be a list")
    for index, child in enumerate(children):
        if not isinstance(child, Mapping):
            raise ValueError("effect_tree child must be an object")
        yield from _walk_with_snapshots(child, child_state, f"{path}/{index}")
        # Sibling leaves execute sequentially.  Reconstruct the next sibling
        # from its own delta when available; nested containers are handled by
        # the recursive call and are intentionally represented once.
        child_delta = child.get("delta")
        if child_delta is not None:
            child_state = _state_from_flat(
                child_state.flat + torch.as_tensor(child_delta, dtype=child_state.flat.dtype),
                child_state,
            )


def _local_target(
    node: Mapping[str, Any], codec: V7TokenCodec,
) -> tuple[
    tuple[int, ...], tuple[tuple[float, ...] | None, ...], int, int,
    tuple[bool, ...],
]:
    rows = node.get("_local_tokens")
    if not isinstance(rows, Sequence):
        raise ValueError("internal local token rows are missing")
    tokens = tuple(str(token) for token in rows)
    try:
        execute_index = tokens.index("<NODE_EXECUTE>")
    except ValueError as exc:
        raise ValueError("local sequence is missing <NODE_EXECUTE>") from exc
    condition_length = execute_index + 1
    if condition_length >= len(tokens):
        raise ValueError("local sequence has no executor target body")
    raw_model_mask = node.get("_model_mask")
    if not isinstance(raw_model_mask, Sequence) or len(raw_model_mask) != len(tokens):
        raise ValueError("local model_mask must align with local tokens")
    model_mask = tuple(bool(value) for value in raw_model_mask)
    if any(model_mask[:condition_length]):
        raise ValueError("NodeSpec condition prefix must be environment-owned")
    targets: list[tuple[float, ...] | None] = [None] * len(tokens)
    raw_delta = node.get("delta")
    raw_delta_slots = node.get("delta_slots")
    if raw_delta_slots is None:
        # Contract-v7 records produced before sparse slots used one aggregate
        # delta. Keep the reader compatible while all new data uses slots.
        raw_delta_slots = [raw_delta] if raw_delta is not None else []
    if not isinstance(raw_delta_slots, Sequence):
        raise ValueError("node delta_slots must be a list")
    delta_positions = [
        index for index, token in enumerate(tokens) if token == "DELTA_SLOT"
    ]
    if len(delta_positions) != len(raw_delta_slots):
        raise ValueError(
            "DELTA_SLOT count must equal the number of independent node delta slots"
        )
    for position, delta in zip(delta_positions, raw_delta_slots):
        if not isinstance(delta, Sequence):
            raise ValueError("each node delta slot must be a numeric vector")
        values = tuple(float(value) for value in delta)
        if not values:
            raise ValueError("each node delta slot must be non-empty")
        targets[position] = values
    unknown = 0
    for token in tokens:
        unknown += int(token not in codec.token_to_id)
    return codec.encode(tokens), tuple(targets), unknown, condition_length, model_mask


def examples_from_v7_record(
    record: Mapping[str, Any],
    codec: V7TokenCodec,
    config: StateVectorConfig | None = None,
    *,
    card_vocab_size: int | None = None,
    relic_vocab_size: int | None = None,
    max_cards: int = 10,
    max_relics: int = 32,
    max_tokens: int = 2048,
) -> tuple[VectorRuleV7Example, ...]:
    """Return one outer-header example plus one example per local node.

    The record is consumed once, but the same shared decoder is trained on its
    short outer program and each independent node body.  Local examples carry
    no action/hand/relic rule programs; their headers already contain the
    extracted typed parameters.
    """

    config = config or StateVectorConfig()
    full = example_from_v7_record(
        record, codec, config,
        card_vocab_size=card_vocab_size, relic_vocab_size=relic_vocab_size,
        max_cards=max_cards, max_relics=max_relics, max_tokens=None,
    )
    root = record.get("effect_tree")
    if not isinstance(root, Mapping):
        raise ValueError("v7 record effect_tree must be an object")

    raw_outer_tokens = record.get("outer_tokens")
    if raw_outer_tokens is None:
        outer_tokens = tuple(
            ["<OUTPUT_BEGIN>", "<TRANSITION_BEGIN>", "<EFFECT_TREE_BEGIN>"]
            + _outer_tree_tokens(root)
            + ["<EFFECT_TREE_END>", "<TRANSITION_END>", "<OUTPUT_END>"]
        )
        outer_model_mask = tuple(True for _ in outer_tokens)
    else:
        if not isinstance(raw_outer_tokens, Sequence):
            raise ValueError("outer_tokens must be a list")
        outer_tokens = tuple(str(token) for token in raw_outer_tokens)
        raw_outer_mask = record.get("outer_model_mask")
        if not isinstance(raw_outer_mask, Sequence) or len(raw_outer_mask) != len(outer_tokens):
            raise ValueError("outer_model_mask must align with outer_tokens")
        outer_model_mask = tuple(bool(value) for value in raw_outer_mask)
    if len(outer_tokens) > max_tokens:
        raise ValueError(f"v7 outer header has {len(outer_tokens)} tokens; max is {max_tokens}")
    outer_ids = codec.encode(outer_tokens)
    outer = replace(
        full, target_ids=outer_ids,
        target_delta=tuple(None for _ in outer_tokens),
        target_model_mask=outer_model_mask,
        unknown_tokens=sum(token not in codec.token_to_id for token in outer_tokens),
        scope_kind="OUTER_HEADERS",
    )
    result: list[VectorRuleV7Example] = [outer]
    for path, node, snapshot in _walk_with_snapshots(root, full.state):
        local_row = next(
            (row for row in record.get("node_local_sequences", ())
             if isinstance(row, Mapping) and row.get("path") == path),
            None,
        )
        if local_row is None:
            raise ValueError(f"missing node_local_sequences row for {path}")
        local_node = dict(node)
        local_node["_local_tokens"] = local_row.get("tokens")
        local_node["_model_mask"] = local_row.get("model_mask")
        ids, deltas, unknown, condition_length, model_mask = _local_target(
            local_node, codec
        )
        if len(ids) > max_tokens:
            raise ValueError(
                f"v7 local node {path} has {len(ids)} tokens; max is {max_tokens}"
            )
        local_state = VectorizedState(
            player=snapshot.player, enemies=snapshot.enemies, piles=snapshot.piles,
            relic_counters=torch.empty(0, dtype=torch.float32),
        )
        result.append(VectorRuleV7Example(
            state=local_state,
            action_program=(0,), action_upgrade=0.0,
            hand_programs=(), hand_upgrades=(), relic_programs=(), relic_counters=(),
            target_ids=ids, target_delta=deltas, unknown_tokens=unknown,
            scope_kind=f"LOCAL_{node['kind']}",
            condition_length=condition_length,
            target_model_mask=model_mask,
        ))
    return tuple(result)


def _programs_to_padded(programs: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    width = max((len(program) for program in programs), default=1)
    result = torch.full((len(programs), width), pad_id, dtype=torch.long)
    for row, program in enumerate(programs):
        if program:
            result[row, :len(program)] = torch.as_tensor(program, dtype=torch.long)
    return result


def collate_vector_rule_v7_batch(
    batch: Sequence[VectorRuleV7Example], *, pad_id: int = 0,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty v7 batch")
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
        width = len(ids) - 1
        target_input[row, :width] = ids[:-1]
        model_mask = example.target_model_mask or tuple(True for _ in example.target_ids)
        if len(model_mask) != len(example.target_ids):
            raise ValueError("target_model_mask must align with target_ids")
        label_mask = torch.as_tensor(model_mask[1:], dtype=torch.bool)
        labels[row, :width][label_mask] = ids[1:][label_mask]
        for position, delta in enumerate(example.target_delta[1:]):
            if delta is None:
                continue
            delta_targets[row, position] = torch.as_tensor(delta, dtype=torch.float32)
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
        "unknown_tokens": sum(example.unknown_tokens for example in batch),
    }


__all__ = [
    "V7_DATA_SCHEMA_VERSION", "V7_EXECUTOR_CONTRACT_VERSION",
    "V7TokenCodec", "VectorRuleV7Example",
    "example_from_v7_record", "collate_vector_rule_v7_batch",
]

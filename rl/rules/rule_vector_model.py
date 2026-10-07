"""Vectorized anonymous rule model (standalone rule-model v6).

The v5 rule model serializes every scalar field as a token.  This module keeps
the old implementation intact and provides the next contract:

* card and relic semantics come from their rule programs, never an id table;
* combat state is a fixed vector for the player and one vector per enemy slot;
* hand changes use a permutation-invariant Transformer over anonymous card-rule vectors;
* transitions are event records whose numeric payload is a continuous delta and
  whose arithmetic is an explicit v5-style calculation trace;
* stochastic operations are explicit model queries followed by environment
  observations that remain in the decoder context; numeric runtime reads use
  inline masked ``CONST_VALUE`` expansions instead.

The module is deliberately independent of the simulator.  A simulator adapter
only needs to provide the canonical dictionaries already used by ``rule_data``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from env.core import vocab as V
from env.core.card_language import (
    TOKEN_VOCAB as CARD_RULE_VOCAB,
    encoded_card_version_programs,
)
from env.core.cards_vocab import CARD_VOCAB
from env.core.rule_language import COMBAT_RELIC_RULES
from env.core.encode import COMBAT_COUNTER_KEYS


# Bumped when the vector/event contract changes. v1 was the initial state
# vector; v2 added the anonymous hand-composition entity; v3 keeps only the
# three combat-pile counts in the default prediction layout; v4 adds the
# example/event vocabulary used by the count-only trace format; v5 adds
# per-event calculation traces and removes source-rule tokens.
RULE_VECTOR_SCHEMA_VERSION = 5


# ---------------------------------------------------------------------------
# Rule-program encoders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramEncoderConfig:
    """Capacity and architecture for one rule-program encoder."""

    token_vocab_size: int
    pad_id: int = 0
    token_dim: int = 64
    output_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward: int = 256
    max_tokens: int = 256
    dropout: float = 0.0


class RuleProgramTransformer(nn.Module):
    """Transformer backbone shared by anonymous rule-program encoders.

    Card and relic programs intentionally keep separate token embeddings because
    their vocabularies are different.  The contextual Transformer itself can be
    shared, which lets both rule domains train the same compositional features.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        n_heads: int,
        n_layers: int,
        feedforward: int,
        max_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % n_heads:
            raise ValueError("token_dim must be divisible by n_heads")
        if n_layers < 1 or max_tokens < 1:
            raise ValueError("rule Transformer layers and capacity must be positive")
        self.token_dim = token_dim
        self.max_tokens = max_tokens
        self.position = nn.Embedding(max_tokens, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(token_dim)

    def contextualize(
        self, hidden: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        hidden = hidden + self.position(positions)[None]
        # Multi-head attention returns NaN when every position is masked.  Keep
        # one harmless placeholder visible, then pooling below zeros the result.
        encoder_padding = padding.clone()
        all_padding = encoder_padding.all(dim=1)
        if bool(all_padding.any()):
            encoder_padding[all_padding, 0] = False
        return self.encoder(hidden, src_key_padding_mask=encoder_padding)


class RuleProgramEncoder(nn.Module):
    """Encode a padded program to one vector using only program tokens.

    There is intentionally no id embedding.  ``forward`` accepts token ids;
    card/relic ids are used only by the environment-side convenience methods
    below to look up the corresponding rule program.
    """

    def __init__(
        self,
        config: ProgramEncoderConfig,
        *,
        backbone: RuleProgramTransformer | None = None,
        register_backbone: bool = True,
    ) -> None:
        super().__init__()
        if config.token_dim % config.n_heads:
            raise ValueError("token_dim must be divisible by n_heads")
        if config.max_tokens < 1 or config.output_dim < 1:
            raise ValueError("program capacity and output_dim must be positive")
        self.config = config
        self.token_embedding = nn.Embedding(
            config.token_vocab_size, config.token_dim, padding_idx=config.pad_id
        )
        if backbone is not None and not register_backbone:
            object.__setattr__(self, "backbone", backbone)
        else:
            self.backbone = backbone
        if backbone is not None:
            if self.backbone.token_dim != config.token_dim:
                raise ValueError("shared rule Transformer token_dim does not match config")
            if self.backbone.max_tokens < config.max_tokens:
                raise ValueError("shared rule Transformer max_tokens is smaller than config")
            # Shared modules are intentionally registered under the backbone;
            # these aliases are for compatibility with diagnostics.
            # Bypass Module.__setattr__ so aliases do not duplicate shared
            # tensors under card_encoder.encoder/relic_encoder.encoder paths
            # in every checkpoint.
            object.__setattr__(self, "position", self.backbone.position)
            object.__setattr__(self, "encoder", self.backbone.encoder)
            object.__setattr__(self, "norm", self.backbone.norm)
        else:
            # Keep the independent layout byte-for-byte compatible with the
            # pre-sharing checkpoints.
            self.position = nn.Embedding(config.max_tokens, config.token_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=config.token_dim,
                nhead=config.n_heads,
                dim_feedforward=config.feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=config.n_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(config.token_dim)
        self.projection = nn.Linear(config.token_dim, config.output_dim)

    def forward(
        self, program_ids: torch.Tensor, *, padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        if program_ids.ndim != 2:
            raise ValueError("program_ids must have shape [batch, length]")
        batch, length = program_ids.shape
        if length > self.config.max_tokens:
            raise ValueError(
                f"program has {length} tokens; max is {self.config.max_tokens}"
            )
        if padding is None:
            padding = program_ids.eq(self.config.pad_id)
        if padding.shape != program_ids.shape:
            raise ValueError("padding must have the same shape as program_ids")
        hidden = self.token_embedding(program_ids)
        if self.backbone is not None:
            hidden = self.backbone.contextualize(hidden, padding)
        else:
            positions = torch.arange(length, device=program_ids.device)
            hidden = hidden + self.position(positions)[None]
            encoder_padding = padding.clone()
            all_padding = encoder_padding.all(dim=1)
            if bool(all_padding.any()):
                encoder_padding[all_padding, 0] = False
            hidden = self.encoder(hidden, src_key_padding_mask=encoder_padding)
        valid = (~padding).unsqueeze(-1)
        pooled = (self.norm(hidden) * valid).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp_min(1)
        pooled = self.projection(pooled)
        # An all-PAD program represents an unavailable rule, not a learned bias.
        return pooled * (~padding).any(dim=1, keepdim=True).to(pooled.dtype)

    def encode_programs(self, programs: Sequence[Sequence[int]], *, device=None):
        if not programs:
            return torch.empty((0, self.config.output_dim), device=device)
        width = max(len(row) for row in programs)
        if width > self.config.max_tokens:
            raise ValueError("program batch exceeds configured max_tokens")
        ids = torch.full(
            (len(programs), width), self.config.pad_id, dtype=torch.long, device=device
        )
        for row, program in enumerate(programs):
            if program:
                ids[row, :len(program)] = torch.as_tensor(
                    program, dtype=torch.long, device=ids.device
                )
        return self(ids)


class CardRuleEncoder(RuleProgramEncoder):
    """Name-free card rule encoder backed by the canonical card DSL."""

    def __init__(
        self,
        config: ProgramEncoderConfig,
        *,
        backbone: RuleProgramTransformer | None = None,
        register_backbone: bool = True,
    ) -> None:
        if config.token_vocab_size != len(CARD_RULE_VOCAB):
            raise ValueError(
                "card encoder token_vocab_size must match CARD_RULE_VOCAB"
            )
        super().__init__(
            config, backbone=backbone, register_backbone=register_backbone
        )
        programs = torch.as_tensor(encoded_card_version_programs(), dtype=torch.long)
        self.register_buffer("version_program_ids", programs, persistent=False)

    def forward_programs(self, program_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(program_ids)

    def card_vectors(
        self, card_ids: torch.Tensor, upgrades: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return vectors by DSL lookup; no card-id embedding participates."""

        ids = card_ids.long()
        if bool((ids < 0).any()) or bool((ids >= self.version_program_ids.shape[0]).any()):
            raise ValueError("card id is outside the canonical card vocabulary")
        if upgrades is None:
            upgrades = torch.zeros_like(ids)
        version = upgrades.long().gt(0).long().clamp(0, 1)
        programs = self.version_program_ids[ids, version]
        shape = programs.shape
        vectors = super().forward(programs.reshape(-1, shape[-1]))
        return vectors.reshape(*shape[:-1], self.config.output_dim)

    @property
    def embedding_dim(self) -> int:
        """Compatibility width for consumers that use embedding modules."""

        return self.config.output_dim

    @property
    def num_embeddings(self) -> int:
        """Number of canonical card programs available to the environment."""

        return int(self.version_program_ids.shape[0])

    def versioned_forward(
        self, card_ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        """Expose the shared-tokenizer interface without reading card ids semantically."""

        return self.card_vectors(card_ids, upgrades)

    def effective_weight(self) -> torch.Tensor:
        """Return all rule vectors for legacy pile-language call sites."""

        ids = torch.arange(self.num_embeddings, device=self.version_program_ids.device)
        return self.card_vectors(ids)


def load_frozen_card_rule_encoder(
    checkpoint_path: str,
    *,
    expected_num_cards: int,
) -> CardRuleEncoder:
    """Load only the v7 anonymous card encoder and freeze its parameters."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint.get("config")
    model_state = checkpoint.get("model")
    if not isinstance(payload, dict) or not isinstance(model_state, dict):
        raise ValueError(
            "card vector checkpoint must contain dict 'config' and 'model' entries"
        )
    card_payload = payload.get("card_encoder")
    if not isinstance(card_payload, dict):
        raise ValueError("card vector checkpoint has no card_encoder config")
    card_config = ProgramEncoderConfig(**card_payload)
    if card_config.token_vocab_size != len(CARD_RULE_VOCAB):
        raise ValueError(
            "card vector checkpoint vocabulary does not match the current card DSL"
        )
    card_state = {
        key.removeprefix("card_encoder."): value
        for key, value in model_state.items()
        if key.startswith("card_encoder.")
    }
    if not card_state:
        raise ValueError("card vector checkpoint contains no card_encoder weights")

    uses_backbone = any(key.startswith("backbone.") for key in card_state)
    backbone = None
    if uses_backbone:
        backbone = RuleProgramTransformer(
            token_dim=card_config.token_dim,
            n_heads=card_config.n_heads,
            n_layers=card_config.n_layers,
            feedforward=card_config.feedforward,
            max_tokens=card_config.max_tokens,
            dropout=card_config.dropout,
        )
    encoder = CardRuleEncoder(card_config, backbone=backbone)
    missing, unexpected = encoder.load_state_dict(card_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "card vector checkpoint is incompatible with CardRuleEncoder: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if encoder.num_embeddings != expected_num_cards:
        raise ValueError(
            "card vector checkpoint card table does not match wholerun: "
            f"checkpoint={encoder.num_embeddings}, environment={expected_num_cards}"
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


FROZEN_CARD_VECTOR_TABLE_SCHEMA = "frozen-card-vector-table-v1"


class FrozenCardVectorTable(nn.Module):
    """A fixed base/upgraded vector for every canonical card id.

    The table is a persistent buffer rather than a Parameter, so neither the
    combat optimizer nor the strategic optimizer can update it. Persisting it
    records the exact table in every policy checkpoint; the source artifact is
    still required while SB3 constructs the policy during load.
    """

    def __init__(self, vectors: torch.Tensor) -> None:
        super().__init__()
        vectors = torch.as_tensor(vectors, dtype=torch.float32).detach().contiguous()
        if vectors.ndim != 3 or vectors.shape[1] != 2 or vectors.shape[2] < 1:
            raise ValueError("card vector table must have shape [cards, 2, width]")
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("card vector table contains non-finite values")
        self.register_buffer("vectors", vectors.clone(), persistent=True)

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.shape[2])

    @property
    def num_embeddings(self) -> int:
        return int(self.vectors.shape[0])

    def versioned_forward(
        self, card_ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        ids = card_ids.long()
        versions = upgrades.long().gt(0).long().clamp(0, 1)
        return self.vectors[ids, versions]

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        return self.vectors[card_ids.long(), 0]

    def effective_weight(self) -> torch.Tensor:
        return self.vectors[:, 0]


def load_frozen_card_vector_table(
    checkpoint_path: str,
    *,
    expected_num_cards: int,
) -> FrozenCardVectorTable:
    """Load a compact frozen table or materialize a standalone rule encoder."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (isinstance(checkpoint, dict)
            and checkpoint.get("schema") == FROZEN_CARD_VECTOR_TABLE_SCHEMA):
        vocabulary = checkpoint.get("card_vocab")
        if vocabulary != list(CARD_VOCAB):
            raise ValueError("card vector table vocabulary does not match the current environment")
        vectors = checkpoint.get("vectors")
        if not isinstance(vectors, torch.Tensor):
            raise ValueError("card vector table checkpoint has no tensor 'vectors'")
        table = FrozenCardVectorTable(vectors)
    else:
        encoder = load_frozen_card_rule_encoder(
            checkpoint_path, expected_num_cards=expected_num_cards
        )
        ids = torch.arange(expected_num_cards, dtype=torch.long)[:, None].expand(-1, 2)
        upgrades = torch.arange(2, dtype=torch.long)[None].expand(expected_num_cards, -1)
        with torch.no_grad():
            table = FrozenCardVectorTable(encoder.versioned_forward(ids, upgrades))
    if table.num_embeddings != expected_num_cards:
        raise ValueError(
            "card vector table does not match the environment: "
            f"checkpoint={table.num_embeddings}, environment={expected_num_cards}"
        )
    return table


_RELIC_WORD_RE = re.compile(r"[A-Za-z0-9]+|%|[^\sA-Za-z0-9]", re.ASCII)


def relic_effect_tokens(relic_id: str) -> tuple[str, ...]:
    """Tokenize only the effect text; the relic id is never emitted."""

    try:
        effect = COMBAT_RELIC_RULES[str(relic_id)]
    except KeyError as exc:
        raise KeyError(f"no rule language for relic {relic_id!r}") from exc
    return tuple(_RELIC_WORD_RE.findall(effect))


RELIC_RULE_VOCAB = (
    "<PAD>",
    *sorted({token for relic in COMBAT_RELIC_RULES for token in relic_effect_tokens(relic)}),
)
RELIC_RULE_INDEX = {token: index for index, token in enumerate(RELIC_RULE_VOCAB)}
RELIC_RULE_IDS = tuple(sorted(COMBAT_RELIC_RULES))
_RELIC_PROGRAMS = tuple(
    tuple(RELIC_RULE_INDEX[token] for token in relic_effect_tokens(relic))
    for relic in RELIC_RULE_IDS
)
_MAX_RELIC_PROGRAM_TOKENS = max(map(len, _RELIC_PROGRAMS), default=1)


class RelicRuleEncoder(RuleProgramEncoder):
    """Name-free relic effect encoder."""

    def __init__(
        self,
        config: ProgramEncoderConfig,
        *,
        backbone: RuleProgramTransformer | None = None,
        register_backbone: bool = True,
    ) -> None:
        if config.token_vocab_size != len(RELIC_RULE_VOCAB):
            raise ValueError(
                "relic encoder token_vocab_size must match RELIC_RULE_VOCAB"
            )
        super().__init__(
            config, backbone=backbone, register_backbone=register_backbone
        )
        programs = torch.full(
            (len(RELIC_RULE_IDS), _MAX_RELIC_PROGRAM_TOKENS),
            config.pad_id,
            dtype=torch.long,
        )
        for row, program in enumerate(_RELIC_PROGRAMS):
            programs[row, :len(program)] = torch.as_tensor(program)
        self.register_buffer("program_ids", programs, persistent=False)

    def relic_vectors(self, relic_indices: torch.Tensor) -> torch.Tensor:
        indices = relic_indices.long()
        if bool((indices < 0).any()) or bool((indices >= self.program_ids.shape[0]).any()):
            raise ValueError("relic index is outside the first-version relic vocabulary")
        shape = indices.shape
        vectors = super().forward(self.program_ids[indices].reshape(-1, self.program_ids.shape[-1]))
        return vectors.reshape(*shape, self.config.output_dim)

    def vectors_for_ids(self, relic_ids: Sequence[str], *, device=None) -> torch.Tensor:
        indices = []
        for relic_id in relic_ids:
            try:
                indices.append(RELIC_RULE_IDS.index(str(relic_id)))
            except ValueError as exc:
                raise KeyError(f"no first-version relic rule for {relic_id!r}") from exc
        return self.relic_vectors(torch.as_tensor(indices, dtype=torch.long, device=device))


# ---------------------------------------------------------------------------
# Fixed state vectors and transition deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateVectorConfig:
    """Stable state layout and normalization contract."""

    max_enemies: int = 5
    player_buff_names: tuple[str, ...] = tuple(V.PLAYER_BUFF_VOCAB)
    monster_buff_names: tuple[str, ...] = tuple(V.MONSTER_BUFF_VOCAB)
    # Card-class play counts and relic-owned counters are deliberately absent.
    # A matching relic/power node owns those counters and requests their exact
    # runtime value when it executes.
    counter_names: tuple[str, ...] = tuple(
        name for name in COMBAT_COUNTER_KEYS
        if name not in {
            "cards_played", "attacks_played", "skills_played",
            "happy_flower", "nunchaku", "pen_nib", "sundial",
        }
    )
    player_scalar_names: tuple[str, ...] = (
        "hp", "max_hp", "block", "block_present", "energy", "turn", "floor",
        "gold", "hp_frac",
    )
    enemy_scalar_names: tuple[str, ...] = (
        "present", "hp", "max_hp", "block", "block_present", "base_damage",
        "adjusted_damage", "hits", "gone"
    )
    pile_names: tuple[str, ...] = (
        "draw_count", "discard_count", "exhaust_count"
    )
    relic_summary_names: tuple[str, ...] = ()
    # Kept under the historical field name for checkpoint/config plumbing.
    # Contract-v7 semantics are binary relic-presence slots, never counters.
    relic_counter_slots: int = 32
    # New v7 configurations map each stable relic id to one presence bit. An
    # old checkpoint without this field is reconstructed with anonymous slots
    # of ``relic_counter_slots`` width by ``from_dict`` below.
    relic_presence_names: tuple[str, ...] = tuple(RELIC_RULE_IDS)
    hp_scale: float = 200.0
    block_scale: float = 100.0
    energy_scale: float = 10.0
    turn_scale: float = 30.0
    floor_scale: float = 50.0
    gold_scale: float = 999.0
    damage_scale: float = 100.0
    hits_scale: float = 10.0
    buff_scale: float = 30.0
    counter_scale: float = 10.0
    pile_scale: float = 100.0

    @property
    def player_dim(self) -> int:
        return (
            len(self.player_scalar_names)
            + len(self.player_buff_names)
            + len(self.counter_names)
            + len(self.relic_summary_names)
            + len(self.relic_presence_names)
        )

    @property
    def enemy_dim(self) -> int:
        return len(self.enemy_scalar_names) + len(self.monster_buff_names)

    @property
    def pile_dim(self) -> int:
        return len(self.pile_names)

    @property
    def world_dim(self) -> int:
        return self.player_dim + self.max_enemies * self.enemy_dim + self.pile_dim


@dataclass
class VectorizedState:
    player: torch.Tensor
    enemies: torch.Tensor
    piles: torch.Tensor
    relic_ids: tuple[str, ...] = ()
    relic_counters: torch.Tensor | None = None

    @property
    def flat(self) -> torch.Tensor:
        return torch.cat((self.player, self.enemies.reshape(-1), self.piles), dim=-1)


def _scaled(value: Any, scale: float) -> float:
    return float(value or 0) / scale


def _counter_value(mapping: Mapping[str, Any], name: str) -> float:
    wanted = name.casefold()
    for key, value in mapping.items():
        if str(key).casefold() == wanted:
            return float(value or 0)
    return 0.0


def vectorize_state(
    state: Mapping[str, Any], config: StateVectorConfig | None = None, *, device=None
) -> VectorizedState:
    """Convert a canonical rule sample state to fixed vectors.

    Names are used only as fixed schema keys while constructing the vector. No
    card, relic or monster id is returned as a model feature.
    """

    config = config or StateVectorConfig()
    if config.relic_counter_slots < 0 or len(config.relic_presence_names) < 0:
        raise ValueError("relic presence slot count cannot be negative")
    combat = state.get("combat", {})
    player = combat.get("player", {})
    global_state = state.get("global", {})
    p: list[float] = []
    scalar_values = {
        "hp": _scaled(player.get("hp", 0), config.hp_scale),
        "max_hp": _scaled(player.get("max_hp", 0), config.hp_scale),
        "block": _scaled(player.get("block", 0), config.block_scale),
        "block_present": float(player.get("block", 0) > 0),
        "energy": _scaled(player.get("energy", 0), config.energy_scale),
        "turn": _scaled(combat.get("turn", 0), config.turn_scale),
        "floor": _scaled(global_state.get("floor", combat.get("floor", 0)), config.floor_scale),
        "gold": _scaled(player.get("gold", global_state.get("gold", 0)), config.gold_scale),
        "hp_frac": (
            float(player.get("hp", 0)) / float(player.get("max_hp", 0))
            if player.get("max_hp", 0) else 0.0
        ),
    }
    p.extend(scalar_values[name] for name in config.player_scalar_names)
    powers = player.get("powers", {})
    # Exact buff magnitudes stay in the runtime ValueStore.  The world vector
    # exposes only presence, which is enough to decide whether a rule must
    # request and use that magnitude.
    p.extend(float(_counter_value(powers, name) != 0) for name in config.player_buff_names)
    counters = player.get("combat_counters", {})
    p.extend(
        (
            float(bool(_counter_value(counters, name)))
            if name.casefold() in {"necronomicon_used", "orange_attack", "orange_skill", "orange_power"}
            else _scaled(_counter_value(counters, name), config.counter_scale)
        )
        for name in config.counter_names
    )

    relic_ids = tuple(str(value) for value in global_state.get("relics", ()))
    raw_relic_counters = list(global_state.get("relic_counters", ()))
    relic_counter_values = [
        float(raw_relic_counters[i]) if i < len(raw_relic_counters) else 0.0
        for i in range(len(relic_ids))
    ]
    relic_summary = {
        "relic_count": len(relic_ids) / config.counter_scale,
        "counter_sum": sum(relic_counter_values) / config.counter_scale,
        "active_counter_count": sum(value != 0 for value in relic_counter_values) / config.counter_scale,
        "max_counter": (max(relic_counter_values) if relic_counter_values else 0.0) / config.counter_scale,
    }
    p.extend(relic_summary[name] for name in config.relic_summary_names)
    relic_set = {str(relic_id) for relic_id in relic_ids}
    p.extend(
        1.0 if relic_name in relic_set else 0.0
        for relic_name in config.relic_presence_names
    )

    enemies = torch.zeros((config.max_enemies, config.enemy_dim), dtype=torch.float32, device=device)
    raw_enemies = list(combat.get("monsters", ()))
    for index, monster in enumerate(raw_enemies[:config.max_enemies]):
        gone = bool(monster.get("gone", False))
        values = {
            "present": float(not gone),
            "hp": _scaled(monster.get("hp", 0), config.hp_scale),
            "max_hp": _scaled(monster.get("max_hp", 0), config.hp_scale),
            "block": _scaled(monster.get("block", 0), config.block_scale),
            "block_present": float(monster.get("block", 0) > 0),
            "base_damage": _scaled(monster.get("base_damage", 0), config.damage_scale),
            "adjusted_damage": _scaled(monster.get("adjusted_damage", monster.get("base_damage", 0)), config.damage_scale),
            "hits": _scaled(monster.get("hits", 0), config.hits_scale),
            "gone": float(gone),
        }
        row = [values[name] for name in config.enemy_scalar_names]
        powers = monster.get("powers", {})
        row.extend(
            float(_counter_value(powers, name) != 0)
            for name in config.monster_buff_names
        )
        enemies[index] = torch.as_tensor(row, dtype=torch.float32, device=device)

    # Pile identity is intentionally outside this contract.  Keep this map
    # limited to the three count features so an old hand/deck field cannot be
    # reintroduced accidentally through a custom state layout.
    pile_counts = {
        "draw_count": len(combat.get("draw", ())),
        "discard_count": len(combat.get("discard", ())),
        "exhaust_count": len(combat.get("exhaust", ())),
    }
    unknown_piles = set(config.pile_names).difference(pile_counts)
    if unknown_piles:
        raise ValueError(
            "pile_names may only contain draw_count, discard_count, exhaust_count; "
            f"got {sorted(unknown_piles)}"
        )
    piles = torch.as_tensor(
        [pile_counts[name] / config.pile_scale for name in config.pile_names],
        dtype=torch.float32,
        device=device,
    )
    return VectorizedState(
        player=torch.as_tensor(p, dtype=torch.float32, device=device),
        enemies=enemies,
        piles=piles,
        relic_ids=relic_ids,
        # The per-relic feature is also presence-only. Exact counters remain
        # in the environment ValueStore and enter generation through masked
        # Environment-owned inline CONST_VALUE spans.
        relic_counters=torch.ones(
            len(relic_ids), dtype=torch.float32, device=device,
        ),
    )


def vector_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], config: StateVectorConfig | None = None,
) -> VectorizedState:
    """Return a component-wise transition delta with the same layout."""

    left = vectorize_state(before, config)
    right = vectorize_state(after, config)
    return VectorizedState(
        player=right.player - left.player,
        enemies=right.enemies - left.enemies,
        piles=right.piles - left.piles,
        relic_ids=right.relic_ids,
        relic_counters=(
            right.relic_counters - left.relic_counters
            if right.relic_counters is not None and left.relic_counters is not None
            else None
        ),
    )


def _raw_state_values(
    state: Mapping[str, Any], config: StateVectorConfig,
) -> dict[str, int | float | bool]:
    """Expose the named scalar values used to explain a vector transition."""

    combat = state.get("combat", {})
    player = combat.get("player", {})
    global_state = state.get("global", {})
    values: dict[str, int | float | bool] = {}
    for name in config.player_scalar_names:
        if name == "floor":
            value = global_state.get("floor", combat.get("floor", 0))
        elif name == "hp_frac":
            maximum = float(player.get("max_hp", 0) or 0)
            value = float(player.get("hp", 0)) / maximum if maximum else 0.0
        elif name == "block_present":
            value = float(player.get("block", 0) > 0)
        else:
            value = player.get(name, global_state.get(name, 0))
        values[f"player.{name}"] = value
    powers = player.get("powers", {})
    values.update({
        f"player.buff.{name}": _counter_value(powers, name)
        for name in config.player_buff_names
    })
    counters = player.get("combat_counters", {})
    values.update({
        f"player.counter.{name}": _counter_value(counters, name)
        for name in config.counter_names
    })
    relic_ids = list(global_state.get("relics", ()))
    relic_counters = [
        float(raw_value) if index < len(global_state.get("relic_counters", ())) else 0.0
        for index, raw_value in enumerate(global_state.get("relic_counters", ()))
    ]
    relic_summary = {
        "relic_count": len(relic_ids),
        "counter_sum": sum(relic_counters),
        "active_counter_count": sum(value != 0 for value in relic_counters),
        "max_counter": max(relic_counters, default=0.0),
    }
    values.update({
        f"player.relic.{name}": relic_summary[name]
        for name in config.relic_summary_names
    })
    relic_set = {str(relic_id) for relic_id in relic_ids}
    values.update({
        f"player.relic_present[{index}]": float(relic_name in relic_set)
        for index, relic_name in enumerate(config.relic_presence_names)
    })

    monsters = list(combat.get("monsters", ()))
    for index in range(config.max_enemies):
        monster = monsters[index] if index < len(monsters) else {}
        gone = bool(monster.get("gone", False))
        values.update({
            f"enemy[{index}].present": float(bool(monster)) and float(not gone),
            f"enemy[{index}].hp": monster.get("hp", 0),
            f"enemy[{index}].max_hp": monster.get("max_hp", 0),
            f"enemy[{index}].block": monster.get("block", 0),
            f"enemy[{index}].block_present": float(monster.get("block", 0) > 0),
            f"enemy[{index}].base_damage": monster.get("base_damage", 0),
            f"enemy[{index}].adjusted_damage": monster.get(
                "adjusted_damage", monster.get("base_damage", 0)
            ),
            f"enemy[{index}].hits": monster.get("hits", 0),
            f"enemy[{index}].gone": gone,
        })
        monster_powers = monster.get("powers", {})
        values.update({
            f"enemy[{index}].buff.{name}": _counter_value(monster_powers, name)
            for name in config.monster_buff_names
        })
    piles = {
        "draw_count": len(combat.get("draw", ())),
        "discard_count": len(combat.get("discard", ())),
        "exhaust_count": len(combat.get("exhaust", ())),
    }
    values.update({f"piles.{name}": piles[name] for name in config.pile_names})
    return values


def _calculation_scalar(value: int | float | bool) -> int | float | bool:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def generic_state_calculations(
    before: Mapping[str, Any], after: Mapping[str, Any],
    config: StateVectorConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build one APPLY_DELTA calculation for each changed named state scalar."""

    config = config or StateVectorConfig()
    left = _raw_state_values(before, config)
    right = _raw_state_values(after, config)
    calculations: list[dict[str, Any]] = []
    for field_name, before_value in left.items():
        if field_name == "player.hp_frac" or field_name.endswith(".block_present"):
            continue
        after_value = right[field_name]
        numeric_before = float(before_value)
        numeric_after = float(after_value)
        if abs(numeric_after - numeric_before) < 1e-8:
            continue
        calculations.append({
            "type": "APPLY_DELTA",
            "field": field_name,
            "left": _calculation_scalar(before_value),
            "delta": _calculation_scalar(numeric_after - numeric_before),
            "result": _calculation_scalar(after_value),
        })
    return tuple(calculations)


# ---------------------------------------------------------------------------
# Anonymous hand composition
# ---------------------------------------------------------------------------


def hand_composition_feature_dim(card_dim: int) -> int:
    """Width of the deterministic, permutation-invariant hand summary.

    The summary contains three scalar counts followed by sum, mean and max of
    the anonymous card-rule vectors.  It is deliberately a bag representation:
    hand indices are action-side references only and never enter this feature.
    """

    if card_dim < 1:
        raise ValueError("card_dim must be positive")
    return 3 + 3 * card_dim


def summarize_hand(
    card_vectors: torch.Tensor,
    upgrades: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    upgrade_scale: float = 5.0,
    max_cards: float = 10.0,
) -> torch.Tensor:
    """Pool a variable-size anonymous hand into one fixed-width vector.

    ``card_vectors`` is ``[H,D]`` or ``[B,H,D]``.  The first three output
    values are normalized card count, total upgrade count and number of
    upgraded instances; the remaining blocks are rule-vector sum, mean and
    elementwise max.  Consequently an upgrade changes the summary even when
    the number of cards is unchanged, while an exhaust/draw changes it through
    both the counts and the pooled rule vectors.
    """

    if card_vectors.ndim == 2:
        card_vectors = card_vectors.unsqueeze(0)
        squeeze = True
    elif card_vectors.ndim == 3:
        squeeze = False
    else:
        raise ValueError("card_vectors must have shape [H,D] or [B,H,D]")
    if not torch.is_floating_point(card_vectors):
        card_vectors = card_vectors.float()
    batch, slots, card_dim = card_vectors.shape
    if upgrades is None:
        upgrades = torch.zeros(batch, slots, device=card_vectors.device)
    elif upgrades.ndim == 1:
        upgrades = upgrades.unsqueeze(0)
    if upgrades.shape != (batch, slots):
        raise ValueError("upgrades must align with the hand card slots")
    if max_cards <= 0:
        raise ValueError("max_cards must be positive")
    if mask is None:
        mask = torch.ones(batch, slots, dtype=torch.bool, device=card_vectors.device)
    elif mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != (batch, slots):
        raise ValueError("mask must align with the hand card slots")
    mask = mask.bool()
    mask_float = mask.to(card_vectors.dtype)
    weights = mask_float.unsqueeze(-1)
    count = mask_float.sum(dim=1, keepdim=True)
    upgrade_values = upgrades.to(card_vectors.dtype)
    upgrade_sum = (upgrade_values * mask_float).sum(dim=1, keepdim=True)
    upgraded_count = ((upgrade_values > 0) & mask).to(card_vectors.dtype).sum(
        dim=1, keepdim=True
    )
    rule_sum = (card_vectors * weights).sum(dim=1)
    rule_mean = rule_sum / count.clamp_min(1.0)
    if slots:
        neg_inf = torch.full_like(card_vectors, float("-inf"))
        rule_max = torch.where(weights.bool(), card_vectors, neg_inf).amax(dim=1)
        rule_max = torch.where(count.gt(0), rule_max, torch.zeros_like(rule_max))
    else:
        rule_max = torch.zeros(batch, card_dim, dtype=card_vectors.dtype, device=card_vectors.device)
    features = torch.cat(
        (
            count / float(max_cards),
            upgrade_sum / max(float(upgrade_scale), 1.0),
            upgraded_count / float(max_cards),
            rule_sum,
            rule_mean,
            rule_max,
        ),
        dim=-1,
    )
    return features.squeeze(0) if squeeze else features


@dataclass(frozen=True)
class HandCompositionConfig:
    """Capacity and Transformer size for the anonymous hand set encoder."""

    card_dim: int
    output_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 4
    feedforward: int = 512
    n_layers: int = 2
    max_cards: int = 10
    upgrade_scale: float = 5.0
    dropout: float = 0.0

    @property
    def feature_dim(self) -> int:
        return hand_composition_feature_dim(self.card_dim)

    @property
    def composition_dim(self) -> int:
        """Width of the learned CLS-space hand composition vector."""

        return self.output_dim


class HandCompositionEncoder(nn.Module):
    """Encode an anonymous hand with a permutation-invariant Transformer.

    There is deliberately no positional embedding: card-rule tokens are a set,
    not a sequence.  A learned CLS token attends to all valid hand instances
    and is projected to the world-entity width.
    """

    def __init__(self, config: HandCompositionConfig) -> None:
        super().__init__()
        if (
            config.card_dim < 1
            or config.output_dim < 1
            or config.hidden_dim < 1
            or config.n_layers < 1
        ):
            raise ValueError("hand encoder dimensions and n_layers must be positive")
        if config.max_cards < 1:
            raise ValueError("hand encoder max_cards must be positive")
        if config.n_heads < 1 or config.feedforward < 1:
            raise ValueError("hand Transformer heads and feedforward must be positive")
        if config.hidden_dim % config.n_heads:
            raise ValueError("hand hidden_dim must be divisible by n_heads")
        self.config = config
        self.card_projection = nn.Linear(config.card_dim, config.hidden_dim)
        self.upgrade_projection = nn.Linear(1, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(
        self,
        card_vectors: torch.Tensor,
        upgrades: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if card_vectors.ndim == 2:
            slots = card_vectors.shape[0]
        elif card_vectors.ndim == 3:
            slots = card_vectors.shape[1]
        else:
            raise ValueError("card_vectors must have shape [H,D] or [B,H,D]")
        if card_vectors.shape[-1] != self.config.card_dim:
            raise ValueError("card vector width does not match HandCompositionConfig.card_dim")
        if card_vectors.ndim == 2:
            card_vectors = card_vectors.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if slots > self.config.max_cards:
            raise ValueError("hand exceeds HandCompositionConfig.max_cards")
        # Rule vectors carry all card semantics. Upgrade count is a numeric
        # attribute of the instance, not a card identity feature.
        if not torch.is_floating_point(card_vectors):
            card_vectors = card_vectors.float()
        if upgrades is None:
            upgrades = torch.zeros(
                card_vectors.shape[:2], device=card_vectors.device
            )
        elif upgrades.ndim == 1:
            upgrades = upgrades.unsqueeze(0)
        if upgrades.shape != card_vectors.shape[:2]:
            raise ValueError("upgrades must align with hand card vectors")
        if mask is None:
            mask = torch.ones(
                card_vectors.shape[:2], dtype=torch.bool, device=card_vectors.device
            )
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != card_vectors.shape[:2]:
            raise ValueError("mask must align with hand card vectors")
        mask = mask.bool()
        if upgrades.device != card_vectors.device:
            upgrades = upgrades.to(card_vectors.device)
        if mask.device != card_vectors.device:
            mask = mask.to(card_vectors.device)
        tokens = self.card_projection(card_vectors)
        tokens = tokens + self.upgrade_projection(
            (
                upgrades.to(device=card_vectors.device, dtype=card_vectors.dtype)
                / max(self.config.upgrade_scale, 1e-6)
            ).unsqueeze(-1)
        )
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        hidden = self.encoder(tokens, src_key_padding_mask=padding)
        encoded = self.projection(self.norm(hidden[:, 0]))
        return encoded.squeeze(0) if squeeze else encoded


def hand_inputs_from_state(
    state: Mapping[str, Any], card_encoder: CardRuleEncoder, *, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve canonical hand instances to anonymous rule vectors.

    Card ids are consumed only by this environment-side adapter.  The returned
    tensors contain rule vectors and upgrade counts, so the model never sees a
    card-name lookup table.
    """

    combat = state.get("combat", state)
    hand = list(combat.get("hand", ()))
    card_dim = card_encoder.config.output_dim
    if not hand:
        return (
            torch.empty((1, 0, card_dim), dtype=torch.float32, device=device),
            torch.empty((1, 0), dtype=torch.float32, device=device),
        )
    from env.core.cards_vocab import CARD_INDEX

    encoder_device = card_encoder.version_program_ids.device
    ids = torch.as_tensor(
        [CARD_INDEX[str(card["id"])] for card in hand],
        dtype=torch.long,
        device=encoder_device,
    )
    upgrades = torch.as_tensor(
        [float(card.get("up", 0)) for card in hand],
        dtype=torch.float32,
        device=encoder_device,
    )
    vectors = card_encoder.card_vectors(ids).unsqueeze(0)
    if device is not None:
        vectors = vectors.to(device)
        upgrades = upgrades.to(device)
    return vectors, upgrades.unsqueeze(0)


def hand_composition_from_state(
    state: Mapping[str, Any],
    card_encoder: CardRuleEncoder,
    *,
    hand_encoder: HandCompositionEncoder | None = None,
    max_cards: int = 10,
    upgrade_scale: float = 5.0,
    device=None,
) -> torch.Tensor:
    """Return the fixed anonymous hand summary for a canonical state."""

    vectors, upgrades = hand_inputs_from_state(state, card_encoder, device=device)
    if hand_encoder is not None:
        encoded = hand_encoder(vectors, upgrades)
        return encoded.squeeze(0) if encoded.ndim == 2 and encoded.shape[0] == 1 else encoded
    return summarize_hand(
        vectors,
        upgrades,
        max_cards=max_cards,
        upgrade_scale=upgrade_scale,
    )


def hand_composition_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    card_encoder: CardRuleEncoder,
    *,
    hand_encoder: HandCompositionEncoder | None = None,
    max_cards: int = 10,
    upgrade_scale: float = 5.0,
    device=None,
) -> torch.Tensor:
    """Compute the target hand-summary delta for one atomic operation."""

    return hand_composition_from_state(
        after,
        card_encoder,
        hand_encoder=hand_encoder,
        max_cards=max_cards,
        upgrade_scale=upgrade_scale,
        device=device,
    ) - hand_composition_from_state(
        before,
        card_encoder,
        hand_encoder=hand_encoder,
        max_cards=max_cards,
        upgrade_scale=upgrade_scale,
        device=device,
    )


# ---------------------------------------------------------------------------
# Event/query protocol
# ---------------------------------------------------------------------------


VECTOR_CALCULATION_TOKENS = (
    # Generic arithmetic operators are the v7 language surface.  The older
    # domain-specific calculation names remain reserved for legacy flat
    # serializers and are not emitted by schema-v7 local sequences.
    "CALCULATION_ADD",
    "CALCULATION_SUBTRACT",
    "CALCULATION_MULTIPLY",
    "CALCULATION_FLOOR_DIVIDE",
    "CALCULATION_MIN",
    "CALCULATION_RELU",
    "CALCULATION_IDENTITY",
    "CALCULATION_APPLY_DELTA",
    "CALCULATION_READ_BASE_DAMAGE",
    "CALCULATION_READ_STRENGTH",
    "CALCULATION_ADD_STRENGTH",
    "CALCULATION_READ_VULNERABLE",
    "CALCULATION_MULTIPLY_VULNERABLE",
    "CALCULATION_FLOOR_DIVIDE_VULNERABLE",
    "CALCULATION_NO_VULNERABLE",
    "CALCULATION_READ_TARGET_BLOCK",
    "CALCULATION_BLOCK_ABSORB",
    "CALCULATION_SUBTRACT_BLOCK",
    "CALCULATION_SUBTRACT_BLOCK_DAMAGE",
    "CALCULATION_READ_TARGET_HP",
    "CALCULATION_SUBTRACT_HP",
    "CALCULATION_READ_TRIGGER_DAMAGE",
    "CALCULATION_READ_DAMAGE_MULTIPLIER",
    "CALCULATION_MULTIPLY_DAMAGE",
    "CALCULATION_RELIC_COUNTER_UPDATE",
    # Legacy flat-stream token types remain reserved for schema-v6
    # checkpoint compatibility. Schema-v7 never emits either operation.
    "CALCULATION_REUSE_ATTACK_DAMAGE",
    "CALCULATION_SUBTRACT_PLAYER_HP",
    "CALCULATION_COUNT_ATTACK_HITS",
)
VECTOR_CALCULATION_FIELD_TOKENS = (
    "CALC_FIELD_field", "CALC_FIELD_value", "CALC_FIELD_left",
    "CALC_FIELD_right", "CALC_FIELD_result", "CALC_FIELD_delta",
    "CALC_FIELD_input", "CALC_FIELD_absorbed",
    "CALC_FIELD_source", "CALC_FIELD_increment", "CALC_FIELD_threshold",
    "CALC_FIELD_reset", "CALC_FIELD_triggered",
)
VECTOR_CALCULATION_MARKERS = (
    "<CALCULATIONS_BEGIN>", "<CALCULATIONS_END>",
    "CALCULATION_BEGIN", "CALCULATION_END",
)
VECTOR_NUMBER_TOKENS = (
    "NUM_BEGIN", "NUM_LSD_BEGIN", "NUM_END", "SIGN_NEG", "DECIMAL",
    *(f"DIGIT_{digit}" for digit in range(10)), "BOOL_0", "BOOL_1",
)


def _calculation_atom(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_") or "EMPTY"


def _default_calculation_value_tokens() -> tuple[str, ...]:
    config = StateVectorConfig()
    labels: list[str] = []
    labels.extend(f"player.{name}" for name in config.player_scalar_names)
    labels.extend(f"player.buff.{name}" for name in config.player_buff_names)
    labels.extend(f"player.counter.{name}" for name in config.counter_names)
    labels.extend(f"player.relic.{name}" for name in config.relic_summary_names)
    labels.extend(
        f"player.relic_present[{index}]"
        for index in range(len(config.relic_presence_names))
    )
    labels.extend(
        f"relic[{index}].counter"
        for index in range(config.relic_counter_slots)
    )
    labels.append("parent.relic_trigger.damage_multiplier")
    for enemy in range(config.max_enemies):
        labels.extend(f"enemy[{enemy}].{name}" for name in config.enemy_scalar_names)
        labels.extend(
            f"enemy[{enemy}].buff.{name}" for name in config.monster_buff_names
        )
    labels.extend(f"piles.{name}" for name in config.pile_names)
    return tuple(f"VALUE_{_calculation_atom(label)}" for label in labels)


VECTOR_CALCULATION_VALUE_TOKENS = _default_calculation_value_tokens()

VECTOR_SPECIAL_TOKENS = (
    "<PAD>", "<UNK>", "<OUTPUT_BEGIN>", "<OUTPUT_END>",
    "<TRANSITION_BEGIN>", "<TRANSITION_END>",
    "<EVENT_BEGIN>", "<EVENT_END>", "EVENT_KIND", "ENTITY_REF", "DELTA_SLOT",
    *VECTOR_CALCULATION_MARKERS, *VECTOR_CALCULATION_TOKENS,
    *VECTOR_CALCULATION_FIELD_TOKENS,
    *VECTOR_NUMBER_TOKENS, *VECTOR_CALCULATION_VALUE_TOKENS,
    "LOOP_KIND", "ATTACK_HITS", "LOOP_COUNT",
    "LOOP_COUNT_2", "LOOP_COUNT_3", "LOOP_COUNT_4", "LOOP_COUNT_5",
    "<QUERY_BEGIN>", "<QUERY_END>",
    "QUERY_RANDOM_TARGET", "QUERY_TOP_DRAW_CARD", "QUERY_RANDOM_HAND_CARD",
    "QUERY_RANDOM_CARD", "QUERY_CREATED_CARD", "<OBS_BEGIN>", "<OBS_END>",
    "OBS_RANDOM_TARGET", "OBS_CARD_INSTANCE", "TARGET_REF", "CARD_REF",
    "RUNTIME_COST", "UPGRADE", "SCOPE_BEGIN", "SCOPE_END", "SCOPE_MAIN", "SCOPE_CHILD",
    "PARENT_QUERY", "CALL_BEGIN", "CALL_END", "CALL_KIND", "PLAY_TOP_DRAW_CARD",
    "COST_MODE_IGNORE", "<CHILD_CARD_RULE_VECTOR>",
    "ENTITY_SELF", "ENTITY_PILES",
    "HAND_OP", "HAND_UPGRADE", "HAND_EXHAUST", "HAND_DRAW", "HAND_REPLACE",
    "HAND_DELTA_SLOT", "QUERY_CHOSEN_HAND_CARD", "OBS_HAND_CARD",
)
VECTOR_EVENT_KIND_TOKENS = (
    "EVENT_CARD_COST", "EVENT_DAMAGE", "EVENT_BLOCK", "EVENT_STATUS",
    "EVENT_COUNTER", "EVENT_PILE_DELTA", "EVENT_PILE_SHUFFLE", "EVENT_DRAW", "EVENT_EXHAUST",
    "EVENT_HEAL", "EVENT_ENERGY", "EVENT_TURN", "EVENT_REPEAT",
    "EVENT_CARD_PLAY", "EVENT_CARD_CREATE", "EVENT_CARD_UPGRADE",
    "EVENT_ACTION_COMPLETE", "EVENT_HAND_COMPOSITION",
)
# Reserve enough slot references for larger experiments while the default state
# contract still uses five enemies.  The state/world config validates the actual
# active count; unused entity tokens simply remain unreachable.
VECTOR_ENTITY_TOKENS = tuple(
    f"ENTITY_ENEMY_SLOT_{index}" for index in range(32)
)
VECTOR_RELIC_ENTITY_TOKENS = tuple(
    f"ENTITY_RELIC_SLOT_{index}" for index in range(32)
)
VECTOR_RUNTIME_ENTITY_VALUE_TOKENS = tuple(
    f"VALUE_ENTITY_ENEMY_SLOT_{index}" for index in range(32)
)
VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS = tuple(
    f"VALUE_runtime_relic_counter_{index}" for index in range(32)
)
VECTOR_EXECUTOR_VALUE_TOKENS = (
    "ENTITY_ENEMY_MASK",
    "VALUE_ENTITY_SELF", "VALUE_ENTITY_PILES", "VALUE_ENTITY_ENEMY_MASK",
    "VALUE_ENTITY_QUERY_TARGET", "VALUE_node_children", "VALUE_node_scope",
    "VALUE_header_ENEMY_ORDER", "VALUE_query_response_target",
    "VALUE_constant_one", "VALUE_PAID_ENERGY", "VALUE_SLOT_ASCENDING",
    "VALUE_RELIC_FIXED", "VALUE_PARENT_RELIC_TRIGGER",
    "VALUE_ATTACK", "VALUE_SKILL", "VALUE_POWER", "VALUE_OTHER",
    "VALUE_PAID_ENERGY", "VALUE_SLOT_ASCENDING",
    "VALUE_EXHAUST", "VALUE_DISCARD", "VALUE_ATTACK_PLAY", "VALUE_SKILL_PLAY",
    "VALUE_SELF", "VALUE_RANDOM_TARGET", "VALUE_parent_trigger_damage",
    "ENTITY_QUERY_TARGET",
    "VALUE_GAIN", "VALUE_SHUFFLE_DISCARD_INTO_DRAW", "VALUE_piles_discard",
    "VALUE_piles_draw",
)
VECTOR_V7_PROTOCOL_TOKENS = (
    # Effect-tree region headers and their typed parameters.
    "<EFFECT_TREE_BEGIN>", "<EFFECT_TREE_END>",
    "<REGION_BEGIN>", "<REGION_END>", "REGION_KIND", "<NODE_EXECUTE>",
    *(f"REGION_{kind}" for kind in (
        "CARD_COMMIT", "CARD_PAYMENT", "CARD_PLAY", "CARD_DESTINATION",
        "COMPLETE", "ATTACK", "DAMAGE", "REPEAT", "CALL",
        "AOE_DAMAGE_ROUND", "RANDOM_HIT", "QUERY", "SELF_DAMAGE",
        "PILE_SHUFFLE", "DRAW", "ENERGY", "RELIC_COUNTER", "STATE_CHANGE",
    )),
    "CARD_ACTION", "PAYMENT_COST", "PAYMENT_SOURCE", "PLAYER_ENERGY",
    "PLAY_KIND", "ATTACK", "SKILL", "POWER", "OTHER",
    "ANONYMOUS_RULE_VECTOR", "REPEAT_COUNT", "REPEAT_SOURCE", "PAID_ENERGY",
    "CALL_TARGET", "SELF", "ARG_DECREMENT", "QUERY_KIND", "RANDOM_TARGET",
    "BASE_DAMAGE", "DAMAGE_SOURCE", "PLAYER_BLOCK", "DAMAGE_SOURCE_KIND",
    "RELIC_FIXED", "DAMAGE_MULTIPLIER_SOURCE", "PARENT_RELIC_TRIGGER",
    "ENEMY_ORDER", "SLOT_ASCENDING", "COUNTER_REF", "COUNTER_EVENT",
    "COUNTER_INCREMENT", "COUNTER_THRESHOLD", "COUNTER_RESET", "TRIGGER_EFFECT",
    "TRIGGER_AMOUNT", "DRAW_COUNT", "ENERGY_OP", "GAIN", "LOSE",
    "ENERGY_AMOUNT", "DESTINATION", "CARD_COUNT", "FIELD_OR_STATUS",
    "CHANGE_AMOUNT", "PILE_OPERATION", "SHUFFLE_DISCARD_INTO_DRAW",
    "INSERT_INTO_DRAW", "CALC_VALUE", "CALCULATION_ADD_FIELD",
    "CALCULATION_SUBTRACT_FIELD",
    # Contract-10 runtime reads: the model names a semantic source, then the
    # environment assigns the binding. Numeric values are inserted later as
    # masked CONST_VALUE expansions; no READ_RESPONSE token exists.
    "<READ_REQUEST_BEGIN>", "<READ_REQUEST_END>", "READ_SOURCE", "READ_BIND",
    # Contract-10 inline constants. The environment inserts ``CONST_VALUE``
    # after runtime registers and earlier CALC_VALUE results used as operands.
    "CONST_VALUE",
    # Generic arithmetic operators used by the v7 local executor.  The
    # operand roles stay in CALC_FIELD_* tokens; the operator is factored out
    # so one arithmetic rule can be shared across HP, block, energy, and
    # counter calculations.
    "CALCULATION_ADD", "CALCULATION_SUBTRACT", "CALCULATION_MULTIPLY",
    "CALCULATION_FLOOR_DIVIDE", "CALCULATION_MIN", "CALCULATION_RELU",
    "CALCULATION_IDENTITY",
    # Symbolic executor operations and fields emitted by NodeSpec bodies.
    *(f"CALCULATION_{kind}" for kind in (
        "CHECK_PRESENCE", "ADD_STRENGTH", "MULTIPLY_DAMAGE",
        "MULTIPLY_VULNERABLE", "FLOOR_DIVIDE_VULNERABLE", "BLOCK_ABSORB",
        "SUBTRACT_BLOCK", "SUBTRACT_BLOCK_DAMAGE", "SUBTRACT_HP",
        "SUBTRACT_PLAYER_HP", "TRANSFER_PILE", "ADD_FIELD", "ADD_COUNTER",
        "CHECK_COUNTER_THRESHOLD", "SELECT_COUNTER_RESULT", "WRITE_RELIC_COUNTER",
        "EXECUTE_CHILDREN", "EXECUTE_ATTACK_CHILDREN", "INITIALIZE_REPEAT",
        "CALL_SELF", "ITERATE_ENEMY_MASK", "EXECUTE_RANDOM_HIT_CHILDREN",
        "COMPLETE_ACTION",
    )),
    *(f"CALC_FIELD_{field}" for field in (
        "scope", "targets", "order", "count", "body", "remaining", "decrement",
        "advanced", "triggered", "reset", "threshold", "increment", "value",
        "damage", "block", "absorbed", "target", "destination",
        "source_count", "destination_count", "source_result", "destination_result",
    )),
    *(f"VALUE_header_{key}" for key in (
        "PAYMENT_COST", "PAYMENT_SOURCE", "PLAY_KIND", "BASE_DAMAGE",
        "DAMAGE_SOURCE", "DAMAGE_SOURCE_KIND", "DAMAGE_MULTIPLIER_SOURCE",
        "REPEAT_COUNT", "REPEAT_SOURCE", "CALL_TARGET", "ARG_DECREMENT",
        "QUERY_KIND", "ENEMY_ORDER", "COUNTER_REF", "COUNTER_EVENT",
        "COUNTER_INCREMENT", "COUNTER_THRESHOLD", "COUNTER_RESET",
        "TRIGGER_EFFECT", "TRIGGER_AMOUNT", "DRAW_COUNT", "ENERGY_OP",
        "ENERGY_AMOUNT", "DESTINATION", "CARD_COUNT", "FIELD_OR_STATUS",
        "CHANGE_AMOUNT", "PILE_OPERATION", "CARD_ACTION",
    )),
    # Register/value atoms used by the shared local executor.
    *(f"VALUE_{name}" for name in (
        "reg_damage", "reg_strength", "reg_has_strength", "reg_damage_after_strength",
        "reg_damage_multiplier", "reg_damage_after_multiplier", "reg_has_vulnerable",
        "reg_damage_vulnerable_numerator", "reg_damage_after_vulnerable",
        "reg_target_block", "reg_has_block", "reg_absorbed_damage",
        "reg_target_block_after", "reg_unblocked_damage", "reg_target_hp",
        "reg_target_hp_after", "reg_target_hp_after_subtract",
        "reg_trigger_damage", "reg_player_hp", "reg_player_hp_after",
        "reg_player_hp_after_subtract", "reg_draw_count", "reg_discard_count",
        "reg_draw_count_after", "reg_counter_advanced", "reg_counter_triggered",
        "reg_discard_count_after",
        "reg_relic_counter", "reg_relic_counter_after", "reg_call_frame",
        "reg_call_remaining", "reg_call_frame_after", "reg_repeat_count",
        "field_before_0", "field_after_0", "field_before_1", "field_after_1",
        "field_before_2", "field_after_2", "field_before_3", "field_after_3",
        "field_before_4", "field_after_4", "field_before_5", "field_after_5",
        "field_before_6", "field_after_6", "field_before_7", "field_after_7",
        "field_before_8", "field_after_8", "node_children", "query_response_target",
        "action_paid_energy", "header_COUNTER_INCREMENT", "header_COUNTER_THRESHOLD",
        "header_COUNTER_RESET", "header_ENERGY_AMOUNT", "call_frame_remaining",
        "PLAYER_ENERGY", "PLAYER_BLOCK", "NEXT_DAMAGE_X2", "GAIN_BLOCK",
        "GAIN_ENERGY", "GAIN_STRENGTH", "GAIN_DEXTERITY", "INSERT_INTO_DRAW",
        "PILE_SHUFFLE", "NONE", "CARD_PLAY", "SKILL_PLAY", "DAMAGE_ALL",
    )),
)
VECTOR_TOKEN_VOCAB = tuple(dict.fromkeys((
    *VECTOR_SPECIAL_TOKENS,
    *VECTOR_EVENT_KIND_TOKENS,
    *VECTOR_ENTITY_TOKENS,
    *VECTOR_RELIC_ENTITY_TOKENS,
    *VECTOR_RUNTIME_ENTITY_VALUE_TOKENS,
    *VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS,
    *VECTOR_EXECUTOR_VALUE_TOKENS,
    *VECTOR_V7_PROTOCOL_TOKENS,
)))
VECTOR_TOKEN_INDEX = {token: index for index, token in enumerate(VECTOR_TOKEN_VOCAB)}
VECTOR_QUERY_TOKENS = frozenset({
    "QUERY_RANDOM_TARGET", "QUERY_TOP_DRAW_CARD", "QUERY_RANDOM_HAND_CARD",
    "QUERY_RANDOM_CARD", "QUERY_CREATED_CARD", "QUERY_CHOSEN_HAND_CARD",
})


def validate_vector_token_limits(
    tokens: Sequence[str],
    *,
    max_events: int | None = None,
    max_queries: int | None = None,
    max_nesting_depth: int | None = None,
) -> None:
    """Validate structural resource limits for a realized transcript."""

    events = sum(token == "<EVENT_BEGIN>" for token in tokens)
    queries = sum(token in VECTOR_QUERY_TOKENS for token in tokens)
    if max_events is not None and events > max_events:
        raise ValueError(f"transcript has {events} events; max is {max_events}")
    if max_queries is not None and queries > max_queries:
        raise ValueError(f"transcript has {queries} queries; max is {max_queries}")
    depth = 0
    observed_depth = 0
    for token in tokens:
        if token == "SCOPE_BEGIN":
            depth += 1
            observed_depth = max(observed_depth, depth)
        elif token == "SCOPE_END":
            depth -= 1
            if depth < 0:
                raise ValueError("transcript closes a scope that was not opened")
    if depth:
        raise ValueError("transcript ends with an open scope")
    if max_nesting_depth is not None and observed_depth > max_nesting_depth:
        raise ValueError(
            f"transcript nesting depth is {observed_depth}; max is {max_nesting_depth}"
        )


class TokenOwner(IntEnum):
    ENVIRONMENT = 0
    MODEL = 1


@dataclass(frozen=True)
class VectorQuery:
    query_id: str
    kind: str


@dataclass(frozen=True)
class VectorResponse:
    query_id: str
    kind: str
    response_token: str
    vector: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    observation_token: str | None = None


@dataclass(frozen=True)
class VectorValueRequest:
    """One model-requested exact runtime field."""

    request_id: str
    source_token: str


@dataclass(frozen=True)
class VectorValueBinding:
    """Environment binding and exact value for one runtime read."""

    request_id: str
    source_token: str
    binding_token: str
    value: int | float

    def __post_init__(self) -> None:
        if not self.source_token.startswith("VALUE_"):
            raise ValueError("runtime source must use a VALUE_* token")
        if not self.binding_token.startswith("VALUE_reg_"):
            raise ValueError("runtime binding must use a VALUE_reg_* token")
        try:
            finite = math.isfinite(float(self.value))
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime binding value must be numeric") from exc
        if not finite:
            raise ValueError("runtime binding value must be finite")


@dataclass(frozen=True)
class VectorEvent:
    """One event header plus an optional full-world continuous delta."""

    kind: str
    entity_ref: str
    delta: tuple[float, ...] | None = None
    repeat_count: int | None = None
    hand_delta: tuple[float, ...] | None = None
    calculations: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SerializedVectorEvents:
    tokens: tuple[str, ...]
    delta_targets: tuple[tuple[float, ...] | None, ...]
    hand_delta_targets: tuple[tuple[float, ...] | None, ...] = ()


def _calculation_number(value: int | float) -> list[str]:
    """Encode calculation numbers using the v5 least-significant-first form."""

    numeric = float(value)
    integral = numeric.is_integer()
    text = str(int(numeric)) if integral else format(numeric, ".8g")
    sign = text.startswith("-")
    if sign:
        text = text[1:]
    if integral:
        text = text[::-1]
    tokens = ["NUM_LSD_BEGIN" if integral else "NUM_BEGIN"]
    if sign:
        tokens.append("SIGN_NEG")
    tokens.extend(
        {".": "DECIMAL"}.get(char, f"DIGIT_{char}") for char in text
    )
    tokens.append("NUM_END")
    return tokens


def _parse_calculation_number(
    tokens: Sequence[str], start: int,
) -> tuple[int | float, int] | None:
    """Parse one v5 number and return its value and exclusive end position."""

    if start >= len(tokens) or tokens[start] not in {"NUM_BEGIN", "NUM_LSD_BEGIN"}:
        return None
    least_significant_first = tokens[start] == "NUM_LSD_BEGIN"
    index = start + 1
    negative = index < len(tokens) and tokens[index] == "SIGN_NEG"
    if negative:
        index += 1
    characters: list[str] = []
    while index < len(tokens) and tokens[index] != "NUM_END":
        token = tokens[index]
        if token.startswith("DIGIT_") and token.removeprefix("DIGIT_").isdigit():
            characters.append(token.removeprefix("DIGIT_"))
        elif token == "DECIMAL" and not least_significant_first:
            characters.append(".")
        else:
            return None
        index += 1
    if index >= len(tokens) or not characters:
        return None
    if least_significant_first:
        if "." in characters:
            return None
        value: int | float = int("".join(reversed(characters)))
    else:
        try:
            value = float("".join(characters))
        except ValueError:
            return None
    return (-value if negative else value), index + 1


def _completed_calculation_result_values(
    tokens: Sequence[str],
) -> dict[str, int | float]:
    """Read model-owned result assignments from the last completed block."""

    if not tokens or tokens[-1] != "CALCULATION_END":
        return {}
    begin = next(
        (
            index for index in range(len(tokens) - 2, -1, -1)
            if tokens[index] == "CALCULATION_BEGIN"
        ),
        None,
    )
    if begin is None:
        return {}
    result_fields = {
        "CALC_FIELD_result", "CALC_FIELD_source_result",
        "CALC_FIELD_destination_result",
    }
    values: dict[str, int | float] = {}
    index = begin + 1
    while index + 3 < len(tokens):
        if tokens[index] not in result_fields:
            index += 1
            continue
        register = tokens[index + 1]
        if not register.startswith("VALUE_reg_") or tokens[index + 2] != "CALC_VALUE":
            index += 1
            continue
        parsed = _parse_calculation_number(tokens, index + 3)
        if parsed is None:
            index += 1
            continue
        value, next_index = parsed
        values[register] = value
        index = next_index
    return values


def serialize_calculations(
    calculations: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Serialize explicit arithmetic blocks using the v5 calculation grammar."""

    tokens: list[str] = ["<CALCULATIONS_BEGIN>"]
    for calculation in calculations:
        calculation_type = _calculation_atom(calculation.get("type", "UNKNOWN"))
        if f"CALCULATION_{calculation_type}" not in VECTOR_CALCULATION_TOKENS:
            raise ValueError(f"unknown vector calculation type {calculation_type!r}")
        tokens.extend(("CALCULATION_BEGIN", f"CALCULATION_{calculation_type}"))
        for key, value in calculation.items():
            if key == "type":
                continue
            field_token = f"CALC_FIELD_{_calculation_atom(key)}"
            if field_token not in VECTOR_CALCULATION_FIELD_TOKENS:
                raise ValueError(f"unknown vector calculation field {key!r}")
            tokens.append(field_token)
            if isinstance(value, bool):
                tokens.append(f"BOOL_{int(value)}")
            elif isinstance(value, (int, float)):
                tokens.extend(_calculation_number(value))
            else:
                value_token = f"VALUE_{_calculation_atom(value)}"
                if value_token not in VECTOR_CALCULATION_VALUE_TOKENS:
                    raise ValueError(f"unknown vector calculation value {value!r}")
                tokens.append(value_token)
        tokens.append("CALCULATION_END")
    tokens.append("<CALCULATIONS_END>")
    return tuple(tokens)


def serialize_vector_events(
    events: Sequence[VectorEvent], *, include_output_markers: bool = True,
    max_events: int | None = None,
    max_queries: int | None = None,
    max_nesting_depth: int | None = None,
) -> SerializedVectorEvents:
    """Serialize event structure while keeping delta values off the token path."""

    tokens: list[str] = []
    delta_targets: list[tuple[float, ...] | None] = []
    hand_delta_targets: list[tuple[float, ...] | None] = []

    def add(
        token: str,
        delta: tuple[float, ...] | None = None,
        hand_delta: tuple[float, ...] | None = None,
    ) -> None:
        tokens.append(token)
        delta_targets.append(delta)
        hand_delta_targets.append(hand_delta)

    if include_output_markers:
        add("<OUTPUT_BEGIN>")
        add("<TRANSITION_BEGIN>")
    for event in events:
        add("<EVENT_BEGIN>")
        add("EVENT_KIND")
        add(event.kind)
        add("ENTITY_REF")
        add(event.entity_ref)
        if event.repeat_count is not None:
            if event.repeat_count not in (2, 3, 4, 5):
                raise ValueError(
                    "repeat_count must be between 2 and 5 for fixed loop vocabulary"
                )
            add("LOOP_KIND")
            add("ATTACK_HITS")
            add("LOOP_COUNT")
            add(f"LOOP_COUNT_{event.repeat_count}")
        for calculation_token in serialize_calculations(event.calculations):
            add(calculation_token)
        if event.delta is not None:
            add("DELTA_SLOT", tuple(float(value) for value in event.delta))
        if event.hand_delta is not None:
            add("HAND_DELTA_SLOT", tuple(float(value) for value in event.hand_delta))
        add("<EVENT_END>")
    if include_output_markers:
        add("<TRANSITION_END>")
        add("<OUTPUT_END>")
    validate_vector_token_limits(
        tokens,
        max_events=max_events,
        max_queries=max_queries,
        max_nesting_depth=max_nesting_depth,
    )
    return SerializedVectorEvents(
        tuple(tokens), tuple(delta_targets), tuple(hand_delta_targets)
    )


def serialize_sword_boomerang(
    hit_count: int, deltas: Sequence[Sequence[float]], *, target_slots: Sequence[int]
) -> SerializedVectorEvents:
    """Build the repeated random-target protocol for Sword Boomerang.

    Each hit gets its own ``QUERY_RANDOM_TARGET``. Responses are inserted by the
    environment at runtime; this helper emits the event skeleton and keeps the
    target slot in the event record for a realized trace.
    """

    if hit_count not in (3, 4):
        raise ValueError("Sword Boomerang hits must be 3 (base) or 4 (upgraded)")
    if len(deltas) != hit_count or len(target_slots) != hit_count:
        raise ValueError("one delta and target slot are required per hit")
    if any(int(slot) < 0 or int(slot) >= len(VECTOR_ENTITY_TOKENS) for slot in target_slots):
        raise ValueError("target slot is outside VECTOR_ENTITY_TOKENS")
    serialized = serialize_vector_events((VectorEvent(
        "EVENT_REPEAT", "ENTITY_SELF", repeat_count=hit_count,
    ),), include_output_markers=True)
    tokens = list(serialized.tokens[:-2])
    delta_targets = list(serialized.delta_targets[:-2])
    hand_delta_targets = list(serialized.hand_delta_targets[:-2])

    def add(
        token: str,
        delta: tuple[float, ...] | None = None,
        hand_delta: tuple[float, ...] | None = None,
    ) -> None:
        tokens.append(token)
        delta_targets.append(delta)
        hand_delta_targets.append(hand_delta)

    for delta, slot in zip(deltas, target_slots):
        # The observation token is a placeholder.  A live environment replaces
        # it with the selected enemy slot and supplies that slot as a vector
        # response through VectorOracle.
        add("<QUERY_BEGIN>")
        add("QUERY_RANDOM_TARGET")
        add("<QUERY_END>")
        add("<OBS_BEGIN>")
        add("OBS_RANDOM_TARGET")
        add("<OBS_END>")
        add("<EVENT_BEGIN>")
        add("EVENT_KIND")
        add("EVENT_DAMAGE")
        add("ENTITY_REF")
        add(f"ENTITY_ENEMY_SLOT_{int(slot)}")
        for calculation_token in serialize_calculations(()):
            add(calculation_token)
        add("DELTA_SLOT", tuple(float(value) for value in delta))
        add("<EVENT_END>")
    add("<TRANSITION_END>")
    add("<OUTPUT_END>")
    return SerializedVectorEvents(
        tuple(tokens), tuple(delta_targets), tuple(hand_delta_targets)
    )


def serialize_hand_operation(
    operation: str,
    hand_delta: Sequence[float],
    *,
    world_delta: Sequence[float] | None = None,
    selection_query: str | None = None,
    selection_queries: Sequence[str] | None = None,
    include_output_markers: bool = True,
) -> SerializedVectorEvents:
    """Serialize one hand bag mutation.

    The decoder receives the pre-operation hand summary in world memory and
    predicts ``HAND_DELTA_SLOT``.  Thus the post-operation composition is
    reconstructed as ``before + delta`` without exposing card ids or pile
    positions.  A selected/random card is an environment response when the
    operation needs one card instance to be identified.
    """

    operation_tokens = {
        "upgrade": "HAND_UPGRADE",
        "exhaust": "HAND_EXHAUST",
        "draw": "HAND_DRAW",
        "replace": "HAND_REPLACE",
    }
    try:
        op_token = operation_tokens[str(operation).casefold()]
    except KeyError as exc:
        raise ValueError(f"unknown hand operation {operation!r}") from exc
    if selection_query is not None and selection_queries is not None:
        raise ValueError("pass selection_query or selection_queries, not both")
    queries = (
        tuple(selection_queries)
        if selection_queries is not None
        else ((selection_query,) if selection_query is not None else ())
    )
    if any(query not in {
        "QUERY_RANDOM_HAND_CARD", "QUERY_CHOSEN_HAND_CARD",
        "QUERY_RANDOM_CARD", "QUERY_TOP_DRAW_CARD",
    } for query in queries):
        raise ValueError(f"not a supported hand/draw query: {queries}")

    tokens: list[str] = []
    world_targets: list[tuple[float, ...] | None] = []
    hand_targets: list[tuple[float, ...] | None] = []

    def add(
        token: str,
        delta: tuple[float, ...] | None = None,
        hand_target: tuple[float, ...] | None = None,
    ) -> None:
        tokens.append(token)
        world_targets.append(delta)
        hand_targets.append(hand_target)

    if include_output_markers:
        add("<OUTPUT_BEGIN>")
        add("<TRANSITION_BEGIN>")
    for query in queries:
        add("<QUERY_BEGIN>")
        add(query)
        add("<QUERY_END>")
        add("<OBS_BEGIN>")
        add("OBS_HAND_CARD" if "HAND_CARD" in query else "OBS_CARD_INSTANCE")
        add("<OBS_END>")
    add("<EVENT_BEGIN>")
    add("EVENT_KIND")
    add("EVENT_HAND_COMPOSITION")
    add("ENTITY_REF")
    add("ENTITY_SELF")
    add("HAND_OP")
    add(op_token)
    for calculation_token in serialize_calculations(()):
        add(calculation_token)
    if world_delta is not None:
        add("DELTA_SLOT", delta=tuple(float(value) for value in world_delta))
    add("HAND_DELTA_SLOT", hand_target=tuple(float(value) for value in hand_delta))
    add("<EVENT_END>")
    if include_output_markers:
        add("<TRANSITION_END>")
        add("<OUTPUT_END>")
    return SerializedVectorEvents(
        tuple(tokens), tuple(world_targets), tuple(hand_targets)
    )


def serialize_armaments_hand_operation(
    upgraded: bool,
    hand_delta: Sequence[float],
) -> SerializedVectorEvents:
    """Hand-composition target for base/upgraded Armaments.

    Base Armaments selects one hand card; upgraded Armaments applies to every
    card and therefore has no selection query.
    """

    return serialize_hand_operation(
        "upgrade",
        hand_delta,
        selection_query="QUERY_CHOSEN_HAND_CARD" if not upgraded else None,
    )


def serialize_burning_pact_hand_operation(
    hand_delta: Sequence[float],
) -> SerializedVectorEvents:
    """Hand-composition target for Burning Pact's chosen exhaust."""

    return serialize_hand_operation(
        "exhaust", hand_delta, selection_query="QUERY_CHOSEN_HAND_CARD"
    )


def serialize_burning_pact_operations(
    exhaust_hand_delta: Sequence[float],
    draw_hand_delta: Sequence[float],
    *,
    exhaust_world_delta: Sequence[float] | None = None,
    draw_world_delta: Sequence[float] | None = None,
    draw_query_kinds: Sequence[str] = (),
) -> SerializedVectorEvents:
    """Serialize Burning Pact as chosen exhaust followed by draw.

    Keeping the two atomic hand deltas separate preserves triggers between the
    operations (for example an on-exhaust draw) instead of learning only the
    action-level net hand change.
    """

    exhaust = serialize_hand_operation(
        "exhaust",
        exhaust_hand_delta,
        world_delta=exhaust_world_delta,
        selection_query="QUERY_CHOSEN_HAND_CARD",
        include_output_markers=False,
    )
    draw = serialize_hand_operation(
        "draw",
        draw_hand_delta,
        world_delta=draw_world_delta,
        selection_queries=draw_query_kinds,
        include_output_markers=False,
    )
    return SerializedVectorEvents(
        ("<OUTPUT_BEGIN>", "<TRANSITION_BEGIN>")
        + exhaust.tokens
        + draw.tokens
        + ("<TRANSITION_END>", "<OUTPUT_END>"),
        (None, None)
        + exhaust.delta_targets
        + draw.delta_targets
        + (None, None),
        (None, None)
        + exhaust.hand_delta_targets
        + draw.hand_delta_targets
        + (None, None),
    )


def serialize_havoc_skeleton() -> tuple[str, ...]:
    """Canonical nested-call skeleton for Havoc (Po Mie)."""

    return (
        "<OUTPUT_BEGIN>", "<TRANSITION_BEGIN>",
        "SCOPE_BEGIN", "SCOPE_MAIN", "CALL_BEGIN", "CALL_KIND",
        "PLAY_TOP_DRAW_CARD", "COST_MODE_IGNORE", "CALL_END",
        "<QUERY_BEGIN>", "QUERY_TOP_DRAW_CARD", "<QUERY_END>", "<OBS_BEGIN>",
        "OBS_CARD_INSTANCE", "<OBS_END>",
        "SCOPE_BEGIN", "SCOPE_CHILD", "PARENT_QUERY", "CARD_REF",
        "<CHILD_CARD_RULE_VECTOR>", "SCOPE_END", "SCOPE_END",
        "<TRANSITION_END>", "<OUTPUT_END>",
    )


@dataclass(frozen=True)
class VectorTranscript:
    tokens: tuple[str, ...]
    owners: tuple[TokenOwner, ...]
    responses: tuple[VectorResponse, ...] = ()
    value_bindings: tuple[VectorValueBinding, ...] = ()

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.owners):
            raise ValueError("transcript tokens/owners length mismatch")
        boundaries = {
            "<OUTPUT_BEGIN>": "<OUTPUT_END>",
            "<REGION_BEGIN>": "<REGION_END>",
        }
        if not self.tokens or self.tokens[0] not in boundaries:
            raise ValueError(
                "vector transcript must start with OUTPUT_BEGIN or REGION_BEGIN"
            )
        if self.tokens[-1] != boundaries[self.tokens[0]]:
            raise ValueError("vector transcript has a mismatched terminal token")

    def teacher_labels(self, token_index: Mapping[str, int], ignore_index: int = -100):
        ids = [token_index[token] for token in self.tokens]
        inputs = ids[:-1]
        labels = [
            ids[index + 1]
            if self.owners[index + 1] == TokenOwner.MODEL else ignore_index
            for index in range(len(ids) - 1)
        ]
        return inputs, labels

    def outputs_for_model_token(
        self, outputs: Sequence[Any], token: str,
    ) -> tuple[Any, ...]:
        """Select decoder rows aligned with one model-owned token kind."""

        model_tokens = tuple(
            value for value, owner in zip(self.tokens, self.owners)
            if owner == TokenOwner.MODEL
        )
        if len(outputs) != len(model_tokens):
            raise ValueError("decoder outputs do not align with model-owned tokens")
        return tuple(
            output for value, output in zip(model_tokens, outputs)
            if value == token
        )


class VectorTranscriptBuilder:
    def __init__(
        self, prefix_tokens: Sequence[str] = ("<OUTPUT_BEGIN>",),
    ) -> None:
        if not prefix_tokens:
            raise ValueError("vector transcript prefix cannot be empty")
        self.tokens = [str(token) for token in prefix_tokens]
        self.owners = [TokenOwner.ENVIRONMENT] * len(self.tokens)
        self.responses: list[VectorResponse] = []
        self.value_bindings: list[VectorValueBinding] = []

    def model(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.owners.extend([TokenOwner.MODEL] * len(tokens))

    def environment(self, *tokens: str) -> None:
        self.tokens.extend(tokens)
        self.owners.extend([TokenOwner.ENVIRONMENT] * len(tokens))

    def exchange(self, query: VectorQuery, response: VectorResponse) -> None:
        if query.kind not in VECTOR_QUERY_TOKENS:
            raise ValueError(f"not a supported query: {query.kind}")
        if response.query_id != query.query_id:
            raise ValueError("query and response ids do not match")
        if response.kind != query.kind:
            raise ValueError("query and response kinds do not match")
        self.model("<QUERY_BEGIN>", query.kind, "<QUERY_END>")
        observation = (
            (response.observation_token,)
            if response.observation_token is not None else ()
        )
        self.environment(
            "<OBS_BEGIN>", *observation, response.response_token, "<OBS_END>"
        )
        self.responses.append(response)

    def bind_value(
        self, request: VectorValueRequest, binding: VectorValueBinding,
    ) -> None:
        if binding.request_id != request.request_id:
            raise ValueError("value request and binding ids do not match")
        if binding.source_token != request.source_token:
            raise ValueError("value request and binding sources do not match")
        if not binding.binding_token.startswith("VALUE_reg_"):
            raise ValueError("runtime binding must use a VALUE_reg_* token")
        self.environment(
            "READ_BIND", binding.binding_token, "<READ_REQUEST_END>",
        )
        self.value_bindings.append(binding)

    def finish(self, end_token: str | None = None) -> VectorTranscript:
        if end_token is None:
            end_token = (
                "<REGION_END>"
                if self.tokens[0] == "<REGION_BEGIN>" else "<OUTPUT_END>"
            )
        if self.tokens[-1] != end_token:
            self.model(end_token)
        return VectorTranscript(
            tuple(self.tokens), tuple(self.owners), tuple(self.responses),
            tuple(self.value_bindings),
        )


class VectorOracle:
    """Ordered environment response provider for interactive decoding."""

    def __init__(self, responses: Sequence[VectorResponse]) -> None:
        self.responses = tuple(responses)
        self.cursor = 0

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.responses)

    def respond(self, query: VectorQuery) -> VectorResponse:
        if self.cursor >= len(self.responses):
            raise RuntimeError(f"unexpected query after trace end: {query.kind}")
        response = self.responses[self.cursor]
        if response.query_id != query.query_id or response.kind != query.kind:
            raise RuntimeError(
                f"oracle expected {response.kind}/{response.query_id}, "
                f"got {query.kind}/{query.query_id}"
            )
        self.cursor += 1
        return response


class VectorValueBinder:
    """Ordered environment-side provider for exact runtime bindings.

    A live environment may instead pass a callable to ``generate_interactive``.
    This ordered implementation is useful for deterministic evaluation and
    verifies that the model requested the expected semantic field.
    """

    def __init__(self, bindings: Sequence[VectorValueBinding]) -> None:
        self.bindings = tuple(bindings)
        self.cursor = 0

    @property
    def complete(self) -> bool:
        return self.cursor == len(self.bindings)

    def bind(self, request: VectorValueRequest) -> VectorValueBinding:
        if self.cursor >= len(self.bindings):
            raise RuntimeError(
                f"unexpected runtime read after binding trace end: {request.source_token}"
            )
        binding = self.bindings[self.cursor]
        if (
            binding.request_id != request.request_id
            or binding.source_token != request.source_token
        ):
            raise RuntimeError(
                f"value binder expected {binding.source_token}/{binding.request_id}, "
                f"got {request.source_token}/{request.request_id}"
            )
        self.cursor += 1
        return binding


class VectorValueStore:
    """Live exact-number store addressed by serialized semantic source tokens.

    ``bindings`` is normally scoped to one local NodeSpec invocation. A mapping
    is sufficient when each semantic source has one local alias; a callable can
    choose aliases from the full request when a node needs a custom policy.
    ``values`` may likewise be a mapping or a live environment callback.
    """

    def __init__(
        self,
        values: Mapping[str, int | float] | Callable[[str], int | float],
        bindings: Mapping[str, str] | Callable[[VectorValueRequest], str],
    ) -> None:
        self._values = (
            {
                self.source_token(source): value
                for source, value in values.items()
            }
            if isinstance(values, Mapping) else values
        )
        self._bindings = (
            {
                self.source_token(source): self.binding_token(binding)
                for source, binding in bindings.items()
            }
            if isinstance(bindings, Mapping) else bindings
        )

    @staticmethod
    def source_token(source: str) -> str:
        source = str(source)
        return source if source.startswith("VALUE_") else f"VALUE_{_calculation_atom(source)}"

    @staticmethod
    def binding_token(binding: str) -> str:
        binding = str(binding)
        return (
            binding
            if binding.startswith("VALUE_reg_")
            else f"VALUE_{_calculation_atom(binding)}"
        )

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        bindings: Mapping[str, str] | Callable[[VectorValueRequest], str],
        *,
        config: StateVectorConfig | None = None,
        extra_values: Mapping[str, int | float] | None = None,
    ) -> "VectorValueStore":
        """Build exact runtime values from a canonical unvectorized state."""

        config = config or StateVectorConfig()
        values: dict[str, int | float] = {
            source: value
            for source, value in _raw_state_values(state, config).items()
        }
        global_state = state.get("global", {})
        for slot, value in enumerate(global_state.get("relic_counters", ())):
            values[f"relic[{slot}].counter"] = value
        if extra_values:
            values.update({str(source): value for source, value in extra_values.items()})
        return cls(values, bindings)

    def read(self, source_token: str) -> int | float:
        source_token = self.source_token(source_token)
        try:
            value = (
                self._values[source_token]
                if isinstance(self._values, dict)
                else self._values(source_token)
            )
        except KeyError as exc:
            raise RuntimeError(
                f"runtime ValueStore has no value for {source_token}"
            ) from exc
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"runtime ValueStore value for {source_token} is not numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise RuntimeError(
                f"runtime ValueStore value for {source_token} is not finite"
            )
        return value

    def write(self, source_token: str, value: int | float) -> None:
        if not isinstance(self._values, dict):
            raise RuntimeError("callback-backed ValueStore cannot be written directly")
        if not math.isfinite(float(value)):
            raise ValueError("runtime ValueStore only accepts finite numbers")
        self._values[self.source_token(source_token)] = value

    def bind(self, request: VectorValueRequest) -> VectorValueBinding:
        try:
            binding_token = (
                self._bindings[request.source_token]
                if isinstance(self._bindings, dict)
                else self._bindings(request)
            )
        except KeyError as exc:
            raise RuntimeError(
                f"runtime ValueStore has no binding for {request.source_token}"
            ) from exc
        return VectorValueBinding(
            request_id=request.request_id,
            source_token=request.source_token,
            binding_token=self.binding_token(binding_token),
            value=self.read(request.source_token),
        )


# ---------------------------------------------------------------------------
# Vector world encoder and event decoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldEncoderConfig:
    player_dim: int
    enemy_dim: int
    pile_dim: int
    card_dim: int
    relic_dim: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    feedforward: int = 1024
    max_enemies: int = 5
    max_relics: int = 32
    dropout: float = 0.1
    hand_dim: int = 0
    action_upgrade_scale: float = 5.0


class VectorWorldEncoder(nn.Module):
    """Encode self, fixed enemy slots, piles, hand, action and relic vectors."""

    def __init__(self, config: WorldEncoderConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if config.hand_dim < 0:
            raise ValueError("hand_dim cannot be negative")
        self.config = config
        self.player_projection = nn.Linear(config.player_dim, config.d_model)
        self.enemy_projection = nn.Linear(config.enemy_dim, config.d_model)
        self.pile_projection = nn.Linear(config.pile_dim, config.d_model)
        self.card_projection = nn.Linear(config.card_dim, config.d_model)
        self.action_upgrade_projection = nn.Linear(1, config.d_model)
        self.relic_projection = nn.Linear(config.relic_dim, config.d_model)
        self.hand_projection = (
            nn.Linear(config.hand_dim, config.d_model)
            if config.hand_dim > 0 else None
        )
        self.relic_counter_projection = nn.Linear(1, config.d_model)
        self.role = nn.Embedding(5, config.d_model)
        # player + enemies + optional action card + piles + optional hand + relics
        self.position = nn.Embedding(4 + config.max_enemies + config.max_relics, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        player: torch.Tensor,
        enemies: torch.Tensor,
        piles: torch.Tensor,
        *,
        action_card: torch.Tensor | None = None,
        action_upgrade: torch.Tensor | None = None,
        hand_composition: torch.Tensor | None = None,
        relic_vectors: torch.Tensor | None = None,
        relic_counters: torch.Tensor | None = None,
        relic_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if player.ndim != 2 or enemies.ndim != 3 or piles.ndim != 2:
            raise ValueError("world vectors must be [B,D], [B,E,D], [B,D]")
        batch, enemy_count, _ = enemies.shape
        if enemy_count != self.config.max_enemies:
            raise ValueError("enemy count does not match WorldEncoderConfig.max_enemies")
        rows = [self.player_projection(player).unsqueeze(1), self.enemy_projection(enemies)]
        roles = [
            torch.zeros(batch, 1, dtype=torch.long, device=player.device),
            torch.ones(batch, enemy_count, dtype=torch.long, device=player.device),
        ]
        masks = [torch.zeros(batch, 1, dtype=torch.bool, device=player.device), torch.zeros(batch, enemy_count, dtype=torch.bool, device=player.device)]
        if action_card is not None:
            if action_card.ndim != 2:
                raise ValueError("action_card must be [B, card_dim]")
            action_hidden = self.card_projection(action_card)
            if action_upgrade is not None:
                if action_upgrade.ndim == 2 and action_upgrade.shape[-1] == 1:
                    action_upgrade = action_upgrade[:, 0]
                if action_upgrade.ndim != 1 or action_upgrade.shape[0] != batch:
                    raise ValueError("action_upgrade must be [B] or [B, 1]")
                action_hidden = action_hidden + self.action_upgrade_projection(
                    (
                        action_upgrade.to(
                            device=action_hidden.device, dtype=action_hidden.dtype
                        )
                        / max(self.config.action_upgrade_scale, 1e-6)
                    ).unsqueeze(-1)
                )
            rows.append(action_hidden.unsqueeze(1))
            roles.append(torch.full((batch, 1), 2, dtype=torch.long, device=player.device))
            masks.append(torch.zeros(batch, 1, dtype=torch.bool, device=player.device))
        rows.append(self.pile_projection(piles).unsqueeze(1))
        roles.append(torch.full((batch, 1), 3, dtype=torch.long, device=player.device))
        masks.append(torch.zeros(batch, 1, dtype=torch.bool, device=player.device))
        if hand_composition is not None:
            if self.hand_projection is None:
                raise ValueError("world encoder was not configured with hand_dim")
            if hand_composition.ndim != 2 or hand_composition.shape != (batch, self.config.hand_dim):
                raise ValueError("hand_composition must be [B, hand_dim]")
            rows.append(self.hand_projection(hand_composition).unsqueeze(1))
            roles.append(torch.full((batch, 1), 4, dtype=torch.long, device=player.device))
            masks.append(torch.zeros(batch, 1, dtype=torch.bool, device=player.device))
        if relic_vectors is not None:
            if relic_vectors.ndim != 3 or relic_vectors.shape[-1] != self.config.relic_dim:
                raise ValueError("relic_vectors must be [B,R,relic_dim]")
            relic_count = relic_vectors.shape[1]
            if relic_count > self.config.max_relics:
                raise ValueError("relic vector count exceeds max_relics")
            if relic_mask is None:
                relic_mask = torch.ones(
                    batch, relic_count, dtype=torch.bool, device=player.device
                )
            elif relic_mask.shape != (batch, relic_count):
                raise ValueError("relic_mask must be [B,R] and align with relic_vectors")
            elif relic_mask.device != player.device:
                relic_mask = relic_mask.to(player.device)
            if relic_counters is None:
                relic_counters = torch.zeros(batch, relic_count, device=player.device)
            elif relic_counters.shape != (batch, relic_count):
                raise ValueError("relic_counters must be [B,R] and align with relic_vectors")
            relic = self.relic_projection(relic_vectors)
            relic = relic + self.relic_counter_projection(relic_counters.unsqueeze(-1))
            rows.append(relic)
            roles.append(torch.full((batch, relic_count), 3, dtype=torch.long, device=player.device))
            masks.append(~relic_mask)
        hidden = torch.cat(rows, dim=1)
        role_ids = torch.cat(roles, dim=1)
        padding = torch.cat(masks, dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device).clamp_max(self.position.num_embeddings - 1)
        hidden = hidden + self.role(role_ids) + self.position(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        return self.norm(hidden), padding


@dataclass(frozen=True)
class DeltaHeadConfig:
    input_dim: int
    delta_dim: int
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.0


class DeltaHead(nn.Module):
    """Project an event hidden state to a continuous world delta."""

    def __init__(self, config: DeltaHeadConfig) -> None:
        super().__init__()
        if config.n_layers < 1:
            raise ValueError("delta head needs at least one layer")
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.n_layers - 1):
            layers.extend((nn.Linear(in_dim, config.hidden_dim), nn.GELU()))
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.delta_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


@dataclass(frozen=True)
class EventDecoderConfig:
    token_vocab_size: int
    pad_id: int = 0
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    feedforward: int = 1024
    max_tokens: int = 2048
    response_dim: int = 0
    dropout: float = 0.1
    max_events: int = 256
    max_queries: int = 128
    max_nesting_depth: int = 8
    hand_delta_dim: int = 0


@dataclass
class EventDecoderOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    delta: torch.Tensor
    hand_delta: torch.Tensor | None = None


class VectorEventDecoder(nn.Module):
    """Causal event decoder with a continuous delta head."""

    def __init__(
        self,
        config: EventDecoderConfig,
        delta_head: DeltaHead,
        hand_delta_head: DeltaHead | None = None,
    ) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if config.response_dim < 0 or config.hand_delta_dim < 0:
            raise ValueError("response_dim and hand_delta_dim cannot be negative")
        self.config = config
        self.delta_head = delta_head
        if config.hand_delta_dim > 0:
            if hand_delta_head is None:
                raise ValueError("hand_delta_head is required when hand_delta_dim > 0")
            if hand_delta_head.config.delta_dim != config.hand_delta_dim:
                raise ValueError("hand delta head width does not match decoder config")
        elif hand_delta_head is not None:
            raise ValueError("hand_delta_head supplied but hand_delta_dim is zero")
        self.hand_delta_head = hand_delta_head
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = nn.Embedding(config.max_tokens, config.d_model)
        self.response_projection = (
            nn.Linear(config.response_dim, config.d_model)
            if config.response_dim > 0 else None
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.token_vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        target_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        *,
        response_vectors: torch.Tensor | None = None,
    ) -> EventDecoderOutput:
        if target_input_ids.ndim != 2:
            raise ValueError("target_input_ids must be [B,T]")
        batch, length = target_input_ids.shape
        if length > self.config.max_tokens:
            raise ValueError("target exceeds EventDecoderConfig.max_tokens")
        hidden = self.token_embedding(target_input_ids)
        hidden = hidden + self.position(torch.arange(length, device=target_input_ids.device))[None]
        if response_vectors is not None:
            if self.response_projection is None:
                raise ValueError("decoder was not configured with response_dim")
            if response_vectors.shape[:2] != (batch, length):
                raise ValueError("response_vectors must align with target_input_ids")
            hidden = hidden + self.response_projection(response_vectors)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=target_input_ids.device), diagonal=1)
        padding = target_input_ids.eq(self.config.pad_id)
        hidden = self.decoder(
            hidden, memory, tgt_mask=causal,
            tgt_key_padding_mask=padding, memory_key_padding_mask=memory_padding,
        )
        hidden = self.norm(hidden)
        hand_delta = (
            self.hand_delta_head(hidden)
            if self.hand_delta_head is not None
            else hidden.new_empty((*hidden.shape[:2], 0))
        )
        return EventDecoderOutput(
            self.output(hidden), hidden, self.delta_head(hidden), hand_delta
        )


@dataclass(frozen=True)
class VectorRuleModelConfig:
    state: StateVectorConfig = field(default_factory=StateVectorConfig)
    card_encoder: ProgramEncoderConfig | None = None
    relic_encoder: ProgramEncoderConfig | None = None
    # New configurations share the contextual rule Transformer by default.
    # ``from_dict`` treats an absent field as False for old checkpoints.
    share_rule_transformer: bool = False
    world_encoder: WorldEncoderConfig | None = None
    decoder: EventDecoderConfig | None = None
    delta_head: DeltaHeadConfig | None = None
    hand_encoder: HandCompositionConfig | None = None
    hand_delta_head: DeltaHeadConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorRuleModelConfig":
        def build(name: str, type_: type):
            value = payload.get(name)
            return type_(**value) if value is not None else None

        state_payload = dict(payload.get("state", {}))
        # Checkpoints produced before v7 used ``relic_counter_slots`` as an
        # anonymous presence width. Preserve that tensor shape when loading
        # them; new configs carry stable relic ids in ``relic_presence_names``.
        if "relic_presence_names" not in state_payload:
            legacy_width = int(state_payload.get("relic_counter_slots", 32))
            state_payload["relic_presence_names"] = tuple(
                f"LEGACY_SLOT_{index}" for index in range(max(0, legacy_width))
            )
        for key in (
            "player_buff_names", "monster_buff_names", "counter_names",
            "player_scalar_names", "enemy_scalar_names", "pile_names",
            "relic_summary_names", "relic_presence_names",
        ):
            if key in state_payload:
                state_payload[key] = tuple(state_payload[key])

        return cls(
            state=StateVectorConfig(**state_payload),
            card_encoder=build("card_encoder", ProgramEncoderConfig),
            relic_encoder=build("relic_encoder", ProgramEncoderConfig),
            share_rule_transformer=bool(payload.get("share_rule_transformer", False)),
            hand_encoder=build("hand_encoder", HandCompositionConfig),
            world_encoder=build("world_encoder", WorldEncoderConfig),
            decoder=build("decoder", EventDecoderConfig),
            delta_head=build("delta_head", DeltaHeadConfig),
            hand_delta_head=build("hand_delta_head", DeltaHeadConfig),
        )

    @classmethod
    def load(cls, path: str) -> "VectorRuleModelConfig":
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


class VectorRuleWorldModel(nn.Module):
    """Full v6 model: anonymous rule encoders, world encoder and event decoder."""

    def __init__(
        self,
        config: VectorRuleModelConfig,
        *,
        card_encoder: CardRuleEncoder | None = None,
        relic_encoder: RelicRuleEncoder | None = None,
        world_encoder: VectorWorldEncoder | None = None,
        decoder: VectorEventDecoder | None = None,
        shared_rule_transformer: RuleProgramTransformer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        state = config.state
        card_cfg = config.card_encoder or ProgramEncoderConfig(
            token_vocab_size=len(CARD_RULE_VOCAB)
        )
        relic_cfg = config.relic_encoder or ProgramEncoderConfig(
            token_vocab_size=len(RELIC_RULE_VOCAB)
        )
        world_cfg = config.world_encoder
        if world_cfg is None:
            hand_cfg = config.hand_encoder or HandCompositionConfig(
                card_dim=card_cfg.output_dim
            )
            world_cfg = WorldEncoderConfig(
                player_dim=state.player_dim, enemy_dim=state.enemy_dim,
                pile_dim=state.pile_dim, card_dim=card_cfg.output_dim,
                relic_dim=relic_cfg.output_dim, hand_dim=hand_cfg.output_dim,
                max_enemies=state.max_enemies,
            )
        else:
            hand_cfg = config.hand_encoder or HandCompositionConfig(
                card_dim=card_cfg.output_dim,
                output_dim=world_cfg.hand_dim if world_cfg.hand_dim > 0 else 1,
            )
        if card_cfg.output_dim != world_cfg.card_dim:
            raise ValueError("card encoder output_dim must match world card_dim")
        if relic_cfg.output_dim != world_cfg.relic_dim:
            raise ValueError("relic encoder output_dim must match world relic_dim")
        if hand_cfg.card_dim != card_cfg.output_dim:
            raise ValueError("hand encoder card_dim must match card encoder output_dim")
        if world_cfg.hand_dim > 0 and hand_cfg.output_dim != world_cfg.hand_dim:
            raise ValueError("hand encoder output_dim must match world hand_dim")
        if (
            world_cfg.player_dim != state.player_dim
            or world_cfg.enemy_dim != state.enemy_dim
            or world_cfg.pile_dim != state.pile_dim
        ):
            raise ValueError("world encoder state dimensions do not match StateVectorConfig")
        decoder_cfg = config.decoder or EventDecoderConfig(
            token_vocab_size=len(VECTOR_TOKEN_VOCAB),
            d_model=world_cfg.d_model,
            hand_delta_dim=(
                hand_cfg.composition_dim
                if config.hand_delta_head is not None else 0
            ),
        )
        delta_cfg = config.delta_head or DeltaHeadConfig(
            input_dim=decoder_cfg.d_model,
            delta_dim=state.world_dim,
        )
        if delta_cfg.input_dim != decoder_cfg.d_model:
            raise ValueError("delta head input_dim must equal decoder d_model")
        hand_delta_cfg = config.hand_delta_head
        if decoder_cfg.hand_delta_dim > 0:
            hand_delta_cfg = hand_delta_cfg or DeltaHeadConfig(
                input_dim=decoder_cfg.d_model,
                delta_dim=decoder_cfg.hand_delta_dim,
            )
            if hand_delta_cfg.input_dim != decoder_cfg.d_model:
                raise ValueError("hand delta head input_dim must equal decoder d_model")
            if hand_delta_cfg.delta_dim != decoder_cfg.hand_delta_dim:
                raise ValueError("hand delta head width must match decoder hand_delta_dim")
        elif hand_delta_cfg is not None:
            raise ValueError("hand_delta_head requires decoder.hand_delta_dim > 0")
        share_rule_transformer = bool(config.share_rule_transformer)
        if share_rule_transformer:
            shared = shared_rule_transformer or (
                getattr(card_encoder, "backbone", None)
                if card_encoder is not None else None
            ) or RuleProgramTransformer(
                token_dim=card_cfg.token_dim,
                n_heads=card_cfg.n_heads,
                n_layers=card_cfg.n_layers,
                feedforward=card_cfg.feedforward,
                max_tokens=max(card_cfg.max_tokens, relic_cfg.max_tokens),
                dropout=card_cfg.dropout,
            )
            compatible = (
                card_cfg.token_dim == relic_cfg.token_dim
                and card_cfg.n_heads == relic_cfg.n_heads
                and card_cfg.n_layers == relic_cfg.n_layers
                and card_cfg.feedforward == relic_cfg.feedforward
                and card_cfg.dropout == relic_cfg.dropout
            )
            if not compatible:
                raise ValueError(
                    "shared rule Transformer requires matching card/relic "
                    "token_dim, heads, layers, feedforward and dropout"
                )
            if card_encoder is not None and getattr(card_encoder, "backbone", None) is not shared:
                raise ValueError(
                    "a supplied card_encoder must use the shared rule Transformer"
                )
            if relic_encoder is not None and getattr(relic_encoder, "backbone", None) is not shared:
                raise ValueError(
                    "a supplied relic_encoder must use the shared rule Transformer"
                )
        else:
            shared = None
        # Expose a convenient diagnostic handle without registering a third
        # state-dict path for the same shared module.
        object.__setattr__(self, "shared_rule_transformer", shared)
        self.card_encoder = card_encoder or CardRuleEncoder(
            card_cfg, backbone=shared
        )
        self.relic_encoder = relic_encoder or RelicRuleEncoder(
            relic_cfg, backbone=shared, register_backbone=(shared is None)
        )
        self.hand_encoder = HandCompositionEncoder(hand_cfg)
        if world_cfg.max_enemies != state.max_enemies:
            raise ValueError("world encoder max_enemies must match state config")
        self.world_encoder = world_encoder or VectorWorldEncoder(world_cfg)
        self.memory_projection = (
            nn.Identity()
            if world_cfg.d_model == decoder_cfg.d_model
            else nn.Linear(world_cfg.d_model, decoder_cfg.d_model)
        )
        self.decoder = decoder or VectorEventDecoder(
            decoder_cfg,
            DeltaHead(delta_cfg),
            DeltaHead(hand_delta_cfg) if hand_delta_cfg is not None else None,
        )

    def encode_world(
        self,
        state: VectorizedState,
        *,
        action_card_vector: torch.Tensor | None = None,
        action_upgrade: torch.Tensor | None = None,
        hand_card_vectors: torch.Tensor | None = None,
        hand_upgrades: torch.Tensor | None = None,
        hand_mask: torch.Tensor | None = None,
        hand_composition: torch.Tensor | None = None,
        relic_vectors: torch.Tensor | None = None,
        relic_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        player = state.player.unsqueeze(0) if state.player.ndim == 1 else state.player
        enemies = state.enemies.unsqueeze(0) if state.enemies.ndim == 2 else state.enemies
        piles = state.piles.unsqueeze(0) if state.piles.ndim == 1 else state.piles
        counters = state.relic_counters
        if counters is not None and counters.ndim == 1:
            counters = counters.unsqueeze(0)
        if hand_card_vectors is not None and hand_composition is not None:
            raise ValueError("pass hand_card_vectors or hand_composition, not both")
        if hand_card_vectors is not None:
            if hand_card_vectors.ndim == 2:
                hand_card_vectors = hand_card_vectors.unsqueeze(0)
            if hand_upgrades is not None and hand_upgrades.ndim == 1:
                hand_upgrades = hand_upgrades.unsqueeze(0)
            if hand_mask is not None and hand_mask.ndim == 1:
                hand_mask = hand_mask.unsqueeze(0)
            hand_composition = self.hand_encoder(
                hand_card_vectors, hand_upgrades, hand_mask
            )
        elif hand_composition is not None and hand_composition.ndim == 1:
            hand_composition = hand_composition.unsqueeze(0)
        memory, padding = self.world_encoder(
            player, enemies, piles,
            action_card=action_card_vector,
            action_upgrade=action_upgrade,
            hand_composition=hand_composition,
            relic_vectors=relic_vectors,
            relic_counters=counters,
            relic_mask=relic_mask,
        )
        return self.memory_projection(memory), padding

    def forward(
        self,
        state: VectorizedState,
        target_input_ids: torch.Tensor,
        *,
        action_card_vector: torch.Tensor | None = None,
        action_upgrade: torch.Tensor | None = None,
        hand_card_vectors: torch.Tensor | None = None,
        hand_upgrades: torch.Tensor | None = None,
        hand_mask: torch.Tensor | None = None,
        hand_composition: torch.Tensor | None = None,
        relic_vectors: torch.Tensor | None = None,
        relic_mask: torch.Tensor | None = None,
        response_vectors: torch.Tensor | None = None,
    ) -> EventDecoderOutput:
        memory, padding = self.encode_world(
            state,
            action_card_vector=action_card_vector,
            action_upgrade=action_upgrade,
            hand_card_vectors=hand_card_vectors,
            hand_upgrades=hand_upgrades,
            hand_mask=hand_mask,
            hand_composition=hand_composition,
            relic_vectors=relic_vectors,
            relic_mask=relic_mask,
        )
        return self.decoder(
            target_input_ids, memory, padding, response_vectors=response_vectors
        )

    @torch.no_grad()
    def generate(
        self,
        state: VectorizedState,
        *,
        output_begin_id: int,
        output_end_id: int,
        max_new_tokens: int | None = None,
        action_card_vector: torch.Tensor | None = None,
        action_upgrade: torch.Tensor | None = None,
        hand_card_vectors: torch.Tensor | None = None,
        hand_upgrades: torch.Tensor | None = None,
        hand_mask: torch.Tensor | None = None,
        hand_composition: torch.Tensor | None = None,
        relic_vectors: torch.Tensor | None = None,
        relic_mask: torch.Tensor | None = None,
        return_hand_deltas: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy non-interactive decoding; returns structure ids and deltas."""

        self.eval()
        memory, memory_padding = self.encode_world(
            state,
            action_card_vector=action_card_vector,
            action_upgrade=action_upgrade,
            hand_card_vectors=hand_card_vectors,
            hand_upgrades=hand_upgrades,
            hand_mask=hand_mask,
            hand_composition=hand_composition,
            relic_vectors=relic_vectors,
            relic_mask=relic_mask,
        )
        batch = memory.shape[0]
        generated = torch.full(
            (batch, 1), output_begin_id, dtype=torch.long, device=memory.device
        )
        deltas = torch.zeros(
            batch, 0, self.config.state.world_dim, device=memory.device
        )
        hand_deltas = torch.zeros(
            batch, 0, self.decoder.config.hand_delta_dim, device=memory.device
        )
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        limit = max_new_tokens or self.decoder.config.max_tokens - 1
        for _ in range(limit):
            output = self.decoder(generated, memory, memory_padding)
            next_id = output.logits[:, -1].argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, output_end_id), next_id)
            generated = torch.cat((generated, next_id[:, None]), dim=1)
            deltas = torch.cat((deltas, output.delta[:, -1:, :]), dim=1)
            predicted_hand_delta = output.hand_delta
            if predicted_hand_delta is None:
                predicted_hand_delta = hand_deltas.new_zeros(
                    batch, 1, self.decoder.config.hand_delta_dim
                )
            hand_deltas = torch.cat((hand_deltas, predicted_hand_delta[:, -1:, :]), dim=1)
            finished |= next_id.eq(output_end_id)
            if bool(finished.all()):
                break
        if return_hand_deltas:
            return generated, deltas, hand_deltas
        return generated, deltas

    @torch.no_grad()
    def generate_interactive(
        self,
        state: VectorizedState,
        *,
        output_begin_id: int,
        output_end_id: int,
        token_index: Mapping[str, int],
        oracle: VectorOracle | Callable[[VectorQuery], VectorResponse] | None = None,
        value_binder: (
            VectorValueBinder
            | VectorValueStore
            | Callable[[VectorValueRequest], VectorValueBinding]
            | None
        ) = None,
        prefix_ids: Sequence[int] | torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        action_card_vector: torch.Tensor | None = None,
        action_upgrade: torch.Tensor | None = None,
        hand_card_vectors: torch.Tensor | None = None,
        hand_upgrades: torch.Tensor | None = None,
        hand_mask: torch.Tensor | None = None,
        hand_composition: torch.Tensor | None = None,
        relic_vectors: torch.Tensor | None = None,
        relic_mask: torch.Tensor | None = None,
        return_hand_deltas: bool = False,
    ) -> (
        tuple[VectorTranscript, tuple[torch.Tensor, ...]]
        | tuple[VectorTranscript, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
    ):
        """Decode one transition with environment-owned query and value reads.

        The oracle response is appended as an environment-owned observation and
        its vector is projected into the decoder context on the next step. For
        a numeric ``READ_REQUEST``, ``value_binder`` supplies the register name
        and exact value. The binding is appended immediately. Model-owned
        ``CALC_VALUE`` results are recorded when their calculation closes;
        every later use of either kind of valued register as an operand is
        followed by an environment-owned ``CONST_VALUE`` expansion. ``prefix_ids`` may
        provide a masked schema-v7 NodeSpec ending in ``<NODE_EXECUTE>``; in
        that mode ``output_end_id`` is the local ``<REGION_END>`` id.
        """

        self.eval()
        memory, memory_padding = self.encode_world(
            state,
            action_card_vector=action_card_vector,
            action_upgrade=action_upgrade,
            hand_card_vectors=hand_card_vectors,
            hand_upgrades=hand_upgrades,
            hand_mask=hand_mask,
            hand_composition=hand_composition,
            relic_vectors=relic_vectors,
            relic_mask=relic_mask,
        )
        if memory.shape[0] != 1:
            raise ValueError("interactive generation currently supports one sample")
        id_to_token = {int(index): str(token) for token, index in token_index.items()}
        terminal_token = id_to_token.get(int(output_end_id))
        if terminal_token is None:
            raise ValueError("output_end_id is absent from token_index")
        if prefix_ids is None:
            initial_ids = (int(output_begin_id),)
        elif isinstance(prefix_ids, torch.Tensor):
            if prefix_ids.ndim != 1:
                raise ValueError("interactive prefix_ids tensor must be one-dimensional")
            initial_ids = tuple(int(value) for value in prefix_ids.tolist())
        else:
            initial_ids = tuple(int(value) for value in prefix_ids)
        if not initial_ids:
            raise ValueError("interactive prefix_ids cannot be empty")
        unknown_prefix = [value for value in initial_ids if value not in id_to_token]
        if unknown_prefix:
            raise ValueError(
                f"interactive prefix contains an unknown token id: {unknown_prefix[0]}"
            )
        if len(initial_ids) >= self.decoder.config.max_tokens:
            raise ValueError("interactive prefix leaves no decoder generation capacity")
        initial_tokens = tuple(id_to_token[value] for value in initial_ids)
        generated = torch.tensor(
            [initial_ids], dtype=torch.long, device=memory.device,
        )
        response_rows: list[torch.Tensor] = [
            torch.zeros(self.decoder.config.response_dim, device=memory.device)
            for _ in initial_ids
        ]
        builder = VectorTranscriptBuilder(initial_tokens)
        delta_rows: list[torch.Tensor] = []
        hand_delta_rows: list[torch.Tensor] = []
        query_number = 0
        pending_query: VectorQuery | None = None
        read_number = 0
        bound_values: dict[str, int | float] = {}
        result_fields = {
            "CALC_FIELD_result", "CALC_FIELD_source_result",
            "CALC_FIELD_destination_result",
        }

        def append_environment_tokens(
            tokens: Sequence[str],
            *, response_vector_at: int | None = None,
            response_vector: torch.Tensor | None = None,
        ) -> None:
            nonlocal generated
            missing = [token for token in tokens if token not in token_index]
            if missing:
                raise RuntimeError(
                    f"interactive environment token is absent from vocabulary: {missing[0]}"
                )
            if generated.shape[1] + len(tokens) > self.decoder.config.max_tokens:
                raise RuntimeError(
                    "interactive environment injection exceeds decoder max_tokens"
                )
            token_ids = torch.tensor(
                [[token_index[token] for token in tokens]],
                dtype=torch.long,
                device=memory.device,
            )
            generated = torch.cat((generated, token_ids), dim=1)
            zero = torch.zeros(self.decoder.config.response_dim, device=memory.device)
            for index in range(len(tokens)):
                if response_vector_at == index and response_vector is not None:
                    response_rows.append(response_vector)
                else:
                    response_rows.append(zero.clone())

        limit = max_new_tokens or self.decoder.config.max_tokens - 1
        for _ in range(limit):
            if generated.shape[1] >= self.decoder.config.max_tokens:
                raise RuntimeError(
                    "interactive generation exhausted decoder max_tokens"
                )
            response_vectors = None
            if self.decoder.config.response_dim:
                response_vectors = torch.stack(response_rows).unsqueeze(0)
            output = self.decoder(
                generated, memory, memory_padding, response_vectors=response_vectors
            )
            next_id = int(output.logits[0, -1].argmax())
            token = id_to_token.get(
                next_id, terminal_token if next_id == output_end_id else "<UNK>"
            )
            delta_rows.append(output.delta[0, -1].detach())
            if output.hand_delta is not None:
                hand_delta_rows.append(output.hand_delta[0, -1].detach())
            else:
                hand_delta_rows.append(
                    torch.zeros(self.decoder.config.hand_delta_dim, device=memory.device)
                )
            generated = torch.cat((
                generated,
                torch.tensor([[next_id]], dtype=torch.long, device=memory.device),
            ), dim=1)
            response_rows.append(torch.zeros(self.decoder.config.response_dim, device=memory.device))
            if token == "<QUERY_BEGIN>":
                if pending_query is not None:
                    raise RuntimeError("nested interactive queries are not supported")
                builder.model(token)
                continue
            if token in VECTOR_QUERY_TOKENS:
                if query_number >= self.decoder.config.max_queries:
                    raise RuntimeError("interactive generation exceeded max_queries")
                if builder.tokens[-1] != "<QUERY_BEGIN>":
                    raise RuntimeError(
                        f"query token is missing QUERY_BEGIN: {token}"
                    )
                pending_query = VectorQuery(f"q{query_number}", token)
                query_number += 1
                builder.model(token)
                continue
            if token == "<QUERY_END>" and pending_query is not None:
                if oracle is None:
                    raise RuntimeError(
                        f"random/card query requires oracle: {pending_query.kind}"
                    )
                query = pending_query
                response = (
                    oracle.respond(query)
                    if isinstance(oracle, VectorOracle) else oracle(query)
                )
                if response.kind != query.kind:
                    raise RuntimeError(
                        f"oracle returned {response.kind} for {query.kind}"
                    )
                builder.model(token)
                observation = (
                    (response.observation_token,)
                    if response.observation_token is not None else ()
                )
                builder.environment(
                    "<OBS_BEGIN>", *observation, response.response_token,
                    "<OBS_END>",
                )
                builder.responses.append(response)
                response_vector = torch.as_tensor(response.vector, dtype=torch.float32, device=memory.device)
                if self.decoder.config.response_dim:
                    if response_vector.numel() != self.decoder.config.response_dim:
                        raise ValueError("oracle response vector has the wrong width")
                append_environment_tokens(
                    (
                        "<OBS_BEGIN>", *observation, response.response_token,
                        "<OBS_END>",
                    ),
                    response_vector_at=(
                        0 if self.decoder.config.response_dim else None
                    ),
                    response_vector=(
                        response_vector if self.decoder.config.response_dim else None
                    ),
                )
                pending_query = None
                continue
            previous_token = builder.tokens[-1]
            builder.model(token)
            if token == "CALCULATION_END":
                bound_values.update(
                    _completed_calculation_result_values(builder.tokens)
                )
            if (
                token.startswith("VALUE_")
                and len(builder.tokens) >= 3
                and builder.tokens[-3] == "<READ_REQUEST_BEGIN>"
                and builder.tokens[-2] == "READ_SOURCE"
            ):
                if value_binder is None:
                    raise RuntimeError(
                        f"numeric runtime read requires value_binder: {token}"
                    )
                request = VectorValueRequest(f"r{read_number}", token)
                read_number += 1
                binding = (
                    value_binder.bind(request)
                    if isinstance(value_binder, (VectorValueBinder, VectorValueStore))
                    else value_binder(request)
                )
                builder.bind_value(request, binding)
                append_environment_tokens(
                    ("READ_BIND", binding.binding_token, "<READ_REQUEST_END>")
                )
                bound_values[binding.binding_token] = binding.value
                continue
            if (
                previous_token.startswith("CALC_FIELD_")
                and previous_token not in result_fields
                and token in bound_values
            ):
                constant_tokens = (
                    "CONST_VALUE", *_calculation_number(bound_values[token]),
                )
                builder.environment(*constant_tokens)
                append_environment_tokens(constant_tokens)
                continue
            if next_id == output_end_id:
                break
        else:
            raise RuntimeError(
                f"interactive generation reached max_new_tokens without {terminal_token}"
            )
        transcript = builder.finish(terminal_token)
        validate_vector_token_limits(
            transcript.tokens,
            max_events=self.decoder.config.max_events,
            max_queries=self.decoder.config.max_queries,
            max_nesting_depth=self.decoder.config.max_nesting_depth,
        )
        if return_hand_deltas:
            return transcript, tuple(delta_rows), tuple(hand_delta_rows)
        return transcript, tuple(delta_rows)


def default_vector_model_config(
    *,
    card_dim: int = 128,
    relic_dim: int = 128,
    hand_dim: int = 128,
    world_dim: int = 256,
    decoder_dim: int = 256,
    card_token_dim: int = 64,
    relic_token_dim: int = 64,
    card_feedforward: int = 256,
    relic_feedforward: int = 256,
    card_max_tokens: int = 256,
    relic_max_tokens: int = 128,
    card_layers: int = 2,
    relic_layers: int = 2,
    world_layers: int = 3,
    decoder_layers: int = 4,
    world_heads: int = 8,
    decoder_heads: int = 8,
    card_heads: int = 4,
    relic_heads: int = 4,
    hand_hidden: int = 256,
    hand_heads: int = 4,
    hand_feedforward: int = 512,
    hand_layers: int = 2,
    hand_max_cards: int = 10,
    hand_dropout: float = 0.0,
    world_feedforward: int = 1024,
    decoder_feedforward: int = 1024,
    delta_hidden: int = 512,
    delta_layers: int = 2,
    hand_delta_hidden: int = 512,
    hand_delta_layers: int = 2,
    card_dropout: float = 0.0,
    relic_dropout: float = 0.0,
    world_dropout: float = 0.1,
    decoder_dropout: float = 0.1,
    delta_dropout: float = 0.0,
    hand_delta_dropout: float = 0.0,
    predict_hand_delta: bool = False,
    response_dim: int = 0,
    max_enemies: int = 5,
    max_relics: int = 32,
    decoder_max_tokens: int = 2048,
    action_upgrade_scale: float = 5.0,
    max_events: int = 256,
    max_queries: int = 128,
    max_nesting_depth: int = 8,
    share_rule_transformer: bool = True,
) -> VectorRuleModelConfig:
    """Build a fully explicit config; every module size is an argument."""

    state = StateVectorConfig(
        max_enemies=max_enemies, relic_counter_slots=max_relics
    )
    card = ProgramEncoderConfig(
        token_vocab_size=len(CARD_RULE_VOCAB), token_dim=card_token_dim,
        output_dim=card_dim, n_heads=card_heads, n_layers=card_layers,
        feedforward=card_feedforward, max_tokens=card_max_tokens,
        dropout=card_dropout,
    )
    relic = ProgramEncoderConfig(
        token_vocab_size=len(RELIC_RULE_VOCAB), token_dim=relic_token_dim,
        output_dim=relic_dim, n_heads=relic_heads, n_layers=relic_layers,
        feedforward=relic_feedforward, max_tokens=relic_max_tokens,
        dropout=relic_dropout,
    )
    hand = HandCompositionConfig(
        card_dim=card_dim, output_dim=hand_dim, hidden_dim=hand_hidden,
        n_heads=hand_heads, feedforward=hand_feedforward,
        n_layers=hand_layers, max_cards=hand_max_cards, dropout=hand_dropout,
    )
    world = WorldEncoderConfig(
        player_dim=state.player_dim, enemy_dim=state.enemy_dim,
        pile_dim=state.pile_dim, card_dim=card_dim, relic_dim=relic_dim,
        hand_dim=hand_dim,
        d_model=world_dim, n_heads=world_heads, n_layers=world_layers,
        feedforward=world_feedforward, max_enemies=state.max_enemies,
        max_relics=max_relics, dropout=world_dropout,
        action_upgrade_scale=action_upgrade_scale,
    )
    decoder = EventDecoderConfig(
        token_vocab_size=len(VECTOR_TOKEN_VOCAB), d_model=decoder_dim,
        n_heads=decoder_heads, n_layers=decoder_layers,
        feedforward=decoder_feedforward, response_dim=response_dim,
        hand_delta_dim=hand.composition_dim if predict_hand_delta else 0,
        max_tokens=decoder_max_tokens, max_events=max_events,
        max_queries=max_queries, max_nesting_depth=max_nesting_depth,
        dropout=decoder_dropout,
    )
    delta = DeltaHeadConfig(
        input_dim=decoder_dim, delta_dim=state.world_dim,
        hidden_dim=delta_hidden, n_layers=delta_layers, dropout=delta_dropout,
    )
    hand_delta = (
        DeltaHeadConfig(
            input_dim=decoder_dim, delta_dim=hand.composition_dim,
            hidden_dim=hand_delta_hidden, n_layers=hand_delta_layers,
            dropout=hand_delta_dropout,
        )
        if predict_hand_delta else None
    )
    return VectorRuleModelConfig(
        state=state, card_encoder=card, relic_encoder=relic,
        share_rule_transformer=share_rule_transformer,
        hand_encoder=hand, world_encoder=world, decoder=decoder,
        delta_head=delta, hand_delta_head=hand_delta,
    )


__all__ = [
    "RULE_VECTOR_SCHEMA_VERSION", "ProgramEncoderConfig", "RuleProgramEncoder",
    "RuleProgramTransformer",
    "CardRuleEncoder", "load_frozen_card_rule_encoder", "RelicRuleEncoder", "CARD_RULE_VOCAB", "RELIC_RULE_VOCAB", "RELIC_RULE_INDEX",
    "RELIC_RULE_IDS", "relic_effect_tokens", "StateVectorConfig", "VectorizedState",
    "vectorize_state", "vector_delta", "generic_state_calculations",
    "serialize_calculations", "VECTOR_CALCULATION_TOKENS",
    "VECTOR_SPECIAL_TOKENS", "VECTOR_EVENT_KIND_TOKENS",
    "VECTOR_ENTITY_TOKENS", "VECTOR_RELIC_ENTITY_TOKENS",
    "VECTOR_RUNTIME_ENTITY_VALUE_TOKENS", "VECTOR_RUNTIME_RELIC_COUNTER_VALUE_TOKENS",
    "VECTOR_EXECUTOR_VALUE_TOKENS", "VECTOR_V7_PROTOCOL_TOKENS",
    "VECTOR_TOKEN_VOCAB", "VECTOR_TOKEN_INDEX", "VECTOR_QUERY_TOKENS",
    "validate_vector_token_limits",
    "hand_composition_feature_dim", "summarize_hand", "HandCompositionConfig",
    "HandCompositionEncoder", "hand_inputs_from_state",
    "hand_composition_from_state", "hand_composition_delta",
    "TokenOwner", "VectorQuery", "VectorResponse", "VectorValueRequest",
    "VectorValueBinding", "VectorTranscript", "VectorTranscriptBuilder",
    "VectorOracle", "VectorValueBinder", "VectorValueStore",
    "WorldEncoderConfig", "VectorWorldEncoder",
    "DeltaHeadConfig", "DeltaHead", "EventDecoderConfig", "EventDecoderOutput",
    "VectorEvent", "SerializedVectorEvents", "serialize_vector_events",
    "serialize_sword_boomerang", "serialize_hand_operation",
    "serialize_armaments_hand_operation", "serialize_burning_pact_hand_operation",
    "serialize_burning_pact_operations",
    "serialize_havoc_skeleton", "VectorEventDecoder",
    "VectorRuleModelConfig", "VectorRuleWorldModel",
    "default_vector_model_config",
]

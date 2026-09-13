"""Whole-run policy with anonymous card programs and interactive reasoning.

The model is initialized from scratch without a rule-checkpoint loading path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, prod

import torch
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from torch import nn

from env.core import action_space as A
from env.core.card_language import (
    MAX_VERSION_PROGRAM_TOKENS,
    TOKEN_VOCAB as CARD_TOKEN_VOCAB,
    encoded_card_version_programs,
)
from rl.policies.model import CardLanguageEmbedding
from rl.policies.pointer_policy import _NullExtractor
from rl.policies.split_policy import SplitStSPolicyCore
from rl.policies.wholerun_reasoning import (
    IGNORE_INDEX,
    InteractiveOracle,
    ReasoningVocab,
    VOCAB,
)


def _width(space, key: str) -> int:
    return int(prod(space[key].shape))


class NumberEncoder(nn.Module):
    """Map a scalar integer to a compositional continuous representation.

    The input deliberately contains no lookup by the literal value.  Sign,
    scale, magnitude and decimal width are shared by every number, so rare
    values can reuse the same arithmetic features as common values.
    """

    def __init__(self, embedding_dim: int, *, scale: float = 256.0) -> None:
        super().__init__()
        if embedding_dim < 1 or scale <= 0:
            raise ValueError("embedding_dim and scale must be positive")
        self.embedding_dim = int(embedding_dim)
        self.scale = float(scale)
        hidden = max(32, min(256, 2 * self.embedding_dim))
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        magnitude = values.abs()
        # Five smooth/shared features: normalized value, sign, log magnitude,
        # zero indicator, and decimal digit count.
        features = torch.stack((
            (values / self.scale).clamp(-32.0, 32.0),
            values.sign(),
            (torch.log1p(magnitude) / torch.log1p(
                values.new_tensor(self.scale)
            )).clamp(0.0, 32.0),
            values.eq(0).float(),
            (torch.floor(torch.log10(magnitude.clamp_min(1.0))) + 1.0) / 8.0,
        ), dim=-1)
        return self.mlp(features)


def _numeric_end_annotations(
    token_ids: torch.Tensor,
    *,
    begin_id: int,
    end_id: int,
    digit_ids: dict[int, int],
    negative_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read complete decimal spans and mark their ``NUM_END`` positions.

    This helper is used for static card programs and teacher-forced reasoning
    batches.  Only completed spans are marked, preventing a number embedding
    from leaking into positions before the final digit has been generated.
    """

    original_shape = token_ids.shape
    if not original_shape:
        raise ValueError("token_ids must have a sequence dimension")
    length = original_shape[-1]
    flat = token_ids.detach().to(device="cpu").reshape(-1, length).tolist()
    values = torch.zeros((len(flat), length), dtype=torch.float32)
    mask = torch.zeros((len(flat), length), dtype=torch.bool)
    for row, sequence in enumerate(flat):
        active = False
        negative = False
        number = 0
        has_digit = False
        for position, token_id in enumerate(sequence):
            token_id = int(token_id)
            if token_id == begin_id:
                active = True
                negative = False
                number = 0
                has_digit = False
            elif not active:
                continue
            elif negative_id is not None and token_id == negative_id:
                negative = True
            elif token_id in digit_ids:
                number = number * 10 + int(digit_ids[token_id])
                has_digit = True
            elif token_id == end_id:
                if has_digit:
                    values[row, position] = -number if negative else number
                    mask[row, position] = True
                active = False
                number = 0
                has_digit = False
            else:
                # A malformed span cannot contribute a value to a later token.
                active = False
                number = 0
                has_digit = False
    return values.reshape(*original_shape), mask.reshape(*original_shape)


class AnonymousCardProgramEmbedding(CardLanguageEmbedding):
    """Card embedding determined only by its complete surface DSL.

    The inherited id table exists solely for API compatibility with the current
    split state encoders.  It is frozen and never read by any forward method.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        shared_token_embedding: nn.Embedding | None = None,
        number_encoder: NumberEncoder | None = None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.weight.requires_grad_(False)
        programs = torch.as_tensor(
            encoded_card_version_programs(), dtype=torch.long
        )
        if programs.shape[:2] != (num_embeddings, 2):
            raise ValueError(
                f"anonymous card table shape {tuple(programs.shape)} does not "
                f"match ({num_embeddings}, 2, L)"
            )
        self.version_program_ids = programs
        token_dim = embedding_dim
        self.token_embedding = shared_token_embedding or nn.Embedding(
            len(CARD_TOKEN_VOCAB), token_dim, padding_idx=0
        )
        if self.token_embedding.embedding_dim != token_dim:
            raise ValueError("shared card token embedding has the wrong width")
        self.program_position = nn.Parameter(
            torch.empty(MAX_VERSION_PROGRAM_TOKENS, token_dim)
        )
        nn.init.normal_(self.program_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=4 * token_dim,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.program_encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.program_norm = nn.LayerNorm(token_dim)
        self.number_encoder = number_encoder or NumberEncoder(token_dim)
        card_token_index = {token: index for index, token in enumerate(CARD_TOKEN_VOCAB)}
        number_values, number_mask = _numeric_end_annotations(
            programs,
            begin_id=card_token_index["NUM_BEGIN"],
            end_id=card_token_index["NUM_END"],
            digit_ids={
                card_token_index[f"DIGIT_{digit}"]: digit for digit in range(10)
            },
            negative_id=card_token_index.get("SIGN_NEG"),
        )
        self.register_buffer("program_number_values", number_values, persistent=False)
        self.register_buffer("program_number_mask", number_mask, persistent=False)

        # Remove inherited convolutional semantic parameters.  Keeping two
        # independent encoders would make it unclear which one is authoritative.
        del self.local_encoder
        del self.effect_scale

    def _programs(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        versions = upgrades.long().gt(0).long()
        return self.version_program_ids[ids.long(), versions]

    def encode_version_tokens(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        versions = upgrades.long().gt(0).long()
        programs = self.version_program_ids[ids.long(), versions]
        number_values = self.program_number_values[ids.long(), versions]
        number_mask = self.program_number_mask[ids.long(), versions]
        shape = programs.shape
        flat = programs.reshape(-1, shape[-1])
        padding = flat.eq(0)
        active = ~padding.all(dim=1)
        active_rows = torch.nonzero(active, as_tuple=False).flatten()
        if active_rows.numel():
            active_flat = flat.index_select(0, active_rows)
            active_padding = padding.index_select(0, active_rows)
            embedded = self.token_embedding(active_flat)
            embedded = embedded + self.program_position[: flat.shape[1]].unsqueeze(0)
            number_values = number_values.reshape(-1, shape[-1]).index_select(
                0, active_rows
            )
            number_mask = number_mask.reshape(-1, shape[-1]).index_select(
                0, active_rows
            )
            embedded = embedded + self.number_encoder(number_values).to(
                dtype=embedded.dtype
            ) * number_mask.unsqueeze(-1).to(dtype=embedded.dtype)
            encoded_active = self.program_encoder(
                embedded, src_key_padding_mask=active_padding
            )
            encoded_active = self.program_norm(encoded_active)
            encoded_active = encoded_active.masked_fill(
                active_padding.unsqueeze(-1), 0.0
            )
            encoded = encoded_active.new_zeros(
                flat.shape[0], flat.shape[1], encoded_active.shape[-1]
            )
            encoded = encoded.index_copy(0, active_rows, encoded_active)
        else:
            encoded = self.token_embedding.weight.new_zeros(
                flat.shape[0], flat.shape[1], self.token_embedding.embedding_dim
            )
        return (
            encoded.reshape(*shape, encoded.shape[-1]),
            padding.reshape(*shape),
        )

    def version_effect_vectors(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        tokens, padding = self.encode_version_tokens(ids, upgrades)
        keep = (~padding).unsqueeze(-1)
        return (tokens * keep).sum(dim=-2) / keep.sum(dim=-2).clamp_min(1)

    def effect_vectors(self, ids: torch.Tensor) -> torch.Tensor:
        return self.version_effect_vectors(ids, torch.zeros_like(ids))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.effect_vectors(ids)

    def versioned_forward(
        self, ids: torch.Tensor, upgrades: torch.Tensor
    ) -> torch.Tensor:
        return self.version_effect_vectors(ids, upgrades)

    def effective_weight(self) -> torch.Tensor:
        ids = torch.arange(self.num_embeddings, device=self.program_position.device)
        return self.effect_vectors(ids)


class _CachedDecoderLayer(nn.Module):
    """Pre-norm decoder layer with an exact one-token cached path."""

    def __init__(self, d_model: int, n_heads: int, feedforward: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, feedforward)
        self.linear2 = nn.Linear(feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        target_padding: torch.Tensor | None,
        memory_padding: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden)
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=target_padding,
            need_weights=False,
        )[0]
        hidden = hidden + attended
        normalized = self.norm2(hidden)
        hidden = hidden + self.cross_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding,
            need_weights=False,
        )[0]
        normalized = self.norm3(hidden)
        return hidden + self.linear2(F.gelu(self.linear1(normalized)))

    def step(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        cached_normalized_inputs: torch.Tensor | None,
        memory_padding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(hidden)
        keys = (
            normalized
            if cached_normalized_inputs is None
            else torch.cat((cached_normalized_inputs, normalized), dim=1)
        )
        hidden = hidden + self.self_attention(
            normalized, keys, keys, need_weights=False
        )[0]
        cross_query = self.norm2(hidden)
        hidden = hidden + self.cross_attention(
            cross_query,
            memory,
            memory,
            key_padding_mask=memory_padding,
            need_weights=False,
        )[0]
        ff_input = self.norm3(hidden)
        hidden = hidden + self.linear2(F.gelu(self.linear1(ff_input)))
        return hidden, keys


@dataclass
class ReasoningKVCache:
    layer_inputs: list[torch.Tensor | None]
    length: int = 0
    number_active: torch.Tensor | None = None
    number_negative: torch.Tensor | None = None
    number_value: torch.Tensor | None = None
    number_has_digit: torch.Tensor | None = None

    @classmethod
    def empty(cls, layers: int) -> "ReasoningKVCache":
        return cls([None] * layers, 0)


class CachedReasoningDecoder(nn.Module):
    """Parallel teacher-forcing decoder plus incremental KV-cache inference."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_tokens: int | None,
        padding_idx: int,
        number_encoder: NumberEncoder | None = None,
    ) -> None:
        super().__init__()
        # ``max_tokens`` is retained as the initial learned-position capacity
        # for checkpoint compatibility. It is no longer a transcript limit;
        # positions beyond this table use deterministic sinusoidal vectors.
        self.max_tokens = None
        self.position_capacity = max(1, int(max_tokens or 1024))
        self.padding_idx = int(padding_idx)
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        self.position = nn.Embedding(self.position_capacity, d_model)
        self.layers = nn.ModuleList([
            _CachedDecoderLayer(d_model, n_heads, 4 * d_model)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.number_encoder = number_encoder or NumberEncoder(d_model)
        self.number_projection = (
            nn.Identity()
            if self.number_encoder.embedding_dim == d_model
            else nn.Linear(self.number_encoder.embedding_dim, d_model)
        )
        token_index = {token: index for index, token in enumerate(VOCAB.tokens)}
        self.number_begin_id = int(token_index["NUM_BEGIN"])
        self.number_end_id = int(token_index["NUM_END"])
        self.number_negative_id = int(token_index["SIGN_NEG"])
        self.number_digit_ids = {
            int(token_index[f"DIGIT_{digit}"]): digit for digit in range(10)
        }

    def _sinusoidal_positions(
        self,
        length: int,
        *,
        start: int = 0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create extra positions on demand without a hard context cap."""
        if length <= 0:
            return torch.empty(
                0, self.position.embedding_dim, device=device, dtype=dtype
            )
        dimension = self.position.embedding_dim
        positions = torch.arange(
            start, start + length, device=device, dtype=torch.float32
        )
        exponents = torch.arange(
            0, dimension, 2, device=device, dtype=torch.float32
        )
        div_term = torch.exp(exponents * (-log(10000.0) / dimension))
        encoded = torch.zeros(
            length, dimension, device=device, dtype=torch.float32
        )
        encoded[:, 0::2] = torch.sin(positions[:, None] * div_term[None])
        odd_width = encoded[:, 1::2].shape[1]
        if odd_width:
            encoded[:, 1::2] = torch.cos(
                positions[:, None] * div_term[:odd_width][None]
            )
        return encoded.to(dtype=dtype)

    def _sinusoidal_position(
        self, index: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create one overflow position for the cached one-token path."""
        dimension = self.position.embedding_dim
        exponents = torch.arange(
            0, dimension, 2, device=device, dtype=torch.float32
        )
        div_term = torch.exp(exponents * (-log(10000.0) / dimension))
        position = torch.tensor(float(index), device=device)
        encoded = torch.zeros(dimension, device=device, dtype=torch.float32)
        encoded[0::2] = torch.sin(position * div_term)
        odd_width = encoded[1::2].shape[0]
        if odd_width:
            encoded[1::2] = torch.cos(position * div_term[:odd_width])
        return encoded.to(dtype=dtype)

    def _positions(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return positions for any length, preserving old learned prefixes."""
        learned_length = min(length, self.position.num_embeddings)
        learned = self.position(
            torch.arange(learned_length, device=device, dtype=torch.long)
        ).to(dtype=dtype)
        if learned_length == length:
            return learned
        extra = self._sinusoidal_positions(
            length - learned_length,
            start=learned_length,
            device=device,
            dtype=dtype,
        )
        return torch.cat((learned, extra), dim=0)

    def _position_at(
        self, index: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if index < self.position.num_embeddings:
            return self.position(
                torch.tensor(index, device=device, dtype=torch.long)
            ).to(dtype=dtype)
        return self._sinusoidal_position(index, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = token_ids.shape
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self._positions(
            length, device=token_ids.device, dtype=hidden.dtype
        )[None]
        number_values, number_mask = _numeric_end_annotations(
            token_ids,
            begin_id=self.number_begin_id,
            end_id=self.number_end_id,
            digit_ids=self.number_digit_ids,
            negative_id=self.number_negative_id,
        )
        number_values = number_values.to(device=token_ids.device)
        number_mask = number_mask.to(device=token_ids.device)
        hidden = hidden + self.number_projection(
            self.number_encoder(number_values)
        ).to(dtype=hidden.dtype) * number_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        target_padding = token_ids.eq(self.padding_idx)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                memory,
                causal_mask=causal,
                target_padding=target_padding,
                memory_padding=memory_padding,
            )
        return self.output(self.output_norm(hidden))

    def step(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None,
        cache: ReasoningKVCache | None = None,
    ) -> tuple[torch.Tensor, ReasoningKVCache]:
        """Consume exactly one token and return logits for the next token."""

        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.shape[1] != 1:
            raise ValueError("cached decoder step consumes exactly one token")
        cache = cache or ReasoningKVCache.empty(len(self.layers))
        batch = token_ids.shape[0]
        if cache.number_active is None or cache.number_active.shape[0] != batch:
            cache.number_active = torch.zeros(
                batch, dtype=torch.bool, device=token_ids.device
            )
            cache.number_negative = torch.zeros(
                batch, dtype=torch.bool, device=token_ids.device
            )
            cache.number_value = torch.zeros(
                batch, dtype=torch.float32, device=token_ids.device
            )
            cache.number_has_digit = torch.zeros(
                batch, dtype=torch.bool, device=token_ids.device
            )
        else:
            cache.number_active = cache.number_active.to(token_ids.device)
            cache.number_negative = cache.number_negative.to(token_ids.device)
            cache.number_value = cache.number_value.to(token_ids.device)
            cache.number_has_digit = cache.number_has_digit.to(token_ids.device)
        token = token_ids[:, 0]
        begin = token.eq(self.number_begin_id)
        end = token.eq(self.number_end_id)
        cache.number_active = torch.where(begin, True, cache.number_active)
        cache.number_negative = torch.where(
            begin, False, cache.number_negative
        )
        cache.number_value = torch.where(
            begin, torch.zeros_like(cache.number_value), cache.number_value
        )
        cache.number_has_digit = torch.where(
            begin, False, cache.number_has_digit
        )
        if self.number_negative_id is not None:
            cache.number_negative = torch.where(
                cache.number_active & token.eq(self.number_negative_id),
                True,
                cache.number_negative,
            )
        for digit_id, digit in self.number_digit_ids.items():
            is_digit = cache.number_active & token.eq(digit_id)
            cache.number_value = torch.where(
                is_digit,
                cache.number_value * 10.0 + float(digit),
                cache.number_value,
            )
            cache.number_has_digit = cache.number_has_digit | is_digit
        completed = end & cache.number_active & cache.number_has_digit
        completed_values = torch.where(
            cache.number_negative, -cache.number_value, cache.number_value
        )
        cache.number_active = torch.where(end, False, cache.number_active)
        cache.number_has_digit = torch.where(end, False, cache.number_has_digit)
        hidden = self.token_embedding(token_ids)
        position = self._position_at(
            cache.length, device=token_ids.device, dtype=hidden.dtype
        )
        hidden = hidden + position.reshape(1, 1, -1)
        if bool(completed.any()):
            numeric = self.number_projection(
                self.number_encoder(completed_values)
            ).to(dtype=hidden.dtype)
            hidden = hidden + numeric[:, None] * completed[:, None, None].to(
                dtype=hidden.dtype
            )
        new_inputs: list[torch.Tensor | None] = []
        for index, layer in enumerate(self.layers):
            hidden, keys = layer.step(
                hidden,
                memory,
                cached_normalized_inputs=cache.layer_inputs[index],
                memory_padding=memory_padding,
            )
            new_inputs.append(keys)
        cache.layer_inputs = new_inputs
        cache.length += 1
        logits = self.output(self.output_norm(hidden[:, -1]))
        return logits, cache


@dataclass(frozen=True)
class InteractiveGeneration:
    token_ids: tuple[int, ...]
    environment_tokens: int
    cache_length: int


class SemanticReasoningPolicyCore(SplitStSPolicyCore):
    """Split whole-run actor-critic with an anonymous card frontend."""

    def __init__(
        self,
        observation_space,
        *,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        card_program_layers: int = 2,
        reasoning_layers: int = 4,
        max_reasoning_tokens: int | None = None,
        combat_language: bool = False,
        map_dag: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            card_language=True,
            combat_language=combat_language,
            map_dag=map_dag,
            **kwargs,
        )
        old_card = self.shared.card
        card_dim = old_card.embedding_dim
        self.number_encoder = NumberEncoder(card_dim)
        self.shared.card = AnonymousCardProgramEmbedding(
            old_card.num_embeddings,
            card_dim,
            n_heads=max(1, min(n_heads, card_dim // 8)),
            n_layers=card_program_layers,
            number_encoder=self.number_encoder,
        )

        # Keep pile modules in the state dict for checkpoint compatibility, but
        # do not encode draw/discard/exhaust card programs in combat. The
        # anonymous combat frontend should read complete rules from hand cards;
        # pile counts remain available in the observation contract without
        # triggering a card-ID semantic shortcut.
        self.combat.encoder.pile_card_language_enabled = False
        self.combat.encoder.exhaust_card_language_enabled = False

        # Raw per-card-index pile projections and hand surface summaries would
        # restore a canonical-card shortcut. Remove the learned projection over
        # id-count bins while retaining the fixed pile role positions.
        with torch.no_grad():
            self.combat.encoder.pile_projection.weight.zero_()
            if self.combat.encoder.pile_projection.bias is not None:
                self.combat.encoder.pile_projection.bias.zero_()
        for parameter in self.combat.encoder.pile_projection.parameters():
            parameter.requires_grad_(False)

        self.action_embedding = nn.Embedding(A.N_ACTIONS, d_model)
        context_keys = (
            "player_scalars",
            "player_buffs",
            "combat_counters",
            "monster_scalars",
            "monster_buffs",
            "run_hp",
            "gold",
            "ascension",
            "act",
            "floor",
        )
        self.reasoning_context_keys = tuple(
            key for key in context_keys if key in observation_space.spaces
        )
        context_width = sum(
            _width(observation_space, key) for key in self.reasoning_context_keys
        )
        self.reasoning_context = nn.Sequential(
            nn.Linear(context_width, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.card_runtime_projection = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.card_memory_projection = nn.Linear(card_dim, d_model)
        self.reasoning_decoder = CachedReasoningDecoder(
            len(VOCAB.tokens),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=reasoning_layers,
            max_tokens=max_reasoning_tokens,
            padding_idx=VOCAB.pad_id,
            number_encoder=self.number_encoder,
        )

    @staticmethod
    def _sanitized(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sanitized = dict(obs)
        for key in ("hand_damage", "hand_block"):
            if key in sanitized:
                sanitized[key] = torch.zeros_like(sanitized[key])
        return sanitized

    def actor_critic(self, obs):
        return super().actor_critic(self._sanitized(obs))

    def _selected_card(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = actions.long().flatten()
        batch = actions.shape[0]
        device = actions.device
        card_ids = torch.zeros(batch, dtype=torch.long, device=device)
        upgrades = torch.zeros(batch, dtype=torch.float32, device=device)
        runtime_costs = torch.zeros(batch, dtype=torch.float32, device=device)
        cost_known = torch.zeros(batch, dtype=torch.bool, device=device)

        def assign(
            segment: str,
            ids_key: str,
            upgrades_key: str,
            offsets=None,
            costs_key: str | None = None,
        ):
            base = A.seg_base(segment)
            width = dict(A._SEGMENTS)[segment]
            mask = (actions >= base) & (actions < base + width)
            if not bool(mask.any()) or ids_key not in obs:
                return
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            if offsets is None:
                columns = actions[rows] - base
            else:
                columns = offsets[rows]
            valid = (columns >= 0) & (columns < obs[ids_key].shape[1])
            rows, columns = rows[valid], columns[valid]
            if rows.numel():
                card_ids[rows] = obs[ids_key][rows, columns].long()
                upgrades[rows] = obs[upgrades_key][rows, columns].float()
                if costs_key is not None and costs_key in obs:
                    runtime_costs[rows] = obs[costs_key][rows, columns].float()
                    cost_known[rows] = True

        assign(
            "PLAY_CARD", "hand_ids", "hand_upgrades", costs_key="hand_costs"
        )
        if "pending_card_slot" in obs:
            pending = obs["pending_card_slot"].long().reshape(batch) - 1
            assign(
                "PICK_TARGET",
                "hand_ids",
                "hand_upgrades",
                offsets=pending,
                costs_key="hand_costs",
            )
        phase = obs["phase"].long().reshape(batch)
        combat_pick = phase.eq(self._combat_phase)
        pick_offsets = actions - A.seg_base("PICK_CARD")
        assign(
            "PICK_CARD", "select_card_ids", "select_card_upgrades",
            offsets=torch.where(combat_pick, torch.full_like(pick_offsets, -1), pick_offsets),
        )
        assign(
            "PICK_CARD", "hand_ids", "hand_upgrades",
            offsets=torch.where(combat_pick, pick_offsets, torch.full_like(pick_offsets, -1)),
        )
        assign("REWARD_CARD", "offer_card_ids", "offer_card_upgrades")
        assign("SHOP_CARD", "shop_card_ids", "shop_card_upgrades")
        return card_ids, upgrades, runtime_costs, cost_known

    def reasoning_memory(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        action = self.action_embedding(actions.long().flatten()).unsqueeze(1)
        context = torch.cat([
            obs[key].float().reshape(batch, -1)
            for key in self.reasoning_context_keys
        ], dim=1)
        context = self.reasoning_context(context).unsqueeze(1)
        ids, upgrades, runtime_costs, cost_known = self._selected_card(obs, actions)
        runtime = self.card_runtime_projection(torch.stack(
            (runtime_costs, cost_known.float()), dim=-1
        )).unsqueeze(1)
        card_tokens, card_padding = self.shared.card.encode_version_tokens(
            ids, upgrades
        )
        card_tokens = self.card_memory_projection(card_tokens)
        memory = torch.cat((action, context, runtime, card_tokens), dim=1)
        fixed_padding = torch.zeros(
            batch, 3, dtype=torch.bool, device=memory.device
        )
        padding = torch.cat((fixed_padding, card_padding), dim=1)
        return memory, padding

    def reasoning_logits(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory, padding = self.reasoning_memory(obs, actions)
        return self.reasoning_decoder(input_ids, memory, padding)

    def reasoning_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.reasoning_logits(obs, actions, input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        with torch.no_grad():
            active = labels.ne(IGNORE_INDEX)
            correct = logits.argmax(-1).eq(labels) & active
            accuracy = correct.sum().float() / active.sum().clamp_min(1)
        return loss, {
            "token_accuracy": float(accuracy.cpu()),
            "supervised_tokens": float(active.sum().cpu()),
        }

    @torch.no_grad()
    def generate_interactive(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        oracle: InteractiveOracle,
        *,
        max_new_tokens: int = 1024,
        vocab: ReasoningVocab = VOCAB,
    ) -> InteractiveGeneration:
        if action.numel() != 1:
            raise ValueError("interactive generation currently requires batch size 1")
        memory, padding = self.reasoning_memory(obs, action.reshape(1))
        cache = ReasoningKVCache.empty(len(self.reasoning_decoder.layers))
        current = torch.tensor([vocab.bos_id], device=memory.device)
        generated = [vocab.bos_id]
        environment_tokens = 0
        query_ids = {
            vocab.index[token]: token
            for token in (
                "QUERY_RANDOM_TARGET",
                "QUERY_DRAW_CARD",
                "QUERY_RANDOM_CARD",
                "QUERY_CREATED_CARD",
            )
        }
        logits, cache = self.reasoning_decoder.step(
            current, memory, padding, cache
        )
        for _ in range(max_new_tokens):
            next_id = int(logits.argmax(-1).item())
            generated.append(next_id)
            current = torch.tensor([next_id], device=memory.device)
            logits, cache = self.reasoning_decoder.step(
                current, memory, padding, cache
            )
            if next_id == vocab.eos_id:
                break
            query = query_ids.get(next_id)
            if query is None:
                continue
            for response_id in vocab.encode(oracle.respond(query)):
                generated.append(response_id)
                environment_tokens += 1
                current = torch.tensor([response_id], device=memory.device)
                logits, cache = self.reasoning_decoder.step(
                    current, memory, padding, cache
                )
        return InteractiveGeneration(
            tuple(generated), environment_tokens, cache.length
        )


class SemanticReasoningPolicy(MaskableMultiInputActorCriticPolicy):
    """SB3 adapter for :class:`SemanticReasoningPolicyCore`."""

    supports_bf16_rollout = True

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        **kwargs,
    ) -> None:
        core_keys = {
            "d_model",
            "n_layers",
            "n_heads",
            "card_program_layers",
            "reasoning_layers",
            "max_reasoning_tokens",
            "combat_language",
            "map_dag",
        }
        self._core_cfg = {
            key: kwargs.pop(key) for key in tuple(kwargs) if key in core_keys
        }
        kwargs["features_extractor_class"] = _NullExtractor
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.core = SemanticReasoningPolicyCore(
            self.observation_space, **self._core_cfg
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values.float(), log_prob.float()

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self.core.actor_critic(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return (
            values.float(),
            distribution.log_prob(actions).float(),
            distribution.entropy().float(),
        )

    def get_distribution(self, obs, action_masks=None):
        distribution = self.action_dist.proba_distribution(
            action_logits=self.core.action_logits(obs)
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs):
        return self.core.value(obs).float()


__all__ = [
    "AnonymousCardProgramEmbedding",
    "CachedReasoningDecoder",
    "InteractiveGeneration",
    "NumberEncoder",
    "ReasoningKVCache",
    "SemanticReasoningPolicy",
    "SemanticReasoningPolicyCore",
]

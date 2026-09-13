"""Standalone encoder/decoder model for combat-rule sequence learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class RuleModelConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 256
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    feedforward: int = 1024
    max_prefix_tokens: int = 2048
    max_target_tokens: int = 2048
    dropout: float = 0.1
    num_begin_id: int | None = None
    num_lsd_begin_id: int | None = None
    num_end_id: int | None = None
    digit_token_ids: tuple[int, ...] = ()
    max_number_digits: int = 16


class RuleWorldModel(nn.Module):
    """Encode state+action once, then autoregress effects and next state."""

    def __init__(self, config: RuleModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.prefix_position = nn.Embedding(config.max_prefix_tokens, config.d_model)
        self.target_position = nn.Embedding(config.max_target_tokens, config.d_model)
        self.number_position = (
            nn.Embedding(config.max_number_digits + 1, config.d_model, padding_idx=0)
            if config.digit_token_ids and config.num_end_id is not None
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers
        )
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _positions(length: int, batch: int, device) -> torch.Tensor:
        return torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)

    def _number_places(
        self, token_ids: torch.Tensor, *, allow_future: bool
    ) -> torch.Tensor:
        """Return 1-based decimal significance; zero means a non-digit token."""
        if self.number_position is None:
            return torch.zeros_like(token_ids)
        digit_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.config.digit_token_ids:
            digit_mask |= token_ids.eq(token_id)

        digit_counts = digit_mask.long().cumsum(dim=1)
        normal_begin = token_ids.eq(self.config.num_begin_id)
        lsd_begin = (
            token_ids.eq(self.config.num_lsd_begin_id)
            if self.config.num_lsd_begin_id is not None
            else torch.zeros_like(digit_mask)
        )
        any_begin = normal_begin | lsd_begin
        begin_counts = torch.where(any_begin, digit_counts, torch.zeros_like(digit_counts))
        last_begin_counts = begin_counts.cummax(dim=1).values
        from_left = digit_counts - last_begin_counts - 1

        reversed_ids = token_ids.flip(1)
        reversed_digits = digit_mask.flip(1)
        reverse_counts = reversed_digits.long().cumsum(dim=1)
        reverse_ends = reversed_ids.eq(self.config.num_end_id)
        end_counts = torch.where(reverse_ends, reverse_counts, torch.zeros_like(reverse_counts))
        last_end_counts = end_counts.cummax(dim=1).values
        from_right = (reverse_counts - last_end_counts - 1).flip(1)

        sequence_positions = torch.arange(
            1, token_ids.shape[1] + 1, device=token_ids.device
        ).unsqueeze(0)
        latest_lsd_begin = torch.where(
            lsd_begin, sequence_positions, torch.zeros_like(sequence_positions)
        ).cummax(dim=1).values
        other_boundary = normal_begin | token_ids.eq(self.config.num_end_id)
        latest_other_boundary = torch.where(
            other_boundary, sequence_positions, torch.zeros_like(sequence_positions)
        ).cummax(dim=1).values
        lsd_active = latest_lsd_begin > latest_other_boundary
        places = torch.where(
            lsd_active | ~torch.tensor(allow_future, device=token_ids.device),
            from_left,
            from_right,
        )
        places = torch.where(digit_mask, places + 1, torch.zeros_like(places))
        if bool((places > self.config.max_number_digits).any()):
            raise ValueError(
                f"number exceeds configured {self.config.max_number_digits} digits"
            )
        return places

    def _embed(
        self, token_ids: torch.Tensor, position_embedding: nn.Embedding,
        *, allow_future: bool,
    ):
        batch, length = token_ids.shape
        embedded = self.token_embedding(token_ids) + position_embedding(
            self._positions(length, batch, token_ids.device)
        )
        if self.number_position is not None:
            embedded = embedded + self.number_position(
                self._number_places(token_ids, allow_future=allow_future)
            )
        return embedded

    def encode(self, prefix_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = prefix_ids.shape
        if length > self.config.max_prefix_tokens:
            raise ValueError(
                f"prefix has {length} tokens; max is {self.config.max_prefix_tokens}"
            )
        padding = prefix_ids.eq(self.config.pad_id)
        source = self._embed(prefix_ids, self.prefix_position, allow_future=True)
        memory = self.encoder(source, src_key_padding_mask=padding)
        return self.memory_norm(memory), padding

    def rule_latent(
        self, memory: torch.Tensor, prefix_padding: torch.Tensor
    ) -> torch.Tensor:
        """Representation intended for later fusion into the split policy."""
        lengths = (~prefix_padding).sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(memory.shape[0], device=memory.device)
        return memory[rows, lengths]

    def decode(
        self,
        target_input_ids: torch.Tensor,
        memory: torch.Tensor,
        prefix_padding: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = target_input_ids.shape
        if length > self.config.max_target_tokens:
            raise ValueError(
                f"target has {length} tokens; max is {self.config.max_target_tokens}"
            )
        target_padding = target_input_ids.eq(self.config.pad_id)
        target = self._embed(
            target_input_ids, self.target_position, allow_future=False
        )
        causal = torch.triu(
            torch.ones(
                length, length, dtype=torch.bool, device=target_input_ids.device
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=prefix_padding,
        )
        return self.output(self.output_norm(decoded))

    def forward(
        self, prefix_ids: torch.Tensor, target_input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, padding = self.encode(prefix_ids)
        logits = self.decode(target_input_ids, memory, padding)
        return logits, self.rule_latent(memory, padding)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        *,
        output_begin_id: int,
        output_end_id: int,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        memory, padding = self.encode(prefix_ids)
        generated = torch.full(
            (prefix_ids.shape[0], 1), output_begin_id,
            dtype=torch.long, device=prefix_ids.device,
        )
        finished = torch.zeros(prefix_ids.shape[0], dtype=torch.bool, device=prefix_ids.device)
        limit = max_new_tokens or self.config.max_target_tokens - 1
        for _ in range(limit):
            logits = self.decode(generated, memory, padding)
            next_id = logits[:, -1].argmax(dim=-1)
            next_id = torch.where(
                finished,
                torch.full_like(next_id, output_end_id),
                next_id,
            )
            generated = torch.cat((generated, next_id.unsqueeze(1)), dim=1)
            finished |= next_id.eq(output_end_id)
            if bool(finished.all()):
                break
        return generated

    def save_config(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, path: str | Path) -> "RuleWorldModel":
        config = RuleModelConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
        return cls(config)

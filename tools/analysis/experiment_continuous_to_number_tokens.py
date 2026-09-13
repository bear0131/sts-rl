"""Test exact numeric-token recovery from a normalized continuous state vector.

The experiment isolates the numeric grounding path used by the vector-rule
model: a complete player vector is projected into Transformer memory, then a
causal Transformer emits the v7 least-significant-first integer tokens.  It
trains one model on the full integer support and another with regularly
interleaved integers held out, separating finite-codebook learning from true
interpolation to unseen values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from rl.rules.rule_vector_model import StateVectorConfig, _calculation_number


TOKENS = (
    "<PAD>",
    "<BOS>",
    "NUM_LSD_BEGIN",
    *(f"DIGIT_{digit}" for digit in range(10)),
    "NUM_END",
)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
PAD_ID = TOKEN_TO_ID["<PAD>"]
BOS_ID = TOKEN_TO_ID["<BOS>"]
END_ID = TOKEN_TO_ID["NUM_END"]


@dataclass(frozen=True)
class ExperimentConfig:
    max_value: int = 200
    holdout_modulus: int = 7
    train_steps: int = 2500
    batch_size: int = 512
    learning_rate: float = 3.0e-4
    d_model: int = 256
    heads: int = 8
    world_layers: int = 3
    decoder_layers: int = 4
    feedforward: int = 1024
    eval_contexts_per_value: int = 8
    seed: int = 5101
    nuisance_mode: str = "random"


class StateToNumberTokens(nn.Module):
    """Small replica of the player-vector to autoregressive-token path."""

    def __init__(self, player_dim: int, config: ExperimentConfig) -> None:
        super().__init__()
        if config.d_model % config.heads:
            raise ValueError("d_model must be divisible by heads")
        self.config = config
        self.player_projection = nn.Linear(player_dim, config.d_model)
        world_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.world = nn.TransformerEncoder(
            world_layer, num_layers=config.world_layers, enable_nested_tensor=False
        )
        self.world_norm = nn.LayerNorm(config.d_model)
        self.token_embedding = nn.Embedding(len(TOKENS), config.d_model, padding_idx=PAD_ID)
        self.position = nn.Embedding(8, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, len(TOKENS), bias=False)
        self.output.weight = self.token_embedding.weight

    def encode(self, player: torch.Tensor) -> torch.Tensor:
        memory = self.player_projection(player).unsqueeze(1)
        return self.world_norm(self.world(memory))

    def forward(self, player: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encode(player)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position(positions)[None]
        causal = torch.triu(
            torch.ones(
                input_ids.shape[1], input_ids.shape[1],
                dtype=torch.bool, device=input_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=input_ids.eq(PAD_ID),
        )
        return self.output(self.decoder_norm(hidden))

    @torch.no_grad()
    def generate(self, player: torch.Tensor, max_new_tokens: int = 5) -> torch.Tensor:
        generated = torch.full(
            (player.shape[0], 1), BOS_ID, dtype=torch.long, device=player.device
        )
        finished = torch.zeros(player.shape[0], dtype=torch.bool, device=player.device)
        for _ in range(max_new_tokens):
            next_id = self(player, generated)[:, -1].argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, END_ID), next_id)
            generated = torch.cat((generated, next_id[:, None]), dim=1)
            finished |= next_id.eq(END_ID)
            if bool(finished.all()):
                break
        return generated


def _number_ids(value: int) -> tuple[int, ...]:
    return tuple(TOKEN_TO_ID[token] for token in _calculation_number(value))


def _targets(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [(BOS_ID, *_number_ids(int(value))) for value in values.cpu().tolist()]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width - 1), PAD_ID, dtype=torch.long)
    labels = torch.full((len(rows), width - 1), PAD_ID, dtype=torch.long)
    for index, row in enumerate(rows):
        ids = torch.tensor(row, dtype=torch.long)
        input_ids[index, :len(row) - 1] = ids[:-1]
        labels[index, :len(row) - 1] = ids[1:]
    return input_ids.to(values.device), labels.to(values.device)


def _player_vectors(
    values: torch.Tensor,
    state: StateVectorConfig,
    *,
    generator: torch.Generator,
    nuisance_mode: str,
) -> torch.Tensor:
    """Build consistent HP fields plus random nuisance state slots."""

    batch = values.shape[0]
    if nuisance_mode == "random":
        player = torch.rand(
            batch, state.player_dim, generator=generator, device=values.device
        )
    elif nuisance_mode == "zero":
        player = torch.zeros(batch, state.player_dim, device=values.device)
    else:
        raise ValueError(f"unsupported nuisance mode: {nuisance_mode}")
    hp_index = state.player_scalar_names.index("hp")
    max_hp_index = state.player_scalar_names.index("max_hp")
    hp_frac_index = state.player_scalar_names.index("hp_frac")
    if nuisance_mode == "random":
        room = state.hp_scale - values.float()
        max_hp = values.float() + torch.floor(
            torch.rand(batch, generator=generator, device=values.device) * (room + 1.0)
        )
        max_hp = max_hp.clamp_min(1.0)
    else:
        max_hp = torch.full_like(values, state.hp_scale, dtype=torch.float32)
    player[:, hp_index] = values.float() / state.hp_scale
    player[:, max_hp_index] = max_hp / state.hp_scale
    player[:, hp_frac_index] = values.float() / max_hp
    return player


def _sample_values(
    support: Sequence[int], batch_size: int, device: torch.device,
    *, generator: torch.Generator,
) -> torch.Tensor:
    support_tensor = torch.tensor(support, dtype=torch.long, device=device)
    indices = torch.randint(
        len(support), (batch_size,), generator=generator, device=device
    )
    return support_tensor[indices]


def _trim_generated(row: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for token in row[1:]:
        result.append(int(token))
        if token == END_ID:
            break
    return tuple(result)


def _decode_tokens(ids: Iterable[int]) -> list[str]:
    return [TOKENS[int(index)] for index in ids]


@torch.no_grad()
def _evaluate(
    model: StateToNumberTokens,
    support: Sequence[int],
    state: StateVectorConfig,
    config: ExperimentConfig,
    device: torch.device,
    *, seed: int,
) -> dict[str, object]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.tensor(
        [value for value in support for _ in range(config.eval_contexts_per_value)],
        dtype=torch.long,
        device=device,
    )
    total_tokens = correct_tokens = exact = 0
    failures: list[dict[str, object]] = []
    eval_batch = min(config.batch_size, 1024)
    for start in range(0, len(values), eval_batch):
        batch_values = values[start:start + eval_batch]
        player = _player_vectors(
            batch_values, state, generator=generator,
            nuisance_mode=config.nuisance_mode,
        )
        input_ids, labels = _targets(batch_values)
        logits = model(player, input_ids)
        valid = labels.ne(PAD_ID)
        correct_tokens += int((logits.argmax(-1).eq(labels) & valid).sum())
        total_tokens += int(valid.sum())
        generated = model.generate(player).cpu().tolist()
        for value, row in zip(batch_values.cpu().tolist(), generated):
            expected = _number_ids(int(value))
            predicted = _trim_generated(row)
            if predicted == expected:
                exact += 1
            elif len(failures) < 20:
                failures.append({
                    "value": int(value),
                    "expected": _decode_tokens(expected),
                    "predicted": _decode_tokens(predicted),
                })
    count = len(values)
    return {
        "support_size": len(support),
        "examples": count,
        "teacher_forced_token_accuracy": correct_tokens / max(total_tokens, 1),
        "autoregressive_exact_accuracy": exact / max(count, 1),
        "exact": exact,
        "failures": failures,
    }


def _train_regime(
    name: str,
    train_support: Sequence[int],
    evaluations: dict[str, Sequence[int]],
    state: StateVectorConfig,
    config: ExperimentConfig,
    device: torch.device,
    *, seed: int,
) -> tuple[StateToNumberTokens, dict[str, object]]:
    torch.manual_seed(seed)
    model = StateToNumberTokens(state.player_dim, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    recent_loss = 0.0
    recent_steps = 0
    for step in range(1, config.train_steps + 1):
        model.train()
        values = _sample_values(
            train_support, config.batch_size, device, generator=generator
        )
        player = _player_vectors(
            values, state, generator=generator,
            nuisance_mode=config.nuisance_mode,
        )
        input_ids, labels = _targets(values)
        logits = model(player, input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, len(TOKENS)), labels.reshape(-1), ignore_index=PAD_ID
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent_loss += float(loss.detach())
        recent_steps += 1
        if step % 100 == 0 or step == config.train_steps:
            print(
                f"[number-token] regime={name} step={step}/{config.train_steps} "
                f"loss={recent_loss / recent_steps:.6f}",
                flush=True,
            )
            recent_loss = 0.0
            recent_steps = 0
    results = {
        key: _evaluate(
            model, support, state, config, device,
            seed=seed + 100 + index,
        )
        for index, (key, support) in enumerate(evaluations.items())
    }
    return model, {
        "name": name,
        "train_support_size": len(train_support),
        "train_support": list(train_support),
        "evaluations": results,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config = ExperimentConfig(
        max_value=args.max_value,
        holdout_modulus=args.holdout_modulus,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        heads=args.heads,
        world_layers=args.world_layers,
        decoder_layers=args.decoder_layers,
        feedforward=args.feedforward,
        eval_contexts_per_value=args.eval_contexts_per_value,
        seed=args.seed,
        nuisance_mode=args.nuisance_mode,
    )
    if config.max_value < 1 or config.holdout_modulus < 2:
        raise ValueError("max-value must be positive and holdout-modulus must be at least 2")
    if config.train_steps < 1 or config.batch_size < 1:
        raise ValueError("train-steps and batch-size must be positive")
    if config.nuisance_mode not in {"random", "zero"}:
        raise ValueError("nuisance-mode must be random or zero")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    state = StateVectorConfig()
    all_values = list(range(config.max_value + 1))
    held_out = [value for value in all_values if value % config.holdout_modulus == 0]
    interpolation_train = [value for value in all_values if value not in set(held_out)]

    regimes: list[dict[str, object]] = []
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    model, result = _train_regime(
        "full_support",
        all_values,
        {"all_values": all_values},
        state,
        config,
        device,
        seed=config.seed,
    )
    regimes.append(result)
    checkpoints["full_support"] = model.state_dict()
    model, result = _train_regime(
        "interpolation_holdout",
        interpolation_train,
        {"seen_values": interpolation_train, "held_out_values": held_out},
        state,
        config,
        device,
        seed=config.seed + 1000,
    )
    regimes.append(result)
    checkpoints["interpolation_holdout"] = model.state_dict()
    report = {
        "experiment": "continuous_player_vector_to_integer_tokens",
        "device": str(device),
        "config": asdict(config),
        "player_dim": state.player_dim,
        "target_field": "player.hp",
        "normalization_scale": state.hp_scale,
        "token_vocabulary": list(TOKENS),
        "deterministic_rounding_baseline_accuracy": 1.0,
        "regimes": regimes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"report": report, "models": checkpoints}, args.checkpoint)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-value", type=int, default=200)
    parser.add_argument("--holdout-modulus", type=int, default=7)
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--world-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--eval-contexts-per-value", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--nuisance-mode", choices=("random", "zero"), default="random")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

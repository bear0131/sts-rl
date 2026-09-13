"""Train the standalone anonymous vector/event rule model.

This entry point consumes independent schema-v5 vector-rule JSONL data.  It
trains the v6 world encoder and causal event decoder jointly, using
cross-entropy for structure tokens and Smooth-L1 only at observed
``DELTA_SLOT`` positions.  New runs share the contextual rule-program
Transformer between the anonymous card and relic encoders; pass
``--no-share-rule-transformer`` for the legacy independent layout.

Example (small local smoke run)::

    python -m rl.rules.training.train_rule_vectors rl/data/vector-rule-smoke-v6.jsonl \
        --out runs/vector-smoke --epochs 2 --batch-size 4 \
        --card-dim 32 --relic-dim 32 --hand-dim 32 --world-dim 64 \
        --decoder-dim 64 --card-token-dim 32 --relic-token-dim 32 \
        --world-heads 4 --decoder-heads 4 --card-heads 4 --relic-heads 4

One-pass online smoke run (the producer and trainer run concurrently)::

    python -m rl.rules.training.train_rule_vectors \
        --stream-command "python -m rl.rules.data.make_vector_rule_smoke_data --stdout --stream" \
        --stream-steps 100 --batch-size 8 --out runs/vector-stream-smoke
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl.rules.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    VectorizedState,
    VectorRuleWorldModel,
    VectorRuleModelConfig,
    default_vector_model_config,
)
from rl.rules.vector_rule_data import (
    VECTOR_DATA_SCHEMA_VERSION,
    VectorRuleV6Dataset,
    collate_vector_rule_v6_batch,
)
from rl.rules.vector_rule_stream import (
    SubprocessV6RecordStream,
    iter_v6_examples,
    take_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", type=Path, nargs="*",
        help="Independent schema-v5 vector-rule JSONL/JSONL.GZ files.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/rules-vector-v6"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--max-relics", type=int, default=32)
    parser.add_argument("--action-upgrade-scale", type=float, default=5.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-data-reuse", action="store_true",
        help="allow fixed files to be revisited for --epochs > 1",
    )
    parser.add_argument(
        "--stream-command",
        help=(
            "one-pass producer command; it must write one schema-v5 JSON object "
            "per stdout line"
        ),
    )
    parser.add_argument("--stream-cwd", type=Path)
    parser.add_argument(
        "--stream-steps", type=int, default=0,
        help="optimizer steps to consume from --stream-command",
    )
    parser.add_argument(
        "--validation-stream-command",
        help="optional independent producer command for one-pass validation",
    )
    parser.add_argument("--validation-stream-steps", type=int, default=0)
    parser.add_argument(
        "--validation-batch-size", type=int, default=0,
        help="fresh samples per validation point; 0 uses --batch-size",
    )
    parser.add_argument(
        "--validation-every-steps", type=int, default=1,
        help="consume fresh validation batches and log a point at this interval",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl-vector-v6")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
    )
    parser.add_argument("--wandb-log-every", type=int, default=1)
    parser.add_argument(
        "--wandb-step-offset", type=int, default=0,
        help="global W&B step offset when appending to an existing run",
    )
    parser.add_argument(
        "--wandb-sample-offset", type=int, default=0,
        help="sample-count offset for W&B when appending to an existing run",
    )

    # Independent module capacities.  These mirror default_vector_model_config
    # so a remote run can be scaled without editing Python source.
    parser.add_argument("--card-dim", type=int, default=128)
    parser.add_argument("--relic-dim", type=int, default=128)
    parser.add_argument("--hand-dim", type=int, default=128)
    parser.add_argument("--world-dim", type=int, default=256)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--card-token-dim", type=int, default=64)
    parser.add_argument("--relic-token-dim", type=int, default=64)
    parser.add_argument("--card-heads", type=int, default=4)
    parser.add_argument("--relic-heads", type=int, default=4)
    parser.add_argument("--world-heads", type=int, default=8)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--card-layers", type=int, default=2)
    parser.add_argument("--relic-layers", type=int, default=2)
    parser.add_argument("--hand-layers", type=int, default=2)
    parser.add_argument("--world-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--card-feedforward", type=int, default=256)
    parser.add_argument("--relic-feedforward", type=int, default=256)
    parser.add_argument("--hand-hidden", type=int, default=256)
    parser.add_argument("--hand-heads", type=int, default=4)
    parser.add_argument("--hand-feedforward", type=int, default=512)
    parser.add_argument("--world-feedforward", type=int, default=1024)
    parser.add_argument("--decoder-feedforward", type=int, default=1024)
    parser.add_argument("--delta-hidden", type=int, default=512)
    parser.add_argument("--delta-layers", type=int, default=2)
    parser.add_argument("--decoder-max-tokens", type=int, default=2048)
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument(
        "--share-rule-transformer", dest="share_rule_transformer",
        action="store_true",
        help="share the contextual Transformer between card and relic rule encoders",
    )
    share_group.add_argument(
        "--no-share-rule-transformer", dest="share_rule_transformer",
        action="store_false",
        help="keep independent card and relic rule Transformers",
    )
    parser.set_defaults(share_rule_transformer=True)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _config(args: argparse.Namespace) -> VectorRuleModelConfig:
    return default_vector_model_config(
        card_dim=args.card_dim,
        relic_dim=args.relic_dim,
        hand_dim=args.hand_dim,
        world_dim=args.world_dim,
        decoder_dim=args.decoder_dim,
        card_token_dim=args.card_token_dim,
        relic_token_dim=args.relic_token_dim,
        card_heads=args.card_heads,
        relic_heads=args.relic_heads,
        world_heads=args.world_heads,
        decoder_heads=args.decoder_heads,
        card_layers=args.card_layers,
        relic_layers=args.relic_layers,
        hand_layers=args.hand_layers,
        world_layers=args.world_layers,
        decoder_layers=args.decoder_layers,
        card_feedforward=args.card_feedforward,
        relic_feedforward=args.relic_feedforward,
        hand_hidden=args.hand_hidden,
        hand_heads=args.hand_heads,
        hand_feedforward=args.hand_feedforward,
        world_feedforward=args.world_feedforward,
        decoder_feedforward=args.decoder_feedforward,
        delta_hidden=args.delta_hidden,
        delta_layers=args.delta_layers,
        max_enemies=5,
        max_relics=args.max_relics,
        hand_max_cards=args.max_cards,
        decoder_max_tokens=args.decoder_max_tokens,
        action_upgrade_scale=args.action_upgrade_scale,
        response_dim=0,
        predict_hand_delta=False,
        share_rule_transformer=args.share_rule_transformer,
    )


def _resume_config_compat(args: argparse.Namespace) -> None:
    """Adopt the architecture flag stored by a checkpoint when resuming.

    Checkpoints created before the shared rule backbone was introduced have no
    flag and therefore use the old independent layout.
    """

    if args.resume is None:
        return
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    payload = checkpoint.get("config")
    if isinstance(payload, dict):
        args.share_rule_transformer = bool(payload.get("share_rule_transformer", False))


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    state = batch.get("state")
    if isinstance(state, VectorizedState):
        batch["state"] = VectorizedState(
            player=state.player.to(device),
            enemies=state.enemies.to(device),
            piles=state.piles.to(device),
            relic_ids=state.relic_ids,
            relic_counters=(
                state.relic_counters.to(device)
                if state.relic_counters is not None else None
            ),
        )
    return batch


def _model_inputs(model: VectorRuleWorldModel, batch: dict[str, Any]) -> dict[str, Any]:
    state = batch["state"]
    action_vectors = model.card_encoder.forward_programs(batch["action_programs"])
    hand_programs = batch["hand_programs"]
    if hand_programs.shape[1]:
        hand_shape = hand_programs.shape
        hand_vectors = model.card_encoder.forward_programs(
            hand_programs.reshape(-1, hand_shape[-1])
        ).reshape(hand_shape[0], hand_shape[1], -1)
    else:
        hand_vectors = hand_programs.new_empty(
            hand_programs.shape[0], 0, model.card_encoder.config.output_dim,
            dtype=torch.float32,
        )
    relic_programs = batch["relic_programs"]
    max_relics = relic_programs.shape[1]
    if max_relics:
        relic_shape = relic_programs.shape
        relic_vectors = model.relic_encoder.forward(
            relic_programs.reshape(-1, relic_shape[-1])
        ).reshape(relic_shape[0], relic_shape[1], -1)
    else:
        relic_vectors = None
    return {
        "state": state,
        "target_input_ids": batch["target_input_ids"],
        "action_card_vector": action_vectors,
        "action_upgrade": batch["action_upgrades"],
        "hand_card_vectors": hand_vectors,
        "hand_upgrades": batch["hand_upgrades"],
        "hand_mask": batch["hand_mask"],
        "relic_vectors": relic_vectors,
        "relic_mask": batch["relic_mask"] if max_relics else None,
    }


def _loss(model: VectorRuleWorldModel, batch: dict[str, Any], delta_loss_weight: float):
    output = model(**_model_inputs(model, batch))
    labels = batch["labels"]
    pad_id = model.decoder.config.pad_id
    token_loss = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad_id,
    )
    mask = batch["delta_mask"]
    if bool(mask.any()):
        delta_loss = F.smooth_l1_loss(output.delta[mask], batch["delta_targets"][mask])
    else:
        delta_loss = output.delta.sum() * 0.0
    return token_loss + delta_loss_weight * delta_loss, token_loss.detach(), delta_loss.detach(), output


def _sequence_length_stats(
    batch: dict[str, Any], pad_id: int,
) -> dict[str, int]:
    """Measure both unpadded targets and the width actually passed to attention.

    ``labels`` contains the shifted target stream, so a non-PAD count is the
    exact number of supervised autoregressive positions for each example.  The
    decoder receives the batch-padded ``target_input_ids`` width, which is kept
    separately to expose padding overhead and the true peak allocation.
    """

    labels = batch["labels"]
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    lengths = labels.ne(pad_id).sum(dim=1)
    return {
        "sequence_length_sum": int(lengths.sum().item()),
        "sequence_length_count": int(lengths.numel()),
        "sequence_length_max": int(lengths.max().item()) if lengths.numel() else 0,
        "model_input_length": int(batch["target_input_ids"].shape[1]),
    }


def _new_metrics() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "token_loss_sum": 0.0,
        "delta_loss_sum": 0.0,
        "tokens": 0,
        "correct_tokens": 0,
        "delta_slots": 0,
        "batches": 0,
        "sequence_length_sum": 0,
        "sequence_length_count": 0,
        "sequence_length_max": 0,
        "model_input_length_max": 0,
    }


@torch.no_grad()
def _evaluate(model, loader, device: torch.device, delta_loss_weight: float) -> dict[str, float]:
    model.eval()
    total_loss = total_token = total_delta = 0.0
    total_tokens = correct_tokens = delta_count = 0
    length_sum = length_count = length_max = model_input_length_max = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        loss, token_loss, delta_loss, output = _loss(model, batch, delta_loss_weight)
        valid = batch["labels"].ne(model.decoder.config.pad_id)
        length_stats = _sequence_length_stats(batch, model.decoder.config.pad_id)
        total_loss += float(loss) * int(valid.sum())
        total_token += float(token_loss) * int(valid.sum())
        total_delta += float(delta_loss)
        total_tokens += int(valid.sum())
        correct_tokens += int((output.logits.argmax(-1).eq(batch["labels"]) & valid).sum())
        delta_count += int(batch["delta_mask"].sum())
        length_sum += length_stats["sequence_length_sum"]
        length_count += length_stats["sequence_length_count"]
        length_max = max(length_max, length_stats["sequence_length_max"])
        model_input_length_max = max(model_input_length_max, length_stats["model_input_length"])
    denominator = max(total_tokens, 1)
    return {
        "loss": total_loss / denominator,
        "token_loss": total_token / denominator,
        "delta_loss": total_delta / max(len(loader), 1),
        "token_accuracy": correct_tokens / denominator,
        "delta_slots": float(delta_count),
        "sequence_length_mean": length_sum / max(length_count, 1),
        "sequence_length_max": float(length_max),
        "model_input_length_max": float(model_input_length_max),
    }


def _stream_examples(args: argparse.Namespace, config: VectorRuleModelConfig, producer):
    return iter_v6_examples(
        iter(producer),
        config.state,
        card_vocab_size=len(CARD_RULE_VOCAB),
        relic_vocab_size=len(RELIC_RULE_VOCAB),
        max_cards=args.max_cards,
        max_relics=args.max_relics,
        max_tokens=args.max_tokens,
    )


def _stream_metrics_update(
    metrics: dict[str, float], model: VectorRuleWorldModel,
    batch: dict[str, Any], output, loss: torch.Tensor,
    token_loss: torch.Tensor, delta_loss: torch.Tensor,
) -> None:
    valid = batch["labels"].ne(model.decoder.config.pad_id)
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise RuntimeError("training batch contains no supervised event tokens")
    metrics["loss_sum"] += float(loss.detach()) * valid_count
    metrics["token_loss_sum"] += float(token_loss.detach()) * valid_count
    metrics["delta_loss_sum"] += float(delta_loss.detach())
    metrics["tokens"] += valid_count
    metrics["correct_tokens"] += int((output.logits.argmax(-1).eq(batch["labels"]) & valid).sum())
    metrics["delta_slots"] += int(batch["delta_mask"].sum())
    metrics["batches"] += 1
    length_stats = _sequence_length_stats(batch, model.decoder.config.pad_id)
    metrics["sequence_length_sum"] = metrics.get("sequence_length_sum", 0) + length_stats[
        "sequence_length_sum"
    ]
    metrics["sequence_length_count"] = metrics.get("sequence_length_count", 0) + length_stats[
        "sequence_length_count"
    ]
    metrics["sequence_length_max"] = max(
        metrics.get("sequence_length_max", 0), length_stats["sequence_length_max"]
    )
    metrics["model_input_length_max"] = max(
        metrics.get("model_input_length_max", 0), length_stats["model_input_length"]
    )


def _finalize_stream_metrics(metrics: dict[str, float]) -> dict[str, float]:
    token_count = max(int(metrics["tokens"]), 1)
    batches = max(int(metrics["batches"]), 1)
    return {
        "loss": metrics["loss_sum"] / token_count,
        "token_loss": metrics["token_loss_sum"] / token_count,
        "delta_loss": metrics["delta_loss_sum"] / batches,
        "token_accuracy": metrics["correct_tokens"] / token_count,
        "delta_slots": float(metrics["delta_slots"]),
        "tokens": float(metrics["tokens"]),
        "batches": float(metrics["batches"]),
        "sequence_length_mean": metrics.get("sequence_length_sum", 0)
        / max(metrics.get("sequence_length_count", 0), 1),
        "sequence_length_max": float(metrics.get("sequence_length_max", 0)),
        "model_input_length_max": float(metrics.get("model_input_length_max", 0)),
    }


def _init_wandb(
    args: argparse.Namespace, config: VectorRuleModelConfig, *, stream_mode: bool,
):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if args.wandb_log_every < 1:
        raise ValueError("--wandb-log-every must be positive")
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("--wandb requires the wandb package") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.out.name,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        mode=args.wandb_mode,
        config={
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "stream_mode": stream_mode,
            "model_config": config.to_dict(),
        },
    )
    wandb.define_metric("step")
    for prefix in ("train/*", "validation/*", "system/*"):
        wandb.define_metric(prefix, step_metric="step")
    return run


def _wandb_system_metrics(device: torch.device) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if device.type == "cuda":
        metrics["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 2**20
        metrics["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 2**20
        metrics["system/gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return metrics


@torch.no_grad()
def _evaluate_stream_batches(
    model: VectorRuleWorldModel,
    args: argparse.Namespace,
    config: VectorRuleModelConfig,
    device: torch.device,
    examples,
    steps: int,
) -> dict[str, float]:
    if steps < 1:
        raise ValueError("validation steps must be positive")
    model.eval()
    validation_batch_size = args.validation_batch_size or args.batch_size
    if validation_batch_size < 1:
        raise ValueError("validation batch size must be positive")
    metrics = _new_metrics()
    collate = lambda rows: collate_vector_rule_v6_batch(rows, pad_id=config.decoder.pad_id)
    for _ in range(steps):
        rows = take_batch(examples, validation_batch_size)
        if len(rows) != validation_batch_size:
            raise RuntimeError("validation producer ended before validation-stream-steps")
        batch = _move_batch(collate(rows), device)
        if not bool(batch["labels"].ne(model.decoder.config.pad_id).any()):
            raise RuntimeError("validation batch contains no supervised event tokens")
        loss, token_loss, delta_loss, output = _loss(model, batch, args.delta_loss_weight)
        _stream_metrics_update(metrics, model, batch, output, loss, token_loss, delta_loss)
    return _finalize_stream_metrics(metrics)


def _train_stream(args: argparse.Namespace, config: VectorRuleModelConfig, device: torch.device) -> Path:
    if args.data:
        raise ValueError("--stream-command cannot be combined with fixed data paths")
    if args.stream_steps < 1:
        raise ValueError("--stream-steps must be positive in stream mode")
    if args.epochs != 1:
        raise ValueError("stream mode is one-pass; use --stream-steps instead of --epochs")
    if bool(args.validation_stream_command) != bool(args.validation_stream_steps):
        raise ValueError(
            "validation-stream-command and validation-stream-steps must be supplied together"
        )
    if args.validation_stream_command and args.validation_every_steps < 1:
        raise ValueError("--validation-every-steps must be positive")
    if args.checkpoint_every_steps < 1:
        raise ValueError("--checkpoint-every-steps must be positive")
    if args.wandb_step_offset < 0 or args.wandb_sample_offset < 0:
        raise ValueError("W&B offsets must be non-negative")
    model = VectorRuleWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if int(checkpoint.get("schema_version", -1)) != VECTOR_DATA_SCHEMA_VERSION:
            raise ValueError("resume checkpoint is not a schema-v5 vector-rule checkpoint")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    args.out.mkdir(parents=True, exist_ok=True)
    config.save(str(args.out / "config.json"))
    metadata = {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "stream_mode": True,
        "stream_command": args.stream_command,
        "stream_steps": args.stream_steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "args": vars(args),
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    wandb_run = _init_wandb(args, config, stream_mode=True)
    collate = lambda rows: collate_vector_rule_v6_batch(rows, pad_id=config.decoder.pad_id)
    metrics = _new_metrics()
    started = time.time()
    last_validation = None
    with ExitStack() as streams:
        producer = streams.enter_context(
            SubprocessV6RecordStream(args.stream_command, cwd=args.stream_cwd)
        )
        examples = _stream_examples(args, config, producer)
        validation_examples = None
        if args.validation_stream_command:
            validation_producer = streams.enter_context(
                SubprocessV6RecordStream(args.validation_stream_command, cwd=args.stream_cwd)
            )
            validation_examples = _stream_examples(args, config, validation_producer)
        for step in range(1, args.stream_steps + 1):
            model.train()
            rows = take_batch(examples, args.batch_size)
            if len(rows) != args.batch_size:
                raise RuntimeError("producer ended before --stream-steps were consumed")
            batch = _move_batch(collate(rows), device)
            optimizer.zero_grad(set_to_none=True)
            loss, token_loss, delta_loss, output = _loss(model, batch, args.delta_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _stream_metrics_update(metrics, model, batch, output, loss, token_loss, delta_loss)
            if (
                validation_examples is not None
                and (
                    step % args.validation_every_steps == 0
                    or step == args.stream_steps
                )
            ):
                last_validation = _evaluate_stream_batches(
                    model, args, config, device, validation_examples,
                    args.validation_stream_steps,
                )
                print(
                    f"[vector-v6-stream] step={step}/{args.stream_steps} "
                    f"validation_loss={last_validation['loss']:.5f} "
                    f"validation_token={last_validation['token_loss']:.5f} "
                    f"validation_acc={last_validation['token_accuracy']:.4f} "
                    f"validation_seq_max={int(last_validation['sequence_length_max'])} "
                    f"validation_model_input_max={int(last_validation['model_input_length_max'])}",
                    flush=True,
                )
            if step % args.log_every == 0 or step == args.stream_steps:
                current = _finalize_stream_metrics(metrics)
                print(
                    f"[vector-v6-stream] step={step}/{args.stream_steps} "
                    f"loss={current['loss']:.5f} token={current['token_loss']:.5f} "
                    f"delta={current['delta_loss']:.5f} "
                    f"seq_max={int(current['sequence_length_max'])} "
                    f"model_input_max={int(current['model_input_length_max'])} "
                    f"samples={step * args.batch_size}",
                    flush=True,
                )
            train_log_due = step % args.wandb_log_every == 0 or step == args.stream_steps
            validation_log_due = last_validation is not None and (
                step % args.validation_every_steps == 0 or step == args.stream_steps
            )
            if wandb_run is not None and (train_log_due or validation_log_due):
                current = _finalize_stream_metrics(metrics)
                wandb_step = args.wandb_step_offset + step
                payload = {
                    "step": wandb_step,
                    "train/loss": current["loss"],
                    "train/token_loss": current["token_loss"],
                    "train/delta_loss": current["delta_loss"],
                    "train/token_accuracy": current["token_accuracy"],
                    "train/delta_slots": current["delta_slots"],
                    "train/sequence_length_mean": current["sequence_length_mean"],
                    "train/sequence_length_max": current["sequence_length_max"],
                    "train/model_input_length_max": current["model_input_length_max"],
                    "train/samples": args.wandb_sample_offset + step * args.batch_size,
                    "train/samples_per_second": (
                        step * args.batch_size / max(time.time() - started, 1e-6)
                    ),
                    **_wandb_system_metrics(device),
                }
                if last_validation is not None and validation_log_due:
                    payload.update({
                        "validation/loss": last_validation["loss"],
                        "validation/token_loss": last_validation["token_loss"],
                        "validation/delta_loss": last_validation["delta_loss"],
                        "validation/token_accuracy": last_validation["token_accuracy"],
                        "validation/delta_slots": last_validation["delta_slots"],
                        "validation/sequence_length_mean": last_validation[
                            "sequence_length_mean"
                        ],
                        "validation/sequence_length_max": last_validation[
                            "sequence_length_max"
                        ],
                        "validation/model_input_length_max": last_validation[
                            "model_input_length_max"
                        ],
                    })
                wandb_run.log(
                    payload,
                    step=wandb_step,
                )
            if step % args.checkpoint_every_steps == 0:
                checkpoint = {
                    "epoch": 1, "step": step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "metrics": _finalize_stream_metrics(metrics),
                    "config": config.to_dict(),
                    "schema_version": VECTOR_DATA_SCHEMA_VERSION,
                }
                torch.save(checkpoint, args.out / "last.pt")
    train_metrics = _finalize_stream_metrics(metrics)
    train_metrics["seconds"] = time.time() - started
    validation = last_validation
    checkpoint = {
        "epoch": 1, "step": args.stream_steps,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        # Keep the validation/train aggregate keys at the top level for older
        # tooling, while exposing the unambiguous split metrics as well.
        "metrics": {
            **train_metrics,
            "train": train_metrics,
            "validation": validation,
        },
        "config": config.to_dict(),
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
    }
    torch.save(checkpoint, args.out / "last.pt")
    torch.save(checkpoint, args.out / "best.pt")
    if wandb_run is not None:
        wandb_run.finish()
    return args.out / "last.pt"


def train(args: argparse.Namespace) -> Path:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.stream_command:
        device = _device(args.device)
        _resume_config_compat(args)
        config = _config(args)
        return _train_stream(args, config, device)
    if not args.data:
        raise ValueError("provide fixed v6 data paths or --stream-command")
    if args.epochs > 1 and not args.allow_data_reuse:
        raise ValueError(
            "fixed v6 data would be reused across epochs; pass --allow-data-reuse "
            "explicitly or use --stream-command"
        )
    if args.validation_stream_command or args.validation_stream_steps:
        raise ValueError("validation stream options require --stream-command")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _resume_config_compat(args)
    device = _device(args.device)
    config = _config(args)
    full_dataset = VectorRuleV6Dataset(
        args.data, config.state, card_vocab_size=len(CARD_RULE_VOCAB),
        relic_vocab_size=len(RELIC_RULE_VOCAB), max_cards=args.max_cards,
        max_relics=args.max_relics, max_tokens=args.max_tokens,
        max_examples=args.max_samples,
    )
    examples = list(full_dataset.examples)
    random.Random(args.seed).shuffle(examples)
    split = int(len(examples) * max(0.0, min(args.validation_fraction, 0.5)))
    val_examples = examples[:split]
    train_examples = examples[split:] or examples
    train_dataset = VectorRuleV6Dataset(train_examples)
    val_dataset = VectorRuleV6Dataset(val_examples) if val_examples else None
    if not train_dataset:
        raise RuntimeError("no usable vector-rule examples were found")
    collate = lambda rows: collate_vector_rule_v6_batch(rows, pad_id=config.decoder.pad_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    ) if val_dataset else None
    model = VectorRuleWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
    args.out.mkdir(parents=True, exist_ok=True)
    config.save(str(args.out / "config.json"))
    metadata = {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "card_rule_vocab": len(CARD_RULE_VOCAB),
        "relic_rule_vocab": len(RELIC_RULE_VOCAB),
        "samples": len(examples),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset) if val_dataset else 0,
        "device": str(device),
        "args": vars(args),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    wandb_run = _init_wandb(args, config, stream_mode=False)
    best = math.inf
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        train_metrics_acc = _new_metrics()
        for step, batch in enumerate(train_loader, 1):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, token_loss, delta_loss, output = _loss(model, batch, args.delta_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _stream_metrics_update(
                train_metrics_acc, model, batch, output, loss, token_loss, delta_loss
            )
            if step % args.log_every == 0 or step == len(train_loader):
                global_step = epoch * len(train_loader) + step
                current_train = _finalize_stream_metrics(train_metrics_acc)
                print(
                    f"[vector-v6] epoch={epoch + 1} step={step}/{len(train_loader)} "
                    f"loss={current_train['loss']:.5f} token={current_train['token_loss']:.5f} "
                    f"delta={current_train['delta_loss']:.5f} "
                    f"seq_max={int(current_train['sequence_length_max'])} "
                    f"model_input_max={int(current_train['model_input_length_max'])}",
                    flush=True,
                )
                if (
                    wandb_run is not None
                    and (global_step % args.wandb_log_every == 0 or step == len(train_loader))
                ):
                    wandb_run.log(
                        {
                            "step": global_step,
                            "train/loss": current_train["loss"],
                            "train/token_loss": current_train["token_loss"],
                            "train/delta_loss": current_train["delta_loss"],
                            "train/token_accuracy": current_train["token_accuracy"],
                            "train/sequence_length_mean": current_train[
                                "sequence_length_mean"
                            ],
                            "train/sequence_length_max": current_train[
                                "sequence_length_max"
                            ],
                            "train/model_input_length_max": current_train[
                                "model_input_length_max"
                            ],
                            "train/samples": epoch * len(train_dataset) + step * args.batch_size,
                            **_wandb_system_metrics(device),
                        },
                        step=global_step,
                    )
        train_metrics = _finalize_stream_metrics(train_metrics_acc)
        metrics = _evaluate(model, val_loader or train_loader, device, args.delta_loss_weight)
        metrics.update({"epoch": epoch + 1, "epoch_seconds": time.time() - started})
        print(
            f"[vector-v6] epoch={epoch + 1} val_loss={metrics['loss']:.5f} "
            f"token_acc={metrics['token_accuracy']:.4f} "
            f"delta_loss={metrics['delta_loss']:.5f} "
            f"val_seq_max={int(metrics['sequence_length_max'])} "
            f"val_model_input_max={int(metrics['model_input_length_max'])} "
            f"seconds={metrics['epoch_seconds']:.1f}", flush=True,
        )
        if wandb_run is not None:
            global_step = (epoch + 1) * len(train_loader)
            wandb_run.log(
                {
                    "step": global_step,
                    "validation/loss": metrics["loss"],
                    "validation/token_loss": metrics["token_loss"],
                    "validation/delta_loss": metrics["delta_loss"],
                    "validation/token_accuracy": metrics["token_accuracy"],
                    "validation/delta_slots": metrics["delta_slots"],
                    "validation/sequence_length_mean": metrics["sequence_length_mean"],
                    "validation/sequence_length_max": metrics["sequence_length_max"],
                    "validation/model_input_length_max": metrics[
                        "model_input_length_max"
                    ],
                    **_wandb_system_metrics(device),
                },
                step=global_step,
            )
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": {
                **metrics,
                "train": train_metrics,
                "validation": metrics,
            },
            "config": config.to_dict(),
            "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        }
        torch.save(checkpoint, args.out / "last.pt")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(checkpoint, args.out / "best.pt")
    if wandb_run is not None:
        wandb_run.finish()
    return args.out / "last.pt"


def main() -> None:
    args = build_parser().parse_args()
    path = train(args)
    print(f"[vector-v6] checkpoint={path}")


if __name__ == "__main__":
    main()

"""Train the anonymous schema-v7 effect-tree language model.

The world/card/relic encoders are the same modules used by the v6 trainer.
Only the autoregressive target language changes: records carry a nested v7
effect tree and the decoder learns its readable tree token stream.  Stream
mode consumes each producer record once, so a long run can generate fresh
combat states while it trains.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    VectorRuleWorldModel,
    VectorRuleModelConfig,
    default_vector_model_config,
)
from rl.vector_rule_v7_data import (
    V7_DATA_SCHEMA_VERSION,
    V7_EXECUTOR_CONTRACT_VERSION,
    V7TokenCodec,
    collate_vector_rule_v7_batch,
    examples_from_v7_record,
)
from rl.vector_rule_stream import SubprocessV6RecordStream, take_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-data", type=Path, required=True,
                        help="v7 JSONL bootstrap data used only to build the token vocabulary")
    parser.add_argument("--tokenizer", type=Path,
                        help="saved v7 tokenizer; defaults to <out>/tokenizer.json")
    parser.add_argument("--out", type=Path, default=Path("rl/runs/rules-vector-v7"))
    parser.add_argument("--stream-command", required=True)
    parser.add_argument("--stream-cwd", type=Path)
    parser.add_argument("--stream-steps", type=int, required=True)
    parser.add_argument("--validation-stream-command")
    parser.add_argument("--validation-stream-steps", type=int, default=0)
    parser.add_argument("--validation-every-steps", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="number of micro-batches per optimizer step",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--max-relics", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--action-upgrade-scale", type=float, default=5.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl-vector-v7")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-log-every", type=int, default=20)
    parser.add_argument("--wandb-sample-offset", type=int, default=0)

    # Keep all module capacities independently tunable, matching v6.
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
    share_group.add_argument("--share-rule-transformer", dest="share_rule_transformer", action="store_true")
    share_group.add_argument("--no-share-rule-transformer", dest="share_rule_transformer", action="store_false")
    parser.set_defaults(share_rule_transformer=True)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _config(args: argparse.Namespace, vocab_size: int) -> VectorRuleModelConfig:
    config = default_vector_model_config(
        card_dim=args.card_dim, relic_dim=args.relic_dim, hand_dim=args.hand_dim,
        world_dim=args.world_dim, decoder_dim=args.decoder_dim,
        card_token_dim=args.card_token_dim, relic_token_dim=args.relic_token_dim,
        card_heads=args.card_heads, relic_heads=args.relic_heads,
        world_heads=args.world_heads, decoder_heads=args.decoder_heads,
        card_layers=args.card_layers, relic_layers=args.relic_layers,
        hand_layers=args.hand_layers, world_layers=args.world_layers,
        decoder_layers=args.decoder_layers, card_feedforward=args.card_feedforward,
        relic_feedforward=args.relic_feedforward, hand_hidden=args.hand_hidden,
        hand_heads=args.hand_heads, hand_feedforward=args.hand_feedforward,
        world_feedforward=args.world_feedforward,
        decoder_feedforward=args.decoder_feedforward,
        delta_hidden=args.delta_hidden, delta_layers=args.delta_layers,
        max_enemies=5, max_relics=args.max_relics, hand_max_cards=args.max_cards,
        decoder_max_tokens=args.decoder_max_tokens,
        action_upgrade_scale=args.action_upgrade_scale, response_dim=0,
        predict_hand_delta=False, share_rule_transformer=args.share_rule_transformer,
    )
    decoder = replace(config.decoder, token_vocab_size=vocab_size, pad_id=0)
    return replace(config, decoder=decoder)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    state = batch["state"]
    state.player = state.player.to(device)
    state.enemies = state.enemies.to(device)
    state.piles = state.piles.to(device)
    if state.relic_counters is not None:
        state.relic_counters = state.relic_counters.to(device)
    return batch


def _model_inputs(model: VectorRuleWorldModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    action_vectors = model.card_encoder.forward_programs(batch["action_programs"])
    hand_programs = batch["hand_programs"]
    if hand_programs.shape[1]:
        shape = hand_programs.shape
        hand_vectors = model.card_encoder.forward_programs(
            hand_programs.reshape(-1, shape[-1])
        ).reshape(shape[0], shape[1], -1)
    else:
        hand_vectors = hand_programs.new_empty(
            hand_programs.shape[0], 0, model.card_encoder.config.output_dim,
            dtype=torch.float32,
        )
    relic_programs = batch["relic_programs"]
    if relic_programs.shape[1]:
        shape = relic_programs.shape
        relic_vectors = model.relic_encoder.forward(
            relic_programs.reshape(-1, shape[-1])
        ).reshape(shape[0], shape[1], -1)
    else:
        relic_vectors = None
    return {
        "state": batch["state"],
        "target_input_ids": batch["target_input_ids"],
        "action_card_vector": action_vectors,
        "action_upgrade": batch["action_upgrades"],
        "hand_card_vectors": hand_vectors,
        "hand_upgrades": batch["hand_upgrades"],
        "hand_mask": batch["hand_mask"],
        "relic_vectors": relic_vectors,
        "relic_mask": batch["relic_mask"] if relic_vectors is not None else None,
    }


def _loss(model: VectorRuleWorldModel, batch: Mapping[str, Any], delta_weight: float):
    output = model(**_model_inputs(model, batch))
    labels = batch["labels"]
    token_loss = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1),
        ignore_index=model.decoder.config.pad_id,
    )
    mask = batch["delta_mask"]
    delta_loss = (
        F.smooth_l1_loss(output.delta[mask], batch["delta_targets"][mask])
        if bool(mask.any()) else output.delta.sum() * 0.0
    )
    return token_loss + delta_weight * delta_loss, token_loss.detach(), delta_loss.detach(), output


def _metrics() -> dict[str, float]:
    return {"loss": 0.0, "token_loss": 0.0, "delta_loss": 0.0, "tokens": 0,
            "correct": 0, "delta_slots": 0, "samples": 0, "unknown": 0,
            "sequence_sum": 0, "sequence_max": 0, "model_input_max": 0,
            "realtime_correct": 0, "realtime_tokens": 0}


def _update(metrics: dict[str, float], model, batch, output, loss, token_loss, delta_loss):
    labels = batch["labels"]
    valid = labels.ne(model.decoder.config.pad_id)
    token_count = int(valid.sum())
    correct = int((output.logits.argmax(-1).eq(labels) & valid).sum())
    metrics["loss"] += float(loss.detach()) * labels.shape[0]
    metrics["token_loss"] += float(token_loss) * token_count
    metrics["delta_loss"] += float(delta_loss) * int(batch["delta_mask"].sum())
    metrics["tokens"] += token_count
    metrics["correct"] += correct
    metrics["delta_slots"] += int(batch["delta_mask"].sum())
    metrics["samples"] += labels.shape[0]
    metrics["unknown"] += int(batch.get("unknown_tokens", 0))
    lengths = valid.sum(dim=1)
    metrics["sequence_sum"] += int(lengths.sum())
    metrics["sequence_max"] = max(metrics["sequence_max"], int(lengths.max()))
    metrics["model_input_max"] = max(metrics["model_input_max"], int(batch["target_input_ids"].shape[1]))
    return correct, token_count


def _finish(
    metrics: Mapping[str, float], *, realtime_accuracy: bool = False
) -> dict[str, float]:
    samples = max(int(metrics["samples"]), 1)
    tokens = max(int(metrics["tokens"]), 1)
    deltas = max(int(metrics["delta_slots"]), 1)
    cumulative_accuracy = metrics["correct"] / tokens
    realtime_tokens = max(int(metrics.get("realtime_tokens", 0)), 1)
    realtime_accuracy_value = (
        int(metrics.get("realtime_correct", 0)) / realtime_tokens
    )
    return {
        "loss": metrics["loss"] / samples,
        "token_loss": metrics["token_loss"] / tokens,
        "delta_loss": metrics["delta_loss"] / deltas,
        "token_accuracy": (
            realtime_accuracy_value if realtime_accuracy else cumulative_accuracy
        ),
        "token_accuracy_cumulative": cumulative_accuracy,
        "samples": int(metrics["samples"]),
        "unknown_tokens": int(metrics["unknown"]),
        "sequence_length_mean": metrics["sequence_sum"] / samples,
        "sequence_length_max": int(metrics["sequence_max"]),
        "model_input_length_max": int(metrics["model_input_max"]),
        "delta_slots": int(metrics["delta_slots"]),
    }


def _init_wandb(args: argparse.Namespace, config: VectorRuleModelConfig):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if args.wandb_log_every < 1:
        raise ValueError("--wandb-log-every must be positive")
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional dependency
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
            "stream_mode": True,
            "model_config": config.to_dict(),
        },
    )
    wandb.define_metric("step")
    for prefix in ("train/*", "validation/*", "system/*"):
        wandb.define_metric(prefix, step_metric="step")
    return run


def _wandb_system_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "system/gpu_memory_allocated_mb": (
            torch.cuda.memory_allocated(device) / 2**20
        ),
        "system/gpu_memory_reserved_mb": (
            torch.cuda.memory_reserved(device) / 2**20
        ),
        "system/gpu_max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
        ),
    }


def _examples(records: Iterable[Mapping[str, Any]], config, codec, args):
    for record in records:
        yield from examples_from_v7_record(
            record, codec, config.state,
            card_vocab_size=len(CARD_RULE_VOCAB),
            relic_vocab_size=len(RELIC_RULE_VOCAB),
            max_cards=args.max_cards, max_relics=args.max_relics,
            max_tokens=args.max_tokens,
        )


def _evaluate_stream(model, args, config, codec, device, examples, steps):
    metrics = _metrics()
    collate = lambda rows: collate_vector_rule_v7_batch(rows, pad_id=codec.pad_id)
    for _ in range(steps):
        rows = take_batch(examples, args.validation_batch_size or args.batch_size)
        if not rows:
            break
        batch = _move_batch(collate(rows), device)
        with torch.no_grad():
            loss, token_loss, delta_loss, output = _loss(model, batch, args.delta_loss_weight)
        _update(metrics, model, batch, output, loss, token_loss, delta_loss)
    return _finish(metrics)


def train(args: argparse.Namespace) -> Path:
    if args.stream_steps < 1 or args.batch_size < 1:
        raise ValueError("stream-steps and batch-size must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.validation_stream_command and args.validation_stream_steps < 1:
        raise ValueError("validation stream requires validation-stream-steps")
    if args.validation_every_steps < 1 or args.checkpoint_every_steps < 1:
        raise ValueError("validation/checkpoint intervals must be positive")
    if args.wandb_sample_offset < 0:
        raise ValueError("--wandb-sample-offset must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or (args.out / "tokenizer.json")
    if tokenizer_path.exists():
        codec = V7TokenCodec.load(tokenizer_path)
    else:
        codec = V7TokenCodec.from_jsonl([args.vocab_data])
        codec.save(tokenizer_path)
    config = _config(args, len(codec.tokens))
    config.save(str(args.out / "config.json"))
    metadata = {
        "schema_version": V7_DATA_SCHEMA_VERSION,
        "executor_contract_version": V7_EXECUTOR_CONTRACT_VERSION,
        "stream_mode": True,
        "stream_command": args.stream_command,
        "stream_steps": args.stream_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "device": str(device),
        "vocab_size": len(codec.tokens),
        "args": vars(args),
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    model = VectorRuleWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if int(checkpoint.get("schema_version", -1)) != V7_DATA_SCHEMA_VERSION:
            raise ValueError("resume checkpoint is not schema-v7")
        if int(checkpoint.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
            raise ValueError("resume checkpoint uses an incompatible executor contract")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
    collate = lambda rows: collate_vector_rule_v7_batch(rows, pad_id=codec.pad_id)
    metrics = _metrics()
    started = time.time()
    last_validation = None
    wandb_run = _init_wandb(args, config)
    try:
        with ExitStack() as streams:
            producer = streams.enter_context(SubprocessV6RecordStream(args.stream_command, cwd=args.stream_cwd))
            examples = _examples(producer, config, codec, args)
            validation_examples = None
            if args.validation_stream_command:
                validation_producer = streams.enter_context(
                    SubprocessV6RecordStream(args.validation_stream_command, cwd=args.stream_cwd)
                )
                validation_examples = _examples(validation_producer, config, codec, args)
            for offset in range(1, args.stream_steps + 1):
                step = start_step + offset
                model.train()
                optimizer.zero_grad(set_to_none=True)
                realtime_correct = 0
                realtime_tokens = 0
                for _ in range(args.gradient_accumulation_steps):
                    rows = take_batch(examples, args.batch_size)
                    if len(rows) != args.batch_size:
                        raise RuntimeError(
                            "v7 producer ended before stream-steps were consumed"
                        )
                    batch = _move_batch(collate(rows), device)
                    loss, token_loss, delta_loss, output = _loss(
                        model, batch, args.delta_loss_weight
                    )
                    (loss / args.gradient_accumulation_steps).backward()
                    batch_correct, batch_tokens = _update(
                        metrics, model, batch, output, loss, token_loss, delta_loss
                    )
                    realtime_correct += batch_correct
                    realtime_tokens += batch_tokens
                metrics["realtime_correct"] = realtime_correct
                metrics["realtime_tokens"] = realtime_tokens
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                validation_due = validation_examples is not None and (
                    offset % args.validation_every_steps == 0
                    or offset == args.stream_steps
                )
                if validation_due:
                    model.eval()
                    last_validation = _evaluate_stream(
                        model, args, config, codec, device, validation_examples,
                        args.validation_stream_steps,
                    )
                    print(
                        f"[vector-v7] step={step}/{start_step + args.stream_steps} "
                        f"validation_loss={last_validation['loss']:.5f} "
                        f"validation_token={last_validation['token_loss']:.5f} "
                        f"validation_acc={last_validation['token_accuracy']:.4f}",
                        flush=True,
                    )
                if offset % args.log_every == 0 or offset == args.stream_steps:
                    current = _finish(metrics, realtime_accuracy=True)
                    print(
                        f"[vector-v7] step={step}/{start_step + args.stream_steps} "
                        f"loss={current['loss']:.5f} token={current['token_loss']:.5f} "
                        f"delta={current['delta_loss']:.5f} acc={current['token_accuracy']:.4f} "
                        f"seq_max={current['sequence_length_max']} "
                        f"model_input_max={current['model_input_length_max']} "
                        f"unknown={current['unknown_tokens']} samples={current['samples']} "
                        f"effective_batch={args.batch_size * args.gradient_accumulation_steps}",
                        flush=True,
                    )
                train_log_due = (
                    step % args.wandb_log_every == 0
                    or offset == args.stream_steps
                )
                if wandb_run is not None and (train_log_due or validation_due):
                    current = _finish(metrics, realtime_accuracy=True)
                    payload = {
                        "step": step,
                        "train/loss": current["loss"],
                        "train/token_loss": current["token_loss"],
                        "train/delta_loss": current["delta_loss"],
                        "train/token_accuracy": current["token_accuracy"],
                        "train/token_accuracy_cumulative": current[
                            "token_accuracy_cumulative"
                        ],
                        "train/delta_slots": current["delta_slots"],
                        "train/sequence_length_mean": current["sequence_length_mean"],
                        "train/sequence_length_max": current["sequence_length_max"],
                        "train/model_input_length_max": current["model_input_length_max"],
                        "train/unknown_tokens": current["unknown_tokens"],
                        "train/micro_batch_size": args.batch_size,
                        "train/gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "train/effective_batch_size": (
                            args.batch_size * args.gradient_accumulation_steps
                        ),
                        "train/samples": (
                            args.wandb_sample_offset
                            + offset * args.batch_size * args.gradient_accumulation_steps
                        ),
                        "train/samples_per_second": (
                            offset * args.batch_size * args.gradient_accumulation_steps
                            / max(time.time() - started, 1e-6)
                        ),
                        **_wandb_system_metrics(device),
                    }
                    if last_validation is not None and validation_due:
                        payload.update({
                            "validation/loss": last_validation["loss"],
                            "validation/token_loss": last_validation["token_loss"],
                            "validation/delta_loss": last_validation["delta_loss"],
                            "validation/token_accuracy": last_validation["token_accuracy"],
                            "validation/delta_slots": last_validation["delta_slots"],
                            "validation/sequence_length_mean": last_validation["sequence_length_mean"],
                            "validation/sequence_length_max": last_validation["sequence_length_max"],
                            "validation/model_input_length_max": last_validation["model_input_length_max"],
                            "validation/unknown_tokens": last_validation["unknown_tokens"],
                        })
                    wandb_run.log(payload, step=step)
                if offset % args.checkpoint_every_steps == 0 or offset == args.stream_steps:
                    checkpoint = {
                        "schema_version": V7_DATA_SCHEMA_VERSION,
                        "executor_contract_version": V7_EXECUTOR_CONTRACT_VERSION,
                        "step": step,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "effective_batch_size": (
                            args.batch_size * args.gradient_accumulation_steps
                        ),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": {
                            "train": _finish(metrics, realtime_accuracy=True),
                            "validation": last_validation,
                        },
                        "config": config.to_dict(),
                    }
                    torch.save(checkpoint, args.out / "last.pt")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    final_metrics = _finish(metrics, realtime_accuracy=True)
    final_metrics["seconds"] = time.time() - started
    (args.out / "metrics.json").write_text(
        json.dumps({"train": final_metrics, "validation": last_validation}, indent=2),
        encoding="utf-8",
    )
    return args.out / "last.pt"


if __name__ == "__main__":
    train(build_parser().parse_args())

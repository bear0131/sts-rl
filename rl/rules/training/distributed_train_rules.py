"""Synchronous multi-machine training for sharded combat-rule data.

Each rank reads only its local JSONL shard. Gradients are accumulated locally,
packed into one FP32 vector, averaged over the existing persistent SSH ring,
and then applied by identical AdamW optimizers on every rank.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import time

import torch
from torch.utils.data import DataLoader

from rl.training.ssh_transport import (
    all_rank_file_barrier,
    parse_peer,
    ssh_cmd,
)
from rl.training.ring_allreduce import SshRingAllReduce
from rl.rules.rule_data import (
    CardBalancedBatchSampler,
    NEXT_STATE_BEGIN,
    RULE_DATA_VERSION,
    RuleDataset,
    RuleTokenizer,
    collate_rule_batch,
    read_jsonl,
    sample_tokens,
)
from rl.rules.rule_model import RuleModelConfig, RuleWorldModel
from rl.rules.training.train_rules import (
    _evaluate,
    _loss_and_counts,
    _split_samples,
    _transition_consistency_loss,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--peer", action="append", default=[])
    parser.add_argument("--ssh-key", default="~/.ssh/sts_dist_test_ed25519")
    parser.add_argument("--ssh-control-dir", default="/tmp/sts_rule_ssh_mux")
    parser.add_argument("--remote-root", default="/tmp/sts_dist_rules")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ring-port", type=int, default=29760)
    parser.add_argument("--barrier-timeout", type=float, default=900.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, required=True,
                        help="Optimizer steps per rank and epoch.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Local micro-batch size on each rank.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--length-bucket-multiplier", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-mode", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--float32-matmul-precision",
                        choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--log-every", type=int, default=25)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-prefix-tokens", type=int, default=4096)
    parser.add_argument("--max-target-tokens", type=int, default=4096)

    parser.add_argument("--critical-number-weight", type=float, default=10.0)
    parser.add_argument("--effect-number-weight", type=float)
    parser.add_argument("--next-number-weight", type=float)
    parser.add_argument("--calculation-number-weight", type=float)
    parser.add_argument("--step-number-weight", type=float, default=8.0)
    parser.add_argument("--causal-token-weight", type=float, default=4.0)
    parser.add_argument("--counter-token-weight", type=float, default=10.0)
    parser.add_argument(
        "--structure-token-weight", type=float, default=1.0,
        help="Weight grammar/schema tokens such as event and zone markers.",
    )
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--consistency-scale", type=float, default=10.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-best-on-resume", action="store_true",
        help="Reset best-loss tracking when the resumed validation set changes.",
    )
    parser.add_argument(
        "--override-optimizer-hyperparameters-on-resume", action="store_true",
        help="Apply command-line learning rate and weight decay after loading "
             "the optimizer state.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-id")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-backfill-log", type=Path,
        help="Rank-0 log whose completed epochs are uploaded before resuming.",
    )
    return parser


def _gradient_parameters(model: RuleWorldModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _pack_gradients(model: RuleWorldModel) -> torch.Tensor:
    parts = [
        parameter.grad.detach().reshape(-1)
        if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1)
        for parameter in _gradient_parameters(model)
    ]
    flat = torch.cat(parts).detach().float().cpu().contiguous()
    return flat


def _install_gradients(model: RuleWorldModel, flat: torch.Tensor) -> None:
    device_flat = flat.to(next(model.parameters()).device)
    offset = 0
    for parameter in _gradient_parameters(model):
        end = offset + parameter.numel()
        parameter.grad = device_flat[offset:end].view_as(parameter)
        offset = end
    if offset != flat.numel():
        raise RuntimeError(f"installed {offset} gradient values from {flat.numel()}")


def _average_metrics(ring, metrics: dict[str, float]) -> dict[str, float]:
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float32)
    averaged, _bytes = ring.average(values)
    return {key: float(value) for key, value in zip(keys, averaged.tolist())}


def _jsonable_args(args) -> dict:
    return json.loads(json.dumps(vars(args), default=str))


def _save_checkpoint(path, model, optimizer, epoch, metrics, args) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "training_config": _jsonable_args(args),
        "distributed": {
            "world_size": args.world_size,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size_per_rank": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
    }, path)


_TRAIN_LOG_PATTERN = re.compile(
    r"^\[rule-dist rank=0\] epoch=(?P<epoch>\d+) "
    r"update=(?P<update>\d+)/(?P<steps>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) "
    r"sync=(?P<sync>[0-9.eE+-]+)s "
    r"grad_mb=(?P<grad_mb>[0-9.eE+-]+)$"
)
_VALIDATION_LOG_PATTERN = re.compile(
    r"^\[rule-dist rank=0\] epoch=(?P<epoch>\d+) "
    r"val_loss=(?P<loss>[0-9.eE+-]+) "
    r"effects=(?P<effects>[0-9.eE+-]+)% "
    r"next=(?P<next>[0-9.eE+-]+)% "
    r"epoch_sec=(?P<epoch_sec>[0-9.eE+-]+) "
    r"sync_sec=(?P<sync_sec>[0-9.eE+-]+) "
    r"elapsed=(?P<elapsed>[0-9.eE+-]+)$"
)


def _historical_wandb_rows(
    path: Path, max_epoch: int, steps_per_epoch: int,
) -> list[tuple[int, dict[str, float | int]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        train_match = _TRAIN_LOG_PATTERN.match(line)
        if train_match:
            values = train_match.groupdict()
            epoch = int(values["epoch"])
            if epoch <= max_epoch:
                update = int(values["update"])
                global_update = (epoch - 1) * steps_per_epoch + update
                rows.append((global_update, {
                    "epoch": epoch,
                    "epoch_update": update,
                    "global_update": global_update,
                    "train/loss": float(values["loss"]),
                    "train/sync_seconds": float(values["sync"]),
                    "train/gradient_megabytes": float(values["grad_mb"]),
                }))
            continue
        validation_match = _VALIDATION_LOG_PATTERN.match(line)
        if not validation_match:
            continue
        values = validation_match.groupdict()
        epoch = int(values["epoch"])
        if epoch > max_epoch:
            continue
        global_update = epoch * steps_per_epoch
        rows.append((global_update, {
            "epoch": epoch,
            "epoch_update": steps_per_epoch,
            "global_update": global_update,
            "validation/loss": float(values["loss"]),
            "validation/effect_token_accuracy": float(values["effects"]) / 100.0,
            "validation/next_state_token_accuracy": float(values["next"]) / 100.0,
            "time/epoch_seconds": float(values["epoch_sec"]),
            "time/sync_seconds": float(values["sync_sec"]),
            "time/elapsed_seconds": float(values["elapsed"]),
        }))
    return rows


def _init_wandb(args, start_epoch: int):
    if not args.wandb or args.rank != 0:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_id,
        name=args.wandb_name or args.run_id,
        resume="allow" if args.wandb_id else None,
        config=_jsonable_args(args),
    )
    wandb.define_metric("global_update")
    wandb.define_metric("train/*", step_metric="global_update")
    wandb.define_metric("train_eval/*", step_metric="global_update")
    wandb.define_metric("validation/*", step_metric="global_update")
    wandb.define_metric("time/*", step_metric="global_update")
    if args.wandb_backfill_log is not None and start_epoch:
        rows = _historical_wandb_rows(
            args.wandb_backfill_log, start_epoch, args.steps_per_epoch
        )
        wandb_step = run.step
        for global_update, payload in rows:
            if global_update >= wandb_step:
                run.log(payload, step=global_update)
        print(
            f"[rule-dist rank=0] wandb_backfilled={len(rows)} "
            f"through_epoch={start_epoch} run={run.url}",
            flush=True,
        )
    else:
        print(f"[rule-dist rank=0] wandb_run={run.url}", flush=True)
    return run


def _validate_args(args) -> None:
    if args.world_size < 2:
        raise ValueError("distributed rule training requires at least two ranks")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, world-size)")
    if min(args.epochs, args.steps_per_epoch, args.batch_size,
           args.gradient_accumulation_steps) < 1:
        raise ValueError("epochs, steps and batch sizes must be positive")
    if args.length_bucket_multiplier < 1:
        raise ValueError("--length-bucket-multiplier must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    if args.validation_samples < 1 or args.validate_every < 1:
        raise ValueError("validation sizes and intervals must be positive")
    if not 1 <= args.ring_port <= 65535:
        raise ValueError("--ring-port must be in [1, 65535]")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    peers = sorted((parse_peer(value) for value in args.peer), key=lambda p: p.rank)
    expected = set(range(args.world_size)) - {args.rank}
    if {peer.rank for peer in peers} != expected:
        raise ValueError(f"peers must be ranks {sorted(expected)}")

    effect_weight = args.effect_number_weight or args.critical_number_weight
    next_weight = args.next_number_weight or args.critical_number_weight
    calculation_weight = args.calculation_number_weight or effect_weight
    step_weight = args.step_number_weight
    causal_weight = args.causal_token_weight
    counter_weight = args.counter_token_weight
    structure_weight = args.structure_token_weight
    if min(
        effect_weight, next_weight, calculation_weight, step_weight,
        causal_weight, counter_weight,
        structure_weight,
    ) < 1.0:
        raise ValueError("number weights must be at least one")
    if args.consistency_loss_weight < 0.0 or args.consistency_scale <= 0.0:
        raise ValueError("invalid consistency loss settings")

    samples = list(read_jsonl(args.data))
    if len(samples) < 2:
        raise ValueError("rule training requires at least two samples per rank")
    train_samples, val_samples, split_unit = _split_samples(
        samples, args.validation_fraction, args.seed
    )
    val_samples = val_samples[:args.validation_samples]
    tokenizer_version = int(json.loads(
        args.tokenizer.read_text(encoding="utf-8")
    ).get("version", -1))
    if tokenizer_version != RULE_DATA_VERSION:
        raise ValueError(
            f"distributed rule training requires a v{RULE_DATA_VERSION} "
            "tokenizer built from all shards with "
            "python -m rl.rules.data.build_rule_tokenizer"
        )
    tokenizer = RuleTokenizer.load(args.tokenizer)
    unknowns = 0
    max_prefix = max_target = 0
    lengths_by_identity = {}
    for sample in samples:
        prefix, target = sample_tokens(sample)
        lengths_by_identity[id(sample)] = len(prefix) + len(target)
        max_prefix = max(max_prefix, len(prefix))
        max_target = max(max_target, len(target))
        unknowns += sum(token == tokenizer.index["<UNK>"] for token in tokenizer.encode(prefix))
        unknowns += sum(token == tokenizer.index["<UNK>"] for token in tokenizer.encode(target))
    if unknowns:
        raise ValueError(f"shared tokenizer produced {unknowns} unknown tokens")
    if max_prefix > args.max_prefix_tokens or max_target > args.max_target_tokens:
        raise ValueError(
            f"rank {args.rank} sequence capacity exceeded: "
            f"prefix={max_prefix}/{args.max_prefix_tokens}, "
            f"target={max_target}/{args.max_target_tokens}"
        )

    config = RuleModelConfig(
        vocab_size=len(tokenizer.tokens), pad_id=tokenizer.pad_id,
        d_model=args.d_model, n_heads=args.n_heads,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        feedforward=args.feedforward, dropout=args.dropout,
        max_prefix_tokens=args.max_prefix_tokens,
        max_target_tokens=args.max_target_tokens,
        num_begin_id=tokenizer.index.get("NUM_BEGIN"),
        num_lsd_begin_id=tokenizer.index.get("NUM_LSD_BEGIN"),
        num_end_id=tokenizer.index.get("NUM_END"),
        digit_token_ids=tuple(tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)),
    )
    model = RuleWorldModel(config)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_loss = math.inf
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.override_optimizer_hyperparameters_on_resume:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
                parameter_group["weight_decay"] = args.weight_decay
        start_epoch = int(checkpoint["epoch"])
        if not args.reset_best_on_resume:
            best_loss = float(checkpoint.get("metrics", {}).get("loss", math.inf))

    collate = partial(collate_rule_batch, pad_id=tokenizer.pad_id)
    micro_batches = args.steps_per_epoch * args.gradient_accumulation_steps
    train_dataset = RuleDataset(
        train_samples, tokenizer, include_critical_mask=True
    )
    batch_sampler = CardBalancedBatchSampler(
        train_samples,
        [lengths_by_identity[id(sample)] for sample in train_samples],
        batch_size=args.batch_size,
        num_batches=micro_batches,
        bucket_multiplier=args.length_bucket_multiplier,
        seed=args.seed + args.rank * 1_000_003,
    )
    if start_epoch:
        batch_sampler.skip_batches(start_epoch * micro_batches)
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        RuleDataset(val_samples, tokenizer, include_critical_mask=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Keep the train/validation comparison on the same fixed-size, full
    # teacher-forced evaluation path. The optimizer still trains on the
    # replacement-sampled loader above; this loader is only for diagnostics.
    train_eval_samples = train_samples[:len(val_samples)]
    train_eval_loader = DataLoader(
        RuleDataset(train_eval_samples, tokenizer, include_critical_mask=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    digit_ids = torch.tensor(
        [tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)], device=device
    )

    local_root = Path(args.remote_root) / args.run_id / f"rank{args.rank}"
    local_root.mkdir(parents=True, exist_ok=True)
    key = Path(args.ssh_key).expanduser()
    control_tag = hashlib.sha1(args.run_id.encode("utf-8")).hexdigest()[:8]
    control_dir = Path(args.ssh_control_dir) / control_tag / f"r{args.rank}"
    control_dir.mkdir(parents=True, exist_ok=True)
    for peer in peers:
        ssh_cmd(
            peer, key, control_dir,
            f"mkdir -p {shlex.quote(str(Path(args.remote_root) / args.run_id / f'rank{args.rank}'))}",
        )

    next_rank = (args.rank + 1) % args.world_size
    next_peer = next(peer for peer in peers if peer.rank == next_rank)

    def ready_barrier() -> None:
        all_rank_file_barrier(
            tag="rule_ring_listening", rank=args.rank, peers=peers, key=key,
            control_dir=control_dir, local_root=local_root,
            remote_root=args.remote_root, run_id=args.run_id,
            timeout=args.barrier_timeout,
        )

    ring = SshRingAllReduce(
        rank=args.rank, world_size=args.world_size,
        next_host=next_peer.host, next_port=next_peer.port,
        next_user=next_peer.user, ssh_key=key, listen_port=args.ring_port,
        timeout=args.barrier_timeout, ready_barrier=ready_barrier,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        tokenizer.save(args.out / "tokenizer.json")
        model.save_config(args.out / "model_config.json")
        (args.out / "run_config.json").write_text(
            json.dumps({
                **_jsonable_args(args), "effect_number_weight": effect_weight,
                "next_number_weight": next_weight,
                "calculation_number_weight": calculation_weight,
                "step_number_weight": step_weight,
                "causal_token_weight": causal_weight,
                "counter_token_weight": counter_weight,
            }, default=str, indent=2), encoding="utf-8",
        )

    local_summary = torch.tensor(
        [len(samples), len(train_samples), len(val_samples), max_prefix, max_target],
        dtype=torch.float32,
    )
    global_summary, _ = ring.average(local_summary)
    global_summary.mul_(args.world_size)
    print(
        f"[rule-dist rank={args.rank}] connected next={next_rank} "
        f"local={len(samples)}/{len(train_samples)}/{len(val_samples)} "
        f"global_samples={int(global_summary[0])} split={split_unit} "
        f"vocab={len(tokenizer.tokens)} lengths={max_prefix}/{max_target} "
        f"balanced_action_card_groups={len(batch_sampler.group_names)}",
        flush=True,
    )
    autocast = (
        partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16)
        if args.amp_mode == "bf16" else nullcontext
    )
    started = time.monotonic()
    wandb_run = None
    try:
        wandb_run = _init_wandb(args, start_epoch)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            sync_seconds = 0.0
            epoch_started = time.monotonic()
            for micro_step, (prefix, target, critical, specs) in enumerate(train_loader, 1):
                prefix = prefix.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                critical = critical.to(device, non_blocking=True)
                with autocast():
                    logits, _latent = model(prefix, target[:, :-1])
                    token_loss, _counts = _loss_and_counts(
                        logits, target[:, 1:], tokenizer.pad_id,
                        tokenizer.index[NEXT_STATE_BEGIN], critical[:, 1:],
                        effect_weight, next_weight, calculation_weight, step_weight,
                        causal_weight, counter_weight,
                        structure_token_weight=structure_weight,
                    )
                    consistency_loss, _terms = _transition_consistency_loss(
                        logits, specs, digit_ids, args.consistency_scale
                    )
                    loss = token_loss + args.consistency_loss_weight * consistency_loss
                (loss / args.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach())
                if micro_step % args.gradient_accumulation_steps:
                    continue

                sync_started = time.monotonic()
                flat = _pack_gradients(model)
                averaged, bytes_sent = ring.average(flat)
                _install_gradients(model, averaged)
                sync_seconds += time.monotonic() - sync_started
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update = micro_step // args.gradient_accumulation_steps
                if args.log_every and update % args.log_every == 0:
                    print(
                        f"[rule-dist rank={args.rank}] epoch={epoch} "
                        f"update={update}/{args.steps_per_epoch} "
                        f"loss={running_loss / micro_step:.4f} "
                        f"sync={sync_seconds:.1f}s grad_mb={bytes_sent / 1048576:.1f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        global_update = (epoch - 1) * args.steps_per_epoch + update
                        wandb_run.log({
                            "epoch": epoch,
                            "epoch_update": update,
                            "global_update": global_update,
                            "train/online_loss": running_loss / micro_step,
                            "train/sync_seconds": sync_seconds,
                            "train/gradient_megabytes": bytes_sent / 1048576,
                        }, step=global_update)

            if epoch % args.validate_every:
                continue
            metrics = _evaluate(
                model, val_loader, device, tokenizer,
                effect_weight, next_weight, calculation_weight,
                args.consistency_loss_weight, args.consistency_scale, step_weight,
                causal_weight, counter_weight,
                structure_token_weight=structure_weight,
            )
            metrics = _average_metrics(ring, metrics)
            train_eval_metrics = _evaluate(
                model, train_eval_loader, device, tokenizer,
                effect_weight, next_weight, calculation_weight,
                args.consistency_loss_weight, args.consistency_scale, step_weight,
                causal_weight, counter_weight,
                structure_weight,
            )
            train_eval_metrics = _average_metrics(ring, train_eval_metrics)
            elapsed = time.monotonic() - started
            print(
                f"[rule-dist rank={args.rank}] epoch={epoch} "
                f"train_eval_loss={train_eval_metrics['loss']:.6f} "
                f"val_loss={metrics['loss']:.6f} "
                f"effects={metrics['effect_accuracy']:.3%} "
                f"next={metrics['next_state_accuracy']:.3%} "
                f"calc={metrics['calculation_accuracy']:.3%} "
                f"calc_num={metrics['calculation_number_accuracy']:.3%} "
                f"card={metrics['card_name_accuracy']:.3%} "
                f"counter={metrics['counter_accuracy']:.3%} "
                f"epoch_sec={time.monotonic() - epoch_started:.1f} "
                f"sync_sec={sync_seconds:.1f} elapsed={elapsed:.1f}",
                flush=True,
            )
            if args.rank == 0:
                if wandb_run is not None:
                    global_update = epoch * args.steps_per_epoch
                    wandb_run.log({
                        "epoch": epoch,
                        "epoch_update": args.steps_per_epoch,
                        "global_update": global_update,
                        "train_eval/loss": train_eval_metrics["loss"],
                        "train_eval/effect_token_accuracy": train_eval_metrics[
                            "effect_accuracy"
                        ],
                        "train_eval/next_state_token_accuracy": train_eval_metrics[
                            "next_state_accuracy"
                        ],
                        "train_eval/calculation_token_accuracy": train_eval_metrics[
                            "calculation_accuracy"
                        ],
                        "train_eval/calculation_number_accuracy": train_eval_metrics[
                            "calculation_number_accuracy"
                        ],
                        "train_eval/card_name_accuracy": train_eval_metrics[
                            "card_name_accuracy"
                        ],
                        "train_eval/counter_accuracy": train_eval_metrics[
                            "counter_accuracy"
                        ],
                        "train_eval/structure_accuracy": train_eval_metrics[
                            "structure_accuracy"
                        ],
                        "validation/loss": metrics["loss"],
                        "validation/effect_token_accuracy": metrics["effect_accuracy"],
                        "validation/next_state_token_accuracy": metrics["next_state_accuracy"],
                        "validation/calculation_token_accuracy": metrics[
                            "calculation_accuracy"
                        ],
                        "validation/calculation_number_accuracy": metrics[
                            "calculation_number_accuracy"
                        ],
                        "validation/card_name_accuracy": metrics[
                            "card_name_accuracy"
                        ],
                        "validation/counter_accuracy": metrics["counter_accuracy"],
                        "validation/structure_accuracy": metrics[
                            "structure_accuracy"
                        ],
                        "time/epoch_seconds": time.monotonic() - epoch_started,
                        "time/sync_seconds": sync_seconds,
                        "time/elapsed_seconds": elapsed,
                    }, step=global_update)
                checkpoint_metrics = dict(metrics)
                checkpoint_metrics.update({
                    f"train_eval_{key}": value
                    for key, value in train_eval_metrics.items()
                })
                _save_checkpoint(
                    args.out / "last.pt", model, optimizer, epoch,
                    checkpoint_metrics, args,
                )
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    _save_checkpoint(
                        args.out / "best.pt", model, optimizer, epoch,
                        checkpoint_metrics, args,
                    )
        # Every optimizer update and validation pass already synchronizes the
        # model across ranks.  Avoid a second full-parameter collective here:
        # at shutdown, one rank can finish its final socket exchange before a
        # slower rank has consumed the last bytes, causing a false peer-close
        # error after all checkpoints have been written.
        print(
            f"[rule-dist rank={args.rank}] training_complete",
            flush=True,
        )
    finally:
        try:
            ring.close()
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()

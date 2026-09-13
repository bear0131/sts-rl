"""Train the standalone state+action -> effects+next-state rule model."""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl.rules.rule_data import (
    CALCULATIONS_BEGIN,
    CALCULATIONS_END,
    CANONICAL_CARD_TOKENS,
    CardBalancedBatchSampler,
    CRITICAL_CALCULATION,
    CRITICAL_CAUSAL,
    CRITICAL_COUNTER,
    CRITICAL_EFFECT,
    CRITICAL_NEXT_STATE,
    CRITICAL_STRUCTURE,
    CRITICAL_STEP,
    NEXT_STATE_BEGIN,
    OUTPUT_BEGIN,
    RuleDataset,
    RuleTokenizer,
    canonicalize_training_tokenizer,
    collate_rule_batch,
    read_jsonl,
    sample_tokens,
)
from rl.rules.rule_model import RuleModelConfig, RuleWorldModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("rl/runs/rules"))
    parser.add_argument(
        "--tokenizer", type=Path,
        help="Shared tokenizer to reuse across independently collected shards.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=0,
        help="Optimizer updates per epoch; zero traverses the full training set.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--length-bucket-multiplier", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-prefix-tokens", type=int, default=2048)
    parser.add_argument("--max-target-tokens", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--critical-number-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--resume", type=Path,
        help="Checkpoint whose model and optimizer state should be continued.",
    )
    parser.add_argument(
        "--reset-best-on-resume", action="store_true",
        help="Reset best-loss tracking when the resumed objective has changed.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sts-rl")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-id")
    parser.add_argument("--wandb-name")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loss_and_counts(
    logits, labels, pad_id, next_state_id,
    critical_regions=None, effect_number_weight=1.0, next_number_weight=1.0,
    calculation_number_weight=1.0, step_number_weight=1.0,
    causal_token_weight=1.0, counter_token_weight=1.0,
    calculation_begin_id=None, calculation_end_id=None,
    card_token_ids=None, structure_token_weight=1.0,
):
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(labels)
    valid = labels.ne(pad_id)
    weights = valid.to(logits.dtype)
    if critical_regions is not None:
        effect_critical = critical_regions.eq(CRITICAL_EFFECT) & valid
        next_critical = critical_regions.eq(CRITICAL_NEXT_STATE) & valid
        calculation_critical = critical_regions.eq(CRITICAL_CALCULATION) & valid
        step_critical = critical_regions.eq(CRITICAL_STEP) & valid
        causal_critical = critical_regions.eq(CRITICAL_CAUSAL) & valid
        counter_critical = critical_regions.eq(CRITICAL_COUNTER) & valid
        structure_critical = critical_regions.eq(CRITICAL_STRUCTURE) & valid
        weights = weights + effect_critical.to(logits.dtype) * (
            effect_number_weight - 1.0
        )
        weights = weights + next_critical.to(logits.dtype) * (
            next_number_weight - 1.0
        )
        weights = weights + calculation_critical.to(logits.dtype) * (
            calculation_number_weight - 1.0
        )
        weights = weights + step_critical.to(logits.dtype) * (
            step_number_weight - 1.0
        )
        weights = weights + causal_critical.to(logits.dtype) * (
            causal_token_weight - 1.0
        )
        weights = weights + counter_critical.to(logits.dtype) * (
            counter_token_weight - 1.0
        )
        weights = weights + structure_critical.to(logits.dtype) * (
            structure_token_weight - 1.0
        )
    else:
        effect_critical = torch.zeros_like(valid)
        next_critical = torch.zeros_like(valid)
        calculation_critical = torch.zeros_like(valid)
        step_critical = torch.zeros_like(valid)
        causal_critical = torch.zeros_like(valid)
        counter_critical = torch.zeros_like(valid)
        structure_critical = torch.zeros_like(valid)
    critical_mask = (
        effect_critical | next_critical | calculation_critical
        | step_critical | causal_critical | counter_critical
        | structure_critical
    )
    loss_weight = weights.sum().clamp_min(1.0)
    loss = (per_token_loss * weights).sum() / loss_weight
    predicted = logits.argmax(dim=-1)
    correct = predicted.eq(labels) & valid
    # The marker itself belongs to the event section; tokens after it are the
    # exact next-state reconstruction target.
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    marker = labels.eq(next_state_id)
    marker_position = torch.where(
        marker,
        positions,
        torch.full_like(positions, labels.shape[1]),
    ).min(dim=1).values
    next_mask = valid & (positions > marker_position.unsqueeze(1))
    event_mask = valid & ~next_mask
    calculation_mask = torch.zeros_like(valid)
    if calculation_begin_id is not None and calculation_end_id is not None:
        calculation_starts = labels.eq(calculation_begin_id)
        calculation_ends = labels.eq(calculation_end_id)
        calculation_depth = (
            calculation_starts.long().cumsum(dim=1)
            - calculation_ends.long().cumsum(dim=1)
        )
        calculation_mask = valid & (
            calculation_depth.gt(0) | calculation_ends
        )
    card_name_mask = torch.zeros_like(valid)
    if card_token_ids is not None and card_token_ids.numel():
        card_name_mask = valid & torch.isin(labels, card_token_ids)
    return loss, {
        "correct": int(correct.sum()), "tokens": int(valid.sum()),
        "event_correct": int((correct & event_mask).sum()),
        "event_tokens": int(event_mask.sum()),
        "next_correct": int((correct & next_mask).sum()),
        "next_tokens": int(next_mask.sum()),
        "critical_correct": int((correct & critical_mask).sum()),
        "critical_tokens": int(critical_mask.sum()),
        "effect_number_correct": int((correct & effect_critical).sum()),
        "effect_number_tokens": int(effect_critical.sum()),
        "next_number_correct": int((correct & next_critical).sum()),
        "next_number_tokens": int(next_critical.sum()),
        "calculation_number_correct": int((correct & calculation_critical).sum()),
        "calculation_number_tokens": int(calculation_critical.sum()),
        "calculation_correct": int((correct & calculation_mask).sum()),
        "calculation_tokens": int(calculation_mask.sum()),
        "card_name_correct": int((correct & card_name_mask).sum()),
        "card_name_tokens": int(card_name_mask.sum()),
        "step_number_correct": int((correct & step_critical).sum()),
        "step_number_tokens": int(step_critical.sum()),
        "causal_correct": int((correct & causal_critical).sum()),
        "causal_tokens": int(causal_critical.sum()),
        "counter_correct": int((correct & counter_critical).sum()),
        "counter_tokens": int(counter_critical.sum()),
        "structure_correct": int((correct & structure_critical).sum()),
        "structure_tokens": int(structure_critical.sum()),
        "loss_weight": float(loss_weight),
    }


def _transition_consistency_loss(logits, specs, digit_ids, scale):
    batch_indices = []
    token_positions = []
    equation_indices = []
    coefficients = []
    before_values = []
    for row, row_specs in enumerate(specs):
        for spec in row_specs:
            equation = len(before_values)
            before_values.append(float(spec["before"]))
            for relation_sign, span in (
                (1, spec["next"]),
                *((-1, delta) for delta in spec["deltas"]),
            ):
                positions = span["positions"]
                for offset, position in enumerate(positions):
                    batch_indices.append(row)
                    # Logit position i-1 predicts target token i.
                    token_positions.append(position - 1)
                    equation_indices.append(equation)
                    coefficients.append(
                        relation_sign
                        * int(span["sign"])
                        * 10 ** (
                            offset
                            if span.get("least_significant_first", False)
                            else len(positions) - offset - 1
                        )
                    )
    if not before_values:
        return logits.sum() * 0.0, 0
    batch_indices = torch.tensor(batch_indices, device=logits.device)
    token_positions = torch.tensor(token_positions, device=logits.device)
    equation_indices = torch.tensor(equation_indices, device=logits.device)
    coefficients = torch.tensor(
        coefficients, dtype=logits.dtype, device=logits.device
    )
    before_values = torch.tensor(
        before_values, dtype=logits.dtype, device=logits.device
    )
    digit_logits = logits[batch_indices, token_positions].index_select(1, digit_ids)
    probabilities = digit_logits.softmax(dim=-1)
    digit_values = torch.arange(10, dtype=logits.dtype, device=logits.device)
    expected_digits = probabilities @ digit_values
    predicted_relation = logits.new_zeros(len(before_values)).index_add(
        0, equation_indices, expected_digits * coefficients
    )
    residual = (predicted_relation - before_values) / scale
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    return loss, len(before_values)


def _evaluate(
    model, loader, device, tokenizer, effect_number_weight,
    next_number_weight, calculation_number_weight, consistency_loss_weight,
    consistency_scale, step_number_weight,
    causal_token_weight, counter_token_weight, structure_token_weight=1.0,
):
    model.eval()
    total_loss = 0.0
    total_consistency_loss = 0.0
    consistency_terms = 0
    totals = CounterLike()
    digit_ids = torch.tensor(
        [tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)],
        device=device,
    )
    card_identity_tokens = set(CANONICAL_CARD_TOKENS)
    card_identity_tokens.update(
        token for token in tokenizer.tokens if token.startswith("CARD_REF_")
    )
    card_token_ids = torch.tensor(
        sorted(
            tokenizer.index[token]
            for token in card_identity_tokens
            if token in tokenizer.index
        ),
        device=device,
    )
    with torch.no_grad():
        for prefix, target, critical, specs in loader:
            prefix, target = prefix.to(device), target.to(device)
            critical = critical.to(device)
            logits, _latent = model(prefix, target[:, :-1])
            loss, counts = _loss_and_counts(
                logits, target[:, 1:], tokenizer.pad_id,
                tokenizer.index[NEXT_STATE_BEGIN],
                critical[:, 1:], effect_number_weight, next_number_weight,
                calculation_number_weight, step_number_weight,
                causal_token_weight, counter_token_weight,
                tokenizer.index[CALCULATIONS_BEGIN],
                tokenizer.index[CALCULATIONS_END], card_token_ids,
                structure_token_weight,
            )
            consistency_loss, term_count = _transition_consistency_loss(
                logits, specs, digit_ids, consistency_scale
            )
            total_loss += float(loss) * counts["loss_weight"]
            total_consistency_loss += float(consistency_loss) * term_count
            consistency_terms += term_count
            totals.add(counts)
    metrics = _metrics(total_loss, totals)
    metrics["cross_entropy_loss"] = metrics["loss"]
    metrics["consistency_loss"] = total_consistency_loss / max(1, consistency_terms)
    metrics["loss"] += consistency_loss_weight * metrics["consistency_loss"]
    return metrics


class CounterLike(dict):
    def add(self, values):
        for key, value in values.items():
            self[key] = self.get(key, 0) + value


def _metrics(total_loss, counts):
    tokens = max(1, counts.get("tokens", 0))
    mean_loss = total_loss / max(1, counts.get("loss_weight", tokens))
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "token_accuracy": counts.get("correct", 0) / tokens,
        "effect_accuracy": counts.get("event_correct", 0) / max(1, counts.get("event_tokens", 0)),
        "next_state_accuracy": counts.get("next_correct", 0) / max(1, counts.get("next_tokens", 0)),
        "critical_number_accuracy": counts.get("critical_correct", 0) / max(
            1, counts.get("critical_tokens", 0)
        ),
        "effect_number_accuracy": counts.get("effect_number_correct", 0) / max(
            1, counts.get("effect_number_tokens", 0)
        ),
        "next_number_accuracy": counts.get("next_number_correct", 0) / max(
            1, counts.get("next_number_tokens", 0)
        ),
        "calculation_number_accuracy": counts.get(
            "calculation_number_correct", 0
        ) / max(1, counts.get("calculation_number_tokens", 0)),
        "calculation_accuracy": counts.get(
            "calculation_correct", 0
        ) / max(1, counts.get("calculation_tokens", 0)),
        "card_name_accuracy": counts.get("card_name_correct", 0) / max(
            1, counts.get("card_name_tokens", 0)
        ),
        "step_number_accuracy": counts.get("step_number_correct", 0) / max(
            1, counts.get("step_number_tokens", 0)
        ),
        "causal_accuracy": counts.get("causal_correct", 0) / max(
            1, counts.get("causal_tokens", 0)
        ),
        "counter_accuracy": counts.get("counter_correct", 0) / max(
            1, counts.get("counter_tokens", 0)
        ),
        "structure_accuracy": counts.get("structure_correct", 0) / max(
            1, counts.get("structure_tokens", 0)
        ),
    }


def _save_checkpoint(path, model, optimizer, epoch, metrics):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def _split_samples(samples, validation_fraction, seed):
    groups = {}
    for index, sample in enumerate(samples):
        metadata = sample.get("metadata", {})
        key = metadata.get("combat_index")
        groups.setdefault(key, []).append(index)

    # Keep complete combats together so validation does not see another action
    # enumerated from a state or loadout present in training. Tiny hand-written
    # datasets without multiple combat ids retain a sample-level fallback.
    if len(groups) >= 2 and None not in groups:
        keys = list(groups)
        random.Random(seed).shuffle(keys)
        val_group_count = min(
            len(keys) - 1,
            max(1, round(len(keys) * validation_fraction)),
        )
        val_keys = set(keys[:val_group_count])
        train = [
            sample for key in keys if key not in val_keys
            for sample in (samples[index] for index in groups[key])
        ]
        val = [
            sample for key in keys if key in val_keys
            for sample in (samples[index] for index in groups[key])
        ]
        target_val_samples = len(samples) * validation_fraction
        if abs(len(train) - target_val_samples) < abs(len(val) - target_val_samples):
            train, val = val, train
        return train, val, "combat"

    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    val_count = min(
        len(samples) - 1,
        max(1, round(len(samples) * validation_fraction)),
    )
    return (
        [samples[index] for index in order[val_count:]],
        [samples[index] for index in order[:val_count]],
        "sample",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.steps_per_epoch < 0 or args.gradient_accumulation_steps < 1:
        raise ValueError("steps must be non-negative and accumulation positive")
    if args.length_bucket_multiplier < 1:
        raise ValueError("--length-bucket-multiplier must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    effect_number_weight = (
        args.effect_number_weight
        if args.effect_number_weight is not None
        else args.critical_number_weight
    )
    next_number_weight = (
        args.next_number_weight
        if args.next_number_weight is not None
        else args.critical_number_weight
    )
    calculation_number_weight = (
        args.calculation_number_weight
        if args.calculation_number_weight is not None
        else effect_number_weight
    )
    step_number_weight = args.step_number_weight
    causal_token_weight = args.causal_token_weight
    counter_token_weight = args.counter_token_weight
    structure_token_weight = args.structure_token_weight
    if (
        effect_number_weight < 1.0
        or next_number_weight < 1.0
        or calculation_number_weight < 1.0
        or step_number_weight < 1.0
        or causal_token_weight < 1.0
        or counter_token_weight < 1.0
        or structure_token_weight < 1.0
    ):
        raise ValueError("number weights must be at least 1")
    if args.consistency_loss_weight < 0.0 or args.consistency_scale <= 0.0:
        raise ValueError("consistency weight must be non-negative and scale positive")
    args.effect_number_weight = effect_number_weight
    args.next_number_weight = next_number_weight
    args.calculation_number_weight = calculation_number_weight
    args.step_number_weight = step_number_weight
    args.causal_token_weight = causal_token_weight
    args.counter_token_weight = counter_token_weight
    args.structure_token_weight = structure_token_weight
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = list(read_jsonl(args.data))
    if len(samples) < 2:
        raise ValueError("rule training requires at least two samples")
    train_samples, val_samples, split_unit = _split_samples(
        samples, args.validation_fraction, args.seed
    )
    if args.validation_samples > 0:
        val_samples = val_samples[:args.validation_samples]
    tokenizer = RuleTokenizer.build(samples)
    if args.tokenizer is not None:
        tokenizer = canonicalize_training_tokenizer(
            RuleTokenizer.load(args.tokenizer), samples,
        )

    lengths = [tuple(map(len, sample_tokens(sample))) for sample in samples]
    max_prefix = max(item[0] for item in lengths)
    max_target = max(item[1] for item in lengths)
    if max_prefix > args.max_prefix_tokens or max_target > args.max_target_tokens:
        raise ValueError(
            "dataset exceeds configured sequence capacity: "
            f"prefix={max_prefix}/{args.max_prefix_tokens}, "
            f"target={max_target}/{args.max_target_tokens}"
        )
    unknowns = sum(
        token == tokenizer.index["<UNK>"]
        for sample in samples
        for part in sample_tokens(sample)
        for token in tokenizer.encode(part)
    )
    if unknowns:
        raise ValueError(f"tokenizer produced {unknowns} unknown tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "tokenizer.json")
    config = RuleModelConfig(
        vocab_size=len(tokenizer.tokens),
        pad_id=tokenizer.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward=args.feedforward,
        max_prefix_tokens=args.max_prefix_tokens,
        max_target_tokens=args.max_target_tokens,
        dropout=args.dropout,
        num_begin_id=tokenizer.index.get("NUM_BEGIN"),
        num_lsd_begin_id=tokenizer.index.get("NUM_LSD_BEGIN"),
        num_end_id=tokenizer.index.get("NUM_END"),
        digit_token_ids=tuple(
            tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)
        ),
    )
    model = RuleWorldModel(config)
    model.save_config(args.out / "model_config.json")
    (args.out / "run_config.json").write_text(
        json.dumps({**vars(args), "data": str(args.data), "out": str(args.out)}, default=str, indent=2),
        encoding="utf-8",
    )

    collate = partial(collate_rule_batch, pad_id=tokenizer.pad_id)
    train_dataset = RuleDataset(
        train_samples, tokenizer, include_critical_mask=True
    )
    train_lengths = [
        sum(map(len, sample_tokens(sample))) for sample in train_samples
    ]
    micro_batches = (
        args.steps_per_epoch * args.gradient_accumulation_steps
        if args.steps_per_epoch
        else math.ceil(len(train_samples) / args.batch_size)
    )
    batch_sampler = CardBalancedBatchSampler(
        train_samples, train_lengths, batch_size=args.batch_size,
        num_batches=micro_batches,
        bucket_multiplier=args.length_bucket_multiplier, seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        RuleDataset(val_samples, tokenizer, include_critical_mask=True), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate,
    )
    device = _device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        if not args.reset_best_on_resume:
            best_loss = float(checkpoint.get("metrics", {}).get("loss", math.inf))
        batch_sampler.skip_batches(start_epoch * micro_batches)
        if args.epochs <= start_epoch:
            raise ValueError(
                f"--epochs must exceed resumed epoch {start_epoch}"
            )
    digit_ids = torch.tensor(
        [tokenizer.index[f"DIGIT_{digit}"] for digit in range(10)],
        device=device,
    )
    print(
        f"[rules] samples={len(samples)} train={len(train_samples)} val={len(val_samples)} "
        f"split={split_unit} vocab={len(tokenizer.tokens)} "
        f"max_lengths={max_prefix}/{max_target} device={device} "
        f"steps_per_epoch={args.steps_per_epoch or len(train_loader)} "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"balanced_action_card_groups={len(batch_sampler.group_names)}",
        flush=True,
    )

    global_step = start_epoch * (args.steps_per_epoch or len(train_loader))
    started = time.monotonic()
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_id,
            name=args.wandb_name or args.out.name,
            resume="allow" if args.wandb_id else None,
            config={**vars(args), "data": str(args.data), "out": str(args.out)},
        )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        for micro_step, (prefix, target, critical, specs) in enumerate(
            train_loader, 1
        ):
            prefix, target = prefix.to(device), target.to(device)
            critical = critical.to(device)
            logits, _latent = model(prefix, target[:, :-1])
            token_loss, counts = _loss_and_counts(
                logits, target[:, 1:], tokenizer.pad_id,
                tokenizer.index[NEXT_STATE_BEGIN],
                critical[:, 1:], effect_number_weight, next_number_weight,
                calculation_number_weight, step_number_weight,
                causal_token_weight, counter_token_weight,
                structure_token_weight=structure_token_weight,
            )
            consistency_loss, _term_count = _transition_consistency_loss(
                logits, specs, digit_ids, args.consistency_scale
            )
            loss = token_loss + args.consistency_loss_weight * consistency_loss
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach()) * prefix.shape[0]
            running_tokens += prefix.shape[0]
            last_micro_batch = micro_step == len(train_loader)
            if (
                micro_step % args.gradient_accumulation_steps
                and not last_micro_batch
            ):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                train_loss = running_loss / max(1, running_tokens)
                print(
                    f"[rules] epoch={epoch} step={global_step} "
                    f"train_loss={train_loss:.4f}", flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "global_update": global_step,
                        "epoch": epoch,
                        "train/loss": train_loss,
                    }, step=global_step)
                running_loss = running_tokens = 0
        metrics = _evaluate(
            model, val_loader, device, tokenizer,
            effect_number_weight, next_number_weight, calculation_number_weight,
            args.consistency_loss_weight, args.consistency_scale, step_number_weight,
            causal_token_weight, counter_token_weight,
            structure_token_weight,
        )
        print(
            f"[rules] epoch={epoch} val_loss={metrics['loss']:.4f} "
            f"effects={metrics['effect_accuracy']:.3%} "
            f"next_state={metrics['next_state_accuracy']:.3%} "
            f"effect_num={metrics['effect_number_accuracy']:.3%} "
            f"next_num={metrics['next_number_accuracy']:.3%} "
            f"calc_num={metrics['calculation_number_accuracy']:.3%} "
            f"calc={metrics['calculation_accuracy']:.3%} "
            f"card={metrics['card_name_accuracy']:.3%} "
        f"counter={metrics['counter_accuracy']:.3%} "
        f"structure={metrics['structure_accuracy']:.3%} "
            f"consistency={metrics['consistency_loss']:.4f} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log({
                "global_update": global_step,
                "epoch": epoch,
                "validation/loss": metrics["loss"],
                "validation/effect_token_accuracy": metrics["effect_accuracy"],
                "validation/next_state_token_accuracy": metrics[
                    "next_state_accuracy"
                ],
                "validation/calculation_number_accuracy": metrics[
                    "calculation_number_accuracy"
                ],
                "validation/calculation_token_accuracy": metrics[
                    "calculation_accuracy"
                ],
                "validation/card_name_accuracy": metrics[
                    "card_name_accuracy"
                ],
                "validation/counter_accuracy": metrics["counter_accuracy"],
                "validation/structure_accuracy": metrics["structure_accuracy"],
                "validation/consistency_loss": metrics["consistency_loss"],
            }, step=global_step)
        _save_checkpoint(args.out / "last.pt", model, optimizer, epoch, metrics)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            _save_checkpoint(args.out / "best.pt", model, optimizer, epoch, metrics)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

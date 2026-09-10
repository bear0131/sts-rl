"""Evaluate a general combat-rule checkpoint with full autoregressive decoding."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import torch

from rl.rule_data import (
    CALCULATIONS_BEGIN,
    CALCULATIONS_END,
    EFFECTS_BEGIN,
    EFFECTS_END,
    NEXT_STATE_BEGIN,
    NEXT_STATE_END,
    OUTPUT_BEGIN,
    OUTPUT_END,
    RuleTokenizer,
    read_jsonl,
    sample_tokens,
)
from rl.rule_model import RuleWorldModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--greedy-samples", type=int, default=400)
    parser.add_argument("--greedy-max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-errors", type=int, default=10)
    parser.add_argument("--out", type=Path)
    return parser


def _pad(rows, pad_id: int, device: torch.device) -> torch.Tensor:
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=pad_id
    ).to(device)


def _trim(tokens: list[str]) -> list[str]:
    try:
        return tokens[:tokens.index(OUTPUT_END) + 1]
    except ValueError:
        return tokens


def _sections(tokens: list[str], begin: str, end: str) -> list[list[str]]:
    sections = []
    position = 0
    while position < len(tokens):
        try:
            left = tokens.index(begin, position)
            right = tokens.index(end, left) + 1
        except ValueError:
            break
        sections.append(tokens[left:right])
        position = right
    return sections


def _sample_tags(sample: dict) -> list[str]:
    action = sample["action"]
    tags = [f"action:{action['type']}"]
    if "card" in action:
        tags.append(f"card:{action['card']['id']}")
    player = sample["state"]["combat"]["player"]
    tags.append(f"player_block:{int(int(player.get('block', 0)) > 0)}")
    tags.append(f"player_powers:{int(bool(player.get('powers', {})))}")
    monsters = sample["state"]["combat"]["monsters"]
    tags.append(f"monster_block:{int(any(int(m.get('block', 0)) > 0 for m in monsters))}")
    tags.append(f"monster_powers:{int(any(bool(m.get('powers', {})) for m in monsters))}")
    tags.append(f"relic_count:{min(len(sample['state']['global']['relics']), 8)}")
    return tags


def main() -> None:
    args = build_parser().parse_args()
    samples = list(read_jsonl(args.data))
    if not samples:
        raise ValueError("evaluation data is empty")
    tokenizer_path = args.run / "tokenizer.json"
    output_version = int(json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    ).get("version", -1))
    tokenizer = RuleTokenizer.load(tokenizer_path)
    model = RuleWorldModel.from_config(args.run / "model_config.json")
    checkpoint = torch.load(
        args.run / args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    encoded = []
    unknowns = 0
    for sample in samples:
        prefix, target = sample_tokens(sample, output_version=output_version)
        prefix_ids = tokenizer.encode(prefix)
        target_ids = tokenizer.encode(target)
        unknown = tokenizer.index["<UNK>"]
        unknowns += sum(value == unknown for value in prefix_ids)
        unknowns += sum(value == unknown for value in target_ids)
        encoded.append((prefix_ids, target_ids, target))

    indices = list(range(len(samples)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:min(args.greedy_samples, len(indices))]
    totals = Counter()
    groups = defaultdict(Counter)
    errors = []
    sections = (
        ("calculations", CALCULATIONS_BEGIN, CALCULATIONS_END),
        ("effects", EFFECTS_BEGIN, EFFECTS_END),
        ("next_state", NEXT_STATE_BEGIN, NEXT_STATE_END),
    )
    with torch.no_grad():
        for begin in range(0, len(indices), args.batch_size):
            batch_indices = indices[begin:begin + args.batch_size]
            prefix = _pad(
                [encoded[index][0] for index in batch_indices],
                tokenizer.pad_id, device,
            )
            generated = model.generate(
                prefix,
                output_begin_id=tokenizer.index[OUTPUT_BEGIN],
                output_end_id=tokenizer.index[OUTPUT_END],
                max_new_tokens=args.greedy_max_tokens,
            ).cpu()
            for row, index in enumerate(batch_indices):
                predicted = _trim(tokenizer.decode(generated[row].tolist()))
                expected = encoded[index][2]
                checks = {
                    "complete": predicted == expected,
                    "terminated": bool(predicted and predicted[-1] == OUTPUT_END),
                }
                for name, left, right in sections:
                    checks[name] = _sections(predicted, left, right) == _sections(
                        expected, left, right
                    )
                totals["samples"] += 1
                for name, passed in checks.items():
                    totals[name] += int(passed)
                for tag in _sample_tags(samples[index]):
                    groups[tag]["samples"] += 1
                    for name, passed in checks.items():
                        groups[tag][name] += int(passed)
                if not checks["complete"] and len(errors) < args.show_errors:
                    mismatch = next(
                        (
                            position for position, (got, want) in enumerate(
                                zip(predicted, expected)
                            ) if got != want
                        ),
                        min(len(predicted), len(expected)),
                    )
                    errors.append({
                        "sample": index,
                        "action": samples[index]["action"],
                        "mismatch_token": mismatch,
                        "expected_length": len(expected),
                        "predicted_length": len(predicted),
                        "expected": expected[mismatch:mismatch + 12],
                        "predicted": predicted[mismatch:mismatch + 12],
                    })

    count = max(1, totals["samples"])
    metrics = {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "data_samples": len(samples),
        "greedy_samples": totals["samples"],
        "unknown_tokens": unknowns,
        **{
            f"greedy_{name}_accuracy": totals[name] / count
            for name in ("complete", "terminated", "calculations", "effects", "next_state")
        },
        "groups": {
            tag: {
                "samples": values["samples"],
                **{
                    name: values[name] / values["samples"]
                    for name in ("complete", "effects", "next_state")
                },
            }
            for tag, values in sorted(groups.items())
        },
        "errors": errors,
    }
    output = json.dumps(metrics, indent=2, sort_keys=True)
    print(output)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

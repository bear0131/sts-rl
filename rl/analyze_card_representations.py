"""Compare card representations averaged over random combat contexts.

The representation is the mean encoder hidden state over one card's
``ACTION_RULE`` span.  Anonymous ``CARD_REF_n`` identity tokens are excluded,
so the result is driven by the card surface and its surrounding context.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from env.core.card_language import card_program
from env.core.rule_language import RULE_CARD_IDS
from rl.rule_data import RuleTokenizer, read_jsonl, sample_tokens
from rl.rule_model import RuleWorldModel


def _effect_flags(card_id: str, upgraded: int) -> dict[str, bool]:
    program = tuple(card_program(card_id, bool(upgraded)))
    return {
        "damage": any(token.startswith("DEAL_DAMAGE") for token in program),
        "block": "GAIN_BLOCK" in program,
    }


def _rule_positions(tokens: Sequence[str]) -> list[int]:
    begin = tokens.index("ACTION_RULE_BEGIN") + 1
    end = tokens.index("ACTION_RULE_END", begin)
    return [
        position for position in range(begin, end)
        if not tokens[position].startswith("CARD_REF_")
        and tokens[position] != "WORD_CARD"
    ]


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def _encode_samples(
    model: RuleWorldModel, tokenizer: RuleTokenizer,
    samples: Sequence[dict[str, Any]], batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    vectors: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            encoded: list[torch.Tensor] = []
            positions: list[list[int]] = []
            for sample in batch:
                prefix, _target = sample_tokens(sample, output_version=5)
                encoded.append(torch.tensor(
                    tokenizer.encode(prefix), dtype=torch.long,
                ))
                positions.append(_rule_positions(prefix))
            prefix_ids = pad_sequence(
                encoded, batch_first=True, padding_value=tokenizer.pad_id,
            ).to(device)
            memory, _padding = model.encode(prefix_ids)
            for row, row_positions in enumerate(positions):
                if not row_positions:
                    raise ValueError("card rule span is empty")
                vectors.append(memory[row, row_positions].mean(dim=0).cpu())
    return vectors


def _centroid(vectors: dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor | None:
    if not keys:
        return None
    return _normalize(torch.stack([vectors[key] for key in keys]).mean(dim=0))


def _cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    return float(torch.dot(_normalize(left), _normalize(right)))


def _mean_cross_cosine(
    vectors: dict[str, torch.Tensor], left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> float | None:
    if not left_keys or not right_keys:
        return None
    values = [
        float(torch.dot(_normalize(vectors[left]), _normalize(vectors[right])))
        for left in left_keys for right in right_keys
    ]
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--samples-per-card", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_card < 1 or args.batch_size < 1:
        raise ValueError("sample and batch sizes must be positive")

    tokenizer = RuleTokenizer.load(args.run / "tokenizer.json")
    model = RuleWorldModel.from_config(args.run / "model_config.json")
    checkpoint = torch.load(
        args.run / args.checkpoint, map_location="cpu", weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)

    samples_by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.data:
        for sample in read_jsonl(path):
            card = sample.get("action", {}).get("card")
            card_id = card.get("id") if isinstance(card, dict) else None
            # Non-card actions are serialized with ``id=None``.  They do not
            # have a card rule span and must not become a synthetic card.
            if isinstance(card_id, str) and card_id in RULE_CARD_IDS:
                key = f"{card_id}@up{int(card.get('up', 0))}"
                samples_by_card[key].append(sample)

    rng = random.Random(args.seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for key, candidates in sorted(samples_by_card.items()):
        candidates = list(candidates)
        rng.shuffle(candidates)
        selected[key] = candidates[:args.samples_per_card]

    vectors: dict[str, torch.Tensor] = {}
    cards: dict[str, dict[str, Any]] = {}
    for key, samples in selected.items():
        if not samples:
            continue
        card = samples[0]["action"]["card"]
        flags = _effect_flags(str(card["id"]), int(card.get("up", 0)))
        sample_vectors = _encode_samples(
            model, tokenizer, samples, args.batch_size, device,
        )
        vectors[key] = _normalize(torch.stack(sample_vectors).mean(dim=0))
        cards[key] = {
            "card_id": str(card["id"]),
            "upgraded": int(card.get("up", 0)),
            "samples_available": len(samples_by_card[key]),
            "samples_used": len(samples),
            **flags,
        }

    damage = [key for key, info in cards.items() if info["damage"]]
    block = [key for key, info in cards.items() if info["block"]]
    damage_only = [key for key in damage if not cards[key]["block"]]
    block_only = [key for key in block if not cards[key]["damage"]]
    both = [key for key in cards if cards[key]["damage"] and cards[key]["block"]]
    neither = [key for key in cards if not cards[key]["damage"] and not cards[key]["block"]]

    damage_centroid = _centroid(vectors, damage)
    block_centroid = _centroid(vectors, block)
    damage_only_centroid = _centroid(vectors, damage_only)
    block_only_centroid = _centroid(vectors, block_only)
    result = {
        "checkpoint": str(args.run / args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dimension": int(next(iter(vectors.values())).numel()) if vectors else 0,
        "samples_per_card_limit": args.samples_per_card,
        "seed": args.seed,
        "cards": cards,
        "groups": {
            "damage": damage,
            "block": block,
            "damage_only": damage_only,
            "block_only": block_only,
            "both": both,
            "neither": neither,
        },
        "cosine": {
            "damage_vs_block_centroid": _cosine(damage_centroid, block_centroid),
            "damage_only_vs_block_only_centroid": _cosine(
                damage_only_centroid, block_only_centroid,
            ),
            "mean_damage_vs_block_card_cosine": _mean_cross_cosine(
                vectors, damage, block,
            ),
            "mean_damage_only_vs_block_only_card_cosine": _mean_cross_cosine(
                vectors, damage_only, block_only,
            ),
        },
        "vectors": {key: vector.tolist() for key, vector in vectors.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8",
    )
    print(
        f"epoch={result['checkpoint_epoch']} cards={len(cards)} "
        f"damage={len(damage)} block={len(block)} "
        f"damage_only_vs_block_only="
        f"{result['cosine']['damage_only_vs_block_only_centroid']}"
    )


if __name__ == "__main__":
    main()

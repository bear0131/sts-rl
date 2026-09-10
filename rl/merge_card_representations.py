"""Merge sampled card vectors and compare damage with block representations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Sequence


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return [value / norm for value in vector]


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(_normalize(left), _normalize(right)))


def _within_cosine(vectors: dict[str, list[float]], keys: Sequence[str]) -> float | None:
    values = [
        _cosine(vectors[left], vectors[right])
        for index, left in enumerate(keys)
        for right in keys[index + 1:]
    ]
    return sum(values) / len(values) if values else None


def _cross_cosine(
    vectors: dict[str, list[float]], left: Sequence[str], right: Sequence[str],
) -> float | None:
    values = [
        _cosine(vectors[left_key], vectors[right_key])
        for left_key in left for right_key in right
    ]
    return sum(values) / len(values) if values else None


def _centroid(vectors: dict[str, list[float]], keys: Sequence[str]) -> list[float]:
    return _normalize(_mean([vectors[key] for key in keys]))


def _nearest_centroid_accuracy(
    vectors: dict[str, list[float]], left: Sequence[str], right: Sequence[str],
) -> dict[str, float]:
    def classify(keys: Sequence[str], other: Sequence[str], expected_left: bool) -> float:
        correct = 0
        for key in keys:
            own = [candidate for candidate in keys if candidate != key]
            own_centroid = _centroid(vectors, own)
            other_centroid = _centroid(vectors, other)
            own_score = _cosine(vectors[key], own_centroid)
            other_score = _cosine(vectors[key], other_centroid)
            correct += int(own_score > other_score)
        return correct / len(keys)

    left_accuracy = classify(left, right, True)
    right_accuracy = classify(right, left, False)
    return {
        "damage_only": left_accuracy,
        "block_only": right_accuracy,
        "balanced": (left_accuracy + right_accuracy) / 2,
    }


def _cross_extremes(
    vectors: dict[str, list[float]], left: Sequence[str], right: Sequence[str],
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    pairs = sorted(
        (
            _cosine(vectors[left_key], vectors[right_key]),
            left_key,
            right_key,
        )
        for left_key in left for right_key in right
    )

    def records(rows: Sequence[tuple[float, str, str]]) -> list[dict[str, Any]]:
        return [
            {"cosine": cosine, "damage": damage, "block": block}
            for cosine, damage, block in rows
        ]

    return {
        "lowest": records(pairs[:limit]),
        "highest": records(list(reversed(pairs[-limit:]))),
    }


def _same_card_upgrade_cosine(vectors: dict[str, list[float]]) -> dict[str, Any]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for key in vectors:
        by_name[key.rsplit("@up", 1)[0]].append(key)
    rows = []
    for name, keys in sorted(by_name.items()):
        if len(keys) < 2:
            continue
        for index, left in enumerate(sorted(keys)):
            for right in sorted(keys)[index + 1:]:
                rows.append({
                    "card": name,
                    "left": left,
                    "right": right,
                    "cosine": _cosine(vectors[left], vectors[right]),
                })
    return {
        "mean": sum(row["cosine"] for row in rows) / len(rows) if rows else None,
        "count": len(rows),
        "lowest": sorted(rows, key=lambda row: row["cosine"])[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    epochs = {int(shard["checkpoint_epoch"]) for shard in shards}
    dimensions = {int(shard["dimension"]) for shard in shards}
    if len(epochs) != 1 or len(dimensions) != 1:
        raise ValueError(f"incompatible shards: epochs={epochs}, dimensions={dimensions}")

    all_keys = sorted({key for shard in shards for key in shard["vectors"]})
    vectors: dict[str, list[float]] = {}
    cards: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        weighted = [0.0] * next(iter(dimensions))
        used_total = 0
        available_total = 0
        card_info = None
        shards_used = 0
        for shard in shards:
            if key not in shard["vectors"]:
                continue
            info = shard["cards"][key]
            weight = int(info["samples_used"])
            for index, value in enumerate(shard["vectors"][key]):
                weighted[index] += weight * float(value)
            used_total += weight
            available_total += int(info["samples_available"])
            card_info = info
            shards_used += 1
        if card_info is None:
            continue
        vectors[key] = _normalize(weighted)
        cards[key] = {
            **card_info,
            "samples_available": available_total,
            "samples_used": used_total,
            "shards_used": shards_used,
        }

    damage = [key for key in all_keys if cards[key]["damage"]]
    block = [key for key in all_keys if cards[key]["block"]]
    damage_only = [key for key in damage if not cards[key]["block"]]
    block_only = [key for key in block if not cards[key]["damage"]]
    both = [key for key in all_keys if cards[key]["damage"] and cards[key]["block"]]
    neither = [
        key for key in all_keys
        if not cards[key]["damage"] and not cards[key]["block"]
    ]

    damage_centroid = _centroid(vectors, damage)
    block_centroid = _centroid(vectors, block)
    damage_only_centroid = _centroid(vectors, damage_only)
    block_only_centroid = _centroid(vectors, block_only)
    within_damage = _within_cosine(vectors, damage_only)
    within_block = _within_cosine(vectors, block_only)
    cross = _cross_cosine(vectors, damage_only, block_only)
    assert within_damage is not None and within_block is not None and cross is not None

    result = {
        "checkpoint_epoch": next(iter(epochs)),
        "dimension": next(iter(dimensions)),
        "merge": {
            "inputs": [str(path) for path in args.input],
            "method": "sample-count weighted mean of normalized shard card means",
        },
        "cards": cards,
        "groups": {
            "damage": damage,
            "block": block,
            "damage_only": damage_only,
            "block_only": block_only,
            "both": both,
            "neither": neither,
        },
        "comparison": {
            "damage_vs_block_centroid_cosine": _cosine(
                damage_centroid, block_centroid,
            ),
            "damage_only_vs_block_only_centroid_cosine": _cosine(
                damage_only_centroid, block_only_centroid,
            ),
            "damage_only_vs_block_only_centroid_euclidean": math.sqrt(
                max(0.0, 2 - 2 * _cosine(
                    damage_only_centroid, block_only_centroid,
                ))
            ),
            "mean_within_damage_only_cosine": within_damage,
            "mean_within_block_only_cosine": within_block,
            "mean_cross_damage_only_block_only_cosine": cross,
            "separation_gap": (within_damage + within_block) / 2 - cross,
            "nearest_centroid_leave_one_out_accuracy": (
                _nearest_centroid_accuracy(vectors, damage_only, block_only)
            ),
            "same_card_upgrade_cosine": _same_card_upgrade_cosine(vectors),
            "cross_group_extremes": _cross_extremes(
                vectors, damage_only, block_only,
            ),
        },
        "centroids": {
            "damage": damage_centroid,
            "block": block_centroid,
            "damage_only": damage_only_centroid,
            "block_only": block_only_centroid,
        },
        "vectors": vectors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8",
    )
    comparison = result["comparison"]
    accuracy = comparison["nearest_centroid_leave_one_out_accuracy"]
    print(
        f"epoch={result['checkpoint_epoch']} cards={len(cards)} "
        f"damage={len(damage)} block={len(block)} "
        f"centroid_cos={comparison['damage_only_vs_block_only_centroid_cosine']:.6f} "
        f"gap={comparison['separation_gap']:.6f} "
        f"balanced_accuracy={accuracy['balanced']:.3%}"
    )


if __name__ == "__main__":
    main()

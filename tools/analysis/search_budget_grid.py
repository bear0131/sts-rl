"""Analyze paired whole-game outcomes for fixed-budget combat search experiments."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_games(folder):
    """Keep one record per arm and verify that duplicates and game identities agree."""
    arms = {}
    actual = None
    weights = None
    for path in sorted(folder.glob("panel-*.json")):
        panel = json.loads(path.read_text(encoding="utf8"))
        ids = tuple(panel["actual_game_ids"])
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("duplicate game in a panel")
        if actual is not None and ids != actual:
            raise ValueError("panels did not play the same games")
        if weights is not None and (panel["weights"], panel["lower"]) != weights:
            raise ValueError("different frozen policies")
        actual, weights = ids, (panel["weights"], panel["lower"])
        for record in panel["arms"]:
            key = tuple(record["config"])
            results = np.asarray(record["terminals"], dtype=int)
            if (
                results.shape != (len(ids), 2)
                or not np.isin(results[:, 0], [0, 1]).all()
            ):
                raise ValueError("invalid terminal outcomes")
            if record["wins"] != results[:, 0].sum() or len(record["counts"]) != len(
                ids
            ):
                raise ValueError("invalid game accounting")
            if key in arms:
                old = arms[key]
                if any(
                    old[name] != record[name]
                    for name in ("terminals", "counts", "work")
                ):
                    raise ValueError(f"repeated control differs for {key}")
            arms[key] = dict(
                record,
                panel_seconds=panel["wall_seconds"],
                panel_arms=len(panel["arms"]),
                rank=panel["rank"],
            )
    if actual is None:
        raise ValueError("empty results folder")
    return actual, arms


def ranking(arms, keys):
    def order(key):
        rows = arms[key]
        return (
            -rows["wins"],
            -sum(floor for _, floor in rows["terminals"]),
            key[2],
            key[3],
            key[0],
        )

    return sorted(keys, key=order)


def compare(records, first, second, draws):
    a = np.asarray(records[first]["terminals"])
    b = np.asarray(records[second]["terminals"])
    delta = a - b
    return dict(
        a=first,
        b=second,
        n=len(a),
        a_wins_only=int((delta[:, 0] == 1).sum()),
        b_wins_only=int((delta[:, 0] == -1).sum()),
        win_difference=float(delta[:, 0].mean()),
        win_ci95=np.quantile(delta[:, 0][draws].mean(1), [0.025, 0.975]).tolist(),
        floor_difference=float(delta[:, 1].mean()),
        floor_ci95=np.quantile(delta[:, 1][draws].mean(1), [0.025, 0.975]).tolist(),
    )


def summarize(folder, output):
    ids, records = load_games(folder)
    indices = np.random.default_rng(2026110551).integers(
        len(ids), size=(10000, len(ids))
    )
    summary = dict(games=len(ids), arms=[], comparisons=[])
    for budget in sorted({x[1] for x in records}):
        ordered = ranking(records, [x for x in records if x[1] == budget])
        for key in ordered:
            row = records[key]
            summary["arms"].append(
                dict(
                    config=key,
                    wins=row["wins"],
                    mean_floor=float(np.mean([floor for _, floor in row["terminals"]])),
                    atomic_actions=sum(row["counts"]),
                    native_work=row["work"],
                    seconds=row["panel_seconds"] if row["panel_arms"] == 1 else None,
                    rank=row["rank"],
                )
            )
            if key != ordered[0]:
                summary["comparisons"].append(
                    compare(records, ordered[0], key, indices)
                )
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.folder, args.output)

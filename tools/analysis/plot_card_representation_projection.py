"""Plot a PCA projection of the merged card representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


GROUP_STYLE = {
    "damage_only": {"color": "#c0392b", "marker": "o", "label": "damage only"},
    "block_only": {"color": "#2471a3", "marker": "s", "label": "block only"},
    "both": {"color": "#8e44ad", "marker": "D", "label": "damage + block"},
    "neither": {"color": "#7f8c8d", "marker": ".", "label": "neither"},
}

DEFAULT_LABELS = {
    "Strike_R", "Defend_R", "Dropkick", "Twin Strike", "Immolate",
    "Combust", "Fire Breathing", "Shrug It Off", "Metallicize",
    "Feel No Pain", "Iron Wave", "Flame Barrier", "Barricade",
}


def _group_by_key(data: dict) -> dict[str, str]:
    groups: dict[str, str] = {}
    for group in ("damage_only", "block_only", "both", "neither"):
        for key in data["groups"][group]:
            groups[key] = group
    return groups


def _short_name(key: str) -> str:
    name, upgraded = key.rsplit("@up", 1)
    return f"{name} +{upgraded}" if upgraded != "0" else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--labels", nargs="*", default=sorted(DEFAULT_LABELS),
        help="card names to annotate; base names match all upgrade versions",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    keys = sorted(data["vectors"])
    matrix = torch.tensor([data["vectors"][key] for key in keys], dtype=torch.float64)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    explained = singular_values.square() / singular_values.square().sum().clamp_min(1e-12)
    groups = _group_by_key(data)

    figure, axis = plt.subplots(figsize=(13, 9), constrained_layout=True)
    for group, style in GROUP_STYLE.items():
        selected = [index for index, key in enumerate(keys) if groups.get(key) == group]
        if not selected:
            continue
        points = coordinates[selected]
        axis.scatter(
            points[:, 0].numpy(), points[:, 1].numpy(),
            s=42 if style["marker"] != "." else 22,
            alpha=0.82, c=style["color"], marker=style["marker"],
            edgecolors="white" if style["marker"] != "." else "none",
            linewidths=0.35, label=style["label"], zorder=3,
        )

    # Plot each group's projected centroid using the same vectors used in the
    # numerical report, so the visual guide is directly comparable.
    for group, style in GROUP_STYLE.items():
        selected = [index for index, key in enumerate(keys) if groups.get(key) == group]
        if not selected:
            continue
        centroid = coordinates[selected].mean(dim=0)
        axis.scatter(
            [float(centroid[0])], [float(centroid[1])], s=150,
            c=style["color"], marker="X", edgecolors="black", linewidths=0.8,
            zorder=5,
        )

    label_names = set(args.labels)
    for index, key in enumerate(keys):
        if key.rsplit("@up", 1)[0] not in label_names:
            continue
        x, y = coordinates[index].tolist()
        axis.annotate(
            _short_name(key), (x, y), xytext=(4, 4), textcoords="offset points",
            fontsize=8, color="#202020", zorder=6,
        )

    handles = [
        Line2D(
            [0], [0], marker=style["marker"], color="none",
            markerfacecolor=style["color"], markeredgecolor="white",
            markersize=8, label=style["label"],
        )
        for style in GROUP_STYLE.values()
    ]
    handles.append(
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor="#ffffff",
            markeredgecolor="black", markersize=9, label="group centroid",
        )
    )
    axis.legend(handles=handles, loc="best", frameon=True)
    axis.axhline(0, color="#d5d8dc", linewidth=0.7, zorder=0)
    axis.axvline(0, color="#d5d8dc", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PCA 1 ({float(explained[0]) * 100:.1f}% variance)")
    axis.set_ylabel(f"PCA 2 ({float(explained[1]) * 100:.1f}% variance)")
    axis.set_title(
        f"Card rule representations, epoch {data['checkpoint_epoch']} "
        f"({len(keys)} card versions)"
    )
    axis.grid(True, color="#ecf0f1", linewidth=0.7)
    figure.savefig(args.out, dpi=180, facecolor="white")
    plt.close(figure)
    print(
        f"saved={args.out} points={len(keys)} "
        f"explained=({float(explained[0]):.6f},{float(explained[1]):.6f})"
    )


if __name__ == "__main__":
    main()

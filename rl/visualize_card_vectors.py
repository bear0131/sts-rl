"""Visualize learned card vectors from an sts-rl MaskablePPO checkpoint.

This script supports three related vector spaces:

1. ``--source deck``
   Builds the complete per-card token used by the overworld model's flattened
   master deck: shared card semantics plus upgrade count, followed by the shared
   card projection.

2. ``--source card``
   Extracts the shared categorical/card-language representation before the
   common card projection.

3. ``--source hand``
   Builds the complete pre-Transformer hand token used at runtime: the shared
   projected card token plus printed cost, damage, block, and an unselected
   select-mask value through ``combat.encoder.hand_context``. Because
   the numeric annotation table currently covers Ironclad, this source plots
   the 75 annotated Ironclad cards only. ``--upgrades`` selects the card version.
   The role embedding is common to every hand card and therefore disappears
   when PCA centers the vectors. Slot-position and contextual Transformer output
   are intentionally excluded because they depend on hand slot and full state,
   not on card identity alone.

The vectors are projected with PCA implemented in NumPy, then written to an
interactive Plotly HTML file and a CSV file.

Example:
    uv run python -m rl.visualize_card_vectors \
        rl/runs/combat/20260804-215612/final.zip \
        --source hand --dim 2 \
        --output rl/runs/combat/20260804-215612/deck_vectors_2d.html
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from env.core.cards_vocab import CARD_VOCAB
from env.core.encode import (
    BLOCK_SCALE,
    DMG_SCALE,
    ENERGY_SCALE,
    UPGRADE_SCALE,
)
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project learned sts-rl card vectors to 2D or 3D with PCA."
    )
    parser.add_argument("checkpoint", type=Path, help="MaskablePPO .zip checkpoint.")
    parser.add_argument(
        "--source",
        choices=("deck", "card", "hand"),
        default="deck",
        help=(
            "deck: complete per-card flattened-deck tokens; "
            "card: shared card semantics before projection; "
            "hand: complete annotated Ironclad hand tokens before the Transformer."
        ),
    )
    parser.add_argument(
        "--upgrades",
        type=int,
        default=0,
        help=(
            "Upgrade count used by card tokens (default: 0). Use 1 for the "
            "normal upgraded variants; values above 1 affect Searing Blow."
        ),
    )
    parser.add_argument(
        "--variants",
        choices=("single", "both"),
        default="single",
        help=(
            "single: plot only --upgrades; both: plot base and +1 variants "
            "together (requires --source hand)."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="Projection dimensionality.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output HTML path. Default: card_vectors_<source>_<dim>d.html "
            "(hand also includes _u<upgrade count>)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: same stem as --output.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used while loading the checkpoint, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each vector before PCA; useful for comparing direction.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include vocabulary index 0 (EMPTY/padding).",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Optional case-insensitive regular expression selecting card names.",
    )
    parser.add_argument(
        "--labels",
        choices=("none", "top", "all"),
        default="none",
        help=(
            "Text labels drawn on the plot. Hover names are always available; "
            "'top' labels the largest-norm vectors."
        ),
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=30,
        help="Number of labels used by --labels top.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "zh"),
        default="en",
        help=(
            "Display language for labels, hover names, and CSV card_name. "
            "Chinese names are available for the annotated Ironclad cards."
        ),
    )
    return parser.parse_args()


def resolve_encoder(model: Any) -> Any:
    """Return the current specialized policy core.

    The historical function name is retained because the dashboard imports it.
    """
    policy = model.policy

    core = getattr(policy, "core", None)
    if core is None:
        raise RuntimeError(
            "Checkpoint policy has no current specialized '.core'."
        )
    if getattr(core, "shared", None) is None:
        raise RuntimeError("Checkpoint policy core has no shared tokenizer.")
    combat = getattr(core, "combat", None)
    if getattr(combat, "encoder", None) is None:
        raise RuntimeError("Checkpoint policy core has no combat encoder.")
    return core


def _semantic_card_vectors(
    embedding: torch.nn.Module,
    upgrades: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(embedding.parameters(), None)
    if parameter is not None:
        device = parameter.device
    else:
        buffer = next(embedding.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")
    ids = torch.arange(len(CARD_VOCAB), device=device)
    normalized = torch.full(
        (len(CARD_VOCAB),),
        upgrades / UPGRADE_SCALE,
        dtype=torch.float32,
        device=device,
    )
    if hasattr(embedding, "versioned_forward"):
        semantic = embedding.versioned_forward(ids, normalized)
    else:
        semantic = embedding(ids)
    return semantic, normalized


def _vectors_from_components(
    embedding: torch.nn.Module,
    card_projection: torch.nn.Linear,
    hand_context: torch.nn.Linear,
    source: str,
    upgrades: int,
) -> tuple[np.ndarray, str]:
    semantic, normalized = _semantic_card_vectors(embedding, upgrades)
    if source == "card":
        vectors = semantic
        description = (
            "shared card ID + versioned effect-language semantics"
            if hasattr(embedding, "versioned_forward")
            else "shared card ID embedding"
        )
    else:
        projected = card_projection(
            torch.cat([semantic, normalized.unsqueeze(-1)], dim=-1)
        )
        if source == "deck":
            vectors = projected
            description = (
                "shared.cards token used by each flattened master-deck slot "
                f"(upgrade count={upgrades})"
            )
        elif source == "hand":
            context = torch.zeros(
                (len(CARD_VOCAB), 4),
                dtype=projected.dtype,
                device=projected.device,
            )
            for index, card_id in enumerate(CARD_VOCAB):
                numbers = card_numbers(card_id, upgrades)
                if numbers is None:
                    continue
                cost, damage, block = numbers
                context[index] = torch.tensor(
                    [
                        cost / ENERGY_SCALE,
                        damage / DMG_SCALE,
                        block / BLOCK_SCALE,
                        0.0,
                    ],
                    dtype=projected.dtype,
                    device=projected.device,
                )
            vectors = projected + hand_context(context)
            description = _hand_description(upgrades)
        else:
            raise ValueError(f"Unknown vector source: {source!r}")

    vectors = vectors.detach().float().cpu().numpy()

    if vectors.ndim != 2:
        raise RuntimeError(f"Expected a 2D vector matrix, got shape {vectors.shape}.")

    if vectors.shape[0] != len(CARD_VOCAB):
        raise RuntimeError(
            "Card vocabulary/checkpoint mismatch: "
            f"checkpoint has {vectors.shape[0]} card rows, "
            f"current CARD_VOCAB has {len(CARD_VOCAB)} entries. "
            "Use the source-code revision that created the checkpoint."
        )

    return vectors, description


def extract_vectors(
    core: Any,
    source: str,
    upgrades: int = 0,
) -> tuple[np.ndarray, str]:
    """Return current runtime card vectors shaped [N_CARDS, vector_dim]."""
    try:
        shared = core.shared
        hand_context = core.combat.encoder.hand_context
        embedding = shared.card
        card_projection = shared.card_projection
    except AttributeError as exc:
        raise RuntimeError(
            "Checkpoint does not contain the current shared tokenizer and "
            "combat hand context."
        ) from exc
    return _vectors_from_components(
        embedding, card_projection, hand_context, source, upgrades
    )


def _hand_description(upgrades: int) -> str:
    return (
        "shared projected card token + combat hand cost/damage/block/select "
        f"context (upgrade count={upgrades}, select mask=0)"
    )


def extract_state_dict_vectors(
    policy_state: dict[str, torch.Tensor],
    source: str,
    upgrades: int = 0,
) -> tuple[np.ndarray, str]:
    """Extract current specialized-policy vectors from an SB3 state dict."""
    from rl.model import CardLanguageEmbedding

    prefix = "core."
    card_prefix = prefix + "shared.card."
    try:
        embedding_weight = policy_state[card_prefix + "weight"]
        projection_weight = policy_state[
            prefix + "shared.card_projection.weight"
        ]
        projection_bias = policy_state[
            prefix + "shared.card_projection.bias"
        ]
        hand_weight = policy_state[
            prefix + "combat.encoder.hand_context.weight"
        ]
    except KeyError as exc:
        raise RuntimeError(
            "Checkpoint is not a current combat/split specialized policy; "
            f"missing {exc.args[0]!r}."
        ) from exc

    embedding_dim = int(embedding_weight.shape[1])
    if card_prefix + "token_embedding.weight" in policy_state:
        embedding = CardLanguageEmbedding(len(CARD_VOCAB), embedding_dim)
    else:
        embedding = torch.nn.Embedding(
            len(CARD_VOCAB), embedding_dim, padding_idx=0
        )
    embedding_state = {
        name: policy_state[card_prefix + name].detach().cpu()
        for name in embedding.state_dict()
        if card_prefix + name in policy_state
    }
    try:
        embedding.load_state_dict(embedding_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shared card tokenizer does not match this checkout."
        ) from exc

    card_projection = torch.nn.Linear(
        int(projection_weight.shape[1]), int(projection_weight.shape[0])
    )
    card_projection.load_state_dict({
        "weight": projection_weight.detach().cpu(),
        "bias": projection_bias.detach().cpu(),
    })
    hand_context = torch.nn.Linear(
        int(hand_weight.shape[1]), int(hand_weight.shape[0]), bias=False
    )
    hand_context.load_state_dict({"weight": hand_weight.detach().cpu()})
    embedding.eval()
    card_projection.eval()
    hand_context.eval()
    with torch.no_grad():
        return _vectors_from_components(
            embedding, card_projection, hand_context, source, upgrades
        )


def select_cards(
    vectors: np.ndarray,
    include_empty: bool,
    match: str | None,
    allowed_names: set[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    indices = np.arange(len(CARD_VOCAB), dtype=np.int64)
    keep = np.ones(len(CARD_VOCAB), dtype=bool)

    if not include_empty:
        keep[0] = False

    if allowed_names is not None:
        if include_empty:
            allowed_names = allowed_names | {CARD_VOCAB[0]}
        keep &= np.array([name in allowed_names for name in CARD_VOCAB])

    if match:
        pattern = re.compile(match, flags=re.IGNORECASE)
        keep &= np.array([bool(pattern.search(name)) for name in CARD_VOCAB])

    indices = indices[keep]
    names = [CARD_VOCAB[i] for i in indices]
    selected = vectors[indices]

    if selected.shape[0] < 4:
        raise ValueError(
            f"Only {selected.shape[0]} cards remain after filtering; "
            "PCA visualization needs at least 4."
        )

    return selected, names, indices


def pca_project(
    vectors: np.ndarray,
    dim: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected coordinates, explained-ratio, and plotted vector norms."""
    x = vectors.astype(np.float64, copy=True)
    original_norms = np.linalg.norm(x, axis=1)

    if normalize:
        denom = np.maximum(original_norms[:, None], 1e-12)
        x /= denom

    x -= x.mean(axis=0, keepdims=True)

    # X = U S V^T. Coordinates on the leading PCs are X V.
    _u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:dim]
    coords = x @ components.T

    variances = singular_values**2
    total = variances.sum()
    if total <= 0:
        explained_ratio = np.zeros(dim, dtype=np.float64)
    else:
        explained_ratio = variances[:dim] / total

    return coords, explained_ratio, original_norms


def choose_text_labels(
    names: list[str],
    norms: np.ndarray,
    mode: str,
    label_top: int,
) -> list[str]:
    if mode == "all":
        return names
    if mode == "none":
        return [""] * len(names)

    n = min(max(label_top, 0), len(names))
    top = set(np.argsort(norms)[-n:].tolist())
    return [name if i in top else "" for i, name in enumerate(names)]


def save_csv(
    path: Path,
    card_ids: list[str],
    display_names: list[str],
    indices: np.ndarray,
    coords: np.ndarray,
    norms: np.ndarray,
    source: str,
    upgrade_counts: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["card_index", "card_id", "card_name", "source", "vector_norm"]
    if upgrade_counts is not None:
        fields += ["upgrade_count", "cost", "damage", "block"]
    fields += [f"pc{i + 1}" for i in range(coords.shape[1])]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if upgrade_counts is None:
            row_upgrades = [None] * len(card_ids)
        else:
            row_upgrades = upgrade_counts.tolist()
        for card_id, display_name, index, point, norm, upgrades in zip(
            card_ids, display_names, indices, coords, norms, row_upgrades
        ):
            row: dict[str, object] = {
                "card_index": int(index),
                "card_id": card_id,
                "card_name": display_name,
                "source": source,
                "vector_norm": float(norm),
            }
            if upgrades is not None:
                upgrades = int(upgrades)
                numbers = card_numbers(card_id, upgrades)
                if numbers is None:
                    cost, damage, block = "", "", ""
                else:
                    cost, damage, block = numbers
                row.update({
                    "upgrade_count": upgrades,
                    "cost": cost,
                    "damage": damage,
                    "block": block,
                })
            row.update({f"pc{i + 1}": float(v) for i, v in enumerate(point)})
            writer.writerow(row)


def _variant_connector_coords(
    card_ids: list[str],
    upgrade_counts: np.ndarray | None,
    coords: np.ndarray,
) -> list[list[float | None]] | None:
    """Return Plotly line coordinates linking each base/+1 card pair."""
    if upgrade_counts is None or not ({0, 1} <= set(upgrade_counts.tolist())):
        return None
    by_card: dict[str, dict[int, int]] = {}
    for i, (card_id, upgrades) in enumerate(zip(card_ids, upgrade_counts)):
        by_card.setdefault(card_id, {})[int(upgrades)] = i

    axes: list[list[float | None]] = [[] for _ in range(coords.shape[1])]
    for variants in by_card.values():
        if 0 not in variants or 1 not in variants:
            continue
        for axis in range(coords.shape[1]):
            axes[axis].extend([
                float(coords[variants[0], axis]),
                float(coords[variants[1], axis]),
                None,
            ])
    return axes


def save_plotly_html(
    path: Path,
    card_ids: list[str],
    display_names: list[str],
    indices: np.ndarray,
    coords: np.ndarray,
    norms: np.ndarray,
    labels: list[str],
    explained_ratio: np.ndarray,
    title: str,
    source_description: str,
    normalize: bool,
    upgrade_counts: np.ndarray | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for the interactive HTML output. Install it with:\n"
            "    uv pip install plotly"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    if upgrade_counts is None:
        customdata = np.column_stack([indices, norms])
        stats_hover = ""
        marker_symbols: str | list[str] = "circle"
    else:
        stats = np.array([
            card_numbers(card_id, int(upgrades)) or (np.nan, np.nan, np.nan)
            for card_id, upgrades in zip(card_ids, upgrade_counts)
        ], dtype=np.float64)
        customdata = np.column_stack([indices, norms, upgrade_counts, stats])
        stats_hover = (
            "Upgrade=%{customdata[2]:.0f}<br>"
            "Cost=%{customdata[3]:.0f}<br>"
            "Damage=%{customdata[4]:.0f}<br>"
            "Block=%{customdata[5]:.0f}<br>"
        )
        marker_symbols = ["diamond" if u > 0 else "circle" for u in upgrade_counts]
    mode = "markers+text" if any(labels) else "markers"
    marker = {
        "size": 7 if coords.shape[1] == 2 else 5,
        "color": norms,
        "colorscale": "Viridis",
        "showscale": True,
        "colorbar": {"title": "Vector norm"},
        "opacity": 0.82,
        "symbol": marker_symbols,
    }
    hover_common = (
        "<b>%{hovertext}</b><br>"
        "CARD_VOCAB index=%{customdata[0]:.0f}<br>"
        "Vector norm=%{customdata[1]:.5f}<br>"
        + stats_hover
    )
    hover_names = [
        name if name == card_id else f"{name} ({card_id})"
        for card_id, name in zip(card_ids, display_names)
    ]
    connector_coords = _variant_connector_coords(
        card_ids, upgrade_counts, coords
    )

    if coords.shape[1] == 2:
        trace = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode=mode,
            text=labels,
            textposition="top center",
            customdata=customdata,
            hovertext=hover_names,
            marker=marker,
            hovertemplate=(hover_common +
                "PC1=%{x:.5f}<br>"
                "PC2=%{y:.5f}<extra></extra>"
            ),
        )
        fig = go.Figure()
        if connector_coords is not None:
            fig.add_trace(go.Scatter(
                x=connector_coords[0],
                y=connector_coords[1],
                mode="lines",
                line={"color": "rgba(90,90,90,0.35)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ))
        fig.add_trace(trace)
        fig.update_xaxes(
            title=f"PC1 ({explained_ratio[0] * 100:.2f}% variance)",
            zeroline=True,
        )
        fig.update_yaxes(
            title=f"PC2 ({explained_ratio[1] * 100:.2f}% variance)",
            zeroline=True,
        )
    else:
        trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode=mode,
            text=labels,
            customdata=customdata,
            hovertext=hover_names,
            marker=marker,
            hovertemplate=(hover_common +
                "PC1=%{x:.5f}<br>"
                "PC2=%{y:.5f}<br>"
                "PC3=%{z:.5f}<extra></extra>"
            ),
        )
        fig = go.Figure()
        if connector_coords is not None:
            fig.add_trace(go.Scatter3d(
                x=connector_coords[0],
                y=connector_coords[1],
                z=connector_coords[2],
                mode="lines",
                line={"color": "rgba(90,90,90,0.35)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ))
        fig.add_trace(trace)
        fig.update_layout(
            scene={
                "xaxis_title": f"PC1 ({explained_ratio[0] * 100:.2f}%)",
                "yaxis_title": f"PC2 ({explained_ratio[1] * 100:.2f}%)",
                "zaxis_title": f"PC3 ({explained_ratio[2] * 100:.2f}%)",
            }
        )

    variance_text = ", ".join(
        f"PC{i + 1}={ratio * 100:.2f}%"
        for i, ratio in enumerate(explained_ratio)
    )
    fig.update_layout(
        title={
            "text": (
                f"{title}<br>"
                f"<sup>{source_description}; PCA variance: {variance_text}; "
                f"L2 normalized: {normalize}</sup>"
            )
        },
        template="plotly_white",
        hovermode="closest",
        margin={"l": 50, "r": 30, "t": 90, "b": 50},
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()

    if args.upgrades < 0:
        raise ValueError("--upgrades must be non-negative")
    if args.variants == "both" and args.source != "hand":
        raise ValueError("--variants both requires --source hand")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output = args.output
    if output is None:
        if args.variants == "both":
            variant = "_both"
        else:
            variant = f"_u{args.upgrades}" if args.source == "hand" else ""
        output = Path(f"card_vectors_{args.source}{variant}_{args.dim}d.html")
    if output.suffix.lower() != ".html":
        raise ValueError("--output must end in .html")

    csv_path = args.csv or output.with_suffix(".csv")

    print(f"[1/5] Loading checkpoint: {args.checkpoint}")
    _data, params, _pytorch_variables = load_from_zip_file(
        args.checkpoint,
        device=args.device,
    )
    if params is None:
        raise RuntimeError("Checkpoint contains no SB3 parameter dictionary.")
    try:
        policy_state = params["policy"]
    except KeyError as exc:
        raise RuntimeError("Checkpoint contains no SB3 policy state dict.") from exc

    print("[2/5] Reading Transformer parameters")

    upgrade_levels = [0, 1] if args.variants == "both" else [args.upgrades]
    print(
        f"[3/5] Extracting source={args.source!r} vectors "
        f"for upgrades={upgrade_levels}"
    )
    allowed_names = set(IRONCLAD_CARDS) if args.source == "hand" else None
    vector_chunks = []
    card_ids = []
    index_chunks = []
    upgrade_chunks = []
    source_description = ""
    for upgrades in upgrade_levels:
        all_vectors, source_description = extract_state_dict_vectors(
            policy_state,
            args.source,
            upgrades=upgrades,
        )
        selected, names, selected_indices = select_cards(
            all_vectors,
            include_empty=args.include_empty,
            match=args.match,
            allowed_names=allowed_names,
        )
        vector_chunks.append(selected)
        card_ids.extend(names)
        index_chunks.append(selected_indices)
        upgrade_chunks.append(np.full(len(names), upgrades, dtype=np.int64))

    vectors = np.concatenate(vector_chunks, axis=0)
    indices = np.concatenate(index_chunks)
    upgrade_counts = (
        np.concatenate(upgrade_chunks) if args.source == "hand" else None
    )
    if args.variants == "both":
        source_description = (
            "complete runtime hand base/+1 tokens; circles=base, "
            "diamonds=upgraded, lines connect each card pair"
        )

    if args.language == "zh":
        display_names = [
            IRONCLAD_CARDS[card_id].name_zh
            if card_id in IRONCLAD_CARDS else card_id
            for card_id in card_ids
        ]
    else:
        display_names = list(card_ids)
    if args.variants == "both" and upgrade_counts is not None:
        display_names = [
            f"{name}+" if upgrades == 1 else name
            for name, upgrades in zip(display_names, upgrade_counts)
        ]

    print(
        f"      selected {len(card_ids)} points "
        f"({len(set(card_ids))} cards); "
        f"vector dimension={vectors.shape[1]}"
    )

    print(f"[4/5] PCA projection to {args.dim}D")
    coords, explained_ratio, norms = pca_project(
        vectors,
        dim=args.dim,
        normalize=args.normalize,
    )
    labels = choose_text_labels(
        display_names,
        norms,
        mode=args.labels,
        label_top=args.label_top,
    )

    titles = {
        "deck": "Flattened master-deck card tokens",
        "card": "Shared card semantic vectors",
        "hand": (
            "Complete Ironclad hand tokens (base and upgraded)"
            if args.variants == "both"
            else f"Complete Ironclad hand tokens (+{args.upgrades})"
        ),
    }
    title = titles[args.source]
    save_csv(
        csv_path,
        card_ids,
        display_names,
        indices,
        coords,
        norms,
        source=args.source,
        upgrade_counts=upgrade_counts,
    )
    save_plotly_html(
        output,
        card_ids,
        display_names,
        indices,
        coords,
        norms,
        labels,
        explained_ratio,
        title,
        source_description,
        normalize=args.normalize,
        upgrade_counts=upgrade_counts,
    )

    print("[5/5] Done")
    print(f"      HTML: {output.resolve()}")
    print(f"      CSV:  {csv_path.resolve()}")
    print(
        "      Explained variance: "
        + ", ".join(
            f"PC{i + 1}={r * 100:.2f}%"
            for i, r in enumerate(explained_ratio)
        )
    )


if __name__ == "__main__":
    main()

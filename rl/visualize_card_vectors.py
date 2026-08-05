"""Visualize learned card vectors from an sts-rl MaskablePPO checkpoint.

This script supports two related vector spaces:

1. ``--source deck``
   Extracts columns of ``encoder.tok["pile"].weight``.  These are the learned
   per-card contribution vectors used when ``deck_counts`` / ``draw_counts`` /
   ``discard_counts`` / ``exhaust_counts`` are linearly projected into one pile
   token.  For a one-hot card count e_i, the contribution is W[:, i].

2. ``--source card``
   Extracts rows of ``encoder.shared["card"].weight``.  These are the ordinary
   categorical card embeddings used for hand/offer/select/shop card tokens.

The vectors are projected with PCA implemented in NumPy, then written to an
interactive Plotly HTML file and a CSV file.

Example:
    uv run python -m rl.visualize_card_vectors \
        rl/runs/combat/20260804-215612/final.zip \
        --source deck --dim 2 \
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
from sb3_contrib import MaskablePPO

from env.core.cards_vocab import CARD_VOCAB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project learned sts-rl card vectors to 2D or 3D with PCA."
    )
    parser.add_argument("checkpoint", type=Path, help="MaskablePPO .zip checkpoint.")
    parser.add_argument(
        "--source",
        choices=("deck", "card"),
        default="deck",
        help=(
            "deck: per-card columns of the deck/pile count projection; "
            "card: categorical card embeddings used by hand/offer tokens."
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
        help="Output HTML path. Default: card_vectors_<source>_<dim>d.html",
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
    return parser.parse_args()


def resolve_encoder(model: MaskablePPO) -> Any:
    """Return the StSTransformerExtractor from the custom pointer policy."""
    policy = model.policy

    core = getattr(policy, "core", None)
    if core is None:
        raise RuntimeError(
            "Checkpoint policy has no '.core'. It may be an older MLP checkpoint "
            "rather than StSPointerPolicy."
        )

    encoder = getattr(core, "encoder", None)
    if encoder is None:
        raise RuntimeError("Checkpoint policy.core has no '.encoder'.")

    return encoder


def extract_vectors(
    encoder: Any,
    source: str,
) -> tuple[np.ndarray, str]:
    """Return an array shaped [N_CARDS, vector_dim]."""
    if source == "deck":
        try:
            layer = encoder.tok["pile"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                'Could not find encoder.tok["pile"]; checkpoint architecture '
                "does not match the current Transformer model."
            ) from exc

        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(
                f'encoder.tok["pile"] is {type(layer).__name__}, expected nn.Linear.'
            )

        # nn.Linear computes y = x @ weight.T + bias.
        # A one-hot card-count input e_i contributes weight[:, i].
        vectors = layer.weight.detach().float().cpu().numpy().T
        description = 'columns of encoder.tok["pile"].weight (deck-count contributions)'
    else:
        try:
            embedding = encoder.shared["card"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                'Could not find encoder.shared["card"]; checkpoint architecture '
                "does not match the current Transformer model."
            ) from exc

        if not isinstance(embedding, torch.nn.Embedding):
            raise TypeError(
                f'encoder.shared["card"] is {type(embedding).__name__}, '
                "expected nn.Embedding."
            )

        vectors = embedding.weight.detach().float().cpu().numpy()
        description = 'rows of encoder.shared["card"].weight (card embeddings)'

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


def select_cards(
    vectors: np.ndarray,
    include_empty: bool,
    match: str | None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    indices = np.arange(len(CARD_VOCAB), dtype=np.int64)
    keep = np.ones(len(CARD_VOCAB), dtype=bool)

    if not include_empty:
        keep[0] = False

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
    names: list[str],
    indices: np.ndarray,
    coords: np.ndarray,
    norms: np.ndarray,
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["card_index", "card_name", "source", "vector_norm"]
    fields += [f"pc{i + 1}" for i in range(coords.shape[1])]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, index, point, norm in zip(names, indices, coords, norms):
            row: dict[str, object] = {
                "card_index": int(index),
                "card_name": name,
                "source": source,
                "vector_norm": float(norm),
            }
            row.update({f"pc{i + 1}": float(v) for i, v in enumerate(point)})
            writer.writerow(row)


def save_plotly_html(
    path: Path,
    names: list[str],
    indices: np.ndarray,
    coords: np.ndarray,
    norms: np.ndarray,
    labels: list[str],
    explained_ratio: np.ndarray,
    title: str,
    source_description: str,
    normalize: bool,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for the interactive HTML output. Install it with:\n"
            "    uv pip install plotly"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    customdata = np.column_stack([indices, norms])
    mode = "markers+text" if any(labels) else "markers"
    marker = {
        "size": 7 if coords.shape[1] == 2 else 5,
        "color": norms,
        "colorscale": "Viridis",
        "showscale": True,
        "colorbar": {"title": "Vector norm"},
        "opacity": 0.82,
    }

    if coords.shape[1] == 2:
        trace = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode=mode,
            text=labels,
            textposition="top center",
            customdata=customdata,
            hovertext=names,
            marker=marker,
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "CARD_VOCAB index=%{customdata[0]:.0f}<br>"
                "Vector norm=%{customdata[1]:.5f}<br>"
                "PC1=%{x:.5f}<br>"
                "PC2=%{y:.5f}<extra></extra>"
            ),
        )
        fig = go.Figure(trace)
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
            hovertext=names,
            marker=marker,
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "CARD_VOCAB index=%{customdata[0]:.0f}<br>"
                "Vector norm=%{customdata[1]:.5f}<br>"
                "PC1=%{x:.5f}<br>"
                "PC2=%{y:.5f}<br>"
                "PC3=%{z:.5f}<extra></extra>"
            ),
        )
        fig = go.Figure(trace)
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

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    output = args.output
    if output is None:
        output = Path(f"card_vectors_{args.source}_{args.dim}d.html")
    if output.suffix.lower() != ".html":
        raise ValueError("--output must end in .html")

    csv_path = args.csv or output.with_suffix(".csv")

    print(f"[1/5] Loading checkpoint: {args.checkpoint}")
    model = MaskablePPO.load(args.checkpoint, device=args.device)

    print("[2/5] Locating the Transformer encoder")
    encoder = resolve_encoder(model)

    print(f"[3/5] Extracting source={args.source!r} vectors")
    all_vectors, source_description = extract_vectors(encoder, args.source)
    vectors, names, indices = select_cards(
        all_vectors,
        include_empty=args.include_empty,
        match=args.match,
    )

    print(
        f"      selected {len(names)} cards; "
        f"vector dimension={vectors.shape[1]}"
    )

    print(f"[4/5] PCA projection to {args.dim}D")
    coords, explained_ratio, norms = pca_project(
        vectors,
        dim=args.dim,
        normalize=args.normalize,
    )
    labels = choose_text_labels(
        names,
        norms,
        mode=args.labels,
        label_top=args.label_top,
    )

    title = (
        "Deck-count card contribution vectors"
        if args.source == "deck"
        else "Shared categorical card embeddings"
    )
    save_csv(
        csv_path,
        names,
        indices,
        coords,
        norms,
        source=args.source,
    )
    save_plotly_html(
        output,
        names,
        indices,
        coords,
        norms,
        labels,
        explained_ratio,
        title,
        source_description,
        normalize=args.normalize,
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

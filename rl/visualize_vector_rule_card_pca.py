"""Project v6 anonymous Ironclad card-rule vectors to two PCA dimensions.

Card names are used only as labels in the output.  The vectors are obtained
from the canonical card rule programs through ``CardRuleEncoder.card_vectors``;
no card-id embedding or natural-language name is supplied to the model.

Example::

    .venv\\Scripts\\python.exe -m rl.visualize_vector_rule_card_pca \\
        --checkpoint .tmp-remote-v6-wide/last.pt \\
        --output-png docs/card-representations-v6-wide-step7500-pca.png \\
        --output-json docs/card-representations-v6-wide-step7500-pca.json \\
        --output-markdown docs/card-representations-v6-wide-step7500-pca.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional layout enhancement
    adjust_text = None

from env.core.cards_vocab import CARD_INDEX
from env.core.ironclad_cards import ATTACK, POWER, SKILL, IRONCLAD_CARDS
from rl.rule_vector_model import VectorRuleModelConfig, VectorRuleWorldModel


KIND_STYLE = {
    ATTACK: {"color": "#c0392b", "label": "攻击牌", "marker": "o"},
    SKILL: {"color": "#2471a3", "label": "技能牌", "marker": "s"},
    POWER: {"color": "#16805c", "label": "能力牌", "marker": "^"},
}

# Two Ironclad cards intentionally share the localized name "燃烧".  Keep the
# card name intact and add a Chinese effect hint so every plotted point is
# uniquely identifiable without exposing an English id in the main label.
LABEL_HINTS_ZH = {
    "Combust": "燃烧（失去生命）",
    "Inflame": "燃烧（获得力量）",
}


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    payload = checkpoint.get("config")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain a model config")
    config = VectorRuleModelConfig.from_dict(payload)
    model = VectorRuleWorldModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint, config


def _extract_vectors(model: VectorRuleWorldModel, device: torch.device):
    cards = []
    for card_id, info in IRONCLAD_CARDS.items():
        if info.kind not in KIND_STYLE:
            raise ValueError(f"unexpected card kind {info.kind!r} for {card_id}")
        cards.append({
            "card_id": card_id,
            "name_zh": info.name_zh,
            "label_zh": LABEL_HINTS_ZH.get(card_id, info.name_zh),
            "kind": info.kind,
            "vocab_index": int(CARD_INDEX[card_id]),
        })
    ids = torch.tensor(
        [row["vocab_index"] for row in cards], dtype=torch.long, device=device,
    )
    upgrades = torch.zeros_like(ids)
    with torch.no_grad():
        vectors = model.card_encoder.card_vectors(ids, upgrades).detach().cpu().numpy()
    if vectors.shape != (len(cards), model.card_encoder.config.output_dim):
        raise ValueError(f"unexpected card vector shape {vectors.shape}")
    return cards, vectors


def _pca(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = vectors.astype(np.float64) - vectors.mean(axis=0, keepdims=True)
    # SVD is used instead of sklearn so the artifact is reproducible with the
    # project's existing NumPy dependency alone.
    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered, full_matrices=False,
    )
    coordinates = centered @ right_singular_vectors[:2].T
    variance = np.square(singular_values)
    explained = variance / max(float(variance.sum()), 1e-12)
    return coordinates, explained[:2], centered


def _write_plot(
    path: Path, cards: list[dict[str, Any]], coordinates: np.ndarray,
    explained: np.ndarray, step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(22, 16), constrained_layout=True)
    texts = []
    for kind, style in KIND_STYLE.items():
        indices = [i for i, card in enumerate(cards) if card["kind"] == kind]
        if not indices:
            continue
        points = coordinates[indices]
        axis.scatter(
            points[:, 0], points[:, 1], s=70, c=style["color"],
            marker=style["marker"], alpha=0.9, edgecolors="white",
            linewidths=0.6, zorder=3,
        )
        for index in indices:
            texts.append(axis.text(
                coordinates[index, 0], coordinates[index, 1],
                cards[index]["label_zh"], fontsize=9, color="#202020",
                zorder=4,
            ))
    # Give every card a visible label while allowing matplotlib to resolve
    # local collisions.  The arrows preserve the point-to-label association.
    if texts and adjust_text is not None:
        adjust_text(
            texts, ax=axis, only_move={"text": "xy"},
            arrowprops={"arrowstyle": "-", "color": "#9aa0a6", "lw": 0.45},
            expand=(1.15, 1.25), force_text=(0.35, 0.45),
        )
    axis.axhline(0, color="#d5d8dc", linewidth=0.75, zorder=0)
    axis.axvline(0, color="#d5d8dc", linewidth=0.75, zorder=0)
    axis.grid(True, color="#ecf0f1", linewidth=0.65, zorder=0)
    handles = [
        Line2D(
            [0], [0], marker=style["marker"], color="none",
            markerfacecolor=style["color"], markeredgecolor="white",
            markersize=9, label=style["label"],
        )
        for style in KIND_STYLE.values()
    ]
    axis.legend(handles=handles, loc="best", frameon=True, fontsize=11)
    axis.set_xlabel(f"主成分 1（解释方差 {explained[0] * 100:.2f}%）", fontsize=12)
    axis.set_ylabel(f"主成分 2（解释方差 {explained[1] * 100:.2f}%）", fontsize=12)
    axis.set_title(
        f"v6 匿名卡牌规则向量 PCA（未升级战士卡牌，step {step}）\n"
        "卡牌名称仅作为图上标签，不参与向量计算",
        fontsize=15,
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _write_markdown(
    path: Path, checkpoint_path: Path, checkpoint: dict[str, Any],
    cards: list[dict[str, Any]], explained: np.ndarray,
) -> None:
    counts = {kind: sum(card["kind"] == kind for card in cards) for kind in KIND_STYLE}
    lines = [
        "# v6 卡牌规则向量 PCA",
        "",
        "本图使用 v6 wide checkpoint 的 `CardRuleEncoder` 提取匿名卡牌规则向量。",
        "模型输入是由 `card_program` 编译出的规则 token；卡牌名称只用于图中中文标注，",
        "没有作为 token 或 ID embedding 输入模型。样本为 75 张未升级战士卡牌。",
        "",
        f"- checkpoint：`{checkpoint_path}`",
        f"- 训练 step：`{int(checkpoint.get('step', -1))}`",
        f"- 卡牌向量维度：`{int(checkpoint['config']['card_encoder']['output_dim'])}`",
        f"- PCA 1 解释方差：`{explained[0] * 100:.4f}%`",
        f"- PCA 2 解释方差：`{explained[1] * 100:.4f}%`",
        f"- 两个主成分合计：`{(explained[0] + explained[1]) * 100:.4f}%`",
        "",
        "## 类别统计",
        "",
        f"攻击牌：{counts[ATTACK]} 张；技能牌：{counts[SKILL]} 张；能力牌：{counts[POWER]} 张。",
        "",
        "PCA 是无监督投影，坐标轴方向本身没有固定语义；类别颜色用于观察规则表示是否自然形成分组，",
        "不能单独作为模型泛化能力或卡牌强弱的结论。",
        "",
        "## 生成文件",
        "",
        "- `card-representations-v6-wide-step7500-pca.png`：带中文标签的二维图。",
        "- `card-representations-v6-wide-step7500-pca.json`：每张卡牌的坐标、类别、向量和 PCA 方差。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    device = torch.device(args.device)
    model, checkpoint, _config = _load_model(args.checkpoint, device)
    cards, vectors = _extract_vectors(model, device)
    coordinates, explained, centered = _pca(vectors)
    _write_plot(
        args.output_png, cards, coordinates,
        explained, int(checkpoint.get("step", -1)),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step", -1)),
        "schema_version": int(checkpoint.get("schema_version", -1)),
        "card_vector_dimension": int(vectors.shape[1]),
        "card_count": int(vectors.shape[0]),
        "upgraded": False,
        "pca": {
            "method": "centered SVD",
            "coordinates": coordinates.tolist(),
            "explained_variance_ratio": explained.tolist(),
            "center": vectors.mean(axis=0).tolist(),
        },
        "cards": [
            {
                **card,
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
                "vector": vectors[index].tolist(),
            }
            for index, card in enumerate(cards)
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    _write_markdown(args.output_markdown, args.checkpoint, checkpoint, cards, explained)
    print(
        f"saved_png={args.output_png} saved_json={args.output_json} "
        f"cards={len(cards)} dim={vectors.shape[1]} "
        f"explained=({explained[0]:.6f},{explained[1]:.6f})",
    )


if __name__ == "__main__":
    main()

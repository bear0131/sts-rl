"""Probe whether a checkpoint plays Bash before Strike against a Cultist."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.core import action_space as A
from env.core import encode
from env.core import _BUILD  # noqa: F401
import slaythespire as sts
from schemas.combat_state import normalize_sim, normalize_sim_global

# Register custom checkpoint classes before loading.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401


DEFAULT_DOWNLOADS = Path.home() / "Downloads"
STARTER_DECK = [
    *([("STRIKE_RED", 0)] * 5),
    *([("DEFEND_RED", 0)] * 4),
    ("BASH", 0),
]
HAND_CARDS = {
    "strike": {"id": "Strike_R", "up": 0, "cost": 1},
    "bash": {"id": "Bash", "up": 0, "cost": 2},
}


def latest_checkpoint(downloads: Path = DEFAULT_DOWNLOADS) -> Path:
    checkpoints = sorted(
        downloads.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no .zip checkpoints found in {downloads}")
    return checkpoints[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="Checkpoint path; default is the newest .zip in ~/Downloads.",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def make_observation(
    hand_order: tuple[str, str],
    energy: int,
) -> dict[str, np.ndarray]:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.floor_num = 1
    gc.set_deck(STARTER_DECK)
    bc = sts.create_battle(gc, "Cultist")
    canon = normalize_sim(bc.to_dict())
    canon["phase"] = "combat"
    canon["arg_step"] = A.ROOT
    canon["global"] = normalize_sim_global(gc)
    canon["player"] = {
        "hp": 80,
        "max_hp": 80,
        "block": 0,
        "energy": energy,
        "powers": {},
    }
    canon["hand"] = [dict(HAND_CARDS[name]) for name in hand_order]
    canon["draw"] = []
    canon["discard"] = (
        [{"id": "Strike_R", "up": 0}] * 4
        + [{"id": "Defend_R", "up": 0}] * 4
    )
    canon["exhaust"] = []
    canon["monsters"] = [
        {
            "id": "Cultist",
            "hp": 52,
            "max_hp": 52,
            "block": 0,
            "gone": False,
            "base_damage": 0,
            "hits": 0,
            "powers": {},
            "move": "CULTIST_INCANTATION",
            "move_id": 50,
        }
    ]
    return encode.encode(canon)


def probe_layout(
    model: MaskablePPO,
    hand_order: tuple[str, str],
    energy: int,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    observation = make_observation(hand_order, energy)
    play_base = A.seg_base("PLAY_CARD")
    actions = [A.seg_base("END_TURN"), play_base, play_base + 1]
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[actions] = True
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = model.policy.get_distribution(
            obs_tensor,
            action_masks=mask,
        )
        probabilities = distribution.distribution.probs[0].cpu().numpy()
        value = float(model.policy.predict_values(obs_tensor)[0].item())

    named_actions = {
        "end_turn": A.seg_base("END_TURN"),
        "strike": play_base + hand_order.index("strike"),
        "bash": play_base + hand_order.index("bash"),
    }
    choice_probabilities = np.asarray(
        [probabilities[named_actions[name]] for name in named_actions],
        dtype=np.float64,
    )
    choice_probabilities /= choice_probabilities.sum()
    counts = rng.multinomial(samples, choice_probabilities)
    deterministic = max(
        named_actions,
        key=lambda name: probabilities[named_actions[name]],
    )
    row: dict[str, object] = {
        "energy": energy,
        "hand_order": "/".join(hand_order),
        "value": value,
        "deterministic": deterministic,
    }
    for index, name in enumerate(named_actions):
        row[f"p_{name}"] = float(choice_probabilities[index])
        row[f"n_{name}"] = int(counts[index])
    return row


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if any(energy < 2 for energy in args.energy):
        raise ValueError("--energy must be at least 2 so Bash is playable")
    checkpoint = args.checkpoint or latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print(f"[cultist-bash-order] checkpoint={checkpoint}")
    print(
        "[cultist-bash-order] state=floor-1 Cultist 52/52 using "
        "Incantation; hand Strike+Bash"
    )
    model = MaskablePPO.load(checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    for energy in args.energy:
        for hand_order in (("strike", "bash"), ("bash", "strike")):
            row = probe_layout(model, hand_order, energy, args.samples, rng)
            rows.append(row)
            print(
                f"energy={energy} hand={row['hand_order']} "
                f"V={row['value']:+.4f} deterministic={row['deterministic']}"
            )
            for name in ("end_turn", "strike", "bash"):
                print(
                    f"  {name:10s} p={row['p_' + name]:7.3%} "
                    f"samples={row['n_' + name]:5d}/{args.samples}"
                )

    print("[cultist-bash-order] hand-order-averaged first action:")
    for energy in args.energy:
        subset = [row for row in rows if row["energy"] == energy]
        print(f"  energy={energy} ({len(subset) * args.samples} samples)")
        for name in ("end_turn", "strike", "bash"):
            probability = np.mean([float(row["p_" + name]) for row in subset])
            count = sum(int(row["n_" + name]) for row in subset)
            print(f"    {name:10s} p={probability:7.3%} samples={count}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[cultist-bash-order] CSV written to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()

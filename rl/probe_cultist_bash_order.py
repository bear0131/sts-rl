"""Probe whether a checkpoint plays Bash before another Ironclad Attack."""

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
from schemas.combat_state import real_to_sim_card_enum
from env.core.ironclad_cards import ATTACK, IRONCLAD_CARDS, X_COST

# Register custom checkpoint classes before loading.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401
from . import split_policy as _split_policy  # noqa: F401


DEFAULT_DOWNLOADS = Path.home() / "Downloads"
BASH_ID = "Bash"
ATTACK_CARD_IDS = tuple(
    card_id
    for card_id, info in IRONCLAD_CARDS.items()
    if info.kind == ATTACK and card_id != BASH_ID
)


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
    parser.add_argument(
        "--energy",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Energy values to test. Defaults to 2 and 3 for the legacy "
            "Strike probe, or 6 with --all-attacks so both cards can be played."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all-attacks",
        action="store_true",
        help="Compare Bash with every other unupgraded Ironclad Attack.",
    )
    selection.add_argument(
        "--attack",
        choices=ATTACK_CARD_IDS,
        default="Strike_R",
        help="One canonical Attack card id to compare with Bash.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary-csv", type=Path, default=None)
    return parser


def hand_card(card_id: str) -> dict[str, object]:
    info = IRONCLAD_CARDS[card_id]
    return {"id": card_id, "up": 0, "cost": info.cost[0]}


def make_observation(
    hand_order: tuple[str, str],
    energy: int,
) -> dict[str, np.ndarray]:
    attack_ids = [card_id for card_id in hand_order if card_id != BASH_ID]
    if len(attack_ids) != 1:
        raise ValueError("hand_order must contain Bash and exactly one other card")
    attack_id = attack_ids[0]
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.floor_num = 1
    gc.set_deck([
        (real_to_sim_card_enum(card_id), 0)
        for card_id in (BASH_ID, attack_id)
    ])
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
    canon["hand"] = [hand_card(card_id) for card_id in hand_order]
    canon["draw"] = []
    canon["discard"] = []
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
    attack_id: str,
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
        "attack": play_base + hand_order.index(attack_id),
        "bash": play_base + hand_order.index(BASH_ID),
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
        "attack_id": attack_id,
        "attack_name_zh": IRONCLAD_CARDS[attack_id].name_zh,
        "attack_cost": IRONCLAD_CARDS[attack_id].cost[0],
        "energy": energy,
        "hand_order": "/".join(hand_order),
        "value": value,
        "deterministic": deterministic,
    }
    for index, name in enumerate(named_actions):
        row[f"p_{name}"] = float(choice_probabilities[index])
        row[f"n_{name}"] = int(counts[index])
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    keys = sorted({(str(row["attack_id"]), int(row["energy"])) for row in rows})
    for attack_id, energy in keys:
        subset = [
            row for row in rows
            if row["attack_id"] == attack_id and row["energy"] == energy
        ]
        means = {
            name: float(np.mean([float(row[f"p_{name}"]) for row in subset]))
            for name in ("end_turn", "attack", "bash")
        }
        play_mass = means["attack"] + means["bash"]
        summaries.append({
            "attack_id": attack_id,
            "attack_name_zh": IRONCLAD_CARDS[attack_id].name_zh,
            "attack_cost": IRONCLAD_CARDS[attack_id].cost[0],
            "energy": energy,
            "layouts": len(subset),
            "p_end_turn": means["end_turn"],
            "p_attack": means["attack"],
            "p_bash": means["bash"],
            "p_bash_given_play": means["bash"] / play_mass if play_mass else 0.0,
            "first_play": "bash" if means["bash"] > means["attack"] else "attack",
            "deterministic_layouts": "/".join(
                str(row["deterministic"]) for row in subset
            ),
        })
    return summaries


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    energies = tuple(args.energy or ((6,) if args.all_attacks else (2, 3)))
    if any(energy < 2 for energy in energies):
        raise ValueError("--energy must be at least 2 so Bash is playable")
    attack_ids = ATTACK_CARD_IDS if args.all_attacks else (args.attack,)
    for attack_id in attack_ids:
        cost = IRONCLAD_CARDS[attack_id].cost[0]
        if cost != X_COST and any(energy < cost for energy in energies):
            raise ValueError(
                f"--energy must be at least {cost} so {attack_id} is playable"
            )
    checkpoint = args.checkpoint or latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print(f"[cultist-bash-order] checkpoint={checkpoint}")
    print(
        "[cultist-bash-order] state=floor-1 Cultist 52/52 using "
        "Incantation; enemy Vulnerable=0; hand Bash+one Attack"
    )
    model = MaskablePPO.load(checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    for attack_id in attack_ids:
        for energy in energies:
            for hand_order in (
                (attack_id, BASH_ID),
                (BASH_ID, attack_id),
            ):
                rows.append(probe_layout(
                    model,
                    attack_id,
                    hand_order,
                    energy,
                    args.samples,
                    rng,
                ))

    summaries = summarize(rows)
    print("[cultist-bash-order] hand-slot-averaged first action:")
    print("attack                 cost energy  end_turn    attack      bash  bash|play first")
    for row in summaries:
        print(
            f"{row['attack_id'][:22]:22s} {row['attack_cost']:>4} "
            f"{row['energy']:>6} {row['p_end_turn']:9.3%} "
            f"{row['p_attack']:9.3%} {row['p_bash']:9.3%} "
            f"{row['p_bash_given_play']:9.3%} {row['first_play']}"
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[cultist-bash-order] CSV written to {args.output_csv.resolve()}")
    if args.output_summary_csv is not None:
        args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_summary_csv.open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(
            "[cultist-bash-order] summary CSV written to "
            f"{args.output_summary_csv.resolve()}"
        )


if __name__ == "__main__":
    main()

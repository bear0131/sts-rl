"""Probe a checkpoint on a controlled two-Louse combat observation.

The hand contains one Strike and one Defend. Both Louses have 6 HP and are
otherwise identical; one intends to attack for 10 and the other intends to use
Grow. The probe covers both hand orders, both monster orders, and configurable
energy values. It reports exact masked-policy probabilities plus repeated
samples of the joint first decision (including Strike's target substep).
"""

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
from rl.policies import model as _model  # noqa: F401
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


DEFAULT_DOWNLOADS = Path.home() / "Downloads"
STARTER_DECK = (
    [("STRIKE_RED", 0)] * 5
    + [("DEFEND_RED", 0)] * 4
    + [("BASH", 0)]
)
HAND_CARDS = {
    "strike": {"id": "Strike_R", "up": 0, "cost": 1},
    "defend": {"id": "Defend_R", "up": 0, "cost": 1},
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
    parser.add_argument("--monster-hp", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def _template() -> dict:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
    gc.floor_num = 1
    gc.set_deck(STARTER_DECK)
    bc = sts.create_battle(gc, "Two Louse")
    canon = normalize_sim(bc.to_dict())
    canon["phase"] = "combat"
    canon["arg_step"] = A.ROOT
    canon["global"] = normalize_sim_global(gc)
    return canon


def _monster(intent: str, hp: int) -> dict:
    if intent == "attack":
        move, move_id, damage, hits = "RED_LOUSE_BITE", 111, 10, 1
    elif intent == "buff":
        move, move_id, damage, hits = "RED_LOUSE_GROW", 112, 0, 0
    else:
        raise ValueError(intent)
    return {
        "id": "FuzzyLouseDefensive",
        "hp": hp,
        "max_hp": 14,
        "block": 0,
        "gone": False,
        "base_damage": damage,
        "hits": hits,
        "powers": {"Curl Up": 4},
        "move": move,
        "move_id": move_id,
    }


def make_observation(
    hand_order: tuple[str, str],
    monster_order: tuple[str, str],
    energy: int,
    monster_hp: int = 6,
    *,
    arg_step: str = A.ROOT,
) -> dict[str, np.ndarray]:
    canon = _template()
    canon["arg_step"] = arg_step
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
        + [{"id": "Defend_R", "up": 0}] * 3
        + [{"id": "Bash", "up": 0}]
    )
    canon["exhaust"] = []
    canon["monsters"] = [
        _monster(intent, monster_hp) for intent in monster_order
    ]
    return encode.encode(canon)


def _probabilities(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    legal_actions: list[int],
) -> tuple[np.ndarray, float]:
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[legal_actions] = True
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = model.policy.get_distribution(
            obs_tensor,
            action_masks=mask,
        )
        probabilities = distribution.distribution.probs[0].cpu().numpy()
        value = float(model.policy.predict_values(obs_tensor)[0].item())
    if not np.isclose(probabilities.sum(), 1.0):
        raise RuntimeError(f"masked probabilities sum to {probabilities.sum()}")
    return probabilities, value


def probe_layout(
    model: MaskablePPO,
    hand_order: tuple[str, str],
    monster_order: tuple[str, str],
    energy: int,
    monster_hp: int,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    play_base = A.seg_base("PLAY_CARD")
    end_turn = A.seg_base("END_TURN")
    root_actions = [end_turn, play_base, play_base + 1]
    root_obs = make_observation(
        hand_order, monster_order, energy, monster_hp
    )
    root_probs, value = _probabilities(model, root_obs, root_actions)

    strike_slot = hand_order.index("strike")
    defend_slot = hand_order.index("defend")
    strike_action = play_base + strike_slot
    defend_action = play_base + defend_slot

    target_base = A.seg_base("PICK_TARGET")
    target_actions = [target_base, target_base + 1]
    target_obs = make_observation(
        hand_order,
        monster_order,
        energy,
        monster_hp,
        arg_step=A.PICK_TARGET,
    )
    target_probs, target_value = _probabilities(model, target_obs, target_actions)
    attacker_slot = monster_order.index("attack")
    buffer_slot = monster_order.index("buff")

    joint = np.asarray(
        [
            root_probs[end_turn],
            root_probs[defend_action],
            root_probs[strike_action] * target_probs[target_base + attacker_slot],
            root_probs[strike_action] * target_probs[target_base + buffer_slot],
        ],
        dtype=np.float64,
    )
    joint /= joint.sum()
    counts = rng.multinomial(samples, joint)
    labels = ("end_turn", "defend", "strike_attacker", "strike_buffer")
    deterministic_root = int(max(root_actions, key=lambda action: root_probs[action]))
    if deterministic_root == end_turn:
        deterministic = "end_turn"
    elif deterministic_root == defend_action:
        deterministic = "defend"
    else:
        deterministic_target = int(
            max(target_actions, key=lambda action: target_probs[action])
        )
        deterministic = (
            "strike_attacker"
            if deterministic_target == target_base + attacker_slot
            else "strike_buffer"
        )

    row: dict[str, object] = {
        "energy": energy,
        "hand_order": "/".join(hand_order),
        "monster_order": "/".join(monster_order),
        "value_root": value,
        "value_target": target_value,
        "deterministic": deterministic,
        "p_end_turn": float(joint[0]),
        "p_defend": float(joint[1]),
        "p_strike_attacker": float(joint[2]),
        "p_strike_buffer": float(joint[3]),
    }
    row.update({f"n_{label}": int(count) for label, count in zip(labels, counts)})
    return row


def _print_row(row: dict[str, object], samples: int) -> None:
    print(
        f"energy={row['energy']} hand={row['hand_order']} "
        f"monsters={row['monster_order']} V={row['value_root']:+.4f} "
        f"deterministic={row['deterministic']}"
    )
    for label in ("end_turn", "defend", "strike_attacker", "strike_buffer"):
        print(
            f"  {label:16s} p={row['p_' + label]:7.3%} "
            f"samples={row['n_' + label]:5d}/{samples}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if not 1 <= args.monster_hp <= 14:
        raise ValueError("--monster-hp must be between 1 and 14")
    if any(energy < 1 for energy in args.energy):
        raise ValueError("--energy values must be at least 1")
    checkpoint = args.checkpoint or latest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print(f"[two-louse] checkpoint={checkpoint}")
    print(
        f"[two-louse] state=starter deck; hand Strike+Defend; both Louses "
        f"{args.monster_hp}/14 HP, Curl Up 4; one attacks 10x1, one uses Grow"
    )
    model = MaskablePPO.load(checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    for energy in args.energy:
        for hand_order in (("strike", "defend"), ("defend", "strike")):
            for monster_order in (("attack", "buff"), ("buff", "attack")):
                row = probe_layout(
                    model,
                    hand_order,
                    monster_order,
                    energy,
                    args.monster_hp,
                    args.samples,
                    rng,
                )
                rows.append(row)
                _print_row(row, args.samples)

    print("[two-louse] permutation-averaged joint policy:")
    for energy in args.energy:
        subset = [row for row in rows if row["energy"] == energy]
        print(f"  energy={energy} ({len(subset) * args.samples} samples)")
        for label in ("end_turn", "defend", "strike_attacker", "strike_buffer"):
            probability = np.mean([float(row["p_" + label]) for row in subset])
            count = sum(int(row["n_" + label]) for row in subset)
            print(f"    {label:16s} p={probability:7.3%} samples={count}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[two-louse] CSV written to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()

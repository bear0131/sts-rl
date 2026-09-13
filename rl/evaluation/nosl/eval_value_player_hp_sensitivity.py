"""Measure value-head output as the player's current HP changes.

The fixture is one real combat observation.  The intervention updates every
encoded current-player-HP field while keeping max HP, combat state, cards,
enemy, energy, and all other fields fixed:

* ``player_scalars[0]`` = HP / 200
* ``player_scalars[5]`` = HP / max HP
* ``run_hp[0]`` = HP / 200
* ``run_hp[2]`` = HP / max HP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from rl.policies.checkpoint import custom_objects as _load_custom_objects
from rl.evaluation.nosl.scenarios import SCENARIOS, make_env

from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def _value(model, observation) -> float:
    tensor, _ = model.policy.obs_to_tensor(observation)
    core = getattr(model.policy, "core", None)
    if core is None or not hasattr(core, "actor_critic"):
        raise RuntimeError("checkpoint does not expose core.actor_critic")
    with torch.inference_mode():
        _logits, values = core.actor_critic(tensor)
    return float(values.detach().float().reshape(-1)[0].cpu().item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument(
        "--hps", type=int, nargs="+", default=None,
        help="player HP values; default spans 1..max HP",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    env = make_env(
        model, SCENARIOS["bash_plus_strike"], False, int(args.seed)
    )
    base = env._obs()
    original = {key: np.asarray(value).copy() for key, value in base.items()}
    max_hp = int(env.thick.bc.to_dict()["player"]["max_hp"])
    original_hp = int(env.thick.bc.to_dict()["player"]["hp"])
    hps = args.hps or [1, 10, 20, 40, 60, max_hp]
    if any(hp < 1 or hp > max_hp for hp in hps):
        raise ValueError(f"--hps must contain values in [1, {max_hp}]")

    rows = []
    for hp in hps:
        observation = {
            key: np.asarray(value).copy() for key, value in base.items()
        }
        observation["player_scalars"][0] = float(hp) / 200.0
        observation["player_scalars"][5] = float(hp) / float(max_hp)
        observation["run_hp"][0] = float(hp) / 200.0
        observation["run_hp"][2] = float(hp) / float(max_hp)
        changed = [
            key for key in observation
            if not np.array_equal(observation[key], original[key])
        ]
        expected_changed = (
            set() if int(hp) == original_hp else {"player_scalars", "run_hp"}
        )
        if set(changed) != expected_changed:
            raise AssertionError(
                f"HP intervention changed unexpected fields: {changed}"
            )
        rows.append({
            "player_hp": int(hp),
            "player_max_hp": max_hp,
            "value": _value(model, observation),
            "changed_fields": changed,
            "player_scalars": observation["player_scalars"].tolist(),
            "run_hp": observation["run_hp"].tolist(),
        })

    order = np.argsort([row["player_hp"] for row in rows])
    hp_sorted = np.asarray([rows[index]["player_hp"] for index in order])
    values_sorted = np.asarray(
        [rows[index]["value"] for index in order], dtype=np.float64
    )
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": int(args.seed),
        "scenario": "Cultist / Bash+ + Strike",
        "original_player_hp": original_hp,
        "player_max_hp": max_hp,
        "intervention": (
            "player_scalars[0,5] and run_hp[0,2] only"
        ),
        "all_other_observation_fields_identical": True,
        "rows": rows,
        "summary": {
            "value_at_min_hp": float(values_sorted[0]),
            "value_at_max_hp": float(values_sorted[-1]),
            "delta_min_minus_max": float(values_sorted[0] - values_sorted[-1]),
            "slope_per_hp": float(np.polyfit(hp_sorted, values_sorted, 1)[0])
            if len(rows) >= 2 else 0.0,
            "monotonic_increasing_with_player_hp": bool(
                np.all(np.diff(values_sorted) >= -1e-12)
            ),
            "monotonic_decreasing_with_player_hp": bool(
                np.all(np.diff(values_sorted) <= 1e-12)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(output), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(output["summary"]), ensure_ascii=False, indent=2))
    print(f"[value-player-hp] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

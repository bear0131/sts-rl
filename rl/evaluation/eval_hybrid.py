"""Paired full-run evaluation of a native RLlib policy and a hybrid route mix.

The baseline uses ``policy_state.pkl`` for every action.  The hybrid uses the
RLlib policy for one chosen phase and an SB3 MaskablePPO checkpoint for every
other phase.  Both conditions run the same deterministic seeds in an
observation environment matching the RLlib policy.

Example:

    uv run python -m rl.evaluation.eval_hybrid \
      C:/Users/Tony2/Downloads/policy_state.pkl \
      --combat-checkpoint C:/Users/Tony2/Downloads/maskable_ppo_6400000_steps.zip \
      --episodes 100 --num-envs 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as model_helpers
from rl.evaluation.eval import is_win
from rl.policies.rllib_inference import RLLibInferenceModel

# Register the custom SB3 policy before loading its checkpoint.
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


@dataclass
class ActiveRun:
    env: SlayTheSpireEnv
    episode: int
    seed: int
    observation: dict
    info: dict
    steps: int = 0
    combat_actions: int = 0
    overworld_actions: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an RLlib policy_state.pkl against a hybrid that routes "
            "one phase to the RLlib policy and everything else to SB3."
        )
    )
    parser.add_argument("policy_state", type=Path)
    parser.add_argument("--combat-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--route-phase",
        choices=["combat", "map"],
        default="combat",
        help=(
            "Phase driven by policy_state.pkl; the SB3 checkpoint is used "
            "for every other phase."
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to rl/runs/eval-hybrid/<timestamp>.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be non-negative")
    if not args.policy_state.is_file():
        raise FileNotFoundError(f"policy state not found: {args.policy_state}")
    if not args.combat_checkpoint.is_file():
        raise FileNotFoundError(
            f"combat checkpoint not found: {args.combat_checkpoint}"
        )


def _space_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if hasattr(left, "low"):
        return bool(
            np.array_equal(left.low, right.low)
            and np.array_equal(left.high, right.high)
        )
    return left == right


def _validate_model_contract(primary, combat) -> None:
    primary_spaces = primary.observation_space.spaces
    combat_spaces = combat.observation_space.spaces
    missing = sorted(set(combat_spaces) - set(primary_spaces))
    mismatched = sorted(
        key
        for key in combat_spaces.keys() & primary_spaces.keys()
        if not _space_equal(combat_spaces[key], primary_spaces[key])
    )
    if missing or mismatched:
        raise ValueError(
            "combat model is incompatible with the primary observation space: "
            f"missing={missing}, mismatched={mismatched}"
        )
    if combat.action_space != primary.action_space:
        raise ValueError(
            f"action spaces differ: combat={combat.action_space}, "
            f"primary={primary.action_space}"
        )
    extra = sorted(set(primary_spaces) - set(combat_spaces))
    print(
        "[hybrid-eval] observation contract: "
        f"common={len(combat_spaces)} primary_only={extra}",
        flush=True,
    )


def _new_env(primary, ascension: int) -> SlayTheSpireEnv:
    return SlayTheSpireEnv(
        ascension=ascension,
        global_map_view=model_helpers.uses_global_map_observation(primary),
        card_language=model_helpers.uses_card_language(primary),
        combat_language=model_helpers.uses_combat_language(primary),
        deck_size_observation=model_helpers.uses_deck_size_observation(primary),
        neow_observation=model_helpers.uses_neow_observation(primary),
        event_option_observation=(
            model_helpers.uses_event_option_observation(primary)
        ),
    )


def _reset_run(
    env: SlayTheSpireEnv,
    episode: int,
    seed: int,
) -> ActiveRun:
    observation, info = env.reset(seed=seed)
    return ActiveRun(env, episode, seed, observation, info)


def _project_batch(runs: list[ActiveRun], observation_space) -> dict:
    return {
        key: np.stack([run.observation[key] for run in runs])
        for key in observation_space.spaces
    }


def _predict_group(model, runs: list[ActiveRun]) -> np.ndarray:
    observations = _project_batch(runs, model.observation_space)
    masks = np.stack([run.env.action_masks() for run in runs])
    if not np.all(masks.any(axis=1)):
        bad = [run.seed for run, mask in zip(runs, masks) if not mask.any()]
        raise RuntimeError(f"all-false action mask for seeds {bad}")
    actions, _ = model.predict(
        observations,
        action_masks=masks,
        deterministic=True,
    )
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if actions.shape != (len(runs),):
        raise RuntimeError(
            f"model returned action shape {actions.shape} for batch {len(runs)}"
        )
    illegal = [
        (run.seed, int(action))
        for run, action, mask in zip(runs, actions, masks)
        if not mask[action]
    ]
    if illegal:
        raise RuntimeError(f"model selected illegal actions: {illegal}")
    return actions


def _finish_row(condition: str, run: ActiveRun, terminated, truncated) -> dict:
    stats = run.env.unwrapped.thick.run_stats()
    floor = int(stats["floor"])
    act = int(stats["act"])
    if is_win(run.env):
        result = "win"
    elif truncated and run.info.get("dead_screen"):
        result = "dead_screen"
    elif truncated:
        result = "other_truncation"
    elif terminated:
        result = "death"
    else:
        raise RuntimeError("attempted to finish a live run")
    return {
        "condition": condition,
        "episode": run.episode + 1,
        "seed": run.seed,
        "result": result,
        "final_floor": floor,
        "final_act": act,
        "act_1_won": int(act > 1),
        "act_2_won": int(act > 2),
        "act_3_won": int(act > 3),
        "steps": run.steps,
        "combat_actions": run.combat_actions,
        "overworld_actions": run.overworld_actions,
        "outcome": stats["outcome"],
    }


def evaluate_condition(
    condition: str,
    primary,
    combat,
    args: argparse.Namespace,
) -> list[dict]:
    hybrid = combat is not None
    route_phase = args.route_phase
    env_count = min(args.num_envs, args.episodes)
    envs = [_new_env(primary, args.ascension) for _ in range(env_count)]
    next_episode = 0
    active: list[ActiveRun] = []
    rows: list[dict] = []
    started = time.monotonic()
    last_report = started

    try:
        for env in envs:
            active.append(
                _reset_run(env, next_episode, args.seed + next_episode)
            )
            next_episode += 1

        while active:
            actions_by_id: dict[int, int] = {}
            route_runs = (
                [run for run in active if run.info.get("phase") == route_phase]
                if hybrid else []
            )
            rest_runs = (
                [run for run in active if run.info.get("phase") != route_phase]
                if hybrid else list(active)
            )
            if rest_runs:
                for run, action in zip(
                    rest_runs, _predict_group(primary, rest_runs)
                ):
                    actions_by_id[id(run)] = int(action)
            if route_runs:
                for run, action in zip(
                    route_runs, _predict_group(combat, route_runs)
                ):
                    actions_by_id[id(run)] = int(action)

            still_active: list[ActiveRun] = []
            for run in active:
                phase = run.info.get("phase")
                action = actions_by_id[id(run)]
                if phase == "combat":
                    run.combat_actions += 1
                else:
                    run.overworld_actions += 1
                (
                    run.observation,
                    _reward,
                    terminated,
                    truncated,
                    run.info,
                ) = run.env.step(action)
                run.steps += 1
                if terminated or truncated:
                    rows.append(
                        _finish_row(condition, run, terminated, truncated)
                    )
                    done = len(rows)
                    if (
                        args.progress_every > 0
                        and done % args.progress_every == 0
                    ):
                        elapsed = time.monotonic() - started
                        eta = elapsed / done * (args.episodes - done)
                        mean_floor = np.mean(
                            [row["final_floor"] for row in rows]
                        )
                        print(
                            f"[hybrid-eval] {condition} {done}/{args.episodes} "
                            f"floor_mean={mean_floor:.2f} "
                            f"rate={done / elapsed:.2f} ep/s "
                            f"ETA={eta / 60:.1f} min",
                            flush=True,
                        )
                        last_report = time.monotonic()
                    if next_episode < args.episodes:
                        still_active.append(
                            _reset_run(
                                run.env,
                                next_episode,
                                args.seed + next_episode,
                            )
                        )
                        next_episode += 1
                else:
                    still_active.append(run)
            active = still_active

            now = time.monotonic()
            if (
                args.heartbeat_seconds > 0
                and now - last_report >= args.heartbeat_seconds
            ):
                floors = [
                    int(run.env.unwrapped.thick.run_stats()["floor"])
                    for run in active
                ]
                print(
                    f"[hybrid-eval] {condition} heartbeat: "
                    f"done={len(rows)}/{args.episodes} active={len(active)} "
                    f"active_floor_mean={np.mean(floors) if floors else 0:.2f}",
                    flush=True,
                )
                last_report = now
    finally:
        for env in envs:
            env.close()

    rows.sort(key=lambda row: int(row["episode"]))
    return rows


def _mean_se(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def summarize(rows: list[dict]) -> dict:
    result = {"episodes": len(rows)}
    metrics = {
        "floor_mean": [row["final_floor"] for row in rows],
        "act_1_won_mean": [row["act_1_won"] for row in rows],
        "act_2_won_mean": [row["act_2_won"] for row in rows],
        "act_3_won_mean": [row["act_3_won"] for row in rows],
        "win_rate": [row["result"] == "win" for row in rows],
        "steps_mean": [row["steps"] for row in rows],
        "combat_actions_mean": [row["combat_actions"] for row in rows],
        "overworld_actions_mean": [row["overworld_actions"] for row in rows],
    }
    for name, values in metrics.items():
        result[name], result[f"{name}_se"] = _mean_se(values)
    result["floor_median"] = float(
        np.median([row["final_floor"] for row in rows])
    )
    result["results"] = {
        name: sum(row["result"] == name for row in rows)
        for name in ("win", "death", "dead_screen", "other_truncation")
    }
    return result


def paired_deltas(pure_rows: list[dict], hybrid_rows: list[dict]) -> dict:
    pure = {int(row["seed"]): row for row in pure_rows}
    hybrid = {int(row["seed"]): row for row in hybrid_rows}
    if pure.keys() != hybrid.keys():
        raise RuntimeError("paired conditions did not evaluate identical seeds")
    result = {}
    for metric in (
        "final_floor", "act_1_won", "act_2_won", "act_3_won"
    ):
        deltas = [
            float(hybrid[seed][metric]) - float(pure[seed][metric])
            for seed in sorted(pure)
        ]
        result[f"{metric}_delta_mean"], result[f"{metric}_delta_se"] = (
            _mean_se(deltas)
        )
    floor_deltas = [
        int(hybrid[seed]["final_floor"]) - int(pure[seed]["final_floor"])
        for seed in sorted(pure)
    ]
    result["floor_pair_outcomes"] = {
        "hybrid_better": sum(delta > 0 for delta in floor_deltas),
        "tied": sum(delta == 0 for delta in floor_deltas),
        "hybrid_worse": sum(delta < 0 for delta in floor_deltas),
    }
    return result


def _print_comparison(pure: dict, hybrid: dict, paired: dict) -> None:
    print("\n[hybrid-eval] paired comparison (hybrid - pure)")
    print(f"{'metric':<22} {'pure':>10} {'hybrid':>10} {'delta':>10} {'paired SE':>10}")
    for summary_name, row_name in (
        ("floor_mean", "final_floor"),
        ("act_1_won_mean", "act_1_won"),
        ("act_2_won_mean", "act_2_won"),
        ("act_3_won_mean", "act_3_won"),
    ):
        print(
            f"{summary_name:<22} {pure[summary_name]:>10.4f} "
            f"{hybrid[summary_name]:>10.4f} "
            f"{paired[f'{row_name}_delta_mean']:>10.4f} "
            f"{paired[f'{row_name}_delta_se']:>10.4f}"
        )
    print(
        f"{'win_rate':<22} {pure['win_rate']:>10.4f} "
        f"{hybrid['win_rate']:>10.4f} "
        f"{hybrid['win_rate'] - pure['win_rate']:>10.4f} {'-':>10}"
    )
    print(f"[hybrid-eval] floor pairs={paired['floor_pair_outcomes']}")


def _write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    pure_rows: list[dict],
    hybrid_rows: list[dict],
    pure_summary: dict,
    hybrid_summary: dict,
    paired: dict,
    hybrid_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pure_rows + hybrid_rows
    with (output_dir / "episodes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "config": {
            "policy_state": str(args.policy_state.resolve()),
            "combat_checkpoint": str(args.combat_checkpoint.resolve()),
            "route_phase": args.route_phase,
            "episodes": args.episodes,
            "seed": args.seed,
            "ascension": args.ascension,
            "num_envs": args.num_envs,
            "deterministic": True,
            "route_condition": f'info["phase"] == "{args.route_phase}"',
        },
        "pure_policy_state": pure_summary,
        "hybrid_label": hybrid_label,
        hybrid_label: hybrid_summary,
        "paired_delta_hybrid_minus_pure": paired,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"[hybrid-eval] outputs={output_dir.resolve()}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    output_dir = args.output_dir or (
        Path("rl/runs/eval-hybrid") / time.strftime("%Y%m%d-%H%M%S")
    )

    load_started = time.monotonic()
    print(f"[hybrid-eval] loading RLlib policy_state={args.policy_state}", flush=True)
    primary = RLLibInferenceModel(args.policy_state, device=args.device)
    print(
        f"[hybrid-eval] loading combat checkpoint={args.combat_checkpoint}",
        flush=True,
    )
    combat = MaskablePPO.load(args.combat_checkpoint, device=args.device)
    _validate_model_contract(primary, combat)
    print(
        f"[hybrid-eval] loaded RLlib steps={primary.global_timestep} and "
        f"SB3 steps={combat.num_timesteps} in "
        f"{time.monotonic() - load_started:.1f}s",
        flush=True,
    )

    primary.set_random_seed(args.seed)
    pure_rows = evaluate_condition("pure_policy_state", primary, None, args)
    primary.set_random_seed(args.seed)
    combat.set_random_seed(args.seed)
    hybrid_label = f"hybrid_{args.route_phase}_sb3"
    hybrid_rows = evaluate_condition(
        hybrid_label, primary, combat, args
    )
    pure_summary = summarize(pure_rows)
    hybrid_summary = summarize(hybrid_rows)
    paired = paired_deltas(pure_rows, hybrid_rows)
    _print_comparison(pure_summary, hybrid_summary, paired)
    _write_outputs(
        output_dir,
        args,
        pure_rows,
        hybrid_rows,
        pure_summary,
        hybrid_summary,
        paired,
        hybrid_label,
    )


if __name__ == "__main__":
    main()

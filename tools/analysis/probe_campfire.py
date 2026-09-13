"""Compare campfire choices from multiple whole-run checkpoints.

The newest checkpoint collects realistic campfire states from deterministic
full runs. Every checkpoint is then evaluated on exactly those same states, so
policy changes are not confounded by different routes or combat outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor

from env.core import action_space as A
from env.core import game_actions as ga
from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as model_utils

# Register custom policy classes before loading checkpoints.
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


CHOICES = tuple(ga.CAMPFIRE_OPTIONS)
REPORT_CHOICES = (*CHOICES, "proceed")
_CAMP_NAME = {offset: name for name, offset in ga.CAMPFIRE_OPTIONS.items()}
_STEP_RE = re.compile(r"_(\d+)_steps$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def checkpoint_step(path: Path) -> int:
    match = _STEP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"checkpoint name has no step count: {path}")
    return int(match.group(1))


def observation_flags(model: MaskablePPO) -> dict[str, bool]:
    return {
        "global_map_view": model_utils.uses_global_map_observation(model),
        "card_language": model_utils.uses_card_language(model),
        "combat_language": model_utils.uses_combat_language(model),
        "deck_size_observation": model_utils.uses_deck_size_observation(model),
        "neow_observation": model_utils.uses_neow_observation(model),
        "event_option_observation": (
            model_utils.uses_event_option_observation(model)
        ),
    }


def copy_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def action_name(action: int) -> str:
    segment, offset = A.segment_of(int(action))
    if segment == "CAMPFIRE":
        return _CAMP_NAME[offset]
    if segment == "PROCEED":
        return "proceed"
    return segment.lower()


def restricted_mask(mask: np.ndarray, segments: set[str]) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for action in np.flatnonzero(mask):
        segment, _offset = A.segment_of(int(action))
        if segment in segments:
            result[action] = True
    if not result.any():
        raise RuntimeError(f"no legal action remains for segments {segments}")
    return result


def hp_bin(hp_fraction: float) -> str:
    if hp_fraction < 0.35:
        return "00-35%"
    if hp_fraction < 0.55:
        return "35-55%"
    if hp_fraction < 0.75:
        return "55-75%"
    return "75-100%"


def collect_states(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    ascension: int,
    device: str,
    max_steps: int,
    progress_every: int,
) -> tuple[list[dict], list[dict], dict]:
    print(f"[campfire] loading reference {checkpoint.name}", flush=True)
    model = MaskablePPO.load(checkpoint, device=device)
    model.policy.eval()
    model.set_random_seed(seed)
    env = SlayTheSpireEnv(ascension=ascension, **observation_flags(model))
    campfire_states: list[dict] = []
    select_states: list[dict] = []
    outcomes = Counter()
    started = time.monotonic()

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            pending_select: str | None = None
            steps = 0

            while not (terminated or truncated) and steps < max_steps:
                mask = get_action_masks(env)
                phase = info.get("phase")
                action, _state = model.predict(
                    obs, action_masks=mask, deterministic=True
                )
                action = int(np.asarray(action).item())

                if phase == "card_select" and pending_select is not None:
                    canon = env.unwrapped.thick.obs_canonical()
                    cards = list(canon.get("options", {}).get("cards", []))
                    pick_mask = restricted_mask(mask, {"PICK_CARD"})
                    select_states.append({
                        "obs": copy_observation(obs),
                        "mask": pick_mask,
                        "kind": pending_select,
                        "episode": episode + 1,
                        "seed": episode_seed,
                        "act": int(canon.get("global", {}).get("act", 0)),
                        "floor": int(canon.get("global", {}).get("floor", 0)),
                        "cards": cards,
                        "reference_action": action,
                    })
                    pending_select = None
                elif pending_select is not None and phase != "rest":
                    pending_select = None

                if phase == "rest":
                    chosen = action_name(action)
                    if chosen in REPORT_CHOICES:
                        canon = env.unwrapped.thick.obs_canonical()
                        global_state = canon.get("global", {})
                        hp = int(global_state.get("hp", 0))
                        max_hp = int(global_state.get("max_hp", 0))
                        camp_mask = restricted_mask(
                            mask, {"CAMPFIRE", "PROCEED"}
                        )
                        legal = [
                            action_name(int(candidate))
                            for candidate in np.flatnonzero(camp_mask)
                        ]
                        campfire_states.append({
                            "obs": copy_observation(obs),
                            "mask": np.array(mask, copy=True),
                            "camp_mask": camp_mask,
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "act": int(global_state.get("act", 0)),
                            "floor": int(global_state.get("floor", 0)),
                            "hp": hp,
                            "max_hp": max_hp,
                            "hp_fraction": hp / max_hp if max_hp else 0.0,
                            "legal": legal,
                            "reference_action": chosen,
                        })
                        if chosen in {"smith", "toke"}:
                            pending_select = chosen

                obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1

            if terminated:
                stats = env.unwrapped.thick.run_stats()
                won = int(stats["act"]) > 3 or (
                    stats["outcome"] == "PLAYER_VICTORY"
                )
                outcomes["win" if won else "death"] += 1
            elif truncated:
                outcomes["truncated"] += 1
            else:
                outcomes["step_limit"] += 1

            done = episode + 1
            if progress_every and done % progress_every == 0:
                print(
                    f"[campfire] collected episodes={done}/{episodes} "
                    f"campfires={len(campfire_states)} "
                    f"selects={len(select_states)}",
                    flush=True,
                )
    finally:
        env.close()

    metadata = {
        "reference": str(checkpoint),
        "episodes": episodes,
        "seed": seed,
        "outcomes": dict(outcomes),
        "campfire_states": len(campfire_states),
        "select_states": len(select_states),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model
    gc.collect()
    return campfire_states, select_states, metadata


def batch_policy(
    model: MaskablePPO,
    states: list[dict],
    mask_key: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_actions = []
    all_probs = []
    for start in range(0, len(states), batch_size):
        chunk = states[start:start + batch_size]
        observations = {
            key: np.stack([state["obs"][key] for state in chunk])
            for key in chunk[0]["obs"]
        }
        masks = np.stack([state[mask_key] for state in chunk])
        with torch.inference_mode():
            distribution = model.policy.get_distribution(
                obs_as_tensor(observations, model.device),
                action_masks=masks,
            )
            probs = distribution.distribution.probs.detach().cpu().numpy()
        all_probs.append(probs)
        all_actions.append(np.argmax(probs, axis=1))
    return np.concatenate(all_actions), np.concatenate(all_probs)


def selected_card(state: dict, action: int) -> str:
    segment, offset = A.segment_of(int(action))
    if segment != "PICK_CARD" or offset >= len(state["cards"]):
        return action_name(action)
    card_id, upgrades = state["cards"][offset]
    return f"{card_id}+{int(upgrades)}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.max_steps < 1:
        raise ValueError("episodes, batch-size, and max-steps must be positive")
    checkpoints = sorted(args.checkpoints, key=checkpoint_step)
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    if len({checkpoint_step(path) for path in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint step counts must be unique")

    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = checkpoints[-1]
    camp_states, select_states, metadata = collect_states(
        reference,
        episodes=args.episodes,
        seed=args.seed,
        ascension=args.ascension,
        device=args.device,
        max_steps=args.max_steps,
        progress_every=args.progress_every,
    )
    if not camp_states:
        raise RuntimeError("reference runs produced no campfire states")

    results = []
    prediction_rows = []
    hp_rows = []
    target_rows = []
    predictions_by_step: dict[int, list[str]] = {}

    for index, checkpoint in enumerate(checkpoints, 1):
        step = checkpoint_step(checkpoint)
        print(
            f"[campfire] evaluating {index}/{len(checkpoints)} "
            f"step={step}",
            flush=True,
        )
        model = MaskablePPO.load(checkpoint, device=args.device)
        model.policy.eval()
        camp_actions, camp_probs = batch_policy(
            model, camp_states, "camp_mask", args.batch_size
        )
        raw_actions, _raw_probs = batch_policy(
            model, camp_states, "mask", args.batch_size
        )
        names = [action_name(int(action)) for action in camp_actions]
        raw_names = [action_name(int(action)) for action in raw_actions]
        predictions_by_step[step] = names
        counts = Counter(names)
        raw_side_actions = sum(name not in REPORT_CHOICES for name in raw_names)
        row = {
            "step": step,
            "states": len(camp_states),
            "raw_side_actions": raw_side_actions,
        }
        for choice in REPORT_CHOICES:
            row[f"{choice}_count"] = counts[choice]
            row[f"{choice}_pct"] = counts[choice] / len(camp_states)
        results.append(row)

        for state_index, (state, name, raw_name, probs) in enumerate(
            zip(camp_states, names, raw_names, camp_probs), 1
        ):
            detail = {
                "step": step,
                "state": state_index,
                "episode": state["episode"],
                "seed": state["seed"],
                "act": state["act"],
                "floor": state["floor"],
                "hp": state["hp"],
                "max_hp": state["max_hp"],
                "hp_fraction": state["hp_fraction"],
                "hp_bin": hp_bin(state["hp_fraction"]),
                "legal": "+".join(state["legal"]),
                "reference_action": state["reference_action"],
                "choice": name,
                "raw_choice": raw_name,
            }
            for choice in REPORT_CHOICES:
                if choice == "proceed":
                    action_id = A.seg_base("PROCEED")
                else:
                    action_id = (
                        A.seg_base("CAMPFIRE")
                        + ga.CAMPFIRE_OPTIONS[choice]
                    )
                detail[f"p_{choice}"] = float(probs[action_id])
            prediction_rows.append(detail)

        for bucket in ("00-35%", "35-55%", "55-75%", "75-100%"):
            indices = [
                i for i, state in enumerate(camp_states)
                if hp_bin(state["hp_fraction"]) == bucket
            ]
            bucket_counts = Counter(names[i] for i in indices)
            for choice in REPORT_CHOICES:
                hp_rows.append({
                    "step": step,
                    "hp_bin": bucket,
                    "states": len(indices),
                    "choice": choice,
                    "count": bucket_counts[choice],
                    "pct": (
                        bucket_counts[choice] / len(indices) if indices else 0.0
                    ),
                })

        if select_states:
            select_actions, _select_probs = batch_policy(
                model, select_states, "mask", args.batch_size
            )
            targets = Counter(
                selected_card(state, int(action))
                for state, action in zip(select_states, select_actions)
                if state["kind"] == "smith"
            )
            total_targets = sum(targets.values())
            for card, count in targets.most_common():
                target_rows.append({
                    "step": step,
                    "card": card,
                    "count": count,
                    "pct": count / total_targets if total_targets else 0.0,
                })

        del model
        gc.collect()

    newest_step = checkpoint_step(reference)
    newest_predictions = predictions_by_step[newest_step]
    for row in results:
        predictions = predictions_by_step[row["step"]]
        changed = sum(
            current != newest
            for current, newest in zip(predictions, newest_predictions)
        )
        row["different_from_newest_count"] = changed
        row["different_from_newest_pct"] = changed / len(camp_states)

    write_csv(args.output_dir / "summary.csv", results)
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "hp_bins.csv", hp_rows)
    write_csv(args.output_dir / "smith_targets.csv", target_rows)
    write_csv(args.output_dir / "collection.csv", [metadata])

    print()
    print(
        "step states rest smith recall lift toke dig proceed "
        "raw_side diff_vs_newest"
    )
    for row in results:
        percentages = [
            f"{100 * row[f'{choice}_pct']:.1f}%"
            for choice in REPORT_CHOICES
        ]
        print(
            f"{row['step']} {row['states']} "
            f"{' '.join(percentages)} {row['raw_side_actions']} "
            f"{100 * row['different_from_newest_pct']:.1f}%"
        )

    print(f"[campfire] collection={metadata}")
    print(f"[campfire] output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

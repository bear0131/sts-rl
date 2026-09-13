"""Measure overworld value-head responses to counterfactual card upgrades."""

from __future__ import annotations

import argparse
import csv
import gc
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor

from env.core import encode
from env.core.cards_vocab import CARD_VOCAB
from env.core.ironclad_cards import IRONCLAD_CARDS
from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as model_utils

# Register custom policy classes before loading checkpoints.
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


_STEP_RE = re.compile(r"_(\d+)_steps$")
_OVERWORLD_PHASES = frozenset({
    "event",
    "map",
    "combat_reward",
    "card_select",
    "rest",
    "chest",
    "boss_relic",
    "shop",
})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--states-per-phase", type=int, default=25)
    parser.add_argument("--seed", type=int, default=40_000)
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


def candidate_cards(obs: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    """One base-copy candidate per distinct upgradeable Ironclad card."""
    result = []
    seen = set()
    for slot, (card_index, upgrades) in enumerate(
        zip(obs["deck_ids"], obs["deck_upgrades"])
    ):
        card_index = int(card_index)
        if card_index <= 0 or float(upgrades) != 0.0:
            continue
        card_id = CARD_VOCAB[card_index]
        if card_id not in IRONCLAD_CARDS or card_id in seen:
            continue
        seen.add(card_id)
        result.append((slot, card_id))
    return result


def collect_states(
    checkpoint: Path,
    *,
    episodes: int,
    states_per_phase: int,
    seed: int,
    ascension: int,
    device: str,
    max_steps: int,
    progress_every: int,
) -> tuple[list[dict], dict]:
    print(f"[upgrade-value] loading reference {checkpoint.name}", flush=True)
    model = MaskablePPO.load(checkpoint, device=device)
    model.policy.eval()
    model.set_random_seed(seed)
    env = SlayTheSpireEnv(ascension=ascension, **observation_flags(model))
    states = []
    phase_counts = Counter()
    outcomes = Counter()
    seen = set()
    started = time.monotonic()

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, info = env.reset(seed=episode_seed)
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated) and steps < max_steps:
                phase = str(info.get("phase", ""))
                if (
                    phase in _OVERWORLD_PHASES
                    and phase_counts[phase] < states_per_phase
                ):
                    canon = env.unwrapped.thick.obs_canonical()
                    global_state = canon.get("global", {})
                    active = obs["deck_ids"] != 0
                    key = (
                        episode,
                        phase,
                        int(global_state.get("floor", 0)),
                        tuple(obs["deck_ids"][active].tolist()),
                        tuple(obs["deck_upgrades"][active].tolist()),
                    )
                    candidates = candidate_cards(obs)
                    if key not in seen and candidates:
                        seen.add(key)
                        states.append({
                            "obs": copy_observation(obs),
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "phase": phase,
                            "act": int(global_state.get("act", 0)),
                            "floor": int(global_state.get("floor", 0)),
                            "hp": int(global_state.get("hp", 0)),
                            "max_hp": int(global_state.get("max_hp", 0)),
                            "candidates": candidates,
                        })
                        phase_counts[phase] += 1

                action, _state = model.predict(
                    obs,
                    action_masks=get_action_masks(env),
                    deterministic=True,
                )
                obs, _reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
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
                counts = ",".join(
                    f"{phase}:{phase_counts[phase]}"
                    for phase in sorted(phase_counts)
                )
                print(
                    f"[upgrade-value] collected episodes={done}/{episodes} "
                    f"states={len(states)} phases={counts}",
                    flush=True,
                )
    finally:
        env.close()

    metadata = {
        "reference": str(checkpoint),
        "episodes": episodes,
        "seed": seed,
        "states": len(states),
        "counterfactual_pairs": sum(len(s["candidates"]) for s in states),
        "phase_counts": dict(phase_counts),
        "outcomes": dict(outcomes),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model
    gc.collect()
    return states, metadata


def values_for_observations(
    model: MaskablePPO,
    observations: list[dict[str, np.ndarray]],
    batch_size: int,
) -> np.ndarray:
    values = []
    for start in range(0, len(observations), batch_size):
        chunk = observations[start:start + batch_size]
        batch = {
            key: np.stack([obs[key] for obs in chunk])
            for key in chunk[0]
        }
        with torch.inference_mode():
            value = model.policy.predict_values(
                obs_as_tensor(batch, model.device)
            )
        values.append(value.detach().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def counterfactual_values(
    model: MaskablePPO,
    states: list[dict],
    records: list[tuple[int, int, str]],
    batch_size: int,
) -> np.ndarray:
    values = []
    increment = 1.0 / encode.UPGRADE_SCALE
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        batch = {
            key: np.stack([states[state_index]["obs"][key] for state_index, _, _ in chunk])
            for key in states[chunk[0][0]]["obs"]
        }
        for row, (state_index, slot, _card_id) in enumerate(chunk):
            ids = batch["deck_ids"][row]
            upgrades = batch["deck_upgrades"][row]
            card_index = int(ids[slot])
            same_card = np.flatnonzero(ids == card_index)
            upgrades[slot] = increment
            upgrades[same_card] = np.sort(upgrades[same_card])
            if not np.isclose(upgrades[same_card], increment).any():
                raise RuntimeError(
                    f"failed to upgrade state={state_index} slot={slot}"
                )
        with torch.inference_mode():
            value = model.policy.predict_values(
                obs_as_tensor(batch, model.device)
            )
        values.append(value.detach().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.episodes,
        args.states_per_phase,
        args.batch_size,
        args.max_steps,
        args.torch_threads,
    ) < 1:
        raise ValueError("numeric probe parameters must be positive")
    checkpoints = sorted(args.checkpoints, key=checkpoint_step)
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = checkpoints[-1]
    states, metadata = collect_states(
        reference,
        episodes=args.episodes,
        states_per_phase=args.states_per_phase,
        seed=args.seed,
        ascension=args.ascension,
        device=args.device,
        max_steps=args.max_steps,
        progress_every=args.progress_every,
    )
    if not states:
        raise RuntimeError("reference runs produced no overworld states")

    records = [
        (state_index, slot, card_id)
        for state_index, state in enumerate(states)
        for slot, card_id in state["candidates"]
    ]
    summary_rows = []
    pair_rows = []
    card_rows = []
    phase_rows = []

    for index, checkpoint in enumerate(checkpoints, 1):
        step = checkpoint_step(checkpoint)
        print(
            f"[upgrade-value] evaluating {index}/{len(checkpoints)} "
            f"step={step}",
            flush=True,
        )
        model = MaskablePPO.load(checkpoint, device=args.device)
        model.policy.eval()
        base_values = values_for_observations(
            model, [state["obs"] for state in states], args.batch_size
        )
        upgraded_values = counterfactual_values(
            model, states, records, args.batch_size
        )
        deltas = upgraded_values - np.asarray([
            base_values[state_index] for state_index, _slot, _card in records
        ])
        base_stats = describe(base_values)
        delta_stats = describe(deltas)
        epsilon = 1e-6
        summary_rows.append({
            "step": step,
            "states": len(states),
            "pairs": len(records),
            "base_value_mean": base_stats["mean"],
            "base_value_std": base_stats["std"],
            "delta_mean": delta_stats["mean"],
            "delta_std": delta_stats["std"],
            "delta_median": delta_stats["median"],
            "delta_mean_abs": float(np.mean(np.abs(deltas))),
            "delta_p05": delta_stats["p05"],
            "delta_p95": delta_stats["p95"],
            "delta_min": delta_stats["min"],
            "delta_max": delta_stats["max"],
            "positive_pct": float(np.mean(deltas > epsilon)),
            "negative_pct": float(np.mean(deltas < -epsilon)),
            "near_zero_pct": float(np.mean(np.abs(deltas) <= epsilon)),
        })

        by_card = defaultdict(list)
        by_phase = defaultdict(list)
        for pair_index, (state_index, slot, card_id) in enumerate(records):
            state = states[state_index]
            delta = float(deltas[pair_index])
            pair_rows.append({
                "step": step,
                "state": state_index + 1,
                "episode": state["episode"],
                "seed": state["seed"],
                "phase": state["phase"],
                "act": state["act"],
                "floor": state["floor"],
                "hp": state["hp"],
                "max_hp": state["max_hp"],
                "slot": slot,
                "card": card_id,
                "base_value": float(base_values[state_index]),
                "upgraded_value": float(upgraded_values[pair_index]),
                "delta": delta,
            })
            by_card[card_id].append(delta)
            by_phase[state["phase"]].append(delta)

        for card_id, card_deltas in sorted(by_card.items()):
            values = np.asarray(card_deltas)
            card_rows.append({
                "step": step,
                "card": card_id,
                "samples": len(values),
                "delta_mean": float(values.mean()),
                "delta_median": float(np.median(values)),
                "delta_mean_abs": float(np.abs(values).mean()),
                "positive_pct": float(np.mean(values > epsilon)),
                "negative_pct": float(np.mean(values < -epsilon)),
            })
        for phase, phase_deltas in sorted(by_phase.items()):
            values = np.asarray(phase_deltas)
            phase_rows.append({
                "step": step,
                "phase": phase,
                "samples": len(values),
                "delta_mean": float(values.mean()),
                "delta_median": float(np.median(values)),
                "delta_mean_abs": float(np.abs(values).mean()),
                "positive_pct": float(np.mean(values > epsilon)),
                "negative_pct": float(np.mean(values < -epsilon)),
            })

        del model
        gc.collect()

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "pairs.csv", pair_rows)
    write_csv(args.output_dir / "cards.csv", card_rows)
    write_csv(args.output_dir / "phases.csv", phase_rows)
    write_csv(args.output_dir / "collection.csv", [metadata])

    print()
    print(
        "step states pairs base_mean base_std delta_mean delta_abs "
        "positive negative near_zero p05 p95"
    )
    for row in summary_rows:
        print(
            f"{row['step']} {row['states']} {row['pairs']} "
            f"{row['base_value_mean']:.6f} {row['base_value_std']:.6f} "
            f"{row['delta_mean']:.8f} {row['delta_mean_abs']:.8f} "
            f"{100 * row['positive_pct']:.1f}% "
            f"{100 * row['negative_pct']:.1f}% "
            f"{100 * row['near_zero_pct']:.1f}% "
            f"{row['delta_p05']:.8f} {row['delta_p95']:.8f}"
        )
    print(f"[upgrade-value] collection={metadata}")
    print(f"[upgrade-value] output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

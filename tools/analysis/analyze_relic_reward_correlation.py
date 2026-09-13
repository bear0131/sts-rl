"""Evaluate relic-count / shaped-return correlation for a whole-run checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _checkpoint_default() -> Path:
    return Path.home() / "Downloads" / "maskable_ppo_27018368_steps.zip"


def _run_worker(args_tuple):
    checkpoint, device, ascension, base_seed, worker, episodes, max_steps = args_tuple

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    from env.core.run_env import SlayTheSpireEnv
    from rl.policies import model as model_utils
    from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
    from rl.policies import split_policy as _split_policy  # noqa: F401
    from rl.training.reward import ShapedRewardWrapper

    model = MaskablePPO.load(checkpoint, device=device)
    base = SlayTheSpireEnv(
        ascension=ascension,
        global_map_view=model_utils.uses_global_map_observation(model),
        card_language=model_utils.uses_card_language(model),
        combat_language=model_utils.uses_combat_language(model),
        deck_size_observation=model_utils.uses_deck_size_observation(model),
        neow_observation=model_utils.uses_neow_observation(model),
        event_option_observation=model_utils.uses_event_option_observation(model),
    )
    env = ShapedRewardWrapper(base)
    rows = []
    try:
        for local_ep in range(episodes):
            seed = base_seed + local_ep
            obs, _info = env.reset(seed=seed)
            initial = list(base.thick.obs_canonical()["global"]["relics"])
            initial_set = set(initial)
            seen_new = set()
            shaped_return = 0.0
            raw_return = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated) and steps < max_steps:
                current = set(base.thick.obs_canonical()["global"]["relics"])
                seen_new |= current - initial_set
                action, _ = model.predict(
                    obs,
                    action_masks=get_action_masks(env),
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = env.step(
                    int(np.asarray(action).item())
                )
                shaped_return += float(reward)
                raw_return += float(info.get("env_reward", 0.0))
                steps += 1

            final_relics = list(base.thick.obs_canonical()["global"]["relics"])
            seen_new |= set(final_relics) - initial_set
            stats = base.thick.run_stats()
            result = (
                "win"
                if int(stats["act"]) > 3 or stats["outcome"] == "PLAYER_VICTORY"
                else "truncated" if truncated else "death"
            )
            rows.append({
                "worker": worker,
                "seed": seed,
                "result": result,
                "shaped_return": shaped_return,
                "raw_env_return": raw_return,
                "final_floor": int(stats["floor"]),
                "final_act": int(stats["act"]),
                "final_hp": int(stats["hp"]),
                "steps": steps,
                "initial_relic_count": len(initial),
                "final_relic_count": len(final_relics),
                "net_final_relics": len(final_relics) - len(initial),
                "unique_new_relics": len(seen_new),
                "final_relics": final_relics,
            })
    finally:
        env.close()
    return rows


def _pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def _spearman(x, y) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _residualize(values, covariates):
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    x = np.column_stack([np.ones(len(values)), x])
    beta = np.linalg.lstsq(x, values, rcond=None)[0]
    return values - x @ beta


def _describe(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _print_summary(rows) -> None:
    print(f"N {len(rows)}")
    print("result_counts", dict(Counter(row["result"] for row in rows)))

    reward = [row["shaped_return"] for row in rows]
    floor = [row["final_floor"] for row in rows]
    act = [row["final_act"] for row in rows]
    cov = np.column_stack([floor, act])
    for key in ("unique_new_relics", "net_final_relics", "final_relic_count"):
        x = [row[key] for row in rows]
        print(
            f"CORR {key} "
            f"pearson={_pearson(x, reward):.4f} "
            f"spearman={_spearman(x, reward):.4f}"
        )
        print(
            f"PARTIAL_floor_act {key} "
            f"pearson={_pearson(_residualize(x, cov), _residualize(reward, cov)):.4f}"
        )

    for key in (
        "shaped_return",
        "unique_new_relics",
        "final_relic_count",
        "final_floor",
        "final_act",
        "steps",
    ):
        print("DESC", key, json.dumps(_describe([row[key] for row in rows])))

    bins = defaultdict(list)
    for row in rows:
        bins[row["unique_new_relics"]].append(row["shaped_return"])
    print("GROUP_BY_unique_new_relics")
    for key in sorted(bins):
        values = np.asarray(bins[key], dtype=np.float64)
        print(
            f"  relics={key:2d} n={len(values):2d} "
            f"mean_return={values.mean():7.3f} std={values.std():6.3f} "
            f"min={values.min():7.3f} max={values.max():7.3f}"
        )

    print("ROWS")
    for row in sorted(rows, key=lambda item: item["seed"]):
        print(
            f"  seed={row['seed']} result={row['result']} "
            f"return={row['shaped_return']:.2f} floor={row['final_floor']:2d} "
            f"act={row['final_act']} unique_new_relics={row['unique_new_relics']:2d} "
            f"final_relics={row['final_relic_count']:2d} steps={row['steps']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=_checkpoint_default())
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    workers = min(args.workers, args.episodes)
    counts = [args.episodes // workers] * workers
    for i in range(args.episodes % workers):
        counts[i] += 1
    jobs = [
        (
            args.checkpoint,
            args.device,
            args.ascension,
            args.seed + sum(counts[:worker]),
            worker,
            count,
            args.max_steps,
        )
        for worker, count in enumerate(counts)
        if count
    ]

    print(
        f"[relic-corr] checkpoint={args.checkpoint} episodes={args.episodes} "
        f"workers={workers} seed={args.seed} ascension={args.ascension}",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            part = future.result()
            rows.extend(part)
            print(
                f"[relic-corr] worker_done={done}/{len(futures)} "
                f"rows={len(rows)}/{args.episodes}",
                flush=True,
            )
    _print_summary(rows)


if __name__ == "__main__":
    main()

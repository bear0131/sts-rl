"""Evaluate complete runs with NOSL-IS-AZ combat decisions.

The combat branch uses the worker-side NOSL search for both checkpoints; map,
event, reward, and shop decisions use the checkpoint policy directly.  This
keeps the comparison aligned with the training-time action path.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.core import vocab as V
from env.core.run_env import SlayTheSpireEnv
from rl import model as model_config

# Register custom policy classes before SB3 unpickles the checkpoint.
from . import pointer_policy as _pointer_policy  # noqa: F401
from . import split_policy as _split_policy  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Deprecated compatibility option; leaf values come directly "
             "from the value network.",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=0x4E4F534C4953415A)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _stats(env: SlayTheSpireEnv) -> dict:
    return env.unwrapped.thick.run_stats()


def _is_win(stats: dict) -> bool:
    return int(stats["act"]) > 3 or stats["outcome"] == "PLAYER_VICTORY"


def _policy_outputs(model, obs: dict) -> tuple[np.ndarray, float]:
    tensor_obs, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        logits, values = model.policy.core.actor_critic(tensor_obs)
    return (
        logits[0].detach().float().cpu().numpy(),
        float(values.reshape(-1)[0].detach().cpu()),
    )


def _search_with_model(
    model,
    env: SlayTheSpireEnv,
    obs: dict,
    *,
    state_nonce: int,
    master_nonce: int,
    simulations: int,
    depth: int,
    c_puct: float,
    temperature: float,
):
    """Run true staged AlphaZero leaf evaluation in the single eval process."""
    root_logits, _root_value = _policy_outputs(model, obs)
    request = env.nosl_is_az_begin(
        root_logits,
        state_nonce=state_nonce,
        master_nonce=master_nonce,
        simulations=simulations,
        depth=depth,
        rollout_steps=0,
        c_puct=c_puct,
        temperature=temperature,
    )
    while not request.get("done", False):
        leaf_logits, leaf_value = _policy_outputs(
            model, request["observation"]
        )
        request = env.nosl_is_az_backprop(leaf_value, leaf_logits)
    return request["result"]


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.simulations < 1 or args.depth < 1 or args.rollout_steps < 0:
        raise ValueError("episodes/simulations/depth must be positive and rollout-steps non-negative")

    model = MaskablePPO.load(args.checkpoint, device=args.device)
    env = SlayTheSpireEnv(
        ascension=0,
        global_map_view=model_config.uses_global_map_observation(model),
        card_language=model_config.uses_card_language(model),
        combat_language=model_config.uses_combat_language(model),
        deck_size_observation=model_config.uses_deck_size_observation(model),
        neow_observation=model_config.uses_neow_observation(model),
        event_option_observation=model_config.uses_event_option_observation(model),
    )
    combat_phase = V.PHASE_VOCAB_INDEX["combat"]
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            decisions = 0
            while not (terminated or truncated):
                phase = int(np.asarray(obs["phase"]).reshape(-1)[0])
                if phase == combat_phase:
                    # State nonce is derived only from public episode counters;
                    # hidden simulator RNG never enters the search seed.
                    nonce = (
                        int(args.master_nonce)
                        ^ (int(seed) * 0x9E3779B97F4A7C15)
                        ^ (int(decisions) * 0xD1B54A32D192ED03)
                    )
                    result = _search_with_model(
                        model,
                        env,
                        obs,
                        state_nonce=nonce,
                        master_nonce=args.master_nonce,
                        simulations=args.simulations,
                        depth=args.depth,
                        c_puct=args.c_puct,
                        temperature=0.0,
                    )
                    visits = np.asarray(result["root_visits"], dtype=np.int64)
                    action = int(np.argmax(visits))
                    decisions += 1
                else:
                    action, _ = model.predict(
                        obs,
                        action_masks=get_action_masks(env),
                        deterministic=True,
                    )
                    action = int(np.asarray(action).item())
                obs, _reward, terminated, truncated, _info = env.step(action)
                steps += 1

            stats = _stats(env)
            result = "win" if _is_win(stats) else ("truncation" if truncated else "death")
            row = {
                "episode": episode + 1,
                "seed": seed,
                "result": result,
                "final_floor": int(stats["floor"]),
                "final_act": int(stats["act"]),
                "steps": steps,
                "combat_search_decisions": decisions,
                "outcome": stats["outcome"],
            }
            rows.append(row)
            print(
                f"[nosl-eval] {episode + 1}/{args.episodes}: "
                f"result={result} floor={stats['floor']} steps={steps} "
                f"combat_decisions={decisions}",
                flush=True,
            )
    finally:
        env.close()

    floors = np.asarray([int(row["final_floor"]) for row in rows], dtype=np.float64)
    counts = Counter(str(row["result"]) for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "seed": args.seed,
        "simulations": args.simulations,
        "depth": args.depth,
        "rollout_steps": args.rollout_steps,
        "wins": counts["win"],
        "win_rate": counts["win"] / args.episodes,
        "deaths": counts["death"],
        "final_floor_mean": float(floors.mean()),
        "final_floor_median": float(np.median(floors)),
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    print(
        f"[nosl-eval] episodes={args.episodes} wins={counts['win']}/{args.episodes} "
        f"win_rate={summary['win_rate']:.4f} final_floor_mean={floors.mean():.2f} "
        f"median={np.median(floors):.1f} elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[nosl-eval] wrote {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()

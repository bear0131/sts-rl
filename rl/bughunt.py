"""Run a checkpoint for many whole-run episodes and catch every crash.

Unlike rl.eval (which assumes a clean run and reports stats), this harness wraps
every env step in try/except: a crash is logged with enough to reproduce (seed,
episode, step, last action, floor/act, full traceback), deduped by a traceback
signature, and then the episode is abandoned and a fresh one started. One bug
never stops the hunt, and identical crashes collapse to a single entry with a
count + a few example seeds.

The point is to surface env/schema/sim edge cases the policy reaches in the long
tail (e.g. an unmapped monster id at an act boss) by sheer volume of play.

Examples:

    python -m rl.bughunt rl/runs/.../maskable_ppo_1000000_steps.zip
    python -m rl.bughunt <ckpt> --episodes 5000 --seed 0
    python -m rl.bughunt <ckpt> --episodes 5000 --mode stochastic
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np

from env.core.run_env import SlayTheSpireEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Import custom model code before loading a checkpoint (see eval.py).
from . import model as _model  # noqa: F401
from .eval import is_win


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crash-hunt a checkpoint over many runs.")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0, help="first seed; episode i uses seed+i")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--mode",
        choices=["deterministic", "stochastic", "mixed"],
        default="mixed",
        help="action selection; 'mixed' alternates per episode to cover both.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output dir for crashes.jsonl + summary.json (default: rl/bughunt_out/<ts>)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=5_000,
        help="hard per-episode step cap; exceeding it is logged as a stall, not a crash.",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="print a running summary every N episodes.",
    )
    p.add_argument(
        "--stop-on-crash",
        action="store_true",
        help="exit as soon as the first crash is caught (fast triage of a new bug).",
    )
    return p


def crash_signature(tb_list) -> str:
    """A stable id for a crash: the deepest frame inside OUR code (env/bridge/rl)
    plus the exception type. Frames in site-packages collapse together so the
    same bug reached via different stacks dedups to one signature."""
    ours = [f for f in tb_list if _is_ours(f.filename)]
    frame = ours[-1] if ours else tb_list[-1]
    rel = os.path.basename(frame.filename)
    return f"{rel}:{frame.lineno}:{frame.name}"


def _is_ours(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(seg in p for seg in ("/env/", "/bridge/", "/rl/", "/sim/")) and (
        "site-packages" not in p
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    out_dir = args.out or os.path.join(
        "rl", "bughunt_out", datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    crashes_path = os.path.join(out_dir, "crashes.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    print(f"[bughunt] output -> {out_dir}")

    model = MaskablePPO.load(args.checkpoint, device=args.device)
    env = SlayTheSpireEnv(
        ascension=args.ascension,
        global_map_view=_model.uses_global_map_observation(model),
        combat_language=_model.uses_combat_language(model),
    )

    # dedup table: signature -> {count, type, message, examples:[{seed,step,floor}], first_tb}
    seen: dict[str, dict] = {}
    stats = {
        "episodes": 0,
        "wins": 0,
        "crashes": 0,
        "stalls": 0,
        "floors_reached": defaultdict(int),
        "steps_total": 0,
    }
    t0 = time.time()
    crashes_file = open(crashes_path, "a", encoding="utf-8")

    def log_crash(kind, seed, ep, step, action, floor, act, exc):
        tb_list = traceback.extract_tb(exc.__traceback__)
        sig = crash_signature(tb_list)
        full_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        rec = {
            "kind": kind,
            "signature": sig,
            "exc_type": type(exc).__name__,
            "message": str(exc),
            "seed": seed,
            "episode": ep,
            "step": step,
            "action": action,
            "floor": floor,
            "act": act,
            "traceback": full_tb,
        }
        crashes_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        crashes_file.flush()
        e = seen.get(sig)
        if e is None:
            seen[sig] = {
                "count": 1,
                "exc_type": type(exc).__name__,
                "message": str(exc),
                "examples": [{"seed": seed, "step": step, "floor": floor}],
                "first_tb": full_tb,
            }
            print(f"\n[NEW CRASH] {sig}  {type(exc).__name__}: {exc}"
                  f"  (seed={seed} floor={floor})")
        else:
            e["count"] += 1
            if len(e["examples"]) < 5:
                e["examples"].append({"seed": seed, "step": step, "floor": floor})

    def write_summary():
        ranked = sorted(seen.items(), key=lambda kv: kv[1]["count"], reverse=True)
        summary = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "elapsed_sec": round(time.time() - t0, 1),
            "episodes": stats["episodes"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / max(1, stats["episodes"]), 4),
            "crashes": stats["crashes"],
            "stalls": stats["stalls"],
            "distinct_signatures": len(seen),
            "floors_reached": dict(sorted(stats["floors_reached"].items())),
            "by_signature": [
                {"signature": s, **{k: v for k, v in d.items() if k != "first_tb"}}
                for s, d in ranked
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def print_progress():
        rate = stats["episodes"] / max(1e-9, time.time() - t0)
        top = sorted(seen.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
        print(
            f"[{stats['episodes']}/{args.episodes}] "
            f"win={stats['wins']} crash={stats['crashes']} "
            f"stall={stats['stalls']} distinct={len(seen)} "
            f"{rate:.1f} ep/s"
        )
        for sig, d in top:
            print(f"    x{d['count']:<5} {sig}  {d['exc_type']}: {d['message'][:80]}")

    for ep in range(args.episodes):
        seed = args.seed + ep
        if args.mode == "mixed":
            deterministic = (ep % 2 == 0)
        else:
            deterministic = (args.mode == "deterministic")

        step = 0
        last_action = None
        floor = act = None
        try:
            obs, info = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                if step >= args.max_steps:
                    stats["stalls"] += 1
                    rs = info.get("run_stats", {})
                    print(f"\n[STALL] seed={seed} ep={ep} hit max-steps "
                          f"floor={rs.get('floor')}")
                    break
                masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=masks,
                                          deterministic=deterministic)
                last_action = int(np.asarray(action).item())
                rs = info.get("run_stats", {})
                floor, act = rs.get("floor"), rs.get("act")
                obs, reward, terminated, truncated, info = env.step(last_action)
                step += 1
        except Exception as exc:  # noqa: BLE001 — the whole point is to catch all
            stats["crashes"] += 1
            log_crash("step", seed, ep, step, last_action, floor, act, exc)
            if args.stop_on_crash:
                stats["episodes"] += 1
                print_progress()
                write_summary()
                crashes_file.close()
                env.close()
                print(f"\n[bughunt] stop-on-crash hit. summary -> {summary_path}")
                return
        else:
            # episode finished cleanly (or stalled); tally outcome
            try:
                stats["wins"] += int(is_win(env))
                rs = info.get("run_stats", {})
                if rs.get("floor") is not None:
                    stats["floors_reached"][int(rs["floor"])] += 1
            except Exception:  # noqa: BLE001
                pass
        stats["episodes"] += 1
        stats["steps_total"] += step

        if (ep + 1) % args.report_every == 0:
            print_progress()
            write_summary()

    print_progress()
    write_summary()
    crashes_file.close()
    env.close()
    print(f"\n[bughunt] done. summary -> {summary_path}")


if __name__ == "__main__":
    main()

"""Paired Gumbel-budget repetitions for the fixed Bash/Clash fixture.

The public combat state is held constant while each repetition receives a
different NOSL nonce.  This makes search-budget and Gumbel-noise effects
measurable without changing the opening hand or encounter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from rl.evaluation.nosl.collect_nosl_corpus import _load_custom_objects
from rl.evaluation.nosl.eval_nosl_order_sensitivity import (
    _jsonable,
    _make_env,
    _run_search,
    SCENARIOS,
)
from env.search.nosl_sampler import NOSLSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument("--simulations", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--gumbel-candidates", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    # The Bash+Clash fixture uses a normal draw pile of eight Strikes after
    # the opening hand, so future draws remain non-unique.
    env = _make_env(
        model,
        SCENARIOS["bash_plus_clash"],
        swapped=False,
        seed=int(args.seed),
        filler_cards=("STRIKE_RED",) * 8,
    )
    sampler = NOSLSampler(0x4E4F534C_7631)
    rows = []
    for repetition in range(max(1, int(args.repeats))):
        nonce = sampler.nonce(
            state_nonce=int(args.seed),
            simulation_index=100_000 + repetition,
        )
        for simulations in args.simulations:
            result = _run_search(
                model,
                env,
                simulations=int(simulations),
                depth=0,
                nonce=int(nonce),
                leaf_batch=int(args.leaf_batch),
                prior_mode="model",
                root_search_mode=1,
                gumbel_candidates=int(args.gumbel_candidates),
            )
            root = result["search"]["tree_root"]
            q = {}
            visits = result["search"]["root_visits"]
            for child in root.get("children", ()):
                label = str(child.get("label", ""))
                if "Bash" in label:
                    q["bash"] = float(child.get("value", 0.0))
                elif "Clash" in label:
                    q["clash"] = float(child.get("value", 0.0))
            rows.append({
                "repetition": repetition,
                "nonce": int(nonce),
                "simulations": int(simulations),
                "action": result["search"]["action"],
                "bash_visits": int(sum(
                    int(v) for k, v in visits.items() if "BASH" in str(k)
                )),
                "clash_visits": int(sum(
                    int(v) for k, v in visits.items() if "CLASH" in str(k)
                )),
                "bash_q": q.get("bash"),
                "clash_q": q.get("clash"),
                "root_value": float(result["search"]["root_value"]),
            })

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": int(args.seed),
        "scenario": "bash_plus_clash",
        "depth": 0,
        "leaf_batch": int(args.leaf_batch),
        "gumbel_candidates": int(args.gumbel_candidates),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

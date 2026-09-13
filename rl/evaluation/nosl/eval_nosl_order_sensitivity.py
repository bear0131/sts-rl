"""Measure C++ NOSL-IS-AZ sensitivity to hand-slot order and search depth.

The fixtures are real simulator combats.  The two relevant cards occupy the
same two deck positions and are swapped between layouts; all other cards,
seed, encounter, HP, and draw-pile composition stay fixed.  The remaining
deck is made of varied ordinary Ironclad cards so the next draw can change the
future damage/block value; the layouts differ only in the two target slots.

Example::

    python -m rl.evaluation.nosl.eval_nosl_order_sensitivity \
        artifacts/latest-nosl-az-s512-20260905/nosl_az_614400_steps.zip \
        --simulations 128 --depths 1 2 3 4 --depth0-simulations 512 \
        --output .tmp-nosl-order-sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from rl.policies.checkpoint import custom_objects as _load_custom_objects
from env.search.nosl_sampler import NOSLSampler


from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


from configs.eval.nosl_scenarios import FILLER_CARDS, SCENARIOS


from rl.evaluation.nosl.scenarios import make_env, card_dict, jsonable, run_search


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--depth0-simulations", type=int, default=512)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument(
        "--prior-mode",
        choices=("model", "uniform", "none"),
        default="model",
        help=(
            "root prior ablation: model logits (default), uniform legal prior, "
            "or none (uniform plus c_puct=0)"
        ),
    )
    parser.add_argument(
        "--root-search-mode",
        choices=("puct", "gumbel"),
        default="puct",
        help="root budget allocator; descendants remain PUCT",
    )
    parser.add_argument(
        "--gumbel-candidates",
        type=int,
        default=0,
        help="number of Gumbel-ranked root candidates; 0 means all legal roots",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = MaskablePPO.load(
        args.checkpoint,
        device=args.device,
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": int(args.seed),
        "encounter": "Cultist",
        "draw_size": 15,
        "draw_is_non_unique": True,
        "scenarios": {},
    }
    started = time.perf_counter()
    for scenario, pair in SCENARIOS.items():
        rows = []
        fillers = (
            ("STRIKE_RED",) * 8
            if scenario == "bash_plus_clash" else FILLER_CARDS
        )
        for swapped in (False, True):
            env = make_env(model, pair, swapped, args.seed, fillers)
            raw = env.thick.bc.to_dict()
            hand = [card_dict(card) for card in raw["hand"]]
            settings = [(int(args.simulations), depth) for depth in args.depths]
            settings.append((int(args.depth0_simulations), 0))
            for simulations, depth in settings:
                print(
                    f"[order] {scenario} layout={'swapped' if swapped else 'base'} "
                    f"depth={depth} simulations={simulations} "
                    f"prior={args.prior_mode}", flush=True
                )
                row = {
                    "scenario": scenario,
                    "layout": "swapped" if swapped else "base",
                    "seed": int(args.seed),
                    "hand": hand,
                    "draw": [card_dict(card) for card in raw["draw"]],
                    "draw_size": len(raw["draw"]),
                }
                search = run_search(
                    model,
                    env,
                    simulations=simulations,
                    depth=depth,
                    prior_mode=args.prior_mode,
                    nonce=NOSLSampler(0x4E4F534C_7631).nonce(
                        state_nonce=args.seed,
                        simulation_index=(len(rows) + (1 if swapped else 0) * 10000),
                    ),
                    leaf_batch=args.leaf_batch,
                    root_search_mode=(1 if args.root_search_mode == "gumbel" else 0),
                    gumbel_candidates=args.gumbel_candidates,
                )
                row.update(search)
                rows.append(row)
            del env
        output["scenarios"][scenario] = rows
    output["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[order] wrote {args.output.resolve()} elapsed={output['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()

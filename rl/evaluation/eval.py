"""Evaluate SB3 strategies using the shared bounded, batched episode runner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as policy_model
from rl.policies.checkpoint import load_checkpoint
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate, summarize
from rl.evaluation.strategies import STRATEGIES, MapRouteStrategy, PhaseStrategy
from rl.search.combat_value_search import add_combat_search_args


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--card-vector-checkpoint', type=Path)
    parser.add_argument('--task', choices=['wholerun', 'combat'], default='wholerun')
    parser.add_argument('--solver', choices=tuple(STRATEGIES), default='policy')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--ascension', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--timeout', type=float, default=1800.0)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--route', choices=['model', 'min-elite-max-rest'], default='model')
    parser.add_argument('--progress-every', type=int, default=1)
    parser.add_argument('--heartbeat-seconds', type=float, default=30.0)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--output', type=Path, help='Structured per-episode rows and summary JSON')
    parser.add_argument('--floor-lo', type=int, default=1)
    parser.add_argument('--floor-hi', type=int, default=50)
    parser.add_argument('--search-mode', choices=['puct', 'gumbel', 'full', 'strict'], default='strict')
    parser.add_argument('--simulations', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--leaf-batch-size', type=int)
    add_combat_search_args(parser)
    return parser


def make_scenario(model, args):
    if args.task == 'combat':
        from rl.tasks.combat import CombatTask
        if args.solver == 'native':
            raise ValueError('native search currently requires the wholerun thin environment')
        if not 1 <= args.floor_lo <= args.floor_hi <= 50:
            raise ValueError('expected 1 <= floor-lo <= floor-hi <= 50')
        options = argparse.Namespace(floor_lo=args.floor_lo, floor_hi=args.floor_hi,
            ascension=args.ascension, combat_hp_loss_coef=0.0,
            card_language=policy_model.uses_card_language(model),
            combat_language=policy_model.uses_combat_language(model))

        def outcome(env, info, terminated, truncated):
            base = env.unwrapped.thick
            return dict(result='other_truncation' if truncated else 'finished',
                        floor=int(base.gc.floor_num), act=int(base.gc.act),
                        combats_cleared=int(base.combats_won))
        return Scenario('combat', CombatTask().make_env_fn(options), outcome)
    return Scenario('wholerun', lambda: SlayTheSpireEnv(
        ascension=args.ascension,
        global_map_view=policy_model.uses_global_map_observation(model),
        card_language=policy_model.uses_card_language(model),
        combat_language=policy_model.uses_combat_language(model),
        deck_size_observation=policy_model.uses_deck_size_observation(model),
        neow_observation=policy_model.uses_neow_observation(model),
        event_option_observation=policy_model.uses_event_option_observation(model)))


def main():
    args = build_parser().parse_args()
    config = EvaluationConfig(args.episodes, args.seed, args.num_envs, args.max_steps, args.timeout)
    if args.progress_every < 0 or args.heartbeat_seconds < 0:
        raise ValueError('progress interval must be non-negative')
    if args.combat_search_depth > 0:
        if args.solver not in ('policy', 'value'):
            raise ValueError('--combat-search cannot be combined with a different solver')
        args.solver = 'value'
    model = load_checkpoint(args.checkpoint, device=args.device, card_vector_checkpoint=args.card_vector_checkpoint)
    model.set_random_seed(args.seed)
    strategy = STRATEGIES[args.solver](model, args)
    if args.route != 'model':
        strategy = PhaseStrategy(strategy, {'map': MapRouteStrategy()})
    started = last_report = time.monotonic()

    def progress(event):
        nonlocal last_report
        if args.heartbeat_seconds and time.monotonic() - last_report >= args.heartbeat_seconds:
            print(f'[eval] {event}', flush=True)
            last_report = time.monotonic()

    rows = []
    failure = None
    try:
        for row in evaluate(strategy, make_scenario(model, args), config, progress=progress):
            rows.append(row)
            if args.progress_every and len(rows) % args.progress_every == 0:
                print(f"[eval] {len(rows)}/{args.episodes}: seed={row['seed']} result={row['result']} floor={row['floor']} steps={row['steps']}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        rows.sort(key=lambda row: row['episode'])
        summary = summarize(rows)
        payload = dict(schema_version=1, task=args.task, strategy=strategy.name,
                       state="error" if failure else "complete", error=failure,
                       checkpoint=str(args.checkpoint.resolve()), config=vars(args),
                       elapsed_seconds=time.monotonic() - started, summary=summary, rows=rows)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted({key for row in rows for key in row})
            with args.output_csv.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)


if __name__ == '__main__':
    main()

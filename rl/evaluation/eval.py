"""Evaluate SB3 strategies using the shared bounded, batched episode runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as policy_model
from rl.policies.checkpoint import load_checkpoint
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate, summarize
from rl.evaluation.trajectory import EpisodeTraceCollector, render_episode_markdown
from rl.evaluation.strategies import STRATEGIES, compose_strategy, strategy_manifest
from rl.search.combat_value_search import add_combat_search_args


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--combat-checkpoint', type=Path,
                        help='Separate combat model; positional checkpoint remains the upper policy')
    parser.add_argument('--compare-combats', type=int, default=0,
                        help='Audit first N combat starts against rules, policy and native on identical forks')
    parser.add_argument('--trace', type=Path, help='Combat states, chosen actions and rule explanations as JSONL')
    parser.add_argument('--card-vector-checkpoint', type=Path)
    parser.add_argument('--task', choices=['wholerun', 'combat'], default='wholerun')
    parser.add_argument('--solver', choices=tuple(STRATEGIES), default='policy')
    parser.add_argument('--upper-strategy', choices=['model', 'rules'], default='model',
                        help='Keep the upper checkpoint, or use overworld rules with explicit model fallback')
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
    parser.add_argument('--trajectory-output', type=Path,
                        help='Directory for sparse episode traces (episodes.jsonl, report.json, summary.md)')
    parser.add_argument('--trajectory-format', choices=['jsonl'], default='jsonl')
    parser.add_argument('--trajectory-detail', choices=['sparse'], default='sparse')
    parser.add_argument('--floor-lo', type=int, default=1)
    parser.add_argument('--floor-hi', type=int, default=50)
    parser.add_argument('--search-mode', choices=['puct', 'gumbel', 'full', 'strict'], default='strict')
    parser.add_argument('--prior-temperature', type=float,
                        help='Override the native policy-prior temperature for evaluation')
    parser.add_argument('--simulations', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--planner-nodes', type=int, default=512)
    parser.add_argument('--planner-worlds', type=int, default=2)
    parser.add_argument('--planner-beam', type=int, default=8)
    parser.add_argument('--planner-depth', type=int, default=16)
    parser.add_argument('--planner-turns', type=int, choices=(1, 2), default=1)
    parser.add_argument('--planner-nonce', type=int, default=20260918)
    parser.add_argument('--leaf-batch-size', type=int)
    parser.add_argument('--leaf-inference', choices=['eager', 'cuda-graph'], default='eager',
                        help='Native leaf backend; cuda-graph requires a CUDA FP32 combat model and keeps exact batch shapes')
    parser.add_argument('--fixed-worldline', action=argparse.BooleanOptionalAction, default=None,
                        help='Keep one redeterminized future per search tree; default inherits the checkpoint setting')
    add_combat_search_args(parser)
    return parser


def make_scenario(model, args, combat_model=None, combat_action_contract='legacy'):
    if args.task == 'combat':
        from rl.tasks.combat import CombatTask
        if args.solver in ('native', 'rules', 'turn-planner'):
            raise ValueError('native/rules/turn-planner currently require the wholerun environment')
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
    models = [model] + ([combat_model] if combat_model is not None else [])
    def uses(feature):
        return any(feature(m) for m in models)
    return Scenario('wholerun', lambda: SlayTheSpireEnv(
        ascension=args.ascension,
        combat_action_contract=combat_action_contract,
        global_map_view=uses(policy_model.uses_global_map_observation),
        card_language=uses(policy_model.uses_card_language),
        combat_language=uses(policy_model.uses_combat_language),
        deck_size_observation=uses(policy_model.uses_deck_size_observation),
        neow_observation=uses(policy_model.uses_neow_observation),
        event_option_observation=uses(policy_model.uses_event_option_observation)))


def file_identity(path):
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.exists() and path.with_suffix('.zip').exists():
        path = path.with_suffix('.zip')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


def main():
    args = build_parser().parse_args()
    config = EvaluationConfig(args.episodes, args.seed, args.num_envs, args.max_steps, args.timeout)
    if args.progress_every < 0 or args.heartbeat_seconds < 0:
        raise ValueError('progress interval must be non-negative')
    if args.compare_combats < 0 or ((args.compare_combats or args.trace) and args.task != 'wholerun'):
        raise ValueError('combat auditing requires wholerun and a non-negative combat count')
    if args.compare_combats and args.stochastic:
        raise ValueError('paired combat auditing requires deterministic policy decisions')
    if args.upper_strategy != 'model' and args.task != 'wholerun':
        raise ValueError('--upper-strategy requires wholerun')
    if args.combat_search_depth > 0:
        if args.solver not in ('policy', 'value'):
            raise ValueError('--combat-search cannot be combined with a different solver')
        args.solver = 'value'
    if args.leaf_inference != 'eager' and (args.solver != 'native' or args.task != 'wholerun'):
        raise ValueError('--leaf-inference cuda-graph requires --solver native --task wholerun')
    model = load_checkpoint(args.checkpoint, device=args.device, card_vector_checkpoint=args.card_vector_checkpoint)
    model.set_random_seed(args.seed)
    combat_model = model
    if args.combat_checkpoint:
        if args.task != 'wholerun' or args.solver not in ('policy', 'native', 'value', 'rules', 'turn-planner'):
            raise ValueError('--combat-checkpoint requires a supported wholerun solver')
        combat_model = load_checkpoint(args.combat_checkpoint, device=args.device,
                                       card_vector_checkpoint=args.card_vector_checkpoint)
        combat_model.set_random_seed(args.seed)
    trace_stream = None
    def trace(event):
        if trace_stream:
            trace_stream.write(json.dumps(event)+'\n')
    strategy = compose_strategy(model, args, combat_model if args.combat_checkpoint else None, trace=trace)
    configured_strategy = strategy
    candidate_strategies = {}
    started = last_report = time.monotonic()

    def progress(event):
        nonlocal last_report
        if args.heartbeat_seconds and time.monotonic() - last_report >= args.heartbeat_seconds:
            print(f'[eval] {event}', flush=True)
            last_report = time.monotonic()

    rows = []
    combat_rows = []
    failure = None
    trace_factory = (lambda env, episode, seed: EpisodeTraceCollector(env, episode, seed)
                     if args.trajectory_output else None)
    try:
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace_stream = args.trace.open('x', encoding='utf-8')
        if args.compare_combats or (args.trace and args.solver != 'turn-planner'):
            from rl.evaluation.baselines.combat_rules import CombatRuleStrategy
            from rl.evaluation.baselines.combat_audit import CombatAuditStrategy
            strategy = CombatAuditStrategy(strategy, rows=combat_rows, limit=args.compare_combats,
                candidates={'rules': CombatRuleStrategy(),
                            'policy': STRATEGIES['policy'](combat_model, args),
                            'native': STRATEGIES['native'](combat_model, args)} if args.compare_combats else {},
                max_steps=args.max_steps, timeout=args.timeout,
                trace=trace if trace_stream else None)
            candidate_strategies = strategy.candidates
        contract = getattr(configured_strategy, 'combat_action_contract', 'legacy')
        for row in evaluate(
                strategy,
                make_scenario(model, args, combat_model, contract),
                config,
                progress=progress, trace_factory=trace_factory):
            rows.append(row)
            if args.progress_every and len(rows) % args.progress_every == 0:
                print(f"[eval] {len(rows)}/{args.episodes}: seed={row['seed']} result={row['result']} floor={row['floor']} steps={row['steps']}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if trace_stream:
            trace_stream.close()
        rows.sort(key=lambda row: row['episode'])
        summary = summarize(rows)
        payload = dict(schema_version=1, task=args.task, strategy=strategy.name,
                       state="error" if failure else "complete", error=failure,
                       checkpoint=str(args.checkpoint.resolve()), config=vars(args),
                       elapsed_seconds=time.monotonic() - started, summary=summary, rows=rows)
        import slaythespire
        from rl.evaluation.baselines.combat_audit import summarize_pairs
        from rl.evaluation.baselines.combat_rules import RULE_VERSION
        payload.update(identities=dict(upper=file_identity(args.checkpoint),
            combat=file_identity(args.combat_checkpoint or args.checkpoint),
            card_encoder=file_identity(args.card_vector_checkpoint),
            native=file_identity(slaythespire.__file__)), rule_version=RULE_VERSION,
            effective_strategy=strategy_manifest(configured_strategy),
            combat_candidates={name: strategy_manifest(solver) for name, solver in candidate_strategies.items()},
            combat_rows=combat_rows, combat_comparison=summarize_pairs(combat_rows))
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
        if args.trajectory_output:
            out_dir = args.trajectory_output
            out_dir.mkdir(parents=True, exist_ok=True)
            episodes_path = out_dir / 'episodes.jsonl'
            with episodes_path.open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row.get('trajectory', {}),
                                            default=str, ensure_ascii=False) + '\n')
            with (out_dir / 'episodes.md').open('w', encoding='utf-8') as stream:
                stream.write('# 对局轨迹记录\n\n')
                for row in rows:
                    stream.write(render_episode_markdown(row.get('trajectory', {})) + '\n')
            report = dict(schema_version=1, task=args.task,
                          strategy=strategy.name, config=vars(args),
                          summary=summary, rows=rows)
            (out_dir / 'report.json').write_text(
                json.dumps(report, default=str, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8')
            result_counts = ', '.join(f'{k}={v}' for k, v in sorted(summary['results'].items()))
            (out_dir / 'summary.md').write_text(
                '# 轨迹评测报告\n\n'
                f'- 对局数：{summary["completed"]}\n'
                f'- 结果：{result_counts}\n'
                f'- 胜率（Wilson 95% 区间）：{summary["win_rate"]:.1%} '
                f'（{summary["win_ci95"][0]:.1%}–{summary["win_ci95"][1]:.1%}）\n'
                f'- 结束楼层均值/中位数：{summary["floor_mean"]:.2f}/{summary["floor_median"]:.2f}\n'
                f'- 到达 Act1/Act2/Act3：{summary["act1_rate"]:.1%}/{summary["act2_rate"]:.1%}/{summary["act3_rate"]:.1%}\n'
                f'- 策略：`{strategy.name}`\n'
                f'- 轨迹文件：[`episodes.jsonl`](episodes.jsonl)、[`episodes.md`](episodes.md)\n',
                encoding='utf-8')


if __name__ == '__main__':
    main()

"""Bounded upper-PPO diagnostic from a protected bundle; never resumes its run.

Uses the production collector/update. Loss ablations operate on cloned in-memory
state. The original mode is local world_size=1; unified candidate mode uses the
real distributed full-state restore and a whole-game holdout, without saving updates.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from env.core import action_space as A, vocab as V
from rl.algorithms import upper_ppo as U


class DiagnosticStop(RuntimeError):
    pass


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'unsupported diagnostic JSON type: {type(value).__name__}')


def subset_learner(state, indices):
    """Retain pending credit and episode identity when selecting live actors."""
    indices = [int(i) for i in indices]
    count = len(state['last_episode_starts'])
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError('invalid distinct source environment indices')
    result = {}
    for key in ('last_episode_starts', 'upper_pending', 'upper_episode_ids', 'nosl_is_az_pending_fights'):
        source = state.get(key)
        if source is None:
            result[key] = None
        else:
            if len(source) != count:
                raise ValueError(f'{key}: recovery actor count mismatch')
            result[key] = [copy.deepcopy(source[i]) for i in indices]
    for index, pending in enumerate(result['upper_pending'] or []):
        if pending is not None:
            pending['env'] = index
    return result


def loss_arm(components, arm):
    if arm not in ('full', 'policy', 'value', 'entropy', 'momentum'):
        raise ValueError(arm)
    # Zero rather than detach: retain the same active Adam parameters so the
    # separate momentum arm exposes movement not caused by the current loss.
    return replace(components, **{name: getattr(components, name) * int(arm in ('full', keep))
        for name, keep in [('policy_loss', 'policy'), ('value_loss', 'value'), ('entropy_loss', 'entropy')]})


def should_stop(baseline, current, drop):
    return next((f'{key}_drop' for key in ('open_probability', 'take_probability')
        if baseline.get(key) is not None and current.get(key) is not None
        and baseline[key] - current[key] >= drop), None)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


@contextmanager
def evaluation_state(model):
    rng, training = rng_state(), model.policy.training
    try:
        model.policy.set_training_mode(False)
        with torch.no_grad(), model.autocast_context():
            yield
    finally:
        model.policy.set_training_mode(training)
        set_rng(rng)


def evaluate(model, rows):
    probabilities, values = [], []
    with evaluation_state(model):
        for start in range(0, len(rows), 128):
            batch = rows[start:start+128]
            obs = {k: torch.as_tensor(np.stack([r['observation'][k] for r in batch]), device=model.device)
                   for k in batch[0]['observation']}
            logits, value = model.policy.core.actor_critic(obs)
            masks = torch.as_tensor(np.stack([r['mask'] for r in batch]), device=model.device)
            probabilities.extend(logits.float().masked_fill(~masks, -1e9).softmax(-1).cpu().numpy())
            values.extend(value.flatten().float().cpu().tolist())
    return np.asarray(probabilities), np.asarray(values)


def make_panel(model, directory):
    """Recreate real take/skip branches; no manufactured feature edits."""
    from env.core import native_actions as C
    from env.core.run_env import SlayTheSpireEnv
    from rl.policies import model as M
    original = torch.load(directory/'decision-records.pt', weights_only=False)
    chest = [r for r in original if int(r['observation']['phase'][0]) == V.PHASE_VOCAB_INDEX['chest']][:128]
    relic = [r for r in original if r['mask'][A._RRELIC:A._RRELIC+A.REWARD_RELIC].any()][:128]
    rows = chest + relic
    pairs, seen = [], set()
    def new_env():
        return SlayTheSpireEnv(combat_action_contract=C.policy_contract(model.observation_space),
            global_map_view=M.uses_global_map_observation(model), card_language=M.uses_card_language(model),
            combat_language=M.uses_combat_language(model), deck_size_observation=M.uses_deck_size_observation(model),
            neow_observation=M.uses_neow_observation(model), event_option_observation=M.uses_event_option_observation(model))
    for panel in torch.load(directory/'panel.pt', weights_only=False):
        if panel['kind'] != 'relic':
            continue
        env = new_env()
        try:
            env.restore_training_state(panel['state'])
            for _ in range(4):
                gold = [int(a) for a in np.flatnonzero(env.action_masks()) if A._RGOLD <= a < A._RGOLD+A.REWARD_GOLD]
                if not gold:
                    break
                env.step(gold[0])
            common = env.export_training_state()
            action = next(int(a) for a in np.flatnonzero(env.action_masks()) if A._RRELIC <= a < A._RRELIC+A.REWARD_RELIC)
            branches = []
            for act in (action, A._PROCEED):
                env.restore_training_state(common)
                assert env.action_masks()[act]
                env.step(act)
                for _ in range(8):
                    if env._phase() != 'combat_reward':
                        break
                    mask = env.action_masks()
                    gold = [int(a) for a in np.flatnonzero(mask) if A._RGOLD <= a < A._RGOLD+A.REWARD_GOLD]
                    if gold:
                        env.step(gold[0])
                    elif mask[A._PROCEED]:
                        env.step(A._PROCEED)
                    else:
                        break
                branches.append((env._phase(), {k: v.copy() for k, v in env._obs().items() if k in U.UPPER_KEYS}, env.action_masks().copy()))
            if any(b[0] != 'map' for b in branches):
                continue
            key = hashlib.sha256(b''.join(v.tobytes() for b in branches for _, v in sorted(b[1].items()))).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((len(rows), len(rows)+1))
            rows.extend(dict(observation=b[1], mask=b[2]) for b in branches)
        finally:
            env.close()
    if not chest or not relic or not pairs:
        raise ValueError('fixed panel missing a required decision/paired-state group')
    return rows, len(chest), len(relic), pairs


def panel_summary(model, panel):
    rows, nc, nr, pairs = panel
    ps, vs = evaluate(model, rows)
    delta = np.array([vs[a]-vs[b] for a, b in pairs])
    return dict(open_probability=float(ps[:nc, A._CHEST].mean()),
        take_probability=float(ps[nc:nc+nr, A._RRELIC:A._RRELIC+A.REWARD_RELIC].sum(1).mean()),
        paired_value_delta=float(delta.mean()), paired_negative_fraction=float((delta < 0).mean()),
        chest_count=nc, relic_count=nr, pairs=len(pairs))


def summarize_targets(model, records, bootstrap, advantages, returns):
    norm = (advantages-advantages.mean())/(advantages.std()+1e-8)
    delta = np.array([r['reward'] + model.policy.core.upper_config.gamma*bootstrap[i]-r['value'] for i, r in enumerate(records)])
    groups = {}
    for name, positive, eligible in (
        ('chest', lambda a: a == A._CHEST, lambda r: int(r['observation']['phase'][0]) == V.PHASE_VOCAB_INDEX['chest']),
        ('relic_offer', lambda a: A._RRELIC <= a < A._RRELIC+A.REWARD_RELIC, lambda r: r['mask'][A._RRELIC:A._RRELIC+A.REWARD_RELIC].any())):
        ids = [i for i, r in enumerate(records) if eligible(r)]
        group = {}
        for label in ('take', 'other'):
            chosen = [i for i in ids if positive(records[i]['action']) == (label == 'take')]
            group[label] = dict(count=len(chosen))
            if chosen:
                group[label].update(raw_advantage=float(advantages[chosen].mean()), normalized_advantage=float(norm[chosen].mean()),
                    positive_fraction=float((advantages[chosen] > 0).mean()), td=float(delta[chosen].mean()),
                    reward=float(np.mean([records[i]['reward'] for i in chosen])), old_value=float(np.mean([records[i]['value'] for i in chosen])),
                    next_value=float(bootstrap[chosen].mean()), target=float(returns[chosen].mean()))
        if ids:
            ps, _ = evaluate(model, [records[i] for i in ids])
            p = ps[:, A._CHEST] if name == 'chest' else ps[:, A._RRELIC:A._RRELIC+A.REWARD_RELIC].sum(1)
            taken = np.array([positive(records[i]['action']) for i in ids])
            group['selected_logit_gradient_proxy'] = float(np.mean(norm[ids]*(taken-p)))
            group['raw_selected_logit_gradient_proxy'] = float(np.mean(advantages[ids]*(taken-p)))
        groups[name] = group
    return dict(count=len(records), mean=float(advantages.mean()), std=float(advantages.std()), groups=groups)


def parameter_group(name):
    if 'upper_tokenizer' in name:
        return 'tokenizer'
    if 'encoder' in name:
        return 'encoder'
    if 'value' in name or 'vf' in name:
        return 'value_head'
    return 'policy_head'


def horizon_advantages(records, bootstrap, gamma, gae_lambda, horizon, original):
    """Change credit horizon on exactly the same completed macro records."""
    if horizon is None:
        # Recomputed BF16 V(s') need not equal the saved next row's V(s).
        # MC must not inherit that numerical residual through lambda=1 GAE.
        returns = np.zeros(len(records), np.float32)
        tails = {}
        for i in range(len(records)-1, -1, -1):
            row = records[i]; key = (row['env'], row['episode'])
            if not row['terminal'] and key not in tails:
                raise ValueError('Monte Carlo target requires complete terminal episodes')
            tails[key] = row['reward'] + (0. if row['terminal'] else gamma*tails[key])
            returns[i] = tails[key]
        return returns-np.array([r['value'] for r in records], np.float32), returns
    adv = np.zeros(len(records), np.float32)
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault((record['probe_end']-1)//horizon, []).append(i)
    for indices in groups.values():
        adv[indices], _ = original([records[i] for i in indices], bootstrap[indices], gamma, gae_lambda)
    return adv, adv + np.array([r['value'] for r in records], np.float32)


def compare_horizons(model, env, callback, args, panel, report, publish, check):
    """Fixed complete episodes: isolate label horizon, not sample count or LR."""
    import rl.algorithms.accumulating_ppo as collector
    initial_ids = list(model._upper_episode_ids)
    capture = collector.capture_step
    tick = 0

    def timed_capture(*a, **kw):
        nonlocal tick
        before = len(model._upper_completed)
        result = capture(*a, **kw)
        tick += 1
        for row in model._upper_completed[before:]:
            row['probe_end'] = tick
        return result

    collector.capture_step = timed_capture
    try:
        callback.on_training_start({}, {})
        assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=512)
    finally:
        collector.capture_step = capture
    assert tick == 512
    all_records = model._upper_completed
    terminal = {(r['env'], r['episode']) for r in all_records if r['terminal']}
    records = [r for r in all_records if r['episode'] > initial_ids[r['env']] and (r['env'], r['episode']) in terminal]
    assert records
    report['collection'] = dict(steps=512, transitions=512*args.envs, all_macros=len(all_records),
        complete_macros=len(records), complete_episodes=len({(r['env'], r['episode']) for r in records}))
    base, opt, rng = cpu_copy(model.policy.state_dict()), cpu_copy(model.policy.upper_optimizer.state_dict()), rng_state()
    cfg = model.policy.core.upper_config
    original_adv, original_loss, original_step = U.macro_advantages, U.ppo_loss_components, U.optimizer_step
    arm, horizon = 'n128', 128

    def reset():
        model.policy.load_state_dict(base)
        model.policy.upper_optimizer.load_state_dict(copy.deepcopy(opt))
        model.policy.upper_optimizer.zero_grad(set_to_none=True)
        model.policy.set_training_mode(False)
        model._upper_completed = records
        set_rng(rng)

    def targets(rows, bootstrap, gamma, lam):
        adv, ret = horizon_advantages(rows, bootstrap, gamma, lam, horizon, original_adv)
        report.setdefault('horizon_targets', {})[arm] = summarize_targets(model, rows, bootstrap, adv, ret)
        np.savez_compressed(args.output/f'horizon-{arm}.npz', advantage=adv, returns=ret, bootstrap=bootstrap,
            reward=[r['reward'] for r in rows], value=[r['value'] for r in rows], end=[r['probe_end'] for r in rows],
            env=[r['env'] for r in rows], episode=[r['episode'] for r in rows], terminal=[r['terminal'] for r in rows],
            action=[r['action'] for r in rows], phase=[int(r['observation']['phase'][0]) for r in rows])
        return adv, ret

    def traced_step(*a, **kw):
        check()
        result = original_step(*a, **kw)
        report['updates'].append(dict(arm=arm, probe=panel_summary(model, panel)))
        publish()
        return result

    try:
        U.macro_advantages, U.optimizer_step = targets, traced_step
        U.ppo_loss_components = lambda **kw: loss_arm(original_loss(**kw), 'momentum' if arm == 'momentum' else args.horizon_loss)
        for arm in args.horizon_arms:
            horizon = {'momentum': 128, 'n128': 128, 'n512': 512, 'mc': None}[arm]
            check(); reset()
            model.policy.core.upper_config = replace(cfg, epochs=args.component_epochs)
            stats = U.update_upper(model)
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
            report['ablations'].append(dict(arm=arm, stats=stats, probe=panel_summary(model, panel)))
            print('HORIZON', arm, json.dumps(report['ablations'][-1]), flush=True)
            publish()
    finally:
        U.macro_advantages, U.ppo_loss_components, U.optimizer_step = original_adv, original_loss, original_step
        reset(); model.policy.core.upper_config = cfg
    report.update(state='complete', stop_reason='horizon_budget', final_probe=panel_summary(model, panel),
        restored_exact=all(torch.equal(v.cpu(), base[k]) for k, v in model.policy.state_dict().items()),
        final_adam_steps=sorted({int(s['step']) for s in model.policy.upper_optimizer.state.values()}))
    publish()


def measure_frozen_rollout(model, collect, *args, **kwargs):
    """Time the real collector, including first-use work, without learner updates."""
    from rl.algorithms.upper_probe import state_digest
    from rl.training.resource_monitor import ResourceMonitor
    def identity():
        return dict(policy=state_digest(model.policy.state_dict()),
                    upper=state_digest(model.policy.upper_optimizer.state_dict()),
                    lower=state_digest(model.policy.small_playout_optimizer.state_dict()))
    before = identity()
    cuda = model.device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    monitor = ResourceMonitor(interval=1., gpu=cuda, torch_device=model.device)
    with monitor:
        started = time.perf_counter()
        ok = collect(model, *args, **kwargs)
        if cuda:
            torch.cuda.synchronize(model.device)
        seconds = time.perf_counter() - started
    after = identity()
    if not ok or before != after:
        raise AssertionError('frozen rollout failed or changed policy/Adam')
    stats = dict(model.episode_rollout_stats)
    if (stats['completed_games'] != model.env.num_envs or stats['censored_games']
            or stats['unfinished_games'] or stats['games'] != model.env.num_envs):
        raise ValueError('frozen rollout requires every natural complete game')
    # Semantic evidence is outside collector timing. Aggregate game/search
    # counts alone cannot detect changed action trajectories or supervision.
    from rl.training.checkpoints import _capture_environment
    digest_started = time.perf_counter()
    source_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if cuda else None)
    semantic_digests = dict(
        environment=state_digest(_capture_environment(model)),
        upper_samples=state_digest(model._upper_completed),
        upper_decisions=state_digest([
            {k: v for k, v in row.items() if k not in ('value', 'log_prob')}
            for row in model._upper_completed]),
        lower_samples=state_digest(model._small_playout_samples),
        rng=state_digest(source_rng))
    digest_seconds = time.perf_counter() - digest_started
    result = dict(cohort=stats, collector_wall_seconds=seconds,
        upper_samples=len(model._upper_completed), lower_samples=len(model._small_playout_samples),
        search_statistics=dict(model._small_playout_stats),
        collector_metrics={k: float(v) for k, v in model.logger.name_to_value.items()
                           if k.startswith('nosl_is_az/') and isinstance(v, (int, float, np.number))},
        unisolated_timing=dict(warmup_seconds=None, upper_forward_seconds=None, encoding_ipc_seconds=None,
                              reason='Included in collector wall, but not separately instrumented by this path.'),
        policy_and_adam_unchanged=True, state_digests=before, optimizer_updates=0,
        semantic_digests=semantic_digests, semantic_digest_seconds=digest_seconds,
        resources=monitor.summary,
        torch_max_allocated_bytes=torch.cuda.max_memory_allocated(model.device) if cuda else None,
        torch_max_reserved_bytes=torch.cuda.max_memory_reserved(model.device) if cuda else None,
        warmup='No discarded warmup cohort; first-use work is included in collector wall time.',
        note='Frozen collection only, not a training round. No update/sync/save cost included in collector wall.')
    return ok, result


def unified_candidates():
    """Use the real four-rank restore/collector/transport, stop before saving.

    This mode fits temporary update candidates to one complete cohort split
    by game. Production source state and lower weights are never advanced.
    All remaining arguments belong to the existing distributed trainer CLI.
    """
    import sys
    from unittest.mock import patch
    from rl.algorithms import small_playout
    from rl.algorithms.upper_probe import run_probe
    from rl.training import distributed_sb3 as trainer
    from rl.training.resource_monitor import ResourceMonitor

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--unified-candidate-protocol', type=Path, required=True)
    parser.add_argument('--probe-output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if '--fork-state' not in remaining or '--continuous' in remaining:
        raise ValueError('candidate diagnosis requires an isolated finite full-state fork')
    protocol = json.loads(args.unified_candidate_protocol.read_text())
    collection_only = protocol['protocol'] == 'unified-upper-rollout-scale-v1'
    if protocol['protocol'] not in ('unified-upper-candidate-screen-v1', 'unified-upper-rollout-scale-v1'):
        raise ValueError('unknown candidate protocol')
    launch = trainer.build_parser().parse_args(remaining)
    limits = protocol['diagnostic']
    cohort_offset = limits.get('fresh_cohort_offset', 0)
    if type(cohort_offset) is not int or cohort_offset < 0:
        raise ValueError('fresh cohort offset must be a nonnegative integer')
    if (launch.continuous or launch.updates != 1 or launch.total_steps is not None
            or launch.fork_reset_environments or launch.resume_reset_environments
            or launch.training_state_dir is not None or launch.checkpoint_dir is not None
            or launch.output is not None or launch.wandb or not launch.episode_rollout
            or launch.n_envs != limits['games_per_rank']
            or launch.world_size != limits['world_size']
            or launch.iteration_timeout_seconds != limits['maximum_wall_seconds']):
        raise ValueError('candidate launch must be finite, unsaved, fully restored and protocol-bounded')
    class ProbeComplete(BaseException):
        pass
    collection = {}
    setup_times = {}
    entry_started = time.perf_counter()
    original_restore = trainer.restore_learner
    def restore(*a, **kw):
        started = time.perf_counter()
        result = original_restore(*a, **kw)
        setup_times['environment_learner_restore_seconds'] = time.perf_counter() - started
        return result
    original_collect = trainer.AccumulatingMaskablePPO.collect_rollouts
    def collect(model, *a, **kw):
        if cohort_offset:
            previous = model.episode_rollout_stats
            if (not model._recovery_environment_restored
                    or previous.get('rollout') != model.episode_rollouts_completed
                    or not previous.get('games')
                    or previous.get('completed_games') != previous.get('games')
                    or previous.get('censored_games') != 0
                    or previous.get('unfinished_games') != 0
                    or any(model._upper_pending)
                    or any(model._nosl_is_az_pending_fights)):
                raise ValueError('fresh cohort offset requires a fully restored natural boundary')
            # Diagnostic-only selection of disjoint, previously unused games.
            # Weights, Adam, RNG and source files remain unchanged. The next
            # real collector begins these games normally; no partial game is dropped.
            model.episode_rollout_seed_offset = (
                getattr(model, 'episode_rollout_seed_offset', 0)
                + cohort_offset * launch.world_size * launch.n_envs)
        if not collection_only:
            return original_collect(model, *a, **kw)
        setup_times['entry_to_collection_seconds'] = time.perf_counter() - entry_started
        ok, result = measure_frozen_rollout(model, original_collect, *a, **kw)
        collection.update(result)
        return ok
    def diagnose(model, **kwargs):
        rank = kwargs['rank']
        if not model._recovery_environment_restored:
            raise ValueError('candidate source environment was not restored')
        if model.episode_rollout_stats['censored_games']:
            raise ValueError('candidate screen cannot accept censored games')
        output = args.probe_output / f'rank{rank}'
        output.mkdir(parents=True, exist_ok=True)
        (output/'cohort.json').write_text(json.dumps(model.episode_rollout_stats, indent=2))
        if collection_only:
            if not collection:
                raise AssertionError('missing measured collection')
            # Keep the transport alive until every rank finishes collecting.
            # This barrier is outside collector time and inside process time.
            joined_at = time.perf_counter()
            joined = kwargs['average'](dict(games=float(model.env.num_envs), complete=1.))
            if joined != dict(games=float(limits['games_per_rank']), complete=1.):
                raise ValueError('frozen rollout completion ranks disagree')
            collection['completion_sync_seconds'] = time.perf_counter() - joined_at
            collection['all_rank_completed_games'] = int(round(joined['games'] * launch.world_size))
            collection['setup_times'] = dict(setup_times)
            (output/'rollout.json').write_text(json.dumps(collection, indent=2, allow_nan=False))
            print('UPPER_ROLLOUT_BENCHMARK_COMPLETE ' + json.dumps(dict(
                rank=rank, games=model.env.num_envs,
                collector_wall_seconds=collection['collector_wall_seconds'],
                policy_and_adam_unchanged=True)), flush=True)
            raise ProbeComplete()
        result = run_probe(model, protocol, output_dir=output, **kwargs)
        print('UPPER_PROBE_COMPLETE ' + json.dumps(dict(rank=rank,
            source_state_restored=result['source_state_restored'],
            control_repeat_exact=result['control_repeat_exact'],
            control_repeat_numerical=result['control_repeat_numerical'])), flush=True)
        raise ProbeComplete()
    original = sys.argv
    sys.argv = [original[0], *remaining]
    try:
        # Initialize on the main thread so allocator telemetry is available.
        # The trainer subsequently restores the checkpoint's CUDA RNG state.
        import os
        import torch
        # Must precede the first CUDA BLAS handle used by the restored model.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        cuda = torch.device(launch.device).type == 'cuda'
        if cuda:
            torch.cuda.init()
        resources = ResourceMonitor(interval=1., gpu=cuda, torch_device=launch.device)
        try:
            with resources:
                from contextlib import ExitStack
                with ExitStack() as stack:
                    stack.enter_context(patch.object(trainer.AccumulatingMaskablePPO, 'collect_rollouts', collect))
                    if collection_only:
                        stack.enter_context(patch.object(trainer, 'restore_learner', restore))
                    stack.enter_context(patch.object(trainer, 'update_upper', diagnose))
                    stack.enter_context(patch.object(
                        small_playout, 'update_online', lambda *a, **kw: {}))
                    try:
                        trainer.main()
                    except ProbeComplete:
                        pass
        finally:
            output = args.probe_output / f'rank{launch.rank}'
            output.mkdir(parents=True, exist_ok=True)
            if hasattr(resources, 'summary'):
                (output/'resources.json').write_text(json.dumps(resources.summary, indent=2))
    finally:
        sys.argv = original


def main():
    import sys
    if '--unified-candidate-protocol' in sys.argv:
        return unified_candidates()
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
    from rl.policies.checkpoint import custom_objects
    from rl.training.environments import make_env
    from rl.training.checkpoints import read_bundle, _restore_environment_states, save_bundle
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=96)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--micro', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=60609)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--stop-drop', type=float, default=.10)
    parser.add_argument('--components-only', action='store_true', help='fixed first-batch loss comparisons only; restore all updates afterwards')
    parser.add_argument('--component-epochs', type=int, default=1)
    parser.add_argument('--horizons-only', action='store_true', help='same complete episodes, four bounded policy-gradient label comparisons')
    parser.add_argument('--horizon-arms', nargs='+', choices=('momentum', 'n128', 'n512', 'mc'),
                        default=['momentum', 'n128', 'n512', 'mc'])
    parser.add_argument('--horizon-loss', choices=('policy', 'value'), default='policy')
    args = parser.parse_args()
    if min(args.envs, args.steps, args.rounds, args.micro) <= 0 or not 0 < args.stop_drop < 1 or not 0 < args.max_seconds <= 3600 or not 1 <= args.component_epochs <= 10:
        raise ValueError('invalid bounded experiment configuration')
    if args.horizons_only and (args.steps != 512 or args.rounds != 1 or args.envs > 96):
        raise ValueError('horizon comparison requires at most 96 actors, exactly 512 steps and one collection')
    if args.output.exists() or args.output.resolve().is_relative_to(args.bundle.resolve()):
        raise ValueError('new output directory outside protected source required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(state='initializing', rounds=[], ablations=[], updates=[], protocol='independent world_size=1 small-batch diagnostic',
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), windows_are_evaluation=False)
    process = psutil.Process()
    process.cpu_percent()
    def publish():
        report['seconds'] = time.monotonic()-started
        report['resources'] = dict(rss_bytes=process.memory_info().rss, available_ram=psutil.virtual_memory().available,
            cpu_seconds=sum(process.cpu_times()[:2]), cpu_percent=process.cpu_percent(),
            torch_allocated=torch.cuda.memory_allocated(), torch_reserved=torch.cuda.memory_reserved(),
            torch_peak=torch.cuda.max_memory_allocated(), free_bytes=psutil.disk_usage(args.output).free)
        temp = args.output/'report.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False, default=json_scalar))
        temp.replace(args.output/'report.json')
    def check():
        if (args.output/'STOP').exists():
            raise DiagnosticStop('requested_stop')
        if time.monotonic()-started >= args.max_seconds:
            raise DiagnosticStop('time_budget')
    class Callback(BaseCallback):
        def _on_step(self):
            check()
            window = report.setdefault('training_outcomes', [])
            for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
                if done:
                    censored = bool(info.get('TimeLimit.truncated') or info.get('dead_screen'))
                    stats = info.get('run_stats', {})
                    window.append(dict(censored=censored, won=not censored and (stats.get('act', 0) > 3 or stats.get('outcome') == 'PLAYER_VICTORY'),
                        floor=stats.get('floor'), act=stats.get('act')))
            return True
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    header = json.loads((args.bundle/'manifest.json').read_text())
    manifest = read_bundle(args.bundle, rank=header['rank'], world_size=header['world_size'])
    report['source'] = dict(step=manifest['global_step'], rank=header['rank'], sha256=manifest['sha256'])
    saved = torch.load(args.bundle/'learner.pt', map_location='cpu', weights_only=False)
    environment = torch.load(args.bundle/'environment.pt', map_location='cpu', weights_only=False)
    assert saved['replay'] is None
    indices = np.sort(np.random.default_rng(args.seed).choice(len(environment['states']), args.envs, replace=False))
    subset = subset_learner(saved, indices)
    report['source_env_indices'] = indices.tolist()
    model = AccumulatingMaskablePPO.load(args.bundle/'model.zip', device='cuda',
        custom_objects=custom_objects(args.bundle/'model.zip'), n_envs=args.envs, n_steps=args.steps)
    model.configure_nosl_is_az_inner_only(); model.configure_rulebase_lower()
    cfg = replace(model.policy.core.upper_config, micro_batch_size=args.micro)
    model.policy.core.upper_config = cfg
    model.policy.upper_config = asdict(cfg)
    model.policy_kwargs['upper_config'] = asdict(cfg)
    config = dict(manifest['config'], n_envs=args.envs, n_steps=args.steps, seed=args.seed, rank=0, world_size=1,
                  upper_micro_batch_size=args.micro)
    env = make_env(model, argparse.Namespace(**config))
    model.set_env(env); model.set_logger(configure(str(args.output/'logs'), []))
    _, callback = model._setup_learn(args.envs*args.steps*args.rounds, Callback(), reset_num_timesteps=False)
    model._last_obs = _restore_environment_states(env, [environment['states'][i] for i in indices])
    model._last_episode_starts = np.asarray(subset['last_episode_starts'], dtype=bool)
    model._upper_pending = subset['upper_pending']
    model._upper_episode_ids = subset['upper_episode_ids']
    model._nosl_is_az_pending_fights = subset['nosl_is_az_pending_fights']
    env.episode_returns[:] = np.asarray(environment['episode_returns'])[indices]
    env.episode_lengths[:] = np.asarray(environment['episode_lengths'])[indices]
    env.episode_count = 0
    model.ep_info_buffer.clear(); model.ep_success_buffer.clear()
    set_rng((saved['python_rng'], saved['numpy_rng'], saved['torch_rng'], saved['cuda_rng']))
    frozen = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items() if k.startswith('core.combat.') or k.endswith('card.vectors')}
    del saved, environment
    saved_rng = rng_state()
    panel = make_panel(model, args.panel)
    set_rng(saved_rng)
    baseline = panel_summary(model, panel)
    report.update(state='running', baseline=baseline, config=asdict(cfg), initial_adam_steps=sorted({int(s['step']) for s in model.policy.upper_optimizer.state.values()}))
    report['source_lower_frozen'] = not model.policy.optimizer.state
    publish()
    print('READY', json.dumps(baseline), flush=True)
    if args.horizons_only:
        try:
            compare_horizons(model, env, callback, args, panel, report, publish, check)
            report['source_reverified'] = read_bundle(args.bundle, rank=header['rank'], world_size=header['world_size'])['sha256'] == manifest['sha256']
            assert all(torch.equal(v.cpu(), frozen[k]) for k, v in model.policy.state_dict().items() if k in frozen)
            publish()
        finally:
            env.close()
        return
    original_loss, original_adv, original_step = U.ppo_loss_components, U.macro_advantages, U.optimizer_step
    arm, round_index, step_in_arm = 'full', 0, 0
    phase = 'online'
    def capture_targets(records, bootstrap, gamma, gae_lambda):
        adv, returns = original_adv(records, bootstrap, gamma, gae_lambda)
        if phase == 'online':
            report['rounds'][-1]['targets'] = summarize_targets(model, records, bootstrap, adv, returns)
            np.savez_compressed(args.output/f'targets-{round_index:02d}.npz', bootstrap=bootstrap, advantage=adv, returns=returns,
                reward=[r['reward'] for r in records], old_value=[r['value'] for r in records],
                action=[r['action'] for r in records], env=[r['env'] for r in records],
                episode=[r['episode'] for r in records], terminal=[r['terminal'] for r in records],
                phase=[int(r['observation']['phase'][0]) for r in records])
        return adv, returns
    def traced_step(*a, **kw):
        nonlocal step_in_arm
        check()
        before = {n: p.detach().clone() for n, p in model.policy.named_parameters() if p.grad is not None}
        grad = {}
        for name, p in model.policy.named_parameters():
            if p.grad is not None:
                group = parameter_group(name)
                grad[group] = grad.get(group, 0.) + float(p.grad.detach().float().square().sum())
        result = original_step(*a, **kw)
        step_in_arm += 1
        delta = {}
        for name, p in model.policy.named_parameters():
            if name in before:
                group = parameter_group(name)
                delta[group] = delta.get(group, 0.) + float((p.detach()-before[name]).float().square().sum())
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise FloatingPointError('nonfinite weights')
        probe = panel_summary(model, panel)
        report['updates'].append(dict(round=round_index, phase=phase, arm=arm, epoch=step_in_arm,
            probe=probe, gradient_l2={k: v**.5 for k, v in grad.items()}, update_l2={k: v**.5 for k, v in delta.items()},
            grad_norm=float(result)))
        publish()
        if phase == 'online':
            reason = should_stop(baseline, probe, args.stop_drop)
            if reason:
                raise DiagnosticStop(reason)
        return result
    try:
        U.macro_advantages = capture_targets
        U.optimizer_step = traced_step
        U.ppo_loss_components = lambda **kw: loss_arm(original_loss(**kw), arm)
        callback.on_training_start({}, {})
        for round_index in range(args.rounds):
            check()
            tick = time.monotonic()
            assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=args.steps)
            records = model._upper_completed
            report['rounds'].append(dict(round=round_index, samples=len(records), collection_seconds=time.monotonic()-tick))
            if round_index == 0:
                # Capture all loss arms from identical weights, Adam, data and RNG.
                base, opt, rng = cpu_copy(model.policy.state_dict()), cpu_copy(model.policy.upper_optimizer.state_dict()), rng_state()
                def reset():
                    model.policy.load_state_dict(base)
                    model.policy.upper_optimizer.load_state_dict(copy.deepcopy(opt))
                    model.policy.upper_optimizer.zero_grad(set_to_none=True)
                    model.policy.set_training_mode(False)
                    model._upper_completed = records
                    set_rng(rng)
                phase = 'ablation'
                try:
                    # The observation hooks must not perturb the original update.
                    reset()
                    model.policy.core.upper_config = replace(cfg, epochs=args.component_epochs)
                    U.ppo_loss_components, U.macro_advantages, U.optimizer_step = original_loss, original_adv, original_step
                    reference_stats = U.update_upper(model)
                    reference = cpu_copy(model.policy.state_dict())
                    U.ppo_loss_components = lambda **kw: loss_arm(original_loss(**kw), arm)
                    U.macro_advantages, U.optimizer_step = capture_targets, traced_step
                    for arm in ('momentum', 'policy', 'value', 'entropy', 'full'):
                        reset(); step_in_arm = 0
                        model.policy.core.upper_config = replace(cfg, epochs=args.component_epochs)
                        stats = U.update_upper(model)
                        report['ablations'].append(dict(arm=arm, stats=stats, probe=panel_summary(model, panel)))
                        publish()
                    report['instrumentation_exact'] = all(torch.equal(v.cpu(), reference[k]) for k, v in model.policy.state_dict().items())
                    report['instrumentation_reference_stats'] = reference_stats
                    if not report['instrumentation_exact']:
                        raise AssertionError('diagnostic hooks changed the full update')
                finally:
                    reset()
                    model.policy.core.upper_config = cfg
                del base, opt
                if args.components_only:
                    report['state'], report['stop_reason'] = 'complete', 'fixed_components_budget'
                    break
            phase, arm, step_in_arm = 'online', 'full', 0
            model.policy.core.upper_config = cfg
            tick = time.monotonic()
            stats = U.update_upper(model)
            assert all(torch.equal(v.cpu(), frozen[k]) for k, v in model.policy.state_dict().items() if k in frozen)
            assert not model.policy.optimizer.state and getattr(model, '_nosl_is_az_replay', None) is None
            episodes = list(model.ep_info_buffer)
            report['rounds'][-1].update(stats=stats, update_seconds=time.monotonic()-tick,
                episode_window_count=len(episodes), episode_window=list(episodes)[-4:])
            publish()
            print('ROUND', round_index, json.dumps(panel_summary(model, panel)), flush=True)
        if not args.components_only:
            report['state'], report['stop_reason'] = 'complete', 'round_budget'
    except DiagnosticStop as exc:
        report['state'], report['stop_reason'] = 'stopped', str(exc)
    except Exception as exc:
        report['state'], report['error'] = 'failed', type(exc).__name__+': '+str(exc)
        publish()
        raise
    finally:
        U.ppo_loss_components, U.macro_advantages, U.optimizer_step = original_loss, original_adv, original_step
        publish()
    report['final_probe'] = panel_summary(model, panel)
    report['final_adam_steps'] = sorted({int(s['step']) for s in model.policy.upper_optimizer.state.values()})
    report['final_bundle'] = None if args.components_only else str(save_bundle(model, args.output/'recovery', rank=0, world_size=1,
        run_id=args.output.name, wandb_id=None, config=config))
    report['source_reverified'] = read_bundle(args.bundle, rank=header['rank'], world_size=header['world_size'])['sha256'] == manifest['sha256']
    publish()
    env.close()
    print('FINISHED', report['state'], report['stop_reason'], flush=True)


if __name__ == '__main__':
    main()

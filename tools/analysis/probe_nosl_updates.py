"""Bounded CPU fit diagnostic on an immutable learner bundle; never resumes a run.

Reuses the real NOSL learner and saved Adam/replay. The audit rows are IN-SAMPLE,
with fixed HP strata. This diagnoses fit/last-update offsets, not generalization
or production batch-size equivalence. No rollout, SSH, W&B or checkpoint writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.checkpoint import custom_objects
from rl.search.native import forward_leaves
from rl.training.checkpoints import read_bundle


def audit_indices(replay, *, seed=5311, per_stratum=64):
    hp = replay['observations']['player_scalars'][:, 0].astype(np.float32) * 200
    roots = replay['arrays']['policy_weights'].reshape(-1) > 0
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(pool, size=min(per_stratum, len(pool)), replace=False)
        for low, high in [(0, 5), (5, 20), (20, 40), (40, float('inf'))]
        for pool in [np.flatnonzero(roots & (hp > low) & (hp <= high))]])


def predictions(model, replay, indices, *, training=False, bf16=False):
    from contextlib import nullcontext
    model.policy.set_training_mode(training)
    logits, values = [], []
    for start in range(0, len(indices), 64):
        batch = {k: v[indices[start:start+64]].astype(np.int64 if v.dtype.kind in 'ibu' else np.float32)
                 for k, v in replay['observations'].items()}
        context = (lambda: torch.autocast('cpu', dtype=torch.bfloat16)) if bf16 else nullcontext
        l, v = forward_leaves(model, batch, compact=True, autocast=context)
        logits.append(l)
        values.append(v)
    return np.concatenate(logits), np.concatenate(values) * 200


def summarize(replay, indices, logits, values, reference_logits):
    if not (np.isfinite(logits).all() and np.isfinite(values).all()):
        raise FloatingPointError('nonfinite diagnostic prediction')
    arrays = replay['arrays']
    hp = replay['observations']['player_scalars'][indices, 0].astype(np.float32)*200
    targets = arrays['value_targets'][indices].reshape(-1)*200
    masks = torch.as_tensor(arrays['action_masks'][indices], dtype=torch.bool)
    if not masks.any(dim=1).all():
        raise ValueError('audit must use policy roots with legal actions')
    def log_policy(data):
        return torch.log_softmax(torch.as_tensor(data).masked_fill(~masks, -1e8), dim=-1)
    old, new = log_policy(reference_logits), log_policy(logits)
    kl = (old.exp()*(old-new)).sum(dim=-1).numpy()
    strata = []
    for low, high in [(0, 5), (5, 20), (20, 40), (40, float('inf'))]:
        take = (hp > low) & (hp <= high)
        no_heal = take & (targets <= hp+.1)
        strata.append(dict(hp_interval=[low, high if np.isfinite(high) else None],
            count=int(take.sum()),
            mae_hp=float(np.abs(values[take]-targets[take]).mean()) if take.any() else None,
            bias_hp=float((values[take]-targets[take]).mean()) if take.any() else None,
            observed_nonhealing_count=int(no_heal.sum()),
            observed_nonhealing_above_hp=float(np.mean(values[no_heal]>hp[no_heal]+.1)) if no_heal.any() else None))
    return dict(strata=strata, policy_kl_from_start=float(kl.mean()),
                policy_argmax_change_fraction=float(np.mean(old.argmax(1).numpy()!=new.argmax(1).numpy())))


def run_probe(bundle, output, *, batch_sizes=(1024,), updates=10, max_seconds=120,
              per_stratum=64, seed=5311, audit_seed=5311):
    if updates < 1 or max_seconds <= 0 or per_stratum < 1 or any(b < 1 for b in batch_sizes):
        raise ValueError('all budgets must be positive')
    bundle, output = Path(bundle), Path(output)
    if output.exists() or output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('use a new output path outside the source bundle')
    header = json.loads((bundle/'manifest.json').read_text())
    manifest = read_bundle(bundle, rank=header['rank'], world_size=header['world_size'])
    replay = torch.load(bundle/'learner.pt', map_location='cpu', weights_only=False)['replay']
    if replay is None:
        raise ValueError('bundle has no replay')
    indices = audit_indices(replay, seed=audit_seed, per_stratum=per_stratum)
    if not len(indices):
        raise ValueError('no policy roots available for audit')
    result = dict(schema_version=1, source_step=manifest['global_step'], source_rank=manifest['rank'],
        source_sha256=manifest['sha256'], seed=seed, audit_seed=audit_seed, audit_rows=len(indices),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch_version=torch.__version__,
        units='fit_value_bias/fit_value_mae: normalized HP/200; strata errors and mode deltas: HP',
        protocol='CPU FP32; single-rank cloned Adam/replay; uniform frozen replay, no new collection; in-sample audit',
        state='running', branches=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def publish():
        result['elapsed_seconds'] = time.monotonic()-started
        temporary = output.with_suffix(output.suffix+'.partial')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output)
    try:
        for batch_size in batch_sizes:
            if time.monotonic()-started >= max_seconds:
                result['state'] = 'budget_exhausted'
                publish()
                return result
            model = AccumulatingMaskablePPO.load(bundle/'model.zip', device='cpu',
                custom_objects=custom_objects(bundle/'model.zip'))
            model.configure_nosl_is_az_inner_only()
            model.amp_mode = 'off'
            model.nosl_is_az_batch_size = batch_size
            model.nosl_is_az_micro_batch_size = min(256, batch_size)
            model.nosl_is_az_replay_max_old_fraction = 1.0
            model._ensure_nosl_is_az_replay().load_state_dict(replay)
            model._nosl_is_az_replay_prepare_stats = {'fresh_roots': 0, 'fresh_start_id': replay['total_added']}
            # Loading a model can reset RNG to its historical training seed.
            # Set the experiment seed after all checkpoint/setup work.
            np.random.seed(seed)
            torch.manual_seed(seed)
            reference_logits, reference_values = predictions(model, replay, indices)
            branch = dict(batch_size=batch_size, rows=[dict(update=0,
                **summarize(replay, indices, reference_logits, reference_values, reference_logits))])
            result['branches'].append(branch)
            # Compare modes on identical tensors before any optimizer change.
            branch['mode_controls'] = []
            for training, bf16 in [(True, False), (False, True)]:
                torch.manual_seed(seed)
                _, values = predictions(model, replay, indices, training=training, bf16=bf16)
                branch['mode_controls'].append(dict(training=training, cpu_bf16=bf16,
                    mean_delta_hp=float((values-reference_values).mean()),
                    max_abs_delta_hp=float(np.abs(values-reference_values).max())))
            publish()
            for update in range(1, updates+1):
                if time.monotonic()-started >= max_seconds:
                    result['state'] = 'budget_exhausted'
                    publish()
                    return result
                model.policy.set_training_mode(True)
                fit = model._train_nosl_is_az(prepare_replay=False, update_count=1)
                logits, values = predictions(model, replay, indices)
                branch['rows'].append(dict(update=update, fit_value_bias=fit.get('value_bias'),
                    fit_value_mae=fit.get('value_mae'), optimizer_steps=fit['updates'],
                    **summarize(replay, indices, logits, values, reference_logits)))
                publish()
        result['state'] = 'complete'
    except BaseException as exc:
        result.update(state='error', error=f'{type(exc).__name__}: {exc}')
        publish()
        raise
    publish()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1024])
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--per-stratum', type=int, default=64)
    parser.add_argument('--seed', type=int, default=5311)
    parser.add_argument('--audit-seed', type=int, default=5311, help='hold audit rows fixed across learner seeds')
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_probe(args.bundle, args.output, batch_sizes=args.batch_sizes, updates=args.updates,
                       max_seconds=args.max_seconds, per_stratum=args.per_stratum, seed=args.seed, audit_seed=args.audit_seed)
    print(json.dumps({k: result[k] for k in ['state', 'source_step', 'elapsed_seconds']}))
    if result['state'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

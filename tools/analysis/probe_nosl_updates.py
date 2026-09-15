"""Bounded cloned-learner diagnostic; never resumes or saves a training run.

Reuses the real NOSL learner and saved Adam/replay. Audit defaults to IN-SAMPLE;
an optional later bundle provides unseen sample IDs under temporal distribution
shift. Optional fixed HP corpora test sensitivity at update milestones. Neither
protocol proves online production equivalence. No rollout, SSH or W&B writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.policies.checkpoint import custom_objects
from rl.search.native import forward_leaves
from rl.training.checkpoints import read_bundle


def audit_indices(replay, *, seed=5311, per_stratum=64, minimum_sample_id=None):
    hp = replay['observations']['player_scalars'][:, 0].astype(np.float32) * 200
    roots = replay['arrays']['policy_weights'].reshape(-1) > 0
    if minimum_sample_id is not None:
        roots &= replay['arrays']['sample_ids'].reshape(-1) >= minimum_sample_id
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
        context = (lambda: torch.autocast(model.device.type, dtype=torch.bfloat16)) if bf16 else nullcontext
        l, v = forward_leaves(model, batch, compact=True, autocast=context)
        logits.append(l)
        values.append(v)
    return np.concatenate(logits), np.concatenate(values) * 200


def supervision_strata(replay):
    """Partition serialized rows without inventing discarded sample_kind metadata."""
    size = int(replay['size'])
    obs, arrays = replay['observations'], replay['arrays']
    # Stored compact observations can be float16; game HP is integral.
    hp = np.rint(obs['player_scalars'][:size, 0].astype(np.float32)*200)
    roots = arrays['policy_weights'][:size].reshape(-1) > 0
    monsters = obs['monster_scalars'][:size]
    living = ((monsters[..., 0] > 0) & (monsters[..., 1] > 0)).any(axis=1)
    kinds = [('policy_root', roots), ('value_only_living_enemy', ~roots & living),
             ('value_only_no_living_enemy', ~roots & ~living)]
    bins = [('dead', hp <= 0), ('1-5', (hp > 0) & (hp <= 5)),
            ('6-20', (hp > 5) & (hp <= 20)), ('21-40', (hp > 20) & (hp <= 40)),
            ('above40', hp > 40)]
    return hp, [(kind, interval, np.flatnonzero(mask & take))
                for kind, mask in kinds for interval, take in bins]


def run_replay_audit(bundle, output, *, per_stratum=64, seed=5311,
                     device='cpu', max_seconds=120):
    """Read-only in-replay coverage/calibration; never calls an optimizer."""
    if per_stratum < 1 or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('positive finite audit budgets required')
    if device not in ('cpu', 'cuda'):
        raise ValueError('audit device must be cpu or cuda')
    bundle, output = Path(bundle), Path(output)
    if output.exists() or output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('use a new output path outside the source bundle')
    started = time.monotonic()
    header = json.loads((bundle/'manifest.json').read_text())
    manifest = read_bundle(bundle, rank=header['rank'], world_size=header['world_size'])
    replay = torch.load(bundle/'learner.pt', map_location='cpu', weights_only=False)['replay']
    if replay is None or replay['size'] < 1:
        raise ValueError('audit requires nonempty replay')
    model = AccumulatingMaskablePPO.load(bundle/'model.zip', device=device,
        custom_objects=custom_objects(bundle/'model.zip'))
    hp, strata = supervision_strata(replay)
    targets = replay['arrays']['value_targets'][:replay['size']].reshape(-1)*200
    if not np.isfinite(hp).all() or not np.isfinite(targets).all():
        raise ValueError('nonfinite replay HP or labels')
    rng = np.random.default_rng(seed)
    result = dict(schema_version=1, state='running', source_step=manifest['global_step'],
        source_rank=manifest['rank'], source_sha256=manifest['sha256'], seed=seed,
        replay_size=int(replay['size']), device=device, torch_version=torch.__version__,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='in-replay calibration, not held-out; FP32 inference; no optimizer or rollout',
        classification='policy_weight separates roots/value-only; living enemies use presence and HP; '
                       'sample_kind was not retained, so value-only strata are not exact boundary/terminal labels',
        rows=[])
    for kind, interval, pool in strata:
        if time.monotonic()-started >= max_seconds:
            result['state'] = 'budget_exhausted'
            break
        ids = rng.choice(pool, size=min(per_stratum, len(pool)), replace=False)
        row = dict(kind=kind, hp_interval=interval, available=len(pool), sampled=len(ids))
        if len(pool):
            delta = targets[pool]-hp[pool]
            row.update(target_mean_hp=float(targets[pool].mean()),
                target_minus_current_hp_mean=float(delta.mean()),
                target_matches_current_count=int((np.abs(delta) <= .1).sum()),
                target_above_current_count=int((delta > .1).sum()),
                target_below_current_count=int((delta < -.1).sum()))
        if len(ids):
            _, values = predictions(model, replay, ids)
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite audit predictions')
            error = values-targets[ids]
            equal = np.abs(targets[ids]-hp[ids]) <= .1
            row.update(sample_ids_sha256=hashlib.sha256(
                replay['arrays']['sample_ids'][ids].astype('<i8').tobytes()).hexdigest(),
                sampled_target_mean_hp=float(targets[ids].mean()),
                predicted_mean_hp=float(values.mean()), mae_hp=float(np.abs(error).mean()),
                bias_hp=float(error.mean()), negative_predictions=int((values < 0).sum()),
                above_current_predictions=int((values > hp[ids]+.1).sum()),
                target_matches_current_sampled=int(equal.sum()),
                target_matches_current_mae_hp=float(np.abs(error[equal]).mean()) if equal.any() else None,
                target_matches_current_bias_hp=float(error[equal].mean()) if equal.any() else None)
        result['rows'].append(row)
    else:
        result['state'] = 'complete'
    # Rehash both source files after inference: audit must not rewrite recovery.
    assert read_bundle(bundle, rank=header['rank'], world_size=header['world_size']) == manifest
    result['elapsed_seconds'] = time.monotonic()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.partial')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    return result


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


def configure_probe_sampling(replay, low_hp_fraction):
    """Override only this private clone's sampler; never the runtime class.

    Zero preserves the native uniform sampler exactly. Positive fractions reserve
    round(count*fraction) rows for living HP1–5, with the remainder from all other
    rows (including terminal/dead rows). This intentionally reweights both losses.
    """
    if not np.isfinite(low_hp_fraction) or not 0 <= low_hp_fraction < 1:
        raise ValueError('low HP fraction must be finite and in [0, 1)')
    hp = replay.observations['player_scalars'][:replay.size, 0].astype(np.float32)*200
    low = np.flatnonzero((hp > 0) & (hp <= 5))
    other = np.flatnonzero(~((hp > 0) & (hp <= 5)))
    metadata = dict(low_hp_fraction=low_hp_fraction, available_low_hp=len(low),
                    available_other=len(other), low_hp_interval=[0, 5])
    if low_hp_fraction == 0:
        return metadata
    if not len(low) or not len(other):
        raise ValueError('quota experiment requires both low HP and other replay rows')
    def sample(count, rng=None, *, fresh_start_id=None, max_old_fraction=1.0):
        if count < 1 or max_old_fraction != 1.0:
            raise ValueError('quota probe requires positive count and frozen replay')
        count = min(int(count), replay.size)
        low_count = int(round(count*low_hp_fraction))
        chooser = np.random if rng is None else rng
        selected = np.concatenate([chooser.choice(pool, size=n, replace=n > len(pool))
            for pool, n in [(low, low_count), (other, count-low_count)]])
        chooser.shuffle(selected)
        return selected
    replay.sample_indices = sample
    return metadata


def run_probe(bundle, output, *, batch_sizes=(1024,), updates=10, max_seconds=120,
              per_stratum=64, seed=5311, audit_seed=5311, low_hp_fractions=(0.,),
              device='cpu', micro_batch_size=256, audit_bundle=None,
              hp_corpus=None, terminal_corpus=None, eval_updates=(5, 10, 20, 40), amp='off',
              learning_rate_scales=(1.,)):
    if updates < 1 or max_seconds <= 0 or per_stratum < 1 or any(b < 1 for b in batch_sizes):
        raise ValueError('all budgets must be positive')
    if not batch_sizes or not low_hp_fractions or any(
            not np.isfinite(f) or not 0 <= f < 1 for f in low_hp_fractions):
        raise ValueError('provide batches and low HP fractions in [0, 1)')
    if not learning_rate_scales or any(not np.isfinite(s) or s < 0 for s in learning_rate_scales):
        raise ValueError('provide finite nonnegative learning rate scales; zero is a no-weight-update control')
    bundle, output = Path(bundle), Path(output)
    if output.exists() or output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('use a new output path outside the source bundle')
    header = json.loads((bundle/'manifest.json').read_text())
    manifest = read_bundle(bundle, rank=header['rank'], world_size=header['world_size'])
    replay = torch.load(bundle/'learner.pt', map_location='cpu', weights_only=False)['replay']
    if replay is None:
        raise ValueError('bundle has no replay')
    if micro_batch_size < 1 or device not in ('cpu', 'cuda'):
        raise ValueError('positive micro batch and cpu/cuda device required')
    if amp not in ('off', 'bf16') or (amp != 'off' and device != 'cuda'):
        raise ValueError('training AMP must be off, or bf16 on CUDA')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA explicitly requested but unavailable')
    audit_replay, audit_metadata, minimum_id = replay, {'kind': 'in-sample'}, None
    if audit_bundle is not None:
        audit_bundle = Path(audit_bundle)
        if output.resolve().is_relative_to(audit_bundle.resolve()):
            raise ValueError('output must be outside the audit bundle')
        other = read_bundle(audit_bundle, rank=manifest['rank'], world_size=manifest['world_size'])
        if any(other[k] != manifest[k] for k in ('run_id', 'wandb_id')) or other['global_step'] <= manifest['global_step']:
            raise ValueError('audit bundle must be later in the same run and rank')
        audit_replay = torch.load(audit_bundle/'learner.pt', map_location='cpu', weights_only=False)['replay']
        minimum_id = replay['total_added']
        audit_metadata = dict(kind='later-replay; unseen sample IDs; temporal distribution shift',
            source_step=other['global_step'], sha256=other['sha256'], minimum_sample_id=minimum_id)
    indices = audit_indices(audit_replay, seed=audit_seed, per_stratum=per_stratum, minimum_sample_id=minimum_id)
    if not len(indices):
        raise ValueError('no policy roots available for audit')
    result = dict(schema_version=1, source_step=manifest['global_step'], source_rank=manifest['rank'],
        source_sha256=manifest['sha256'], seed=seed, audit_seed=audit_seed, audit_rows=len(indices),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch_version=torch.__version__,
        units='fit_value_bias/fit_value_mae: normalized HP/200; strata errors and mode deltas: HP',
        protocol='single-rank cloned Adam/replay; frozen replay, no new collection; FP32 audit; per-branch sampling quota',
        device=device, training_amp=amp, micro_batch_size=micro_batch_size, audit=audit_metadata,
        state='running', branches=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    from rl.evaluation.nosl.hp_sensitivity import load_corpus, evaluate_corpus, evaluate_terminal_corpus
    hp_data = load_corpus(hp_corpus) if hp_corpus is not None else None
    terminal_data = load_corpus(terminal_corpus) if terminal_corpus is not None else None
    def health(model):
        result = {}
        if hp_data is not None:
            result['hp_sensitivity'] = evaluate_corpus(model, hp_data)
        if terminal_data is not None:
            result['terminal'] = evaluate_terminal_corpus(model, terminal_data)
        return result
    def publish():
        result['elapsed_seconds'] = time.monotonic()-started
        temporary = output.with_suffix(output.suffix+'.partial')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output)
    try:
        for batch_size, low_hp_fraction, lr_scale in product(batch_sizes, low_hp_fractions, learning_rate_scales):
            if time.monotonic()-started >= max_seconds:
                result['state'] = 'budget_exhausted'
                publish()
                return result
            model = AccumulatingMaskablePPO.load(bundle/'model.zip', device=device,
                custom_objects=custom_objects(bundle/'model.zip'))
            model.configure_nosl_is_az_inner_only()
            # Keep the checkpoint's Adam moments and schedule; scale every
            # schedule evaluation so the learner cannot overwrite this control.
            original_schedule = model.lr_schedule
            model.lr_schedule = lambda progress, schedule=original_schedule, scale=lr_scale: schedule(progress)*scale
            for group in model.policy.optimizer.param_groups:
                group['lr'] *= lr_scale
            model.amp_mode = amp
            model.nosl_is_az_batch_size = batch_size
            model.nosl_is_az_micro_batch_size = min(micro_batch_size, batch_size)
            model.nosl_is_az_replay_max_old_fraction = 1.0
            buffer = model._ensure_nosl_is_az_replay()
            buffer.load_state_dict(replay)
            sampling = configure_probe_sampling(buffer, low_hp_fraction)
            model._nosl_is_az_replay_prepare_stats = {'fresh_roots': 0, 'fresh_start_id': replay['total_added']}
            # Loading a model can reset RNG to its historical training seed.
            # Set the experiment seed after all checkpoint/setup work.
            np.random.seed(seed)
            torch.manual_seed(seed)
            reference_logits, reference_values = predictions(model, audit_replay, indices)
            branch = dict(batch_size=batch_size, sampling=sampling, learning_rate_scale=lr_scale,
                initial_learning_rates=[float(g['lr']) for g in model.policy.optimizer.param_groups], rows=[dict(update=0,
                **summarize(audit_replay, indices, reference_logits, reference_values, reference_logits))],
                health=[dict(update=0, **health(model))])
            result['branches'].append(branch)
            # Compare modes on identical tensors before any optimizer change.
            branch['mode_controls'] = []
            for training, bf16 in [(True, False), (False, True)]:
                torch.manual_seed(seed)
                _, values = predictions(model, audit_replay, indices, training=training, bf16=bf16)
                branch['mode_controls'].append(dict(training=training, bf16=bf16, device=device,
                    mean_delta_hp=float((values-reference_values).mean()),
                    max_abs_delta_hp=float(np.abs(values-reference_values).max())))
            publish()
            for update in range(1, updates+1):
                if time.monotonic()-started >= max_seconds:
                    result['state'] = 'budget_exhausted'
                    publish()
                    return result
                model.policy.set_training_mode(True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                fit_started = time.monotonic()
                fit = model._train_nosl_is_az(prepare_replay=False, update_count=1)
                if device == 'cuda':
                    torch.cuda.synchronize()
                fit_seconds = time.monotonic()-fit_started
                logits, values = predictions(model, audit_replay, indices)
                branch['rows'].append(dict(update=update, fit_value_bias=fit.get('value_bias'),
                    fit_value_mae=fit.get('value_mae'), optimizer_steps=fit['updates'],
                    learning_rates=[float(g['lr']) for g in model.policy.optimizer.param_groups],
                    fit_seconds=fit_seconds, sampled_rows=min(batch_size, replay['size']),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                    **summarize(audit_replay, indices, logits, values, reference_logits)))
                if update in eval_updates or update == updates:
                    branch['health'].append(dict(update=update, **health(model)))
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
    parser.add_argument('--audit-only', action='store_true',
                        help='read-only supervision/HP coverage and calibration; no learner updates')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1024])
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--per-stratum', type=int, default=64)
    parser.add_argument('--seed', type=int, default=5311)
    parser.add_argument('--audit-seed', type=int, default=5311, help='hold audit rows fixed across learner seeds')
    parser.add_argument('--learning-rate-scales', nargs='+', type=float, default=[1.],
                        help='multiply saved LR schedule; zero controls for drift without weight updates (Adam moments still advance)')
    parser.add_argument('--low-hp-fractions', nargs='+', type=float, default=[0.],
                        help='0 keeps uniform replay; positive values reserve that batch fraction for HP1–5')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--amp', choices=['off', 'bf16'], default='off', help='training precision; audit stays FP32')
    parser.add_argument('--micro-batch-size', type=int, default=256)
    parser.add_argument('--audit-bundle', type=Path)
    parser.add_argument('--hp-corpus', type=Path)
    parser.add_argument('--terminal-corpus', type=Path)
    parser.add_argument('--eval-updates', nargs='+', type=int, default=[5, 10, 20, 40])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.audit_only:
        result = run_replay_audit(args.bundle, args.output, per_stratum=args.per_stratum,
                                 seed=args.audit_seed, device=args.device, max_seconds=args.max_seconds)
    else:
        result = run_probe(args.bundle, args.output, batch_sizes=args.batch_sizes, updates=args.updates,
                       max_seconds=args.max_seconds, per_stratum=args.per_stratum, seed=args.seed,
                       audit_seed=args.audit_seed, low_hp_fractions=args.low_hp_fractions,
                       device=args.device, micro_batch_size=args.micro_batch_size, audit_bundle=args.audit_bundle,
                       hp_corpus=args.hp_corpus, terminal_corpus=args.terminal_corpus,
                       eval_updates=args.eval_updates, amp=args.amp, learning_rate_scales=args.learning_rate_scales)
    print(json.dumps({k: result[k] for k in ['state', 'source_step', 'elapsed_seconds']}))
    if result['state'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

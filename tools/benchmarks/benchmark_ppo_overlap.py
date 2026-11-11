"""Frozen four-rank gradient-overlap experiment for COMBAT49. Never writes a training checkpoint.

A is the production serial epoch. B computes upper chunks first but restores the
original local FP32 addition order. C overlaps a balanced upper-only ring, D
overlaps that ring using the original per-element rank addition order.
Qualification captures all 16 gradients; separate timing passes omit the copies.

Run on each stopped rank with its existing runtime, --plan, --checkpoint-dir
(its own recovery bundle), a new --output directory, and the same unused
--ring-port. An external supervisor must impose a wall timeout on all ranks.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import shlex
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.packed_ppo import configure_packed_updates, packed_ppo_epoch
from rl.algorithms.ppo import configure_advantage_normalization, prepare_advantage_normalization
from rl.training.checkpoints import restore_learner
from rl.training.distributed_sb3 import (average_scalar_dict, pack_gradients, parse_peer,
                                         synchronize_gradient_step, ssh_cmd)
from rl.training.environments import make_env
from rl.training.resource_monitor import ResourceMonitor
from rl.training.ring_allreduce import ParallelRingAllReduce, SshRingAllReduce
from tools.benchmarks.ppo_overlap import EarlyUpperReduction, UpperFirstSchedule


def clone_state(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: clone_state(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_state(v) for v in x]
    return deepcopy(x)


def digest(tensors):
    h = hashlib.sha256()
    for value in tensors:
        h.update(value.contiguous().numpy())
    return h.hexdigest()


def difference(a, b):
    assert a.keys() == b.keys()
    delta = norm = maximum = 0.
    for k, v in a.items():
        d = v.double() - b[k].double()
        maximum = max(maximum, float(d.abs().max()))
        delta += float(d.square().sum())
        norm += float(b[k].double().square().sum())
    return dict(max_abs=maximum, relative_l2=(delta/max(norm, 1e-30))**.5)


def run(args, stack):
    plan = json.loads(args.plan.read_text())
    argv, rank = plan['argv'], plan['rank']
    peers = [parse_peer(argv[i+1]) for i, v in enumerate(argv) if v == '--peer']
    def arg(k):
        return argv[argv.index(k)+1]
    assert int(arg('--world-size')) == 4 and int(arg('--ring-streams')) == 8
    peer = next(peer for peer in peers if peer.rank == (rank+1) % 4)
    run_dir = Path(plan['run'])
    assert (run_dir/'STOP').exists() and json.loads((run_dir/f'rank{rank}.exit.json').read_text())['exit_code'] == 0
    args.output.mkdir(parents=True, exist_ok=False)
    control = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='ppo-ol-')))
    def wait_for_neighbor(name):
        # Publish locally so a slower neighbor can create its own output
        # directory at any point. Each experiment requires a fresh directory.
        marker = args.output.resolve()/name
        marker.touch(exist_ok=False)
        poll = 'until test -f "$1"; do sleep 0.2; done'
        command = shlex.join(['timeout', '120', 'sh', '-c', poll, 'sh', str(marker)])
        ssh_cmd(peer, Path(arg('--ssh-key')), control, command, timeout=125)
    links = []
    with ExitStack() as partial:
        for i in range(8):
            # Wait until our neighbor is listening, on a range separate from production.
            link = SshRingAllReduce(rank=rank, world_size=4, next_host=peer.host, next_user=peer.user,
                                    next_port=peer.port, ssh_key=Path(arg('--ssh-key')),
                                    listen_port=args.ring_port+i, timeout=120,
                                    ready_barrier=lambda: wait_for_neighbor(f'listening-{i}'))
            partial.callback(link.close)
            links.append(link)
        ring = ParallelRingAllReduce(links)
        partial.pop_all()  # ring.close now owns all eight links
    stack.callback(ring.close)
    def avg(values):
        return average_scalar_dict(values, ring)

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = args.checkpoint_dir
    started = perf_counter()
    model = AccumulatingMaskablePPO.load(source/'model.zip', device='cpu')
    env = make_env(model, SimpleNamespace(ascension=0, interactive_reasoning=False, vec='native-subproc',
        rank=rank, seed=args.seed, native_search_threads=24, n_envs=256, envs_per_worker=3,
        start_method='forkserver'))
    try:
        model = AccumulatingMaskablePPO.load(source/'model.zip', env=env, device='cuda')
        model.set_logger(configure(folder=None, format_strings=[]))
        _, callback = model._setup_learn(256*768, callback=lambda *_: True, reset_num_timesteps=False)
        restore_learner(model, source, rank=rank, world_size=4)
        callback.on_training_start({}, {})
        assert model.policy.combat_value_target == 'whole-run' and getattr(model, 'deck_search_config', None) is None
        setup = perf_counter()-started
        started = perf_counter()
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 768)
        torch.cuda.synchronize()
        collection = perf_counter()-started
    finally:
        env.close()
    assert model.batch_size * model.gradient_accumulation_steps == 12288
    configure_packed_updates(model, dict(token_budget=196608, max_rows=1024, precision='fp32',
        context_grouping=True, upper_screen_max_rows=384, max_contexts_per_chunk=320))
    configure_advantage_normalization(model, 'rollout')
    prepare_advantage_normalization(model, average=avg, world_size=4)
    weights = clone_state(model.policy.state_dict())
    adam = clone_state(model.policy.optimizer.state_dict())
    report = dict(rank=rank, source=str(source), seed=args.seed, setup_sec=setup, collection_sec=collection,
                  normalization='global rollout', torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  cpu=next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
                  config=model.ppo_update_config, fp32='highest', world=4, streams=8, arms=[])
    reference = None
    ref_grads = []
    for qualify, order in ((True, ('A','B','C','D')), (False, ('D','C','B','A'))):
        for label in order:
            print('BEGIN', qualify, label, flush=True)
            torch.use_deterministic_algorithms(qualify)
            model.policy.load_state_dict(weights)
            model.policy.optimizer.load_state_dict(clone_state(adam))
            model.policy.set_training_mode(True)
            np.random.seed(args.seed+rank)
            torch.manual_seed(args.seed)
            model._ppo_microbatch_caps = {}
            model._ppo_update_metric_count = 0
            model._ppo_update_metric_sums = None
            model._ppo_microbatch_stats = dict(chunks=0, rows=0, oom_retries=0, min_rows=12288, max_rows=0)
            reducer = EarlyUpperReduction(model, ring, exact_ring=label=='D') if label in ('C','D') else None
            scheduler = reducer or (UpperFirstSchedule(model) if label=='B' else None)
            trial_grads = []
            def sync(step):
                if reducer:
                    ret = reducer.synchronize(step)
                else:
                    ret = synchronize_gradient_step(model, rank=rank, world_size=4, peers=[], key=None,
                        control_dir=None, local_root=None, remote_root=None, run_id=None, tag=None,
                        ring_transport=ring, barrier_timeout=120)
                if qualify:
                    trial_grads.append(pack_gradients(model))
                return ret
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with ResourceMonitor(interval=1., gpu=True, torch_device='cuda') as monitor:
                torch.cuda.synchronize()
                avg(dict(ready=1.))
                started = perf_counter()
                try:
                    result = packed_ppo_epoch(model, model.clip_range(model._current_progress_remaining),
                        None if model.clip_range_vf is None else model.clip_range_vf(model._current_progress_remaining),
                        synchronize=sync, average=avg,
                        schedule_chunks=scheduler.schedule if scheduler else None,
                        after_physical_chunk=scheduler.after_physical_chunk if scheduler else None)
                    torch.cuda.synchronize()
                    wall = perf_counter()-started
                finally:
                    if reducer:
                        reducer.close()
            assert result['optimizer_steps']==16 and model._ppo_microbatch_stats['rows']==model._ppo_update_metric_count==196608
            arm = dict(label=label, qualification=qualify, epoch_sec=wall, result=result,
                       chunks=model._ppo_microbatch_stats.copy(), resources=monitor.summary,
                       buckets=reducer.steps if reducer else None,
                       peak_reserved=torch.cuda.max_memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated())
            if qualify:
                policy = clone_state(model.policy.state_dict())
                state = clone_state(model.policy.optimizer.state_dict())['state']
                moments = {(n,k):v for n,s in state.items() for k,v in s.items() if k!='step'}
                steps = {n:float(s['step']) for n,s in state.items()}
                if reference is None:
                    reference = policy, moments, steps
                    ref_grads = trial_grads
                arm['policy_sha'] = digest(policy.values())
                arm['adam_sha'] = digest(v for s in state.values() for v in s.values())
                arm['policy_diff'] = difference(policy, reference[0])
                arm['adam_diff'] = difference(moments, reference[1])
                arm['same_steps'] = steps == reference[2]
                arm['gradient_close'] = [bool(torch.allclose(v,w,rtol=3e-4,atol=2e-6))
                                         for v,w in zip(trial_grads, ref_grads, strict=True)]
                arm['gradient_max_abs'] = [float((v-w).abs().max()) for v,w in zip(trial_grads,ref_grads,strict=True)]
                arm['passed'] = (arm['gradient_close'][0] and arm['same_steps']
                                 and arm['policy_diff']['max_abs']<5e-5 and arm['adam_diff']['relative_l2']<1e-4)
            report['arms'].append(arm)
            (args.output/'result.tmp').write_text(json.dumps(report, indent=2))
            (args.output/'result.tmp').replace(args.output/'result.json')
            print('END', qualify, label, round(wall, 3), arm.get('passed'),flush=True)
            if qualify and label in ('B','D') and not arm['passed']:
                raise AssertionError('distributed qualification failed; see result.json')
    # The local send can finish before the SSH tunnel delivers its last frame.
    wait_for_neighbor('finished')
    print('DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'checkpoint-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ring-port', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261111)
    with ExitStack() as stack:
        run(parser.parse_args(), stack)

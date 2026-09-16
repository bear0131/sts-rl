"""Fixed combat leaf inference workload for the shared benchmark CLI."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.benchmarks.resources import ResourceMonitor
from rl.policies import model as model_utils
from rl.policies.split_policy import SplitStSPointerPolicy
from rl.search.native import forward_leaves


def run(args):
    if args.leaf_repeats < 1:
        raise ValueError('leaf-repeats must be positive')
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    started = time.perf_counter()
    env = SlayTheSpireEnv(global_map_view=True)
    try:
        if args.checkpoint:
            model = AccumulatingMaskablePPO.load(args.checkpoint, device=args.device)
            env.close()
            env = SlayTheSpireEnv(global_map_view=True,
                card_language=model_utils.uses_card_language(model),
                combat_language=model_utils.uses_combat_language(model))
        else:
            model = AccumulatingMaskablePPO(SplitStSPointerPolicy, env, device=args.device,
                seed=args.seed, n_steps=8, batch_size=4,
                policy_kwargs=dict(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads))
        model.amp_mode = args.amp
        model.policy.set_training_mode(False)
        if args.leaf_inference == 'cuda-graph':
            from rl.search.inference import CombatLeafGraphs
            model._native_leaf_inference = CombatLeafGraphs(model.policy.core, model.device, amp=args.amp,
                profile_interval=args.graph_profile_interval)
        observations = []
        for index in range(0 if args.leaf_input else args.n_envs):
            env.reset(seed=args.seed + index)
            for _ in range(100):
                if env.thick.bc is not None:
                    break
                env.step(int(np.flatnonzero(env.action_masks())[0]))
            if env.thick.bc is None:
                raise RuntimeError('failed to reach combat in bounded fixture')
            observations.append(env._obs())
        if args.leaf_input:
            with np.load(args.leaf_input, allow_pickle=False) as saved:
                batch = {key:saved[key] for key in saved.files}
        else:
            batch = {key: np.stack([o[key] for o in observations])
                     for key in model.policy.core.combat_observation_keys
                     if key in observations[0] and key in model.observation_space.spaces}
        batch_size = next(iter(batch.values())).shape[0]
        def forward():
            return forward_leaves(model, batch, autocast=model.autocast_context)
        setup = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(10):
            expected = forward()
        warmup = time.perf_counter() - started
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with ResourceMonitor(interval=args.sample_interval, gpu=args.device == 'cuda') as monitor:
            started = time.perf_counter()
            for _ in range(args.leaf_repeats):
                result = forward()
            elapsed = time.perf_counter() - started
        for a, b in zip(expected, result):
            np.testing.assert_array_equal(a, b)
            if not np.isfinite(b).all():
                raise RuntimeError('non-finite leaf inference')
        profile = None
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                for _ in range(10):
                    forward()
            trace = args.output.with_suffix('.trace.json')
            prof.export_chrome_trace(str(trace))
            profile = str(trace)
            table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25)
            args.output.with_suffix('.operators.txt').write_text(table, encoding='utf-8')
        cuda = None if args.device != 'cuda' else {
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        import slaythespire
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        return dict(status='ok', schema_version=1, workload='leaves', config=config,
            identity=dict(python=platform.python_version(), torch=torch.__version__,
                device=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                native_sha256=hashlib.sha256(Path(slaythespire.__file__).read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None),
            timing=dict(setup_seconds=setup, warmup_seconds=warmup, measured_seconds=elapsed,
                leaf_batches=args.leaf_repeats, leaves=args.leaf_repeats * batch_size,
                milliseconds_per_batch=elapsed * 1000 / args.leaf_repeats),
            output_sha256=hashlib.sha256(b''.join(x.tobytes() for x in result)).hexdigest(),
            graphs=model._native_leaf_inference.statistics() if args.leaf_inference == 'cuda-graph' else None,
            memory=dict(cuda=cuda), resources=monitor.summary, samples=monitor.samples, profile=profile)
    finally:
        env.close()

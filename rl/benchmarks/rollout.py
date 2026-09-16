"""Fixed-checkpoint production VecEnv/search workload, without optimizer updates."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.benchmarks.resources import ResourceMonitor
from rl.search.native import NativeSearchConfig
from rl.training.environments import make_env


def run(args):
    if args.checkpoint is None or args.search_config is None:
        raise ValueError('rollout workload requires --checkpoint and --search-config')
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    started = time.perf_counter()
    model = AccumulatingMaskablePPO.load(args.checkpoint, device=args.device)
    env = make_env(model, SimpleNamespace(ascension=0, interactive_reasoning=False,
        vec='dummy' if args.vec_env == 'dummy' else 'subproc', rank=0, seed=args.seed,
        n_envs=args.n_envs, envs_per_worker=args.envs_per_worker if args.vec_env == 'grouped' else 1,
        start_method='spawn'))
    try:
        model = AccumulatingMaskablePPO.load(args.checkpoint, device=args.device, env=env,
            custom_objects=dict(n_steps=args.n_steps, batch_size=args.batch_size))
        config = NativeSearchConfig.from_mapping(json.loads(args.search_config.read_text()))
        for name, value in asdict(config).items():
            setattr(model, 'nosl_is_az_' + name, value)
        model.nosl_is_az_search_mode = {0:'puct', 1:'gumbel', 2:'full_gumbel'}[config.root_search_mode]
        model.nosl_is_az = model.nosl_is_az_cpp = True
        model.amp_mode = args.amp
        model.nosl_is_az_inference_schedule = args.inference_schedule
        model.nosl_is_az_inference_batch_size = args.inference_batch_size
        model.nosl_is_az_inference_max_wait_ms = args.inference_wait_ms
        model.set_random_seed(args.seed)
        if args.leaf_inference == 'cuda-graph':
            from rl.search.inference import CombatLeafGraphs
            model._native_leaf_inference = CombatLeafGraphs(model.policy.core, model.device, amp=args.amp,
                failure_path=args.output.with_suffix('.failed-leaves.npz'), profile_interval=args.graph_profile_interval)
        _, callback = model._setup_learn(args.n_steps * args.n_envs * args.updates,
            callback=lambda *_: True, reset_num_timesteps=True)
        callback.on_training_start({}, {})
        setup = time.perf_counter() - started
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with ResourceMonitor(interval=args.sample_interval, gpu=args.device == 'cuda') as monitor:
            started = time.perf_counter()
            for _ in range(args.updates):
                if not model.collect_rollouts(env, callback, model.rollout_buffer, args.n_steps):
                    raise RuntimeError('rollout stopped before budget')
            elapsed = time.perf_counter() - started
        digest = hashlib.sha256()
        def hash_value(value):
            if isinstance(value, dict):
                for key in sorted(value):
                    digest.update(key.encode())
                    hash_value(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value: hash_value(item)
            elif isinstance(value, np.ndarray):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(repr(value).encode())
        hash_value({key:getattr(model.rollout_buffer,key) for key in
                    ('observations','actions','rewards','values','log_probs','episode_starts')})
        hash_value(model._nosl_is_az_samples)
        metrics = {k:float(v) for k,v in model.logger.name_to_value.items()
                   if k.startswith('nosl_is_az/')}
        if not metrics.get('nosl_is_az/cpp_leaf_evaluations', 0):
            raise RuntimeError('budget did not exercise native search')
        import slaythespire
        cache = getattr(model, '_native_leaf_inference', None)
        return dict(status='ok', schema_version=1, workload='rollout',
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            search_config=asdict(config), output_sha256=digest.hexdigest(),
            identity=dict(torch=torch.__version__, native_sha256=hashlib.sha256(Path(slaythespire.__file__).read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()),
            timing=dict(setup_seconds=setup, measured_seconds=elapsed, transitions=model.num_timesteps,
                transitions_per_second=model.num_timesteps/elapsed),
            search_metrics=metrics, graphs=None if cache is None else cache.statistics(),
            memory=dict(cuda=None if args.device!='cuda' else dict(
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())),
            resources=monitor.summary,samples=monitor.samples)
    finally:
        env.close()

"""Bounded lower-only training workload for ``python -m rl.benchmarks.train``."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil
import torch
import env.core
import slaythespire as sts

from rl.algorithms.small_playout import BattleCollector, BattleFixture, SmallPlayoutLearner
from rl.policies.small_playout import collate_features
from rl.search.small_playout import SmallSearchConfig
from rl.training.resource_monitor import ResourceMonitor


def run(args):
    if args.amp != 'off' or args.accumulation != 1 or args.search != 'none' or args.checkpoint:
        raise ValueError('small-playout benchmark uses FP32, accumulation=1 and fresh initialization')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    started = time.perf_counter()
    exploration = .8 if args.simulations < 500 else 1.41 if args.simulations < 1000 else 2.5
    learner = SmallPlayoutLearner(seed=args.seed, device=args.device,
                                 config=SmallSearchConfig(simulations=args.simulations, exploration=exploration),
                                 threads=args.native_search_threads)
    fixture_config = BattleFixture(encounter=args.encounter)
    if args.battle_deck:
        from dataclasses import replace
        fixture_config = replace(fixture_config, deck=tuple(args.battle_deck))
    collector = BattleCollector(fixture_config, n_envs=args.n_envs, seed=args.seed)
    learner.synchronize()
    setup_seconds = time.perf_counter() - started
    before_warmup = time.perf_counter()
    for _ in range(args.warmup_updates):
        samples, _ = collector.collect(learner.search, args.n_steps)
        learner.update(samples, batch_size=args.batch_size, epochs=args.epochs)
    learner.synchronize()
    warmup_seconds = time.perf_counter() - before_warmup
    # Fixed-root measurements are diagnostic only, outside full collection timing.
    fixture = collector.fixture.create(args.seed)
    features = sts.small_playout_features(fixture)
    tensors = collate_features([features], learner.device)
    with torch.inference_mode():
        expected = learner.policy(*tensors).cpu().numpy()[0]
    actual = np.asarray(learner.search.net.logits(fixture))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    repeats = 1000
    tick = time.perf_counter()
    for _ in range(repeats):
        learner.search.net.logits(fixture)
    native_us = (time.perf_counter() - tick) * 1e6 / repeats
    learner.synchronize()
    tick = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            learner.policy(*tensors)
    learner.synchronize()
    torch_us = (time.perf_counter() - tick) * 1e6 / repeats
    checkpoint = args.output.with_suffix('.pt')
    reports = []
    save_seconds = restore_seconds = 0.
    monitor = ResourceMonitor(interval=args.sample_interval, gpu=True, torch_device=args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with monitor, args.output.with_suffix('.jsonl').open('w', encoding='utf-8') as log:
        tick = time.perf_counter()
        for index in range(args.updates):
            round_start = time.perf_counter()
            samples, report = collector.collect(learner.search, args.n_steps)
            report['learner'] = learner.update(samples, batch_size=args.batch_size, epochs=args.epochs)
            save_start = time.perf_counter()
            learner.save(checkpoint, collector)
            save_time = time.perf_counter() - save_start
            save_seconds += save_time
            # Exercise actual saved optimizer + pending battle replay, then use it.
            if index == 0:
                restore_start = time.perf_counter()
                restored, pending = SmallPlayoutLearner.load(checkpoint, device=args.device,
                                                           threads=args.native_search_threads)
                np.testing.assert_array_equal(restored.policy.export(), learner.policy.export())
                if [b.nosl_future_key() for b in pending.roots] != [b.nosl_future_key() for b in collector.roots]:
                    raise AssertionError('checkpoint battle journal replay changed the public state')
                learner, collector = restored, pending
                restore_seconds += time.perf_counter() - restore_start
            report.update(round=index+1, save_seconds=save_time,
                          round_seconds=time.perf_counter() - round_start, global_steps=collector.steps)
            log.write(json.dumps(report) + '\n')
            log.flush()
            reports.append(report)
        learner.synchronize()
        measured_seconds = time.perf_counter() - tick
    counts = {}
    for report in reports:
        for key, value in report['counts'].items():
            if key == 'playout_lengths':
                counts.setdefault(key, np.zeros(201, np.int64))
                counts[key] += value
            else:
                counts[key] = counts.get(key, 0) + value
    histogram = counts['playout_lengths']
    counts['playout_lengths'] = histogram.tolist()
    counts['mean_playout_actions'] = float(np.dot(histogram, np.arange(201)) / max(1, histogram.sum()))
    counts['depth_cut_fraction'] = counts['depth_cuts'] / counts['simulations']
    executed = counts['real_transitions'] + counts['tree_transitions'] + counts['playout_transitions']
    cuda = None
    if args.device == 'cuda':
        cuda = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(status='ok', schema_version=1,
                scope='lower-only fixed-deck battle fixtures; not whole-game PPO throughput',
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                identity=dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                              dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                              native_module=str(Path(sts.__file__).resolve()),
                              native_sha256=hashlib.sha256(Path(sts.__file__).read_bytes()).hexdigest(),
                              python=platform.python_version(), platform=platform.platform(), cpu=platform.processor(),
                              cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False),
                              torch=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
                architecture=dict(learner.search.manifest(), key_count=learner.policy.key_count,
                                  parameters=learner.policy.export().size, state_features=32, action_features=24),
                timing=dict(setup_seconds=setup_seconds, warmup_seconds=warmup_seconds,
                            measured_seconds=measured_seconds, collection_seconds=counts['collection_seconds'],
                            optimization_seconds=sum(r['learner']['optimization_seconds'] for r in reports),
                            save_seconds=save_seconds, restore_seconds=restore_seconds,
                            executed_env_transitions=executed, all_transitions_per_second=executed/measured_seconds,
                            collection_us_per_executed_transition=counts['collection_seconds']*1e6/executed,
                            real_actions_per_second=counts['real_transitions']/measured_seconds,
                            native_feature_encode_plus_logits_us=native_us, torch_cached_feature_logits_us=torch_us),
                counts=counts, rounds=reports, checkpoint=str(checkpoint),
                memory=dict(parameter_bytes=learner.policy.export().nbytes, cuda=cuda),
                resources=monitor.summary, samples=monitor.samples)

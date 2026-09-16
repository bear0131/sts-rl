"""Bounded CUDA graph cache for deterministic combat leaf inference.

Graphs retain model parameter addresses, not a weight snapshot. AMP weight casts
are captured with caching disabled so optimizer updates remain visible. Call
clear() after replacing parameters or changing model structure. This backend is
owned by the runtime, excluded from checkpoints, and never used for backward.
"""
from collections import OrderedDict
import json
import time
from pathlib import Path

import numpy as np
import torch


class CombatLeafGraphs:
    def __init__(self, core, device, *, amp='off', capacity=32, failure_path=None,
                 profile_interval=0, batch_multiple=0):
        if torch.device(device).type != 'cuda':
            raise ValueError('combat CUDA graphs require a CUDA device')
        if capacity < 1 or amp not in ('off', 'bf16'):
            raise ValueError('invalid graph capacity or AMP mode')
        self.core, self.device, self.amp, self.capacity = core, device, amp, capacity
        self.entries = OrderedDict()
        self.hits = self.captures = 0
        self.failure_path = failure_path
        self.profile_interval = int(profile_interval)
        if self.profile_interval < 0:
            raise ValueError('profile interval must be nonnegative')
        self.evictions = self.profile_samples = 0
        self.capture_seconds = self.copy_seconds = self.output_seconds = 0.0
        self.h2d_ms = self.replay_ms = self.d2h_ms = 0.0
        self.batch_counts = {}
        self.batch_multiple = int(batch_multiple)
        if self.batch_multiple < 0:
            raise ValueError('batch multiple must be nonnegative')

    def statistics(self):
        return dict(captures=self.captures, hits=self.hits, evictions=self.evictions,
            cached_graphs=len(self.entries), capture_seconds=self.capture_seconds,
            copy_seconds=self.copy_seconds, output_seconds=self.output_seconds,
            profile_samples=self.profile_samples, h2d_ms=self.h2d_ms,
            replay_ms=self.replay_ms, d2h_ms=self.d2h_ms, batch_counts=dict(self.batch_counts))

    def clear(self):
        self.entries.clear()

    def __call__(self, observations, width):
        try:
            return self._forward(observations, width)
        except RuntimeError:
            # CUDA errors can poison the context: save only CPU inputs here.
            if self.failure_path is not None:
                target = Path(self.failure_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, **observations)
                target.with_suffix('.json').write_text(json.dumps(dict(
                    width=width, amp=self.amp, captures=self.captures, hits=self.hits,
                    signatures=[repr(s) for s in self.entries])), encoding='utf-8')
            raise

    def _forward(self, observations, width):
        if self.core.training:
            raise RuntimeError('leaf graphs require eval mode')
        batch = len(next(iter(observations.values())))
        if batch < 1 or any(len(v) != batch for v in observations.values()):
            raise ValueError('leaf observations require equal nonempty batches')
        padded = ((batch + self.batch_multiple - 1) // self.batch_multiple * self.batch_multiple
                  if self.batch_multiple else batch)
        signature = (width, tuple((k, (padded, *v.shape[1:]), v.dtype.str)
                                  for k, v in observations.items()))
        if signature not in self.entries:
            if len(self.entries) >= self.capacity:
                self.entries.popitem(last=False)
                self.evictions += 1
            started = time.perf_counter()
            self.entries[signature] = self._capture(observations, width, padded)
            self.capture_seconds += time.perf_counter()-started
            self.captures += 1
        else:
            self.hits += 1
        self.entries.move_to_end(signature)
        entry = self.entries[signature]
        self.batch_counts[batch] = self.batch_counts.get(batch, 0)+1
        events = ([torch.cuda.Event(enable_timing=True) for _ in range(4)]
                  if self.profile_interval and (self.hits+self.captures) % self.profile_interval == 0 else None)
        if events: events[0].record()
        started = time.perf_counter()
        self._copy(entry, observations)
        self.copy_seconds += time.perf_counter()-started
        if events: events[1].record()
        entry['graph'].replay()
        if events: events[2].record()
        started = time.perf_counter()
        result = entry['logits'][:batch].cpu().numpy(), entry['values'][:batch].cpu().numpy()
        self.output_seconds += time.perf_counter()-started
        if events:
            events[3].record(); events[3].synchronize()
            self.h2d_ms += events[0].elapsed_time(events[1])
            self.replay_ms += events[1].elapsed_time(events[2])
            self.d2h_ms += events[2].elapsed_time(events[3])
            self.profile_samples += 1
        return result

    @staticmethod
    def _copy(entry, observations):
        for host, device, layout in entry['buffers']:
            array = host.numpy()
            for key, start, end, shape in layout:
                target = array[start:end].reshape(shape)
                value = observations[key]
                np.copyto(target[:len(value)], value)
                # Valid duplicate rows avoid fully masked dummy observations.
                # Combat inference is independent across batch rows; padding
                # never enters the search or its returned action/value arrays.
                target[len(value):] = value[0]
            device.copy_(host, non_blocking=True)

    def _capture(self, observations, width, padded):
        groups = {}
        for key, value in observations.items():
            groups.setdefault(value.dtype, []).append((key, value))
        buffers, tensors = [], {}
        for dtype, fields in groups.items():
            count = sum(padded * int(np.prod(v.shape[1:])) for _, v in fields)
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            host = torch.empty(count, dtype=torch_dtype, pin_memory=True)
            device = torch.empty(count, dtype=torch_dtype, device=self.device)
            offset, layout = 0, []
            for key, value in fields:
                shape = (padded, *value.shape[1:])
                end = offset + int(np.prod(shape))
                tensors[key] = device[offset:end].view(shape)
                layout.append((key, offset, end, shape))
                offset = end
            buffers.append((host, device, layout))
        entry = dict(buffers=buffers)
        self._copy(entry, observations)
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        def forward():
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                    enabled=self.amp == 'bf16', cache_enabled=False):
                logits, values = self.core.combat_actor_critic(tensors, active_relic_width=width)
                return logits.float(), values.float().reshape(-1)
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        # Capture each graph's cuBLAS workspace in its own private pool. CUDA
        # streams come from a finite reusable pool; an evicted graph must not
        # release workspace still referenced by another graph on that stream.
        torch._C._cuda_clearCublasWorkspaces()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits, values = forward()
        current.wait_stream(stream)
        torch._C._cuda_clearCublasWorkspaces()
        entry.update(graph=graph, logits=logits, values=values)
        return entry

"""Bounded CUDA graph cache for deterministic combat leaf inference.

Graphs retain model parameter addresses, not a weight snapshot. AMP weight casts
are captured with caching disabled so optimizer updates remain visible. Call
clear() after replacing parameters or changing model structure. This backend is
owned by the runtime, excluded from checkpoints, and never used for backward.
"""
from collections import OrderedDict

import numpy as np
import torch


class CombatLeafGraphs:
    def __init__(self, core, device, *, amp='off', capacity=32):
        if torch.device(device).type != 'cuda':
            raise ValueError('combat CUDA graphs require a CUDA device')
        if capacity < 1 or amp not in ('off', 'bf16'):
            raise ValueError('invalid graph capacity or AMP mode')
        self.core, self.device, self.amp, self.capacity = core, device, amp, capacity
        self.entries = OrderedDict()
        self.hits = self.captures = 0

    def clear(self):
        self.entries.clear()

    def __call__(self, observations, width):
        if self.core.training:
            raise RuntimeError('leaf graphs require eval mode')
        signature = (width, tuple((k, v.shape, v.dtype.str) for k, v in observations.items()))
        if signature not in self.entries:
            if len(self.entries) >= self.capacity:
                self.entries.popitem(last=False)
            self.entries[signature] = self._capture(observations, width)
            self.captures += 1
        else:
            self.hits += 1
        self.entries.move_to_end(signature)
        entry = self.entries[signature]
        self._copy(entry, observations)
        entry['graph'].replay()
        return entry['logits'].cpu().numpy(), entry['values'].cpu().numpy()

    @staticmethod
    def _copy(entry, observations):
        for host, device, layout in entry['buffers']:
            array = host.numpy()
            for key, start, end in layout:
                np.copyto(array[start:end], observations[key].reshape(-1))
            device.copy_(host, non_blocking=True)

    def _capture(self, observations, width):
        groups = {}
        for key, value in observations.items():
            groups.setdefault(value.dtype, []).append((key, value))
        buffers, tensors = [], {}
        for dtype, fields in groups.items():
            count = sum(v.size for _, v in fields)
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
            host = torch.empty(count, dtype=torch_dtype, pin_memory=True)
            device = torch.empty(count, dtype=torch_dtype, device=self.device)
            offset, layout = 0, []
            for key, value in fields:
                end = offset + value.size
                tensors[key] = device[offset:end].view(value.shape)
                layout.append((key, offset, end))
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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits, values = forward()
        current.wait_stream(stream)
        entry.update(graph=graph, logits=logits, values=values)
        return entry

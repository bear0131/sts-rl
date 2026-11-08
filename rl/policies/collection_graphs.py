"""Bounded FP32 phase graphs owned by one shared-PPO collection.

Only masked trailing padding and independent batch padding change physical shape.
Sampling, legal-action masking and log probabilities remain in the policy.
No graph or temporary CUDA allocation is saved with a checkpoint.
"""
from collections import OrderedDict
import time

import torch

from env.core import encode, vocab
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS


class SharedCollectionGraphs:
    def __init__(self, core, *, capacity=192):
        if capacity < 1 or next(core.parameters()).device.type != 'cuda':
            raise ValueError('collection graphs require CUDA and positive capacity')
        if getattr(core, 'training_objective', None) != 'shared-wholerun-ppo-room-v1':
            raise ValueError('collection graphs require the shared whole-run policy')
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in core.parameters()):
            raise ValueError('collection graphs require FP32 parameters')
        self.core, self.capacity = core, capacity
        self.entries = OrderedDict()
        self.closed = False
        self.stats = dict(captures=0, hits=0, evictions=0, capture_seconds=0.)

    @staticmethod
    def _width(ids, minimum):
        columns = ids.ne(0).any(dim=0).nonzero()
        width = int(columns[-1, 0]) + 1 if columns.numel() else 1
        return min(ids.shape[1], max(minimum, 1 << (width - 1).bit_length()))

    def _capture(self, batch, padded, forward):
        count = len(next(iter(batch.values())))
        inputs = {key: torch.cat([value, value[:1].expand(padded-count, *value.shape[1:])]).clone()
                  for key, value in batch.items()}
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = forward(inputs)
        current.wait_stream(stream)
        torch.cuda.synchronize()
        # cuBLAS scratch must belong to the graph, not an eager allocation that
        # can be freed/reused when another shape is captured or evicted.
        torch._C._cuda_clearCublasWorkspaces()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward(inputs)
        current.wait_stream(stream)
        torch._C._cuda_clearCublasWorkspaces()
        return inputs, graph, output

    def __call__(self, observations):
        if self.closed or self.core.training or torch.is_grad_enabled():
            raise RuntimeError('collection graphs require a live eval/no-grad scope')
        if torch.is_autocast_enabled('cuda'):
            raise RuntimeError('collection graphs require FP32, without autocast')
        # Stable lookup-table storage is refreshed even on cache hits. Direct
        # parameter writes/Adam updates are visible; replacing parameters needs
        # a fresh collection runtime.
        self.core.prepare_inference_cache()
        phase = observations['phase'].long().flatten()
        indices, logits, values = [], [], []
        for phase_id in torch.unique(phase).tolist():
            selected = (phase == phase_id).nonzero().flatten()
            combat = phase_id == vocab.PHASE_VOCAB_INDEX['combat']
            keys = (COMBAT_OBSERVATION_KEYS if combat
                    else encode.upper_observation_keys(phase_id, observations))
            batch = {key: value[selected] for key, value in observations.items() if key in keys}
            relic = self._width(batch['relics'], 4)
            deck = None if combat else self._width(batch['deck_ids'], 16)
            for key in ('relics', 'relic_counters'):
                batch[key] = batch[key][:, :relic]
            if deck is not None:
                for key in ('deck_ids', 'deck_upgrades'):
                    batch[key] = batch[key][:, :deck]
            count = len(selected)
            padded = 1 << (count - 1).bit_length()
            signature = (phase_id, padded, deck, relic,
                         tuple((key, tuple(value.shape[1:]), value.dtype) for key, value in batch.items()))

            def forward(tensors):
                if combat:
                    return self.core.combat_actor_critic(tensors, active_relic_width=relic)
                return self.core.overworld(tensors, self.core.upper_tokenizer, phase_id,
                                           upper_active_widths=(deck, relic))

            if signature not in self.entries:
                if len(self.entries) >= self.capacity:
                    self.entries.popitem(last=False)
                    self.stats['evictions'] += 1
                started = time.perf_counter()
                self.entries[signature] = self._capture(batch, padded, forward)
                self.stats['capture_seconds'] += time.perf_counter() - started
                self.stats['captures'] += 1
            else:
                self.stats['hits'] += 1
            self.entries.move_to_end(signature)
            inputs, graph, output = self.entries[signature]
            for key, value in batch.items():
                inputs[key][:count].copy_(value)
            graph.replay()
            # Results own storage; subsequent replay/eviction cannot mutate
            # them. Register the clone's stream before releasing private pools.
            out, value = [tensor[:count].clone() for tensor in output]
            for tensor in output:
                tensor.record_stream(torch.cuda.current_stream())
            indices.append(selected)
            logits.append(out.float())
            values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    def close(self):
        if not self.closed:
            torch.cuda.synchronize()
            self.entries.clear()
            self.closed = True

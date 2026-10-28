"""Exact, bounded raw-network memoization owned by one collection.

Keys contain canonical input bytes plus physical execution widths, never tree,
environment or worldline identity. They do not stand for simulator equivalence.
"""
from collections import OrderedDict
import time

import numpy as np
import torch

from rl.search.inference import CombatLeafGraphs


CACHE_METRICS = ('requests', 'misses', 'hits', 'batch_duplicates', 'evictions',
                 'bytes', 'peak_bytes', 'key_seconds', 'forward_seconds')
GRAPH_METRICS = ('captures', 'hits', 'evictions', 'cached_graphs',
                 'capture_seconds', 'copy_seconds', 'output_seconds')
ROLLOUT_INFERENCE_METRICS = (
    *(f'{side}_cache_{key}' for side in ('upper', 'lower') for key in CACHE_METRICS),
    *(f'upper_graph_{key}' for key in GRAPH_METRICS),
    'outer_settlement_seconds', 'upper_prepare_seconds', 'outer_overlap_batches', 'outer_overlap_rows',
    'outer_exit_rows', 'outer_value_seconds', 'outer_value_rows', 'outer_value_batches',
)


class ExactInputCache:
    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError('cache capacity must be nonnegative')
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.scopes = OrderedDict()
        self.next_scope = 0
        self.stats = dict.fromkeys(CACHE_METRICS, 0)

    def clear(self):
        self.entries.clear()
        self.scopes.clear()
        self.next_scope = 0
        self.stats['bytes'] = 0

    def forward(self, observations, forward, *, namespace=()):
        started = time.perf_counter()
        count = len(next(iter(observations.values())))
        if not count or any(len(v) != count for v in observations.values()):
            raise ValueError('cache inputs require an equal nonempty batch')
        if self.max_bytes == 0:
            self.stats['requests'] += count
            self.stats['misses'] += count
            result = forward(observations)
            self.stats['forward_seconds'] += time.perf_counter()-started
            return result
        schema = tuple((k, v.shape[1:], v.dtype.str) for k, v in sorted(observations.items()))
        # Batch-wise packing avoids a Python field loop for every leaf. Python
        # bytes keys use hash tables with full equality on collision.
        columns = [np.ascontiguousarray(observations[k]).view(np.uint8).reshape(count, -1)
                   for k, _, _ in schema]
        packed = np.concatenate(columns, axis=1)
        scope = (namespace, schema)
        if scope not in self.scopes:
            if len(self.scopes) >= 64:
                self.scopes.popitem(last=False)
            # Hash the full schema once per batch, not once per row lookup.
            # Never recycle a number while an older entry can still exist.
            self.scopes[scope] = self.next_scope
            self.next_scope += 1
        scope = self.scopes[scope]
        keys = [(scope, row.tobytes()) for row in packed]
        found, missing, pending = {}, [], {}
        for i, key in enumerate(keys):
            if key in self.entries:
                self.entries.move_to_end(key)
                found[i] = self.entries[key][0]
                self.stats['hits'] += 1
            elif key in pending:
                self.stats['batch_duplicates'] += 1
            else:
                pending[key] = len(missing)
                missing.append(i)
        self.stats['requests'] += count
        self.stats['misses'] += len(missing)
        self.stats['key_seconds'] += time.perf_counter()-started
        if missing:
            started = time.perf_counter()
            logits, values = forward({k: v[missing] for k, v in observations.items()})
            results = np.concatenate((np.asarray(logits, np.float32),
                                      np.asarray(values, np.float32).reshape(-1, 1)), axis=1)
            self.stats['forward_seconds'] += time.perf_counter()-started
            for i, result in zip(missing, results, strict=True):
                value = result.copy()
                key = keys[i]
                # Conservative allowance for entry/key/index Python objects.
                cost = len(key[1]) + value.nbytes + 512
                if cost <= self.max_bytes:
                    while self.entries and self.stats['bytes'] + cost > self.max_bytes:
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.stats['bytes'] -= removed
                        self.stats['evictions'] += 1
                    self.entries[key] = value, cost
                    self.stats['bytes'] += cost
                    self.stats['peak_bytes'] = max(self.stats['peak_bytes'], self.stats['bytes'])
        # Results own their memory; later mutation, eviction or graph replay
        # cannot modify the cache or a previously returned search response.
        output = np.stack([found[i] if i in found else results[pending[key]]
                           for i, key in enumerate(keys)])
        return output[:, :-1], output[:, -1]


class _UpperPhaseCore:
    def __init__(self, teacher):
        self.teacher = teacher

    @property
    def training(self):
        return self.teacher.training

    def combat_actor_critic(self, observations, *, active_relic_width):
        phase, deck, relics = active_relic_width
        return self.teacher.forward_phase(observations, phase, (deck, relics))


def _upper_graph_inputs(observations, widths, capacities):
    """Group masked padding widths so small deck changes do not recapture.

    Token IDs, positions and counters stay intact. Only already masked PAD
    tokens are added, within the frozen checkpoint's positional capacity.
    """
    padded = tuple(min(cap, max(minimum, 1 << (width - 1).bit_length()))
                   for width, cap, minimum in zip(widths, capacities, (16, 4), strict=True))
    result = dict(observations)
    for keys, width, target in zip((('deck_ids', 'deck_upgrades'), ('relics', 'relic_counters')),
                                   widths, padded, strict=True):
        for key in keys:
            if width != target:
                result[key] = np.pad(observations[key], ((0, 0), (0, target-width)))
    return result, padded


class RolloutInference:
    def __init__(self, model, *, upper_bytes=128*1024**2, lower_bytes=0):
        self.model = model
        self.upper = ExactInputCache(upper_bytes)
        self.lower = ExactInputCache(lower_bytes)
        self.upper_graphs = None
        if torch.device(model.device).type == 'cuda' and getattr(model, '_native_leaf_inference', None) is not None:
            self.upper_graphs = CombatLeafGraphs(_UpperPhaseCore(model.policy.core.teacher),
                model.device, amp='off', capacity=32, batch_multiple=8, compile_forward=False)
        self.outer_settlement_seconds = 0.
        self.upper_prepare_seconds = 0.
        self.outer_overlap_batches = self.outer_overlap_rows = 0
        self.outer_exit_rows = self.outer_value_rows = self.outer_value_batches = 0
        self.outer_value_seconds = 0.

    def forward_upper(self, observations):
        started = time.perf_counter()
        groups = self.model.policy.core.teacher.prepare_numpy(observations)
        self.upper_prepare_seconds += time.perf_counter()-started
        return self._forward_upper(groups, len(observations['phase']))

    def forward_upper_rows(self, observations):
        started = time.perf_counter()
        groups = self.model.policy.core.teacher.prepare_search_numpy(observations)
        self.upper_prepare_seconds += time.perf_counter()-started
        return self._forward_upper(groups, len(observations))

    def _forward_upper(self, groups, count):
        teacher = self.model.policy.core.teacher
        output = None
        for indices, phase, widths, obs in groups:
            if self.upper_graphs is not None:
                capacities = tuple(teacher.observation_space.spaces[key].shape[0]
                                   for key in ('deck_ids', 'relics'))
                obs, widths = _upper_graph_inputs(obs, widths, capacities)
            signature = (phase, *widths)
            def forward(batch):
                if self.upper_graphs is not None:
                    return self.upper_graphs(batch, signature)
                tensors = {k: torch.as_tensor(v, device=self.model.device) for k, v in batch.items()}
                logits, values = teacher.forward_phase(tensors, phase, widths)
                return logits.cpu().numpy(), values.reshape(-1).cpu().numpy()
            logits, values = self.upper.forward(obs, forward, namespace=signature)
            if output is None:
                output = np.empty((count, logits.shape[1]), np.float32)
                result_values = np.empty(len(output), np.float32)
            output[indices], result_values[indices] = logits, values
        return output, result_values

    def statistics(self):
        result = {f'{side}_cache_{k}': v for side in ('upper', 'lower')
                  for k, v in getattr(self, side).stats.items()}
        graph = self.upper_graphs.statistics() if self.upper_graphs else dict.fromkeys(GRAPH_METRICS, 0)
        result.update({f'upper_graph_{k}': graph[k] for k in GRAPH_METRICS})
        result.update(outer_settlement_seconds=self.outer_settlement_seconds,
                      upper_prepare_seconds=self.upper_prepare_seconds,
                      outer_overlap_batches=self.outer_overlap_batches,
                      outer_overlap_rows=self.outer_overlap_rows,
                      outer_exit_rows=self.outer_exit_rows,
                      outer_value_seconds=self.outer_value_seconds,
                      outer_value_rows=self.outer_value_rows,
                      outer_value_batches=self.outer_value_batches)
        return result

    def clear(self):
        self.upper.clear()
        self.lower.clear()
        if self.upper_graphs is not None:
            self.upper_graphs.clear()

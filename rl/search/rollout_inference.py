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


class _UnifiedUpperCore:
    """Static, all-upper forward consumed by the existing graph transport."""
    def __init__(self, core):
        self.core = core

    @property
    def training(self):
        return self.core.training

    def combat_actor_critic(self, observations, *, active_relic_width):
        return (self.core.upper_actor(observations, compact=False),
                self.core.upper_critic(observations, compact=False))


class UpperPolicyInference:
    """One collection's upper inference; no graph state enters a checkpoint.

    CUDA unified-upper batches of at most 64 rows use bounded graphs.
    Deck consumers use the same transport for cached context and dynamic heads.
    Other devices and older policies retain their existing prediction path.
    The caller owns phase scheduling and categorical sampling, as before.
    """
    def __init__(self, core, device, *, context_bytes=128*1024**2):
        self.core, self.device = core, device
        self.graphs = None
        self.closed = False
        self.deck = None
        if getattr(core, 'model_contract', None) == 'deck-latent-combat-v1':
            self.deck = _DeckConsumers(core, device, context_bytes)
        if (torch.device(device).type == 'cuda'
                and getattr(getattr(core, 'upper_config', None), 'architecture', None)
                    == 'upper-latent-actions-v1'):
            self.graphs = CombatLeafGraphs(_UnifiedUpperCore(core), device, amp='off',
                capacity=8, batch_multiple=8, compile_forward=False, device_outputs=True)

    def __enter__(self):
        if self.closed:
            raise RuntimeError('upper inference collection already closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        if self.graphs is not None:
            self.graphs.clear()
        if self.deck is not None:
            self.deck.close()

    def __call__(self, observations):
        if self.closed:
            raise RuntimeError('upper inference collection already closed')
        if self.deck is not None:
            return self.deck(observations)
        from env.core import vocab as V
        from rl.search.native import observation_as_tensor
        count = len(observations['phase'])
        if (self.graphs is None or count > 64
                or np.any(observations['phase'] == V.PHASE_VOCAB_INDEX['combat'])):
            with torch.inference_mode():
                return self.core.actor_critic(observation_as_tensor(observations, self.device))
        logits, values = self.graphs(observations, None)
        # Keep the existing device-side masking/log-probability computation and
        # per-game CPU categorical RNG. These small tensors own their storage.
        return logits, values[:, None]

    def statistics(self):
        if self.deck is not None:
            return self.deck.statistics()
        values = self.graphs.statistics() if self.graphs is not None else dict.fromkeys(GRAPH_METRICS, 0)
        return {f'upper_graph_{key}': values[key] for key in GRAPH_METRICS}

    def search_context(self, observations):
        """The current consumer's frozen context, shared by its branch rows."""
        if self.closed or self.deck is None:
            raise RuntimeError('search context requires an open deck consumer')
        from rl.policies.deck_latent import SearchContext
        _, combat, anchor = self.deck.context(observations)
        return SearchContext(torch.as_tensor(combat, device=self.device),
                             torch.as_tensor(anchor, device=self.device))


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


class _DeckForward:
    """Only tensor interfaces for the old graph executor; no scheduling here."""
    inference_scratch_only = True

    def __init__(self, core, context=False):
        self.core, self.context = core, context

    @property
    def training(self):
        return self.core.training

    def prepare_inference_cache(self):
        self.core.combat.inference_card_table()

    def combat_actor_critic(self, obs, *, active_relic_width):
        core = self.core
        if self.context:
            z = core.context(obs, active_widths=active_relic_width)
            return torch.cat((z, core.combat.context_projection(z),
                              core.anchor.context_projection(z)), -1), z.new_zeros(len(z), 1)
        phase, anchor, deck_width, relic_width, choice_width = active_relic_width
        from env.core import vocab as V, native_actions as C
        if anchor:
            value = core.anchor_with_context(obs, obs['_anchor_context'])
            return value.new_empty(len(value), 0), value
        if phase == V.PHASE_VOCAB_INDEX['combat']:
            return core.combat(dict(obs, _choice_width=choice_width))[:2]
        return core.overworld(obs, core.upper_tokenizer, phase,
                              upper_active_widths=(deck_width, relic_width))


def _observed_width(values):
    """Retain interior slots, including legal candidates whose card ID is PAD."""
    present = np.asarray(values, bool).any(axis=0)
    return max(1, int(np.flatnonzero(present)[-1]) + 1 if present.any() else 1)


class _DeckConsumers:
    """Current model heads on the original exact cache and graph transport.

    The large encoder and both static projections are memoized together.
    Dynamic HP, counters, piles and candidate slots only reach the small heads.
    All cached rows and returned results have independent storage.
    """
    def __init__(self, core, device, context_bytes):
        if context_bytes <= 0:
            raise ValueError('deck sampling requires a positive context cache capacity')
        self.core, self.device = core, device
        self.cache = ExactInputCache(context_bytes)
        self.anchor_cache = ExactInputCache(context_bytes)
        self.encoder = _DeckForward(core, context=True)
        self.head = _DeckForward(core)
        self.enc_graphs = self.head_graphs = None
        self.versions = self.addresses = None
        if torch.device(device).type == 'cuda':
            self.enc_graphs = CombatLeafGraphs(self.encoder, device, amp='off',
                capacity=8, batch_multiple=8, compile_forward=False)
            self.head_graphs = CombatLeafGraphs(self.head, device, amp='off',
                capacity=32, batch_multiple=8, compile_forward=False, device_outputs=True)

    def _check_weights(self):
        if self.core.training or torch.is_grad_enabled():
            raise RuntimeError('deck collection requires frozen inference without gradients')
        tensors = tuple(self.core.parameters()) + tuple(self.core.buffers())
        addresses = tuple((id(p), p.data_ptr(), str(p.device), p.dtype) for p in tensors)
        versions = tuple(p._version for p in tensors)
        if addresses != self.addresses:
            for graphs in (self.enc_graphs, self.head_graphs):
                if graphs is not None:
                    graphs.clear()
        if addresses != self.addresses or versions != self.versions:
            self.cache.clear()
            self.anchor_cache.clear()
        self.addresses, self.versions = addresses, versions

    def _forward(self, forward, graphs, obs, signature):
        if graphs is not None:
            return graphs(obs, signature)
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
        logits, values = forward.combat_actor_critic(tensors, active_relic_width=signature)
        return logits.float().cpu().numpy(), values.float().flatten().cpu().numpy()

    def context(self, obs):
        self._check_weights()
        static = {k: np.asarray(obs[k]) for k in self.core.context.keys}
        width = self.core.combat.context_projection.in_features
        combat_width = self.core.combat.context_projection.out_features
        anchor_width = self.core.anchor.context_projection.out_features
        row_cost = sum(v[0].nbytes for v in static.values()) + (width+combat_width+anchor_width+1)*4 + 512
        if self.cache.max_bytes < row_cost:
            raise ValueError('context cache cannot hold even one complete context row')

        def encode(missing):
            widths = (_observed_width(missing['deck_ids']), _observed_width(missing['relics']))
            return self._forward(self.encoder, self.enc_graphs, missing, widths)

        encoded, _ = self.cache.forward(static, encode)
        return encoded[:, :width], encoded[:, width:width+combat_width], encoded[:, width+combat_width:]

    def __call__(self, obs):
        from env.core import vocab as V, native_actions as C
        run, combat, anchor = self.context(obs)
        batch = dict(obs, _run_context=run, _combat_context=combat, _anchor_context=anchor)
        phases = np.asarray(obs['phase']).reshape(-1)
        anchored = np.asarray(obs['combat_exit_anchor'], bool).any(axis=1)
        logits = torch.empty((len(phases), C.AUTOREGRESSIVE_SIZE), device=self.device)
        values = torch.empty(len(phases), device=self.device)
        for phase in np.unique(phases):
            for at_anchor in ((False, True) if phase == V.PHASE_VOCAB_INDEX['combat'] else (False,)):
                rows = np.flatnonzero((phases == phase) & (anchored == at_anchor))
                if not len(rows):
                    continue
                data = {k: v[rows] for k, v in batch.items()}
                pick = C.JOINT_PICK - C.JOINT_END
                required = ((data['select_native_indices'] > 0)
                            | data['combat_legal'][:, pick:pick+data['select_native_indices'].shape[1]].astype(bool))
                signature = (int(phase), at_anchor, _observed_width(data['deck_ids']),
                             _observed_width(data['relics']), _observed_width(required))
                index = torch.as_tensor(rows, device=self.device)
                if at_anchor:
                    keys = ('player_scalars', 'potions', 'relics', 'relic_counters',
                            '_anchor_context', *self.core.anchor.dynamic_keys)
                    inputs = {k: data[k] for k in keys}
                    inputs['player_scalars'] = inputs['player_scalars'][:, :2]
                    def score_anchor(missing):
                        out, value = self._forward(self.head, self.head_graphs, missing,
                                                  (int(phase), True, 0, 0, 0))
                        if isinstance(value, torch.Tensor):
                            out, value = out.cpu().numpy(), value.cpu().numpy()
                        return out, value
                    _, value = self.anchor_cache.forward(inputs, score_anchor)
                    logits[index] = 0
                else:
                    out, value = self._forward(self.head, self.head_graphs, data, signature)
                    logits[index] = torch.as_tensor(out, device=self.device)
                values[index] = torch.as_tensor(value, device=self.device)
        return logits, values[:, None]

    def statistics(self):
        stats = {f'context_cache_{key}': value for key, value in self.cache.stats.items()}
        stats.update({f'anchor_cache_{key}': value for key, value in self.anchor_cache.stats.items()})
        for name, graphs in (('context', self.enc_graphs), ('upper', self.head_graphs)):
            values = graphs.statistics() if graphs is not None else dict.fromkeys(GRAPH_METRICS, 0)
            stats.update({f'{name}_graph_{key}': values[key] for key in GRAPH_METRICS})
        return stats

    def close(self):
        for graphs in (self.enc_graphs, self.head_graphs):
            if graphs is not None:
                graphs.clear()
        self.cache.clear()
        self.anchor_cache.clear()


class _UpperPhaseCore:
    def __init__(self, teacher, value_only=False):
        self.teacher = teacher
        self.value_only = value_only

    @property
    def training(self):
        return self.teacher.training

    def combat_actor_critic(self, observations, *, active_relic_width):
        phase, deck, relics = active_relic_width
        if self.value_only:
            values = self.teacher.forward_phase_values(observations, phase, (deck, relics))
            return values.new_empty((len(values), 0)), values
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
    def __init__(self, model, *, upper_bytes=128*1024**2, lower_bytes=0, upper_value_only=False):
        self.model = model
        self.upper_value_only = upper_value_only
        self.upper = ExactInputCache(upper_bytes)
        self.lower = ExactInputCache(lower_bytes)
        self.upper_graphs = None
        self._native_upper = False
        if torch.device(model.device).type == 'cuda' and getattr(model, '_native_leaf_inference', None) is not None:
            self.upper_graphs = CombatLeafGraphs(_UpperPhaseCore(model.policy.core.teacher, upper_value_only),
                model.device, amp='off', capacity=32, batch_multiple=8, compile_forward=False)
            if upper_value_only:
                from env.core.native_observation import upper_value_runtime
                native = upper_value_runtime(model.policy.core.teacher, upper_bytes, pad=True)
                if native is not None:
                    self.upper, self._native_upper = native, True
        self.outer_settlement_seconds = 0.
        self.upper_prepare_seconds = 0.
        self.outer_overlap_batches = self.outer_overlap_rows = 0
        self.outer_exit_rows = self.outer_value_rows = self.outer_value_batches = 0
        self.outer_value_seconds = 0.

    def forward_upper(self, observations):
        if self._native_upper:
            observations = self.model.policy.core.teacher._validated_observations(observations)
            return self.forward_upper_rows([{key: value[i] for key, value in observations.items()}
                                            for i in range(len(observations['phase']))])
        started = time.perf_counter()
        groups = self.model.policy.core.teacher.prepare_numpy(observations)
        self.upper_prepare_seconds += time.perf_counter()-started
        return self._forward_upper(groups, len(observations['phase']))

    def forward_upper_rows(self, observations):
        if self._native_upper:
            values = self.upper.forward(observations, self.model.policy.core.teacher,
                                        self._forward_upper_values)
            return np.empty((len(values), 0), np.float32), values
        started = time.perf_counter()
        groups = self.model.policy.core.teacher.prepare_search_numpy(observations)
        self.upper_prepare_seconds += time.perf_counter()-started
        return self._forward_upper(groups, len(observations))

    def _forward_upper_values(self, batch, signature):
        return self.upper_graphs(batch, signature)[1]

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
                if self.upper_value_only:
                    values = teacher.forward_phase_values(tensors, phase, widths)
                    return np.empty((len(values), 0), np.float32), values.reshape(-1).cpu().numpy()
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
                      upper_prepare_seconds=self.upper_prepare_seconds +
                          (self.upper.prepare_seconds if self._native_upper else 0.),
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

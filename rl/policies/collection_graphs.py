"""Bounded FP32 phase graphs reused by shared-PPO collections.

Only masked trailing padding and independent batch padding change physical shape.
Sampling, legal-action masking and log probabilities remain in the policy.
No graph or temporary CUDA allocation is saved with a checkpoint.
"""
from collections import OrderedDict
import time

import numpy as np
import torch

from env.core import encode, native_actions as C, vocab
from rl.policies.split_policy import COMBAT_OBSERVATION_KEYS
from rl.policies.shared_wholerun import MERGED_UPPER_PHASE


# Synthetic collection route, separate from real screen IDs and MERGED_UPPER_PHASE.
_ANCHOR_PHASE = -2


class SharedCollectionGraphs:
    def __init__(self, core, *, capacity=128):
        if capacity < 1 or next(core.parameters()).device.type != 'cuda':
            raise ValueError('collection graphs require CUDA and positive capacity')
        if getattr(core, 'training_objective', None) != 'shared-wholerun-ppo-room-v1':
            raise ValueError('collection graphs require the shared whole-run policy')
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in core.parameters()):
            raise ValueError('collection graphs require FP32 parameters')
        self.core, self.capacity = core, capacity
        self.parameters = tuple(core.parameters())
        self.device = self.parameters[0].device
        self.entries = OrderedDict()
        self.temporary_pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.sequence_done = torch.cuda.Event()
        self.sequence_done.record(torch.cuda.current_stream(self.device))
        self.closed = False
        self.map_cache = None
        self.host_map_cache = None
        self.host_inputs = None
        self.map_entries = OrderedDict()
        self.context_cache = None
        self.weight_versions = None
        self.begin_collection()

    def begin_collection(self):
        self.host_inputs = None
        self.stats = dict(captures=0, hits=0, evictions=0, capture_seconds=0.,
                          context_rows=0, map_rows=0)

    def compatible(self, core):
        return (self.core is core and not self.closed
                and tuple(map(id, core.parameters())) == tuple(map(id, self.parameters))
                and all(p.device.type == 'cuda' and p.dtype == torch.float32
                        for p in self.parameters))

    @staticmethod
    def _width(ids, minimum):
        columns = np.flatnonzero(np.any(ids != 0, axis=0))
        width = int(columns[-1]) + 1 if len(columns) else 1
        return min(ids.shape[1], max(minimum, 1 << (width - 1).bit_length()))

    def _capture(self, batch, padded, forward):
        count = len(next(iter(batch.values())))
        with torch.inference_mode(False), torch.no_grad():
            inputs = {key: torch.cat([value, value[:1].expand(padded-count, *value.shape[1:])]).clone()
                      for key, value in batch.items()}
        current = torch.cuda.current_stream(self.device)
        # One capture stream permits reuse of its temporary allocation blocks.
        stream = self.capture_stream
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = forward(inputs)
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        # Each entry owns inputs and published outputs OUTSIDE the shared pool.
        # No tensor from the pool survives a forward. Unlike sharing ordinary
        # graph outputs, this permits replay in arbitrary entry/LRU order; the
        # complete calls are serialized below, including across CUDA streams.
        with torch.inference_mode(False), torch.no_grad():
            published = tuple(torch.empty_like(tensor) for tensor in output)
        del output
        # cuBLAS scratch must belong to the graph, not an eager allocation that
        # can be freed/reused when another shape is captured or evicted.
        torch._C._cuda_clearCublasWorkspaces()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=self.temporary_pool):
            temporary = forward(inputs)
            for target, source in zip(published, temporary, strict=True):
                target.copy_(source)
            del target, source, temporary
        current.wait_stream(stream)
        torch._C._cuda_clearCublasWorkspaces()
        return inputs, graph, published

    def _replay(self, cache, signature, batch, padded, forward, capacity):
        if signature not in cache:
            if len(cache) >= capacity:
                # Evicting the last graph may release the temporary pool's
                # backing blocks. Finish queued replay before dropping it.
                torch.cuda.synchronize(self.device)
                cache.popitem(last=False)
                if not self.entries and not self.map_entries:
                    # The last graph ended this pool's allocator lifetime.
                    # A retired pool token cannot start a new capture.
                    self.temporary_pool = torch.cuda.graph_pool_handle()
                self.stats['evictions'] += 1
            started = time.perf_counter()
            cache[signature] = self._capture(batch, padded, forward)
            self.stats['capture_seconds'] += time.perf_counter() - started
            self.stats['captures'] += 1
        else:
            self.stats['hits'] += 1
        cache.move_to_end(signature)
        inputs, graph, output = cache[signature]
        count = len(next(iter(batch.values())))
        for key, value in batch.items():
            inputs[key][:count].copy_(value)
            inputs[key].record_stream(torch.cuda.current_stream(self.device))
        graph.replay()
        for tensor in output:
            tensor.record_stream(torch.cuda.current_stream(self.device))
        return output

    def bind_host_observations(self, tensors, observations, *, context_slots=None):
        """Bind the source of one immediate forward, never a later tensor view.

        The collector calls this after converting this exact observation.
        Subsets, bootstrap values and arbitrary GPU callers use the fallback.
        """
        if tensors.keys() != observations.keys() or any(
                tuple(tensors[key].shape) != observations[key].shape for key in tensors):
            raise ValueError('host observation binding must match the converted batch')
        if context_slots is not None:
            indices, capacity = context_slots
            indices = np.asarray(indices, dtype=np.intp)
            if (len(indices) != len(observations['phase']) or len(np.unique(indices)) != len(indices)
                    or (indices < 0).any() or (indices >= capacity).any()):
                raise ValueError('invalid collection context slots')
            context_slots = indices, capacity
        self.host_inputs = tensors, observations, context_slots

    def __call__(self, observations):
        if self.closed:
            raise RuntimeError('collection graphs require a live eval/no-grad scope')
        # One collector owns this runtime. Its serial host calls may use
        # different CUDA streams; protect caches, outputs and scratch until
        # the preceding call has copied all of its published results.
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.sequence_done)
        try:
            pending, self.host_inputs = self.host_inputs, None
            host = pending[1] if pending is not None and pending[0] is observations else None
            return self._forward(observations, host, pending[2] if host is not None else None)
        finally:
            self.sequence_done.record(stream)

    def _reset_map_cache(self, rooms, features):
        # These owned buffers survive switches between inference_mode and
        # no_grad callers, as well as changes to model weights.
        with torch.inference_mode(False), torch.no_grad():
            self.map_cache = (torch.zeros_like(rooms), torch.zeros_like(features),
                torch.zeros(len(rooms), rooms.shape[1], self.core.upper_config.width,
                            device=rooms.device))
            self.map_valid = torch.zeros(len(rooms), dtype=torch.bool, device=rooms.device)

    def _run_context(self, observations, host, context_slots=None):
        encoder = self.core.context
        versions = tuple(p._version for p in encoder.parameters())
        cached = self.context_cache
        count = len(host['phase'])
        slots, capacity = (np.arange(count), count) if context_slots is None else context_slots
        if cached is None or cached[0] != versions or any(
                cached[1][k].shape != (capacity, *host[k].shape[1:])
                or cached[1][k].dtype != host[k].dtype for k in encoder.keys):
            previous = {k: np.zeros((capacity, *host[k].shape[1:]), dtype=host[k].dtype) for k in encoder.keys}
            with torch.inference_mode(False), torch.no_grad():
                latent = torch.empty(capacity, self.core.upper_config.width, device=self.device)
            valid = np.zeros(capacity, dtype=bool)
        else:
            _, previous, latent, valid = cached
        dirty = ~valid[slots]
        for k in encoder.keys:
            dirty |= (host[k] != previous[k][slots]).reshape(count, -1).any(1)
        rows = np.flatnonzero(dirty)
        if len(rows):
            # Only the expensive attention is deduplicated. Identity includes
            # every public key byte and cache entries are owned by this runtime.
            keys = np.concatenate([np.ascontiguousarray(host[k][rows]).view(np.uint8)
                                   .reshape(len(rows), -1) for k in encoder.keys], axis=1)
            records = keys.view(np.dtype((np.void, keys.shape[1]))).ravel()
            _, first, inverse = np.unique(records, return_index=True, return_inverse=True)
            selected = rows[first]
            indices = torch.as_tensor(selected, device=self.device)
            batch = {k: observations[k][indices] for k in encoder.keys}
            widths = tuple(self._width(host[k][selected], 1) for k in ('deck_ids', 'relics'))
            encoded = encoder(batch, active_widths=widths)
            latent[torch.as_tensor(slots[rows], device=self.device)] = encoded[
                torch.as_tensor(inverse, device=self.device)]
            for k in encoder.keys:
                previous[k][slots[rows]] = host[k][rows]
            valid[slots[rows]] = True
            self.stats['context_rows'] = self.stats.get('context_rows', 0) + len(selected)
        self.context_cache = versions, previous, latent, valid
        return latent if context_slots is None else latent[torch.as_tensor(slots, device=self.device)]

    def _forward(self, observations, host=None, context_slots=None):
        if self.closed or self.core.training or torch.is_grad_enabled():
            raise RuntimeError('collection graphs require a live eval/no-grad scope')
        if torch.is_autocast_enabled('cuda'):
            raise RuntimeError('collection graphs require FP32, without autocast')
        # Stable lookup-table storage is refreshed even on cache hits. Direct
        # parameter writes/Adam updates are visible; replacing parameters needs
        # a fresh collection runtime.
        self.core.prepare_inference_cache()
        deck_model = getattr(self.core, 'model_contract', None) == 'deck-latent-combat-v1'
        if deck_model and host is None:
            return self.core.eager_actor_critic(observations)
        context = self._run_context(observations, host, context_slots) if deck_model else None
        if deck_model:
            self.core.publish_actor_context(context)
        phase = observations['phase'].long().flatten()
        use_map = self.core.map_dag_enabled and 'upper_public_context' in observations
        if host is not None:
            host_phase = host['phase'].reshape(-1)
            host_dirty = np.zeros(len(host_phase), dtype=bool)
            if use_map:
                rooms, features = observations['map_full_rooms'], observations['map_full_features']
                slots, capacity = (np.arange(len(host_phase)), len(host_phase)) if context_slots is None else context_slots
                slot_tensor = torch.as_tensor(slots, device=self.device)
                versions = tuple(p._version for p in self.parameters)
                valid_shape = (self.map_cache is not None
                               and self.map_cache[0].shape == (capacity, *rooms.shape[1:])
                               and self.map_cache[1].shape == (capacity, *features.shape[1:]))
                if not valid_shape:
                    self._reset_map_cache(rooms[:1].expand(capacity, *rooms.shape[1:]),
                                          features[:1].expand(capacity, *features.shape[1:]))
                previous_rooms, previous_features, encoded_map = self.map_cache
                cached = self.host_map_cache
                if not valid_shape or versions != self.weight_versions or cached is None:
                    cached = (np.zeros((capacity, *host['map_full_rooms'].shape[1:]), dtype=host['map_full_rooms'].dtype),
                              np.zeros((capacity, *host['map_full_features'].shape[1:]), dtype=host['map_full_features'].dtype),
                              np.zeros(capacity, dtype=bool))
                    self.host_map_cache = cached
                    self.map_valid.zero_()
                self.weight_versions = versions
                same = ((host['map_full_rooms'] == cached[0][slots]).reshape(len(phase), -1).all(1)
                        & (host['map_full_features'] == cached[1][slots]).reshape(len(phase), -1).all(1))
                host_dirty = (~same | ~cached[2][slots]) & (host_phase != vocab.PHASE_VOCAB_INDEX['combat'])
                cached[0][slots[host_dirty]] = host['map_full_rooms'][host_dirty]
                cached[1][slots[host_dirty]] = host['map_full_features'][host_dirty]
                cached[2][slots[host_dirty]] = True
            meta = np.concatenate([host_phase[:, None], host['relics'], host['deck_ids'],
                                   host_dirty[:, None]], axis=1)
        else:
            self.host_map_cache = None
            dirty = torch.zeros_like(phase, dtype=torch.bool)
            if use_map:
                rooms, features = observations['map_full_rooms'], observations['map_full_features']
                versions = tuple(p._version for p in self.parameters)
                if (self.map_cache is None or self.map_cache[0].shape != rooms.shape
                        or self.map_cache[1].shape != features.shape):
                    self._reset_map_cache(rooms, features)
                if versions != self.weight_versions:
                    self.map_valid.zero_()
                    self.weight_versions = versions
                previous_rooms, previous_features, encoded_map = self.map_cache
                # Equality includes every map feature (current/choice/reachable,
                # topology and room identity), not just floor or enemy type.
                same = ((rooms == previous_rooms).flatten(1).all(1)
                        & (features == previous_features).flatten(1).all(1))
                dirty = (~same | ~self.map_valid) & (phase != vocab.PHASE_VOCAB_INDEX['combat'])

            # One small host read replaces repeated per-phase CUDA nonzero/width
            # synchronizations. All actual observation tensors stay on the device.
            meta = torch.cat([phase[:, None], observations['relics'].long(),
                              observations['deck_ids'].long(), dirty[:, None].long()],
                             dim=1).cpu().numpy()
        relic_end = 1 + observations['relics'].shape[1]
        if use_map:
            changed = np.flatnonzero(meta[:, -1])
            if len(changed):
                selected = torch.from_numpy(changed).to(device=phase.device, non_blocking=True)
                count = len(changed)
                padded = 1 << (count - 1).bit_length()
                batch = dict(rooms=rooms[selected], features=features[selected])
                signature = (padded, tuple(rooms.shape[1:]), rooms.dtype,
                             tuple(features.shape[1:]), features.dtype)
                encoder = self.core.overworld.encoder
                def map_forward(tensors):
                    return (encoder.map_dag(encoder.room_embedding(tensors['rooms'].long()),
                                           tensors['features'].float()),)
                output = self._replay(self.map_entries, signature, batch, padded,
                                      map_forward, min(16, self.capacity))
                destination = slot_tensor[selected] if host is not None else selected
                encoded_map[destination] = output[0][:count]
                output[0].record_stream(torch.cuda.current_stream(self.device))
                previous_rooms[destination] = rooms[selected]
                previous_features[destination] = features[selected]
                self.map_valid[destination] = True
                self.stats['map_rows'] = self.stats.get('map_rows', 0) + count
            if host is not None and context_slots is not None:
                encoded_map = encoded_map[slot_tensor]
        # Pack by dtype, preserving categorical integers exactly. Each phase
        # submits only a few gathers/copies; views are reconstructed in its graph.
        schema = tuple((key, tuple(value.shape[1:]), value.dtype)
                       for key, value in observations.items())
        if getattr(self, '_packed_schema', None) != schema:
            groups = {}
            for key, value in observations.items():
                groups.setdefault(value.dtype, []).append(key)
            layout, grouped = {}, {}
            for index, keys in enumerate(groups.values()):
                group = 'packed' + str(index)
                grouped[group] = keys
                offset = 0
                for key in keys:
                    shape = observations[key].shape[1:]
                    end = offset + int(np.prod(shape))
                    layout[key] = (group, offset, end, shape)
                    offset = end
            self._packed_schema, self._packed_layout, self._packed_groups = schema, layout, grouped
        layout = self._packed_layout
        packed = {group: torch.cat([observations[k].reshape(len(phase), -1) for k in keys], dim=1)
                  for group, keys in self._packed_groups.items()}
        indices, logits, values = [], [], []
        combat_id = vocab.PHASE_VOCAB_INDEX['combat']
        selection_id = vocab.PHASE_VOCAB_INDEX['card_select']
        ordinary_upper = (meta[:, 0] != combat_id) & (meta[:, 0] != selection_id)
        routes = meta[:, 0].copy()
        if deck_model:
            routes[(routes == combat_id) & host['combat_exit_anchor'].any(1)] = _ANCHOR_PHASE
        if host is not None and use_map:
            routes[ordinary_upper] = MERGED_UPPER_PHASE
        for phase_id in np.unique(routes).tolist():
            rows = np.flatnonzero(routes == phase_id)
            selected = torch.from_numpy(rows).to(device=phase.device, non_blocking=True)
            combat = phase_id in (combat_id, _ANCHOR_PHASE)
            keys = (COMBAT_OBSERVATION_KEYS if combat
                    else observations if phase_id == MERGED_UPPER_PHASE
                    else encode.upper_observation_keys(phase_id, observations))
            if combat and deck_model:
                keys = keys | {'pile_upgraded_counts', 'pile_upgrade_sums'}
            keys = tuple(key for key in observations if key in keys)
            needed = {layout[key][0] for key in keys}
            batch = {group: value[selected] for group, value in packed.items() if group in needed}
            relic = self._width(meta[rows, 1:relic_end], 4)
            deck = None if combat else self._width(meta[rows, relic_end:-1], 16)
            choice_width = None
            if deck_model:
                batch['_run_context'] = context[selected]
                if phase_id == combat_id:
                    choices = host['select_native_indices'][rows] > 0
                    pick = C.JOINT_PICK - C.JOINT_END
                    choices |= host['combat_legal'][rows, pick:pick + choices.shape[1]].astype(bool)
                    choice_width = self._width(choices, 1)
            if use_map and not combat:
                batch['_map_encoding'] = encoded_map[selected]
            union_width = (self.core.overworld.encoder.merged_width(host, rows, multiple=16)
                if phase_id == MERGED_UPPER_PHASE else None)
            count = len(selected)
            padded = (1 << (count - 1).bit_length()) if count <= 16 else ((count + 15) // 16) * 16
            signature = (phase_id, padded, deck, relic, union_width, choice_width, schema,
                         tuple((key, tuple(value.shape[1:]), value.dtype) for key, value in batch.items()))

            def forward(packed_batch):
                tensors = {}
                for key in keys:
                    group, start, end, shape = layout[key]
                    tensors[key] = packed_batch[group][:, start:end].reshape(padded, *shape)
                for key in ('relics', 'relic_counters'):
                    tensors[key] = tensors[key][:, :relic]
                if deck is not None:
                    for key in ('deck_ids', 'deck_upgrades'):
                        tensors[key] = tensors[key][:, :deck]
                if deck_model:
                    tensors['_run_context'] = packed_batch['_run_context']
                    if phase_id == _ANCHOR_PHASE:
                        z = tensors['_run_context']
                        return (z.new_zeros(padded, C.AUTOREGRESSIVE_SIZE),
                                self.core.anchor_with_context(tensors, self.core.anchor.context_projection(z)))
                    if combat:
                        tensors['_choice_width'] = choice_width
                if '_map_encoding' in packed_batch:
                    tensors['_map_encoding'] = packed_batch['_map_encoding']
                if combat:
                    return self.core.combat_actor_critic(tensors, active_relic_width=relic)
                return self.core.overworld(tensors, self.core.upper_tokenizer, phase_id,
                                           upper_active_widths=((deck, relic, union_width)
                                               if phase_id == MERGED_UPPER_PHASE else (deck, relic)),
                                           upper_map_encoding=tensors.get('_map_encoding'))

            output = self._replay(self.entries, signature, batch, padded, forward, self.capacity)
            # Results own storage; subsequent replay/eviction cannot mutate
            # them. Register the clone's stream before releasing private pools.
            out, value = [tensor[:count].clone() for tensor in output]
            for tensor in output:
                tensor.record_stream(torch.cuda.current_stream(self.device))
            indices.append(selected)
            logits.append(out.float())
            values.append(value.float())
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.device)
            self.entries.clear()
            self.context_cache = None
            self.map_entries.clear()
            self.map_cache = None
            self.host_map_cache = None
            self.host_inputs = None
            self.closed = True

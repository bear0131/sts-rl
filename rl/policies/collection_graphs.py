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
        self.context_entries = OrderedDict()
        self.packed_bank = None
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

    def _capture(self, batch, padded, forward, keep_alive=()):
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
        graph._external_owners = keep_alive
        with torch.cuda.graph(graph, stream=stream, pool=self.temporary_pool):
            temporary = forward(inputs)
            for target, source in zip(published, temporary, strict=True):
                target.copy_(source)
            del target, source, temporary
        current.wait_stream(stream)
        torch._C._cuda_clearCublasWorkspaces()
        return inputs, graph, published

    def _replay(self, cache, signature, batch, padded, forward, capacity, *, keep_alive=()):
        if signature not in cache:
            if len(cache) >= capacity:
                # Evicting the last graph may release the temporary pool's
                # backing blocks. Finish queued replay before dropping it.
                torch.cuda.synchronize(self.device)
                cache.popitem(last=False)
                if not self.entries and not self.map_entries and not self.context_entries:
                    # The last graph ended this pool's allocator lifetime.
                    # A retired pool token cannot start a new capture.
                    self.temporary_pool = torch.cuda.graph_pool_handle()
                self.stats['evictions'] += 1
            started = time.perf_counter()
            cache[signature] = self._capture(batch, padded, forward, keep_alive)
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

    def bind_host_observations(self, tensors, observations, *, context_slots=None, packed=None):
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
        self.host_inputs = tensors, observations, context_slots, packed

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
            return self._forward(observations, host, pending[2] if host is not None else None,
                                 pending[3] if host is not None else None,
                packed_layout=pending[4] if host is not None and len(pending) > 4 else None)
        finally:
            self.sequence_done.record(stream)

    def host_actor(self, observations, host, slots, packed, masks, *, layout=None):
        """The same phase graphs, with categorical output inside each replay.

        The frozen owner has one exact host batch. Unused observation fields
        stay packed; no per-field CUDA views, masking or validation rendezvous
        need to execute in Python between graph replays.
        """
        if self.closed:
            raise RuntimeError('collection graphs require a live runtime')
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.sequence_done)
        try:
            return self._forward(observations, host, slots, packed, masks, packed_layout=layout)[0]
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

    def _bind_packed_bank(self, packed, layout, masks, slots):
        """Own one column bank for every graph consumer of this actor batch.

        Graphs gather their own fields and rows. Repeated small actor/context
        calls do not launch one eager gather/copy for each observation field.
        Addresses remain stable until geometry/capacity changes; the signature
        then changes and old graph entries keep their own storage references.
        """
        count = len(next(iter(packed.values())))
        capacity = max(256, 1 << (count-1).bit_length())
        schema = tuple((k, tuple(v.shape[1:]), v.dtype) for k,v in packed.items())
        previous = self.packed_bank
        if previous is None or previous['schema'] != schema or previous['capacity'] < count:
            with torch.inference_mode(False), torch.no_grad():
                columns = {k: torch.empty((capacity,*v.shape[1:]),device=self.device,dtype=v.dtype)
                           for k,v in packed.items()}
                bank = dict(schema=schema, capacity=capacity, columns=columns, layout=dict(layout),
                    context=torch.empty((capacity,self.core.upper_config.width),device=self.device),
                    slots=torch.empty(capacity,device=self.device,dtype=torch.long), mask=None)
            self.packed_bank = bank
        else:
            bank = previous
            if bank['layout'] != layout:
                raise ValueError('packed observation layout changed without a schema change')
        for k,v in packed.items():
            bank['columns'][k][:count].copy_(v)
        bank['slots'][:count].copy_(torch.as_tensor(slots,device=self.device))
        if masks is not None:
            if bank['mask'] is None or bank['mask'].shape[1:] != masks.shape[1:]:
                bank['mask'] = torch.empty((bank['capacity'],*masks.shape[1:]),device=self.device,dtype=masks.dtype)
            bank['mask'][:count].copy_(masks)
        # Captured closures own this generation, including buffers replaced by
        # a later capacity/layout. Never let an old graph dereference self.bank.
        bound = dict(bank)
        bound['key'] = tuple(v.data_ptr() for v in bank['columns'].values()) + (
            bank['context'].data_ptr(), bank['slots'].data_ptr(),
            0 if masks is None else bank['mask'].data_ptr())
        return bound

    @staticmethod
    def _packed_fields(bank, rows, keys):
        groups = {bank['layout'][k][0] for k in keys}
        columns = {g: bank['columns'][g][rows] for g in groups}
        result = {}
        for key in keys:
            group,start,end,shape = bank['layout'][key]
            result[key] = columns[group][:,start:end].reshape(len(rows),*shape)
        return result

    def _run_context(self, observations, host, context_slots=None, packed_bank=None):
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
            batch = ({k: observations[k][indices] for k in encoder.keys} if packed_bank is None
                     else {'_indices': indices})
            widths = tuple(self._width(host[k][selected], 1) for k in ('deck_ids', 'relics'))
            live = (1+(host['deck_ids'][selected] != 0).sum(1)
                      +(host['relics'][selected] != 0).sum(1)
                      +((host['map_full_rooms'][selected] != 0)
                        | host['map_full_features'][selected].any(-1)).sum(1))
            packed_width = max(16, 1 << (int(live.max())-1).bit_length())
            count = len(selected)
            padded = max(8, 1 << (count - 1).bit_length()) if count <= 16 else ((count + 15) // 16) * 16
            signature = ('run-context', padded, widths, packed_width,
                         None if packed_bank is None else packed_bank['key'],
                         tuple((k, tuple(v.shape[1:]), v.dtype) for k, v in batch.items()))
            def encode_context(inputs):
                fields = (inputs if packed_bank is None else
                          self._packed_fields(packed_bank,inputs['_indices'],encoder.keys))
                return (encoder(fields, active_widths=widths, packed_width=packed_width),)
            encoded = self._replay(self.context_entries, signature, batch, padded,
                                  encode_context, min(128,self.capacity),
                                  keep_alive=() if packed_bank is None else (packed_bank,))[0][:count]
            latent[torch.as_tensor(slots[rows], device=self.device)] = encoded[
                torch.as_tensor(inverse, device=self.device)]
            for k in encoder.keys:
                previous[k][slots[rows]] = host[k][rows]
            valid[slots[rows]] = True
            self.stats['context_rows'] = self.stats.get('context_rows', 0) + len(selected)
        self.context_cache = versions, previous, latent, valid
        return latent if context_slots is None else latent[torch.as_tensor(slots, device=self.device)]

    def _forward(self, observations, host=None, context_slots=None, packed=None, actor_masks=None, *, packed_layout=None):
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
        packed_bank = (None if packed_layout is None else self._bind_packed_bank(
            packed,packed_layout,actor_masks,np.arange(len(host['phase'])) if context_slots is None else context_slots[0]))
        context = self._run_context(observations,host,context_slots,packed_bank) if deck_model else None
        if packed_bank is not None:
            packed_bank['context'][:len(context)].copy_(context)
        if deck_model:
            self.core.publish_actor_context(context)
        phase = observations['phase'].long().flatten()
        use_map = (self.core.map_dag_enabled and 'upper_public_context' in observations
                   and (host is None or np.any(host['phase'] != vocab.PHASE_VOCAB_INDEX['combat'])))
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
                batch = (dict(rooms=rooms[selected], features=features[selected])
                         if packed_bank is None else dict(_indices=selected))
                signature = (padded, tuple(rooms.shape[1:]), rooms.dtype,
                             tuple(features.shape[1:]), features.dtype,
                             None if packed_bank is None else packed_bank['key'])
                encoder = self.core.overworld.encoder
                def map_forward(tensors):
                    if packed_bank is not None:
                        fields = self._packed_fields(packed_bank,tensors['_indices'],
                                                     ('map_full_rooms','map_full_features'))
                        tensors = dict(rooms=fields['map_full_rooms'],features=fields['map_full_features'])
                    return (encoder.map_dag(encoder.room_embedding(tensors['rooms'].long()),
                                           tensors['features'].float()),)
                output = self._replay(self.map_entries, signature, batch, padded,
                                      map_forward, min(16, self.capacity),
                                      keep_alive=() if packed_bank is None else (packed_bank,))
                destination = slot_tensor[selected] if host is not None else selected
                encoded_map[destination] = output[0][:count]
                output[0].record_stream(torch.cuda.current_stream(self.device))
                previous_rooms[destination] = rooms[selected]
                previous_features[destination] = features[selected]
                self.map_valid[destination] = True
                self.stats['map_rows'] = self.stats.get('map_rows', 0) + count
            if host is not None and context_slots is not None and packed_bank is None:
                encoded_map = encoded_map[slot_tensor]
        # Pack by dtype, preserving categorical integers exactly. Each phase
        # submits only a few gathers/copies; views are reconstructed in its graph.
        geometry = host if host is not None and packed is not None else observations
        schema = tuple((key, tuple(value.shape[1:]), value.dtype)
                       for key, value in geometry.items())
        if getattr(self, '_packed_schema', None) != schema:
            groups = {}
            for key, value in geometry.items():
                groups.setdefault(value.dtype, []).append(key)
            layout, grouped = {}, {}
            for index, keys in enumerate(groups.values()):
                group = 'packed' + str(index)
                grouped[group] = keys
                offset = 0
                for key in keys:
                    shape = geometry[key].shape[1:]
                    end = offset + int(np.prod(shape))
                    layout[key] = (group, offset, end, shape)
                    offset = end
            self._packed_schema, self._packed_layout, self._packed_groups = schema, layout, grouped
        layout = self._packed_layout
        if packed is None:
            packed = {group: torch.cat([observations[k].reshape(len(phase), -1) for k in keys], dim=1)
                      for group, keys in self._packed_groups.items()}
        elif packed.keys() != self._packed_groups.keys():
            raise ValueError('packed actor input schema mismatch')
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
            batch = ({group: value[selected] for group, value in packed.items() if group in needed}
                     if packed_bank is None else dict(_indices=selected))
            relic = self._width(meta[rows, 1:relic_end], 4)
            deck = None if combat else self._width(meta[rows, relic_end:-1], 16)
            choice_width = None
            if deck_model:
                if packed_bank is None:
                    batch['_run_context'] = context[selected]
                if phase_id == combat_id:
                    choices = host['select_native_indices'][rows] > 0
                    pick = C.JOINT_PICK - C.JOINT_END
                    choices |= host['combat_legal'][rows, pick:pick + choices.shape[1]].astype(bool)
                    choice_width = self._width(choices, 1)
            if use_map and not combat and packed_bank is None:
                batch['_map_encoding'] = encoded_map[selected]
            if actor_masks is not None and packed_bank is None:
                batch['_actor_mask'] = actor_masks[selected]
            union_width = (self.core.overworld.encoder.merged_width(host, rows, multiple=16)
                if phase_id == MERGED_UPPER_PHASE else None)
            count = len(selected)
            padded = (max(8, 1 << (count - 1).bit_length()) if count <= 16
                      else ((count + 15) // 16) * 16) if actor_masks is not None else (
                      (1 << (count - 1).bit_length()) if count <= 16 else ((count + 15) // 16) * 16)
            signature = (phase_id, padded, deck, relic, union_width, choice_width, schema,
                         None if packed_bank is None else packed_bank['key'],
                         encoded_map.data_ptr() if packed_bank is not None and use_map and not combat else None,
                         tuple((key, tuple(value.shape[1:]), value.dtype) for key, value in batch.items()))

            def forward(packed_batch):
                tensors = ({} if packed_bank is None else
                           self._packed_fields(packed_bank,packed_batch['_indices'],keys))
                if packed_bank is None:
                    for key in keys:
                        group, start, end, shape = layout[key]
                        tensors[key] = packed_batch[group][:, start:end].reshape(padded, *shape)
                for key in ('relics', 'relic_counters'):
                    tensors[key] = tensors[key][:, :relic]
                if deck is not None:
                    for key in ('deck_ids', 'deck_upgrades'):
                        tensors[key] = tensors[key][:, :deck]
                if deck_model:
                    tensors['_run_context'] = (packed_batch['_run_context'] if packed_bank is None else
                                               packed_bank['context'][packed_batch['_indices']])
                    if phase_id == _ANCHOR_PHASE:
                        z = tensors['_run_context']
                        return (z.new_zeros(padded, C.AUTOREGRESSIVE_SIZE),
                                self.core.anchor_with_context(tensors, self.core.anchor.context_projection(z)))
                    if combat:
                        tensors['_choice_width'] = choice_width
                if packed_bank is not None and use_map and not combat:
                    tensors['_map_encoding'] = encoded_map[packed_bank['slots'][packed_batch['_indices']]]
                elif '_map_encoding' in packed_batch:
                    tensors['_map_encoding'] = packed_batch['_map_encoding']
                if combat:
                    return self.core.combat_actor_critic(tensors, active_relic_width=relic)
                return self.core.overworld(tensors, self.core.upper_tokenizer, phase_id,
                                           upper_active_widths=((deck, relic, union_width)
                                               if phase_id == MERGED_UPPER_PHASE else (deck, relic)),
                                           upper_map_encoding=tensors.get('_map_encoding'))

            if actor_masks is not None:
                raw_forward = forward
                def forward(packed_batch):
                    out, value = raw_forward(packed_batch)
                    # Same two Categorical normalizations as MaskableCategorical:
                    # original logits first, then the mask and normalization.
                    normalized = out.float()-out.float().logsumexp(-1, keepdim=True)
                    mask = (packed_batch['_actor_mask'] if packed_bank is None else
                            packed_bank['mask'][packed_batch['_indices']])
                    masked = normalized.masked_fill(~mask.bool(), -1e8)
                    logprob = masked-masked.logsumexp(-1, keepdim=True)
                    return torch.cat((logprob.softmax(-1), logprob, value.reshape(-1,1)),1), value
            output = self._replay(self.entries, signature, batch, padded, forward, self.capacity,
                keep_alive=() if packed_bank is None else (packed_bank, encoded_map if use_map else None))
            # Results own storage; subsequent replay/eviction cannot mutate
            # them. Register the clone's stream before releasing private pools.
            out, value = [tensor[:count].clone() for tensor in output]
            for tensor in output:
                tensor.record_stream(torch.cuda.current_stream(self.device))
            indices.append(selected)
            logits.append(out.float())
            values.append(value.float())
        if len(indices) == 1:
            return logits[0], values[0]
        order = torch.argsort(torch.cat(indices))
        return torch.cat(logits)[order], torch.cat(values)[order]

    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.device)
            self.entries.clear()
            self.context_entries.clear()
            self.context_cache = None
            self.map_entries.clear()
            self.map_cache = None
            self.host_map_cache = None
            self.host_inputs = None
            self.packed_bank = None
            self.closed = True

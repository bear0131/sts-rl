"""Frozen DeckLatent scoring directly from resident CUDA observations."""
from collections import OrderedDict
import numpy as np
import torch

from env.core.cuda_observation import CudaObservationDecoder, canonical_misc
from env.core import encode
from rl.policies.combat_summary import CombatSummary
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.search.rollout_inference import ExactInputCache


class ResidentDeckScorer:
    # The waiting branch pool can exceed a neural frame. Bound materialization
    # before decoding/projecting anchors, including exact-cache hits.
    anchor_batch_size = 4096

    def __init__(self, policy, cache_bytes=64*1024**2, graph_capacity=128, native_host=None,
                 *, compiled_gather=None):
        self.policy, self.core = policy, policy.core
        self.device = next(policy.parameters()).device
        self.decoder = CudaObservationDecoder(policy.observation_space, self.device)
        self.runtime = SharedCollectionGraphs(self.core, capacity=graph_capacity)
        self.anchor_cache = (ExactInputCache(cache_bytes // 4) if native_host is None
                             else native_host.ExactValueCache(cache_bytes // 4))
        self.anchor_contexts = OrderedDict()
        self.anchor_context_bytes = 0
        self.anchor_context_limit = cache_bytes // 2
        self.native_host = native_host
        self.native_anchor_intern = None if native_host is None else native_host.ByteIntern(self.anchor_context_limit)
        self.native_context_inputs = None if native_host is None else native_host.ByteIntern(cache_bytes // 2)
        self.next_anchor_context = 0
        # Keep registration owners as well as versions: replacing a parameter
        # or a child module must invalidate the frozen collection, without
        # recursively enumerating the entire encoder at every combat depth.
        modules = tuple(self.core.modules())
        self.parameter_bindings = tuple((m, name, p, p._version) for m in modules
            for name, p in m._parameters.items() if p is not None)
        self.module_bindings = tuple((m, name, child) for m in modules
            for name, child in m._modules.items() if child is not None)
        self.refreshes = 0
        self.choice_columns = torch.arange(1, 97, device=self.device)
        self.root_banks = {}
        self.bound_arena = None
        self.step_pipeline = None
        with torch.no_grad():
            self.card_table = self.core.combat.inference_card_table()
            projection = self.core.combat.card_projection
            self.projected_cards = torch.nn.functional.linear(
                self.card_table, projection.weight[:, :-1], projection.bias)
            self.card_upgrade = projection.weight[:, -1].contiguous()
            net = self.core.combat
            weight = net.monster_projection.weight
            self.monster_tables = (
                torch.nn.functional.linear(net.monster_id.weight, weight[:, :32]),
                torch.nn.functional.linear(net.monster_move.weight, weight[:, 32:64]),
                torch.nn.functional.linear(net.monster_intent.weight, weight[:, 64:80]),
                weight[:, 80:].contiguous())
            summaries = self.core.combat.summaries
            ids = torch.arange(summaries.card.num_embeddings, device=self.device)
            self.card_base = summaries.card_terms(ids, torch.zeros_like(ids))
            self.card_delta = (summaries.card_projection(summaries.card.versioned_forward(ids,
                torch.ones_like(ids)))-self.card_base)*ids.ne(0)[:, None]
            relic_ids = torch.arange(summaries.relic_identity.num_embeddings, device=self.device)
            counters = torch.arange(256, device=self.device).float()/10
            self.relic_table = summaries.relic_terms(relic_ids[:, None], counters[None, :]).contiguous()
        # Compile the current input adapter and heads around the migrated
        # ragged kernels. Keeping their dynamic capacities outside Dynamo
        # avoids recompiling the entire model for every CUDA graph bucket.
        # The continuous service constructs this once in its CUDA owner.
        self.compiled_gather = compiled_gather if compiled_gather is not None else torch.compile(
            self._forward_gather, fullgraph=False, dynamic=True, options={'triton.cudagraphs': False})

    def step_forward(self, batch, width, **kwargs):
        # Geometry variants belong to this bounded frozen runtime. Do not
        # silently hit the process-wide default limit and fall back to eager.
        with torch._dynamo.config.patch(recompile_limit=128, accumulated_recompile_limit=256):
            return self.compiled_gather(batch, width, **kwargs)

    @staticmethod
    def root_fields(core):
        return set(core.context.keys) | set(core.combat.scalar_keys) | {
            'phase', 'relic_counters', 'combat_is_boss'}

    def _intern_anchor_contexts(self, contexts):
        """Full-byte equality once per admission; IDs are never recycled.

        Evicting the host intern table can only cause a cache miss. Existing
        roots retain their IDs and miss forwarding reads their current GPU
        context, so no table entry owns a live root's tensor or lifetime.
        """
        host = contexts.detach().cpu().numpy() if isinstance(contexts,torch.Tensor) else contexts
        if self.native_anchor_intern is not None:
            self.native_anchor_intern.set_limit(self.anchor_context_limit)
            ids = self.native_anchor_intern.intern([host])
            self.anchor_context_bytes = self.native_anchor_intern.bytes
            return ids
        ids = []
        for row in host:
            key = (row.dtype.str, row.shape, row.tobytes())
            if key in self.anchor_contexts:
                self.anchor_contexts.move_to_end(key)
                identity = self.anchor_contexts[key]
            else:
                identity = self.next_anchor_context
                self.next_anchor_context += 1
                cost = len(key[2]) + 256
                if cost <= self.anchor_context_limit:
                    while self.anchor_contexts and self.anchor_context_bytes + cost > self.anchor_context_limit:
                        removed, _ = self.anchor_contexts.popitem(last=False)
                        self.anchor_context_bytes -= len(removed[2]) + 256
                    self.anchor_contexts[key] = identity
                    self.anchor_context_bytes += cost
            ids.append(identity)
        return np.asarray(ids, np.int64)

    @torch.no_grad()
    def roots(self, observations, context=None):
        keys = self.root_fields(self.core)
        self.root = {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                     for k in keys if k in observations[0]}
        self.context = self.core.prepare_search_context(self.root) if context is None else context
        self.anchor_context_ids = torch.as_tensor(self._intern_anchor_contexts(self.context.anchor),
                                                  device=self.device)
        # Match the existing collection graph's bounded relic projection. A
        # maximum-capacity PAD suffix otherwise creates a multi-GiB basis
        # tensor for thousands of search lanes, despite only a few real relics.
        relics = np.stack([o['relics'] for o in observations])
        self.relic_width = SharedCollectionGraphs._width(relics, 4)
        # Port the old consumer-field pack: cached context already owns the
        # deck/map encoder result. Never expand those large static tensors for
        # every hypothetical lane just to discard them in the combat head.
        head_root = {k: self.root[k] for k in
                     ('ascension', 'run_keys', 'combat_is_boss', 'relics', 'relic_counters')}
        for key in ('relics', 'relic_counters'):
            head_root[key] = head_root[key][:, :self.relic_width]
        ids = head_root['relics'].long()
        self.root_relic_identity = (self.core.combat.summaries.relic_identity(ids)
                                   * ids.ne(0)[..., None]).sum(1)
        # The frozen identity sum is a pure function of these exact IDs. Key
        # them on the host, avoiding a GPU-to-host rendezvous for every root.
        self.anchor_relic_ids = torch.as_tensor(self._intern_anchor_contexts(relics[:, :self.relic_width]),
                                                device=self.device)
        # Inventory and non-live counters are invariant inside this battle.
        # Live fields (including consumed Lizard Tail) keep their own lookup.
        static_relics = (self.core.combat.summaries.relic_terms(ids, head_root['relic_counters'])
                         * (~self.decoder.live[ids])[..., None]).sum(1)
        groups = {}
        for key, value in head_root.items():
            groups.setdefault(value.dtype, []).append(key)
        capacity = max(256, 1 << (len(observations)-1).bit_length())
        self.bank_key = (self.relic_width, capacity)
        bank = self.root_banks.setdefault(self.bank_key, {})
        self.root_layout, self.packed_root = {}, {}
        for index, keys in enumerate(groups.values()):
            group, offset = 'rootgroup:'+str(index), 0
            for key in keys:
                shape = head_root[key].shape[1:]
                end = offset + int(np.prod(shape))
                self.root_layout[key] = group, offset, end, shape
                offset = end
            packed = torch.cat([head_root[k].reshape(len(observations), -1) for k in keys], 1)
            if group not in bank:
                bank[group] = torch.empty((capacity, packed.shape[1]), device=packed.device, dtype=packed.dtype)
            bank[group][:len(observations)].copy_(packed)
            self.packed_root[group] = bank[group]
        if 'context' not in bank:
            bank['context'] = torch.empty((capacity, self.context.combat.shape[1]), device=self.device)
        bank['context'][:len(observations)].copy_(self.context.combat)
        self.banked_context = bank['context']
        if 'static_relics' not in bank:
            bank['static_relics'] = torch.empty((capacity, 64), device=self.device)
        bank['static_relics'][:len(observations)].copy_(static_relics)
        self.banked_static_relics = bank['static_relics']
        for key in ('deck_ids', 'outer_deck_misc'):
            name = 'metadata:'+key
            if name not in bank:
                bank[name] = torch.empty((capacity, *self.root[key].shape[1:]),
                    device=self.device, dtype=self.root[key].dtype)
            bank[name][:len(observations)].copy_(self.root[key])
        self.metadata_root = {key: bank['metadata:'+key] for key in ('deck_ids', 'outer_deck_misc')}
        self.changed_context = {}
        if self.native_context_inputs is not None:
            self.context_input_ids = self._context_input_ids(observations)

    def _context_input_ids(self, observations):
        groups = {}
        for key in self.core.context.keys:
            if key != 'outer_deck_misc':
                groups.setdefault(observations[0][key].dtype,[]).append(key)
        fields = [self.native_host.stack_fields(observations,keys) for keys in groups.values()]
        return self.native_context_inputs.intern(fields)

    def _pack_stream_roots(self):
        groups = {}
        for key,value in self.root.items():
            groups.setdefault(value.dtype,[]).append(key)
        self.stream_root_banks=[]
        for keys in groups.values():
            count=len(self.root[keys[0]])
            storage=torch.cat([self.root[k].reshape(count,-1) for k in keys],1)
            layout={};offset=0
            for key in keys:
                shape=self.root[key].shape[1:];end=offset+int(np.prod(shape))
                layout[key]=(offset,end,shape)
                self.root[key]=storage[:,offset:end].reshape(count,*shape)
                offset=end
            self.stream_root_banks.append((storage,layout))

    @torch.no_grad()
    def update_roots(self, slots, observations, context, capacity):
        """Current-model adapter for the old pool's versioned root bank."""
        from rl.policies.collection_graphs import SharedCollectionGraphs
        identities=self._intern_anchor_contexts(context.anchor)
        if not hasattr(self, 'stream_observations'):
            initial_context=SearchContext(*(torch.as_tensor(x,device=self.device)
                                          for x in (context.combat,context.anchor)))
            self.stream_observations = [observations[0]] * capacity
            for slot, observation in zip(slots, observations, strict=True):
                self.stream_observations[slot] = observation
            initial = SearchContext(initial_context.combat[:1].expand(capacity, -1).clone(),
                                    initial_context.anchor[:1].expand(capacity, -1).clone())
            self.roots(self.stream_observations, initial)
            self._pack_stream_roots()
            self.root_generations = np.zeros(capacity, np.int64)
        for slot, observation in zip(slots, observations, strict=True):
            self.stream_observations[slot] = observation
            self.root_generations[slot] += 1
        if self.native_context_inputs is not None:
            self.context_input_ids[slots] = self._context_input_ids(observations)
        relics = np.stack([o['relics'] for o in observations])
        relic_identities = self._intern_anchor_contexts(relics[:, :self.relic_width])
        if self.native_host is not None:
            from env.search.cuda_battle import upload_arrays
            packed=[self.native_host.stack_fields(observations,list(layout))
                    for _,layout in self.stream_root_banks]
            identity_rows,relic_identity_rows,index,*incoming_banks=upload_arrays(self.native_host,self.device,
                [identities,relic_identities,np.asarray(slots,np.int64),*packed])
            context=SearchContext(*(torch.as_tensor(x,device=self.device)
                                    for x in (context.combat,context.anchor)))
        else:
            context=SearchContext(*(torch.as_tensor(x,device=self.device) for x in (context.combat,context.anchor)))
            index = torch.as_tensor(slots, device=self.device, dtype=torch.long)
            identity_rows=torch.as_tensor(identities,device=self.device)
            relic_identity_rows=torch.as_tensor(relic_identities,device=self.device)
            incoming_banks=[torch.as_tensor(np.concatenate([np.stack([o[k] for o in observations]).reshape(len(slots),-1)
                           for k in layout],1),device=self.device) for _,layout in self.stream_root_banks]
        width = SharedCollectionGraphs._width(relics, 4)
        if width > self.relic_width:
            self.context.combat.index_copy_(0, index, context.combat)
            self.context.anchor.index_copy_(0, index, context.anchor)
            self.anchor_context_ids.index_copy_(0, index, identity_rows)
            self.roots(self.stream_observations, self.context)
            self._pack_stream_roots()
            return
        batch = dict(index=index, combat_context=context.combat, anchor_context=context.anchor,
                     anchor_identity=identity_rows, relic_identity=relic_identity_rows)
        batch.update({'bank'+str(i):v for i,v in enumerate(incoming_banks)})
        def update(inputs):
            index = inputs['index']
            self.context.combat.index_copy_(0,index,inputs['combat_context'])
            self.context.anchor.index_copy_(0,index,inputs['anchor_context'])
            self.anchor_context_ids.index_copy_(0,index,inputs['anchor_identity'])
            incoming = {}
            for i,(storage,layout) in enumerate(self.stream_root_banks):
                uploaded=inputs['bank'+str(i)]
                storage.index_copy_(0,index,uploaded)
                for key,(start,end,shape) in layout.items():
                    incoming[key]=uploaded[:,start:end].reshape(len(index),*shape)
            groups = {}
            for key,(group,start,end,shape) in self.root_layout.items():
                value=incoming[key]
                if key in ('relics','relic_counters'):value=value[:,:self.relic_width]
                groups.setdefault(group,[]).append(value.reshape(len(index),-1))
            for group,values in groups.items():
                self.packed_root[group].index_copy_(0,index,torch.cat(values,1))
            ids=incoming['relics'][:,:self.relic_width].long()
            summaries=self.core.combat.summaries
            identity=(summaries.relic_identity(ids)*ids.ne(0)[...,None]).sum(1)
            static=(summaries.relic_terms(ids,incoming['relic_counters'][:,:self.relic_width])
                    *(~self.decoder.live[ids])[...,None]).sum(1)
            self.root_relic_identity.index_copy_(0,index,identity)
            self.anchor_relic_ids.index_copy_(0,index,inputs['relic_identity'])
            self.banked_static_relics.index_copy_(0,index,static)
            self.banked_context.index_copy_(0,index,inputs['combat_context'])
            for key,bank in self.metadata_root.items():bank.index_copy_(0,index,incoming[key])
            return (inputs['combat_context'],)
        # Admission is idempotent, unlike simulator advancement. Capture its
        # gathers/projections together; exact row counts avoid duplicate writes
        # to live root slots in padded batches.
        signature=('admit-roots',len(slots),self.bank_key,
                   tuple((k,tuple(v.shape[1:]),v.dtype) for k,v in batch.items()))
        self.runtime._replay(self.runtime.entries,signature,batch,len(slots),update,self.runtime.capacity)

    def bind_wave(self, arena, owners, offset):
        if self.bound_arena is not arena:
            if self.bound_arena is not None:
                self.runtime.close()
                self.runtime = SharedCollectionGraphs(self.core, capacity=self.runtime.capacity)
            self.bound_arena = arena
            self.wave_owners = torch.empty(arena.capacity, dtype=torch.long, device=self.device)
            self.wave_offset = torch.zeros((), dtype=torch.long, device=self.device)
            self.arena_inputs = dict(categorical=arena.categorical, scalars=arena.scalars,
                mask=arena.mask, outcome=arena.outcome, prefix=arena.prefix,
                upgrades=arena.upgrades, permanent_misc=arena.misc)
        self.wave_owners[:len(owners)].copy_(owners)
        self.wave_offset.fill_(offset)

    def _forward_gather(self, batch, width, contexts=None, row_count=None, workspace=None):
        rows = batch['rows']
        owner = self.wave_owners[rows]
        physical = rows + self.wave_offset
        excluded = ('permanent_misc', 'upgrades') if row_count is not None else ('permanent_misc',)
        compact = (physical if self.bound_arena.observation_slots is None
                   else self.bound_arena.observation_slots[physical])
        piles = None
        if row_count is not None:
            from rl.policies.resident_ragged import frozen_indexed_piles
            piles = frozen_indexed_piles(self.arena_inputs['scalars'],
                self.arena_inputs['upgrades'], compact, row_count, self.card_base, self.card_delta,
                self.core.combat.summaries.upgrade_projection, self.decoder.pile_start, encode.UPGRADE_SCALE)
        # Keep the custom-kernel boundary before the head gathers so the
        # compiler can fuse them into their consumers instead of retaining
        # a full histogram tensor across that boundary.
        gathered = {'raw:'+k: (v if k == 'scalars' and row_count is not None
                               else v[compact if k in ('scalars','upgrades') else physical])
                    for k, v in self.arena_inputs.items() if k not in excluded}
        if piles is not None:
            gathered['resident_piles'] = piles
            gathered['scalar_rows'] = compact
        gathered.update({k: v[owner] for k, v in self.packed_root.items()})
        gathered['context'] = self.banked_context[owner] if contexts is None else contexts[rows]
        gathered['static_relics'] = self.banked_static_relics[owner]
        if row_count is not None:
            gathered['row_count'] = row_count
        if workspace is not None:
            gathered['workspace'] = workspace
        return self._forward(gathered, width, False)

    def resolve_changed_context(self, owners, misc, projected, changed, anchor, *, with_ids=False):
        """Batch cache misses and copies; never index a CUDA scalar in Python."""
        if self.native_context_inputs is None:
            keys = torch.cat((owners[changed, None], misc[changed]), 1).cpu().numpy()
            keys = [(int(self.root_generations[int(key[0])]), key.tobytes())
                    if hasattr(self, 'root_generations') else key.tobytes() for key in keys]
        else:
            # Root identity is not an encoder input. Intern every actual static
            # encoder field, then use the complete projected misc bytes. Equal
            # inputs across roots/siblings share a result; real changes do not.
            root_ids=self.context_input_ids[owners[changed].cpu().numpy()]
            projected_host=projected[changed].cpu().numpy()
            keys=self.native_context_inputs.intern([root_ids[:,None],projected_host]).tolist()
        missing = {}
        for j, key in enumerate(keys):
            if key not in self.changed_context and key not in missing:
                missing[key] = j
        if missing:
            selected = changed[torch.as_tensor(list(missing.values()), device=self.device)]
            obs = {k: v[owners[selected]] for k, v in self.root.items()}
            obs['outer_deck_misc'] = projected[selected]
            new = self.core.prepare_search_context(obs)
            identities = self._intern_anchor_contexts(new.anchor)
            for j, key in enumerate(missing):
                self.changed_context[key] = new.combat[j].clone(), new.anchor[j].clone(), identities[j]
            self.refreshes += len(missing)
        values = torch.stack([self.changed_context[key][int(anchor)] for key in keys])
        identities = np.asarray([self.changed_context[key][2] for key in keys], np.int64) if with_ids else None
        # Materialize this batch before eviction; live pipeline rows own copies.
        while len(self.changed_context)>4096:
            del self.changed_context[next(iter(self.changed_context))]
        if with_ids:
            if not anchor:
                raise ValueError('context identities belong to anchor scoring')
            return values, identities
        return values

    def run_wave(self, arena, owners, randoms, max_depth, deadline):
        from rl.search.resident_step import ResidentStepPipeline
        if self.step_pipeline is not None and (self.step_pipeline.arena is not arena
                or self.step_pipeline.max_depth != max_depth):
            self.step_pipeline.close()
            self.step_pipeline = None
        if self.step_pipeline is None:
            self.step_pipeline = ResidentStepPipeline(self, arena, max_depth)
        self.step_pipeline.begin(owners, randoms)
        return self.step_pipeline.run(deadline)

    @staticmethod
    def _sample(logits, mask, uniforms):
        # Preserve the paired-chain FP64 CDF and the exact PCG64 uniforms.
        scores = logits.double().masked_fill(~mask.bool(), -torch.inf)
        probability = (scores-scores.max(1, keepdim=True).values).exp()
        probability = probability/probability.sum(1, keepdim=True)
        cdf = probability.cumsum(1)
        cdf = cdf/cdf[:, -1:]
        return (cdf <= uniforms[:, None]).sum(1).int()

    def _forward_sample(self, batch, width):
        logits, values = self._forward_gather(batch, width)
        physical = batch['rows'] + self.wave_offset
        return self._sample(logits, self.arena_inputs['mask'][physical], batch['uniforms']), values

    def _metadata_gather(self, batch):
        rows = batch['rows']
        owner, physical = self.wave_owners[rows], rows + self.wave_offset
        c, mask = self.arena_inputs['categorical'][physical], self.arena_inputs['mask'][physical]
        required = (c[:, 127:223] > 0) | mask[:, 91:187].bool()
        width = (required * self.choice_columns).amax(1).clamp_min(1)
        root = {key: value[owner] for key, value in self.metadata_root.items()}
        misc = self.arena_inputs['permanent_misc'][physical]
        changed = ((misc >= 0) & (root['deck_ids'] != 0)
                   & (canonical_misc(misc) != root['outer_deck_misc'])).any(1)
        return (torch.stack((self.arena_inputs['outcome'][physical], mask.sum(1), width, changed), 1),)

    @staticmethod
    def _pad_indices(batch, padded):
        # Cached row indices are references into the CURRENT arena/wave.
        # Unlike copied observations, last wave's padding cannot stay valid
        # when its root count or active prefix shrinks.
        count = len(batch['rows'])
        return {k: torch.cat((v, v[:1].expand(padded-count, *v.shape[1:])))
                for k, v in batch.items()}

    @torch.no_grad()
    def metadata(self, raw, owners):
        """One host rendezvous for scheduling, choice shape and permanent growth."""
        if self.bound_arena is not None:
            count = len(owners)
            padded = max(16, 1 << (count-1).bit_length())
            rows = torch.arange(count, device=self.device)
            output = self.runtime._replay(self.runtime.map_entries,
                ('resident-metadata', padded, *self.bank_key),
                self._pad_indices({'rows': rows}, padded), padded,
                self._metadata_gather, 16)
            return output[0][:count].cpu().numpy()
        required = (raw['categorical'][:, 127:223] > 0) | raw['mask'][:, 91:187].bool()
        width = (required * self.choice_columns).amax(1).clamp_min(1)
        misc = raw['permanent_misc']
        original = self.root['outer_deck_misc'][owners]
        changed = ((misc >= 0) & (self.root['deck_ids'][owners] != 0)
                   & (canonical_misc(misc) != original)).any(1)
        return torch.stack((raw['outcome'], raw['mask'].sum(1), width, changed), 1).cpu().numpy()

    def _forward(self, batch, width, anchor):
        raw = {k[4:]: v for k, v in batch.items() if k.startswith('raw:')}
        roots = {key: batch[group][:, start:end].reshape(len(raw['outcome']), *shape)
                 for key, (group, start, end, shape) in self.root_layout.items()}
        rows = torch.arange(len(raw['outcome']), device=self.device)
        o = (self.decoder.anchor(raw, roots, rows) if anchor
             else self.decoder(raw, roots, rows, head_only=True,
                               choice_width=width if 'row_count' in batch else 96,
                               include_piles='resident_piles' not in batch,
                               scalar_rows=batch.get('scalar_rows')))
        if anchor:
            # Counter features belong after exact-cache lookup. Merely asking
            # for a previously scored anchor must not rebuild its projection.
            return o, batch['relic_identity']
        o['_choice_width'] = width
        o['_resident_fused_blocks'] = True
        if 'row_count' in batch:
            o['_resident_rows'] = batch['row_count']
            o['_resident_project_selection'] = True
            o['_resident_card_table'] = self.card_table
            o['_resident_card_projection'] = (self.projected_cards, self.card_upgrade)
            o['_resident_monster_projection'] = self.monster_tables
        if 'workspace' in batch:
            o['_resident_workspace'] = batch['workspace']
        context = SearchContext(batch['context'], None)
        if 'resident_piles' in batch:
            piles, sizes = batch['resident_piles']
        else:
            counts = torch.stack([o[k+'_counts'] for k in ('draw', 'discard', 'exhaust')], 1)
            piles = counts @ self.card_base + o['pile_upgraded_counts'] @ self.card_delta
            piles = piles + o['pile_upgrade_sums'].sum(2, keepdim=True)*self.core.combat.summaries.upgrade_projection
            sizes = counts[:, :, 1:].sum(2, keepdim=True)
        if 'row_count' in batch:
            from rl.policies.resident_ragged import frozen_relic_summary
            summaries = self.core.combat.summaries
            relics = frozen_relic_summary(o['relics'], o['relic_counters'], batch['static_relics'],
                self.decoder.live, self.relic_table, summaries.relic_identity.weight,
                summaries.relic_counter.weight)
        else:
            relics = self.core.combat.summaries.relic_terms(o['relics'], o['relic_counters']).sum(1)
        summary = CombatSummary(piles, sizes, relics)
        return self.core.combat_with_context(o, context, summary)

    @torch.no_grad()
    def score(self, raw, owners, rows, *, anchor=False, metadata=None, uniforms=None, return_host=False):
        if return_host and not anchor:
            raise ValueError('host results belong only to the completed anchor consumer')
        if anchor and uniforms is not None:
            raise ValueError('anchor values do not sample combat actions')
        if (any(m._parameters.get(name) is not p or p._version != version
                for m, name, p, version in self.parameter_bindings)
                or any(m._modules.get(name) is not child for m, name, child in self.module_bindings)):
            raise RuntimeError('resident search weights changed during collection')
        if anchor and len(rows) > self.anchor_batch_size:
            values = (np.empty(len(rows),np.float32) if return_host
                      else torch.empty(len(rows), device=self.device))
            for start in range(0, len(rows), self.anchor_batch_size):
                end = start + self.anchor_batch_size
                _, values[start:end] = self.score(raw, owners, rows[start:end], anchor=True,return_host=return_host)
            return None, values
        if anchor:
            rows = torch.as_tensor(rows, device=self.device)
            return self._score_anchor(raw, owners[rows], rows, rows, return_host=return_host)
        if metadata is not None:
            widths = np.asarray(metadata)[np.asarray(rows), 2]
            widths = np.minimum(96, 2 ** np.ceil(np.log2(widths)).astype(np.int64))
            unique = np.unique(widths)
            if len(unique) > 1:
                # The old ragged path never made ordinary combat rows pay for
                # one large choice screen. Keep each native candidate slot,
                # score homogeneous physical widths and restore original order.
                logits = (torch.empty((len(rows), self.decoder.space['combat_legal'].shape[0]), device=self.device)
                          if uniforms is None else torch.empty(len(rows), dtype=torch.int32, device=self.device))
                values = torch.empty(len(rows), device=self.device)
                rows = np.asarray(rows)
                for width in unique:
                    take = np.flatnonzero(widths == width)
                    out, value = self.score(raw, owners, rows[take], metadata=metadata,
                        uniforms=None if uniforms is None else uniforms[take])
                    logits[take], values[take] = out, value
                return logits, values
        hint = None if metadata is None else np.asarray(metadata)[np.asarray(rows)]
        rows = torch.as_tensor(rows, device=self.device)
        if hint is not None and not hint[:, 3].any() and self.bound_arena is not None:
            width = min(96, 1 << (int(hint[:, 2].max())-1).bit_length())
            padded = max(16, 1 << (len(rows)-1).bit_length())
            # Arena/root addresses remain owned throughout the collection.
            # Replay gathers on device instead of launching and copying every
            # raw/static field from Python at each search depth.
            batch = {'rows': rows}
            if uniforms is not None:
                batch['uniforms'] = uniforms
            batch = self._pad_indices(batch, padded)
            out = self.runtime._replay(self.runtime.entries,
                ('resident-gather', width, padded, uniforms is not None, *self.bank_key), batch, padded,
                lambda x: (self._forward_gather(x, width) if uniforms is None
                           else self._forward_sample(x, width)), self.runtime.capacity)
            return out[0][:len(rows)], out[1][:len(rows)].flatten()
        owner = owners[rows]
        batch = {'raw:'+k: v[rows] for k, v in raw.items()}
        batch.update({k: v[owner] for k, v in self.packed_root.items()})
        context = self.context.combat[owner].clone()
        # Permanent growth is rare; it owns a refreshed projection. No root or
        # sibling cache is mutated, and every card slot remains in its key.
        misc = batch['raw:permanent_misc']
        original = self.root['outer_deck_misc'][owner]
        projected = None
        if hint is None:
            projected = torch.where((misc >= 0) & (self.root['deck_ids'][owner] != 0), canonical_misc(misc), original)
            changed = (projected != original).any(1).nonzero().flatten()
        else:
            changed = torch.as_tensor(np.flatnonzero(hint[:, 3]), device=self.device)
        if len(changed):
            if projected is None:
                projected = torch.where((misc >= 0) & (self.root['deck_ids'][owner] != 0), canonical_misc(misc), original)
            refreshed = self.resolve_changed_context(owner, misc, projected, changed, False)
            context.index_copy_(0, changed, refreshed)
        batch['context'] = context
        if hint is None:
            required = (batch['raw:categorical'][:, 127:223] > 0) | batch['raw:mask'][:, 91:187].bool()
            columns = required.any(0).nonzero().flatten()
            width = min(96, max(1, int(columns[-1])+1)) if len(columns) else 1
        else:
            width = int(hint[:, 2].max())
        # Only masked trailing padding changes physical shape; no candidate cap.
        width = min(96, 1 << (width-1).bit_length())
        padded = max(16, 1 << (len(rows)-1).bit_length())
        outputs = self.runtime._replay(self.runtime.entries, ('resident', width, padded, self.relic_width), batch,
            padded, lambda x: self._forward(x, width, False), self.runtime.capacity)
        logits = outputs[0][:len(rows)]
        if uniforms is not None:
            logits = self._sample(logits, raw['mask'][rows], uniforms)
        return logits, outputs[1][:len(rows)].flatten()

    @torch.no_grad()
    def _score_anchor(self, raw, owner, rows, scalar_rows, *, return_host):
        if (any(m._parameters.get(name) is not p or p._version != version
                for m,name,p,version in self.parameter_bindings)
                or any(m._modules.get(name) is not child for m,name,child in self.module_bindings)):
            raise RuntimeError('resident search weights changed during collection')
        # Context and relic vectors stay on device; exact byte-interned IDs
        # participate in the key. Only cache misses gather those large vectors.
        misc = raw['permanent_misc'][rows]
        original = self.root['outer_deck_misc'][owner]
        projected = torch.where((misc >= 0) & (self.root['deck_ids'][owner] != 0), canonical_misc(misc), original)
        changed = (projected != original).any(1).nonzero().flatten()
        context_ids = self.anchor_context_ids[owner]
        context = None
        if len(changed):
            refreshed, identities = self.resolve_changed_context(owner,misc,projected,changed,True,with_ids=True)
            context = self.context.anchor[owner].clone()
            context.index_copy_(0,changed,refreshed)
            context_ids.index_copy_(0,changed,torch.as_tensor(identities,device=self.device))
        roots = {k: self.root[k][:, :self.relic_width] if k in ('relics','relic_counters') else self.root[k]
                 for k in ('relics','relic_counters','run_keys')}
        obs = self.decoder.anchor_indexed(raw,roots,owner,rows,scalar_rows)
        fields = {k:obs[k] for k in ('gold','run_keys','potions','combat_exit_anchor','relics','relic_counters')}
        fields.update(hp=obs['player_scalars'][:,:2],context_id=context_ids[:,None],
                      relic_identity_id=self.anchor_relic_ids[owner,None])
        groups = {}
        for key,value in fields.items(): groups.setdefault(value.dtype,[]).append(key)
        inputs = {}
        for keys in groups.values():
            host = torch.cat([fields[k].reshape(len(rows),-1) for k in keys],1).cpu().numpy()
            offset = 0
            for key in keys:
                shape = fields[key].shape[1:];end = offset+int(np.prod(shape))
                inputs[key]=host[:,offset:end].reshape(len(rows),*shape);offset=end
        def forward(host, selected):
            take = torch.as_tensor(selected,device=self.device)
            # Cache supplies original miss rows in first-occurrence order; no
            # host sorting, deduplication or lookup by root identity is needed.
            x = {k:fields[k][take] for k in ('gold','run_keys','potions','combat_exit_anchor','relics','relic_counters')}
            x['player_scalars'] = fields['hp'][take]
            ctx = self.context.anchor[owner[take]] if context is None else context[take]
            dynamic = self.core.combat.summaries.relic_counter_terms(x['relics'],x['relic_counters'])
            relics = self.root_relic_identity[owner[take]]+(dynamic*x['relics'].ne(0)[...,None]).sum(1)
            values = self.core.anchor_with_context(x,ctx,relics)
            return np.empty((len(values),0),np.float32),values.flatten().cpu().numpy()
        _,values = self.anchor_cache.forward(inputs,forward,indexed=True)
        return None,values if return_host else torch.as_tensor(values,device=self.device)

    def close(self):
        if self.step_pipeline is not None:
            self.step_pipeline.close()
            self.step_pipeline = None
        self.runtime.close()
        self.anchor_cache.clear()
        self.anchor_contexts.clear()
        if self.native_anchor_intern is not None:self.native_anchor_intern.clear()
        if self.native_context_inputs is not None:self.native_context_inputs.clear()
        self.anchor_context_bytes = 0
        self.compiled_gather = None

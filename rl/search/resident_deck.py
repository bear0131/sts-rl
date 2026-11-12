"""Frozen DeckLatent scoring directly from resident CUDA observations."""
import numpy as np
import torch

from env.core.cuda_observation import CudaObservationDecoder
from env.core.outer_return import MISC_SCALE
from rl.policies.combat_summary import CombatSummary
from rl.policies.collection_graphs import SharedCollectionGraphs
from rl.policies.deck_latent import SearchContext
from rl.search.rollout_inference import ExactInputCache


class ResidentDeckScorer:
    def __init__(self, policy, cache_bytes=64*1024**2, graph_capacity=128):
        self.policy, self.core = policy, policy.core
        self.device = next(policy.parameters()).device
        self.decoder = CudaObservationDecoder(policy.observation_space, self.device)
        self.runtime = SharedCollectionGraphs(self.core, capacity=graph_capacity)
        self.anchor_cache = ExactInputCache(cache_bytes // 4)
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
            summaries = self.core.combat.summaries
            ids = torch.arange(summaries.card.num_embeddings, device=self.device)
            self.card_base = summaries.card_terms(ids, torch.zeros_like(ids))
            self.card_delta = (summaries.card_projection(summaries.card.versioned_forward(ids,
                torch.ones_like(ids)))-self.card_base)*ids.ne(0)[:, None]

    @torch.no_grad()
    def roots(self, observations, context=None):
        keys = set(self.core.context.keys) | set(self.core.combat.scalar_keys) | {
            'phase', 'relic_counters', 'combat_is_boss'}
        self.root = {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                     for k in keys if k in observations[0]}
        self.context = self.core.prepare_search_context(self.root) if context is None else context
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
        for key in ('deck_ids', 'outer_deck_misc'):
            name = 'metadata:'+key
            if name not in bank:
                bank[name] = torch.empty((capacity, *self.root[key].shape[1:]),
                    device=self.device, dtype=self.root[key].dtype)
            bank[name][:len(observations)].copy_(self.root[key])
        self.metadata_root = {key: bank['metadata:'+key] for key in ('deck_ids', 'outer_deck_misc')}
        self.changed_context = {}

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

    def _forward_gather(self, batch, width, contexts=None, row_count=None):
        rows = batch['rows']
        owner = self.wave_owners[rows]
        physical = rows + self.wave_offset
        gathered = {'raw:'+k: v[physical] for k, v in self.arena_inputs.items() if k != 'permanent_misc'}
        gathered.update({k: v[owner] for k, v in self.packed_root.items()})
        gathered['context'] = self.banked_context[owner] if contexts is None else contexts[rows]
        if row_count is not None:
            gathered['row_count'] = row_count
        return self._forward(gathered, width, False)

    def resolve_changed_context(self, owners, misc, projected, changed, anchor):
        """Batch cache misses and copies; never index a CUDA scalar in Python."""
        keys = torch.cat((owners[changed, None], misc[changed]), 1).cpu().numpy()
        keys = [key.tobytes() for key in keys]
        missing = {}
        for j, key in enumerate(keys):
            if key not in self.changed_context and key not in missing:
                missing[key] = j
        if missing:
            selected = changed[torch.as_tensor(list(missing.values()), device=self.device)]
            obs = {k: v[owners[selected]] for k, v in self.root.items()}
            obs['outer_deck_misc'] = projected[selected]
            new = self.core.prepare_search_context(obs)
            for j, key in enumerate(missing):
                self.changed_context[key] = new.combat[j].clone(), new.anchor[j].clone()
            self.refreshes += len(missing)
        return torch.stack([self.changed_context[key][int(anchor)] for key in keys])

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
                   & (misc/MISC_SCALE != root['outer_deck_misc'])).any(1)
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
                   & (misc/MISC_SCALE != original)).any(1)
        return torch.stack((raw['outcome'], raw['mask'].sum(1), width, changed), 1).cpu().numpy()

    def _forward(self, batch, width, anchor):
        raw = {k[4:]: v for k, v in batch.items() if k.startswith('raw:')}
        roots = {key: batch[group][:, start:end].reshape(len(raw['outcome']), *shape)
                 for key, (group, start, end, shape) in self.root_layout.items()}
        rows = torch.arange(len(raw['outcome']), device=self.device)
        o = (self.decoder.anchor(raw, roots, rows) if anchor
             else self.decoder(raw, roots, rows, head_only=True))
        if anchor:
            # Cache identity is the complete consumed dynamic state plus the
            # frozen projected context. Combat turn/block do not enter it.
            relics = self.core.combat.summaries.relic_terms(o['relics'], o['relic_counters']).sum(1)
            return o, relics
        o['_choice_width'] = width
        o['_resident_fused_blocks'] = True
        if 'row_count' in batch:
            o['_resident_rows'] = batch['row_count']
        context = SearchContext(batch['context'], None)
        counts = torch.stack([o[k+'_counts'] for k in ('draw', 'discard', 'exhaust')], 1)
        piles = counts @ self.card_base + o['pile_upgraded_counts'] @ self.card_delta
        piles = piles + o['pile_upgrade_sums'].sum(2, keepdim=True)*self.core.combat.summaries.upgrade_projection
        relics = self.core.combat.summaries.relic_terms(o['relics'], o['relic_counters']).sum(1)
        summary = CombatSummary(piles, counts[:, :, 1:].sum(2, keepdim=True), relics)
        return self.core.combat_with_context(o, context, summary)

    @torch.no_grad()
    def score(self, raw, owners, rows, *, anchor=False, metadata=None, uniforms=None):
        if anchor and uniforms is not None:
            raise ValueError('anchor values do not sample combat actions')
        if (any(m._parameters.get(name) is not p or p._version != version
                for m, name, p, version in self.parameter_bindings)
                or any(m._modules.get(name) is not child for m, name, child in self.module_bindings)):
            raise RuntimeError('resident search weights changed during collection')
        if metadata is not None and not anchor:
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
        if hint is not None and not anchor and not hint[:, 3].any() and self.bound_arena is not None:
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
        context = (self.context.anchor if anchor else self.context.combat)[owner].clone()
        # Permanent growth is rare; it owns a refreshed projection. No root or
        # sibling cache is mutated, and every card slot remains in its key.
        misc = batch['raw:permanent_misc']
        original = self.root['outer_deck_misc'][owner]
        projected = None
        if hint is None:
            projected = torch.where((misc >= 0) & (self.root['deck_ids'][owner] != 0), misc/MISC_SCALE, original)
            changed = (projected != original).any(1).nonzero().flatten()
        else:
            changed = torch.as_tensor(np.flatnonzero(hint[:, 3]), device=self.device)
        if len(changed):
            if projected is None:
                projected = torch.where((misc >= 0) & (self.root['deck_ids'][owner] != 0), misc/MISC_SCALE, original)
            context.index_copy_(0, changed, self.resolve_changed_context(owner, misc, projected, changed, anchor))
        batch['context'] = context
        if anchor:
            obs, relics = self._forward(batch, 1, True)
            inputs = {k: obs[k].cpu().numpy() for k in ('gold', 'run_keys', 'potions', 'combat_exit_anchor')}
            inputs.update(hp=obs['player_scalars'][:, :2].cpu().numpy(),
                          context=context.cpu().numpy(), relics=relics.cpu().numpy())
            def forward(host):
                x = {k: torch.as_tensor(v, device=self.device) for k, v in host.items()}
                x['player_scalars'] = x.pop('hp')
                values = self.core.anchor_with_context(x, x.pop('context'), x.pop('relics'))
                return np.empty((len(values), 0), np.float32), values.flatten().cpu().numpy()
            _, values = self.anchor_cache.forward(inputs, forward)
            return None, torch.as_tensor(values, device=self.device)
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

    def close(self):
        if self.step_pipeline is not None:
            self.step_pipeline.close()
            self.step_pipeline = None
        self.runtime.close()
        self.anchor_cache.clear()

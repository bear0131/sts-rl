"""Frozen batched policy/anchor adapter for the model-free chain scheduler."""
import numpy as np
import torch

from rl.policies.deck_latent import SearchContext
from rl.policies.combat_summary import CombatSummary
from env.core import native_actions as C
from rl.search.rollout_inference import ExactInputCache


_CACHE_COUNTERS = ('requests', 'misses', 'hits', 'batch_duplicates', 'evictions',
                   'key_seconds', 'forward_seconds')
DECK_SEARCH_CACHE_METRICS = tuple(f'{side}_cache_{key}' for side in ('combat', 'anchor')
                                for key in (*_CACHE_COUNTERS, 'forward_batches'))


class DeckSearchInferenceCache:
    """One frozen collection, exact network inputs; never a transition cache."""
    def __init__(self, core, max_bytes=64 * 1024**2):
        self.core = core
        self.versions = tuple((id(p), p._version) for p in core.parameters())
        self.combat = ExactInputCache(max_bytes - max_bytes // 4)
        self.anchor = ExactInputCache(max_bytes // 4)
        self.batches = dict(combat=0, anchor=0)

    def validate(self, core):
        if core is not self.core or tuple((id(p), p._version) for p in core.parameters()) != self.versions:
            raise RuntimeError('search weights changed during a frozen collection')

    def statistics(self):
        return {f'{side}_cache_{key}': (self.batches[side] if key == 'forward_batches'
                else getattr(self, side).stats[key]) for side in ('combat', 'anchor')
                for key in (*_CACHE_COUNTERS, 'forward_batches')}

    def clear(self):
        self.combat.clear()
        self.anchor.clear()


class DeckLatentScorer:
    def __init__(self, policy, root_observations, context=None, tables=None, inference_cache=None):
        if policy.training:
            raise ValueError('search scorer requires a frozen inference policy')
        self.core = policy.core
        self.device = next(policy.parameters()).device
        self.roots = root_observations
        self.context_refreshes = 0
        self.context_keys = (*self.core.context.keys, 'phase')
        self.combat_keys = self.core.combat.search_input_keys
        self.anchor_keys = ('player_scalars', 'potions', *self.core.anchor.dynamic_keys)
        self.observation_keys = tuple(sorted(set(self.context_keys) | set(self.combat_keys)
            | set(self.anchor_keys) | {'draw_counts', 'discard_counts', 'exhaust_counts',
                                      'pile_upgraded_counts', 'pile_upgrade_sums', 'relic_counters'}))
        with torch.no_grad():
            self.context = (self.core.prepare_search_context(self.pack(root_observations, self.context_keys))
                            if context is None else context)
            self.tables = self.core.combat.summaries.export_inference_tables() if tables is None else tables
            self.summaries = [self.tables.new_state(o) for o in root_observations]
            # Copy root projections once, not on every scored frontier. Refreshed
            # permanent contexts acquire their own host copy below.
            self.host_context = SearchContext(*(v.detach().cpu().numpy().copy()
                for v in (self.context.combat, self.context.anchor)))
        self.inference_cache = (DeckSearchInferenceCache(self.core, 0) if inference_cache is None
                                else inference_cache)
        self.inference_cache.validate(self.core)
        self.initial_cache_stats = self.inference_cache.statistics()

    def cache_statistics(self):
        return {k: v - self.initial_cache_stats[k] for k, v in self.inference_cache.statistics().items()}

    def pack(self, observations, keys):
        return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=self.device)
                for k in keys}

    @torch.no_grad()
    def __call__(self, requests):
        self.inference_cache.validate(self.core)
        if not requests:
            return []
        roots = [r.root for r in requests]
        context = self.host_context.index(roots)
        # A branch retains its refreshed static context after permanent changes.
        # Counter changes/card moves do not enter this key or invoke the encoder.
        changed, summaries = [], []
        for i, r in enumerate(requests):
            if 'summary' not in r.cache:
                r.cache.update(summary=self.summaries[r.root].fork(), static=self.roots[r.root])
            summaries.append(r.cache['summary'].update(r.observation))
            if any(not np.array_equal(r.observation[k], r.cache['static'][k]) for k in self.core.context.keys):
                changed.append(i)
            elif 'host_context' in r.cache:
                context.combat[i], context.anchor[i] = r.cache['host_context']
        if changed:
            refreshed = self.core.prepare_search_context(
                self.pack([requests[i].observation for i in changed], self.context_keys))
            host = SearchContext(*(v.detach().cpu().numpy().copy() for v in (refreshed.combat, refreshed.anchor)))
            context.combat[changed] = host.combat
            context.anchor[changed] = host.anchor
            self.context_refreshes += len(changed)
            for j, i in enumerate(changed):
                requests[i].cache['static'] = {k: requests[i].observation[k].copy() for k in self.core.context.keys}
                requests[i].cache['host_context'] = (host.combat[j], host.anchor[j])
        summary = CombatSummary(*(np.stack([getattr(s, key) for s in summaries])
                                  for key in ('piles', 'counts', 'relics')))
        output = [None] * len(requests)
        for anchor in (False, True):
            rows = [i for i, r in enumerate(requests) if (r.kind == 'anchor') == anchor]
            if not rows:
                continue
            # The context and summaries already own static deck/map/pile data.
            # Pack only inputs consumed by this head, preserving row order and
            # the exact values/shape seen by the previous full-dict projection.
            batch = {k: np.stack([requests[i].observation[k] for i in rows])
                     for k in (self.anchor_keys if anchor else self.combat_keys)}
            batch['_context'] = (context.anchor if anchor else context.combat)[rows]
            batch['_relic_summary'] = summary.relics[rows]
            side = 'anchor' if anchor else 'combat'
            width = None
            if anchor:
                # Battle block/energy/turn do not enter the continuation head.
                batch['player_scalars'] = batch['player_scalars'][:, :2].copy()
            else:
                batch['_piles'], batch['_pile_counts'] = summary.piles[rows], summary.pile_counts[rows]
                if self.core.combat_encoder == 'transformer64x2':
                    pick = C.JOINT_PICK - C.JOINT_END
                    required = ((batch['select_native_indices'] > 0)
                                | batch['combat_legal'][:, pick:pick + batch['select_native_indices'].shape[1]].astype(bool))
                    columns = np.flatnonzero(required.any(0))
                    width = int(columns[-1]) + 1 if len(columns) else 1

            def forward(inputs):
                self.inference_cache.batches[side] += 1
                tensors = {k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()}
                projected = tensors.pop('_context')
                relics = tensors.pop('_relic_summary')
                if anchor:
                    values = self.core.anchor_with_context(tensors, projected, relics)
                    return np.empty((len(values), 0), np.float32), values.flatten().cpu().numpy()
                partial = CombatSummary(tensors.pop('_piles'), tensors.pop('_pile_counts'), relics)
                if width is not None:
                    tensors['_choice_width'] = width
                logits, values = self.core.combat_with_context(tensors,
                    SearchContext(projected, None), partial)
                return logits.cpu().numpy(), values.flatten().cpu().numpy()

            # Physical choice width affects rounding; do not reuse across it.
            # Missing-row batching is checked numerically on the same device,
            # not claimed bitwise equal to a differently sized GEMM.
            namespace = (width, str(self.device), torch.get_float32_matmul_precision(),
                         torch.is_autocast_enabled(self.device.type), str(torch.get_autocast_dtype(self.device.type)))
            logits, values = getattr(self.inference_cache, side).forward(batch, forward, namespace=namespace)
            if anchor:
                for i, value in zip(rows, values, strict=True):
                    output[i] = (None, float(value))
            else:
                for i, logit, value in zip(rows, logits, values, strict=True):
                    output[i] = (logit, float(value))
        return output

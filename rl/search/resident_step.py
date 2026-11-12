"""Transactional observation-to-action graphs over resident paired chains.

Ports the old StepPipeline's stable device compaction, one status rendezvous,
side-effect-free capacity retry and owned graph workspaces to deck-latent heads.
The model/observation contract and paired PCG64 streams remain unchanged.
"""
from collections import OrderedDict
import time

import numpy as np
import torch
import triton

from env.core.outer_return import MISC_SCALE
from env.search import resident_step_kernels as K


class ResidentStepPipeline:
    def __init__(self, scorer, arena, max_depth, capacity=128):
        if 'observe_live' not in arena.functions:
            raise ValueError('resident step graphs require observe_live simulator artifacts')
        self.scorer, self.arena, self.max_depth = scorer, arena, max_depth
        self.n, self.capacity = arena.capacity, capacity
        self.graphs = OrderedDict()
        self.workspace_capacities = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stats = dict(captures=0, replays=0, retries=0, host_reads=0, evictions=0)
        def empty(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=arena.device)
        self.live, self.depth, self.cursor, self.kind, self.forced, self.need, self.prefix, self.ids = (
            empty((self.n,)) for _ in range(8))
        self.anchors = empty((self.n,))
        self.final = empty((self.n,), torch.float64)
        self.totals = empty((triton.cdiv(self.n, 1024),))
        self.status, self.enabled, self.counters = empty((10,)), empty(()), empty((5,), torch.int64)
        self.randoms = empty((self.n, max_depth), torch.float64)
        self.expected = empty((self.n, 96), torch.float32)
        self.context = empty((self.n, scorer.context.combat.shape[1]), torch.float32)
        # Root banks may change shape with relic width; the card identity bank
        # has a fixed capacity and address for all waves of this collection.
        self.deck = empty((self.n, 96), torch.int64)
        self.enabled.zero_()

    def begin(self, owners, randoms):
        n = len(owners)
        self.live.zero_(); self.live[:n].fill_(1)
        self.depth.zero_(); self.depth[:n].fill_(1)
        self.cursor.zero_(); self.anchors.zero_(); self.final.zero_(); self.counters.zero_()
        self.randoms[:n].copy_(randoms)
        self.expected[:n].copy_(self.scorer.root['outer_deck_misc'][owners])
        self.context[:n].copy_(self.scorer.context.combat[owners])
        self.deck[:len(self.scorer.root['deck_ids'])].copy_(self.scorer.root['deck_ids'])
        self.wave_n, self.owners = n, owners
        self.enabled.fill_(1)

    def execute(self, rows, uniforms, width, token_capacity, query_capacity):
        s, a, n, F = self.scorer, self.arena, self.n, len(rows)
        a.selected.fill_(-1)
        a._launch('observe_live', a.observe_args+[self.live, s.wave_offset], n)
        self.status.zero_()
        K.classify[(triton.cdiv(n, 32),)](a.outcome, a.mask, a.categorical, a.misc,
            self.expected, self.deck, s.wave_owners, self.live, self.depth,
            self.kind, self.forced, self.need, self.status, s.wave_offset,
            n, self.max_depth, MISC_SCALE, 32, num_warps=4)
        nb = triton.cdiv(n, 1024)
        K.compact_local[(nb,)](self.need, self.prefix, self.totals, n, 1024)
        K.compact_scatter[(nb,)](self.need, self.prefix, self.totals, self.ids,
            self.status, n, 1024, nb, triton.next_power_of_2(nb))
        K.prepare[(triton.cdiv(F, 256),)](self.ids, rows, self.randoms, self.cursor,
            uniforms, self.status, F, self.max_depth, 256)
        logits, values = s._forward_gather({'rows': rows}, width, contexts=self.context,
            row_count=self.status, workspace=(token_capacity, query_capacity, self.status))
        sampled = torch.empty(F, dtype=torch.int32, device=a.device)
        K.sample_fp64[(F,)](logits, a.mask, rows, s.wave_offset, uniforms, sampled,
                            self.status, a.actions, triton.next_power_of_2(a.actions), num_warps=4)
        K.validate[(triton.cdiv(F, 256),)](values, self.status, F, 256)
        K.commit[(triton.cdiv(n, 256),)](self.kind, self.forced, self.live, self.depth,
            self.cursor, self.anchors, self.final, a.selected, self.status,
            self.enabled, self.counters, s.wave_offset, n, F, width, 256)
        K.commit_scores[(triton.cdiv(F, 256),)](rows, sampled, values, self.kind,
            self.final, a.selected, self.status, self.enabled, s.wave_offset, F, width, 256)
        a._launch('apply', a.apply_args, n)

    def graph(self, count, width):
        # Persistent kernels consume device counts. Bounded physical buckets
        # also keep the current head adapter from projecting a huge PAD tail.
        # Ragged blocks skip the remaining padded rows/tokens on device.
        F = max(32, triton.next_power_of_2(max(1, count)))
        width = next(w for w in (1, 4, 16, 96) if width <= w)
        base = F, width, self.scorer.bank_key
        ct,cq = self.workspace_capacities.get(base, (F*32, F*16))
        key = base, ct, cq
        if key not in self.graphs:
            if len(self.graphs) >= self.capacity:
                torch.cuda.synchronize()
                _, (old, _, _) = self.graphs.popitem(last=False)
                old.reset()
                self.stats['evictions'] += 1
                if not self.graphs:
                    self.pool = torch.cuda.graph_pool_handle()
            rows = torch.zeros(F, device=self.arena.device, dtype=torch.long)
            uniforms = torch.empty(F, device=self.arena.device, dtype=torch.float64)
            self.enabled.zero_()
            # Warmup observes but cannot commit actions, depths, counters or RNG.
            for _ in range(2):
                self.execute(rows, uniforms, width, ct, cq)
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self.execute(rows, uniforms, width, ct, cq)
            torch._C._cuda_clearCublasWorkspaces()
            self.enabled.fill_(1)
            self.graphs[key] = graph, rows, uniforms
            self.stats['captures'] += 1
        self.graphs.move_to_end(key)
        return self.graphs[key][0], F, width

    def refresh_context(self):
        s, a, n = self.scorer, self.arena, self.wave_n
        offset = int(s.wave_offset.cpu())
        misc = a.misc[offset:offset+n]
        original = s.root['outer_deck_misc'][self.owners]
        projected = torch.where((misc >= 0) & (s.root['deck_ids'][self.owners] != 0),
                                misc/MISC_SCALE, original)
        changed = (projected != self.expected[:n]).any(1).nonzero().flatten()
        if not len(changed):
            raise RuntimeError('device context retry did not identify a changed row')
        refreshed = s.resolve_changed_context(self.owners, misc, projected, changed, False)
        self.context.index_copy_(0, changed, refreshed)
        self.expected.index_copy_(0, changed, projected[changed])

    def run(self, deadline):
        count, width = self.wave_n, 1
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('resident paired budget incomplete; no partial teacher accepted')
            graph, F, W = self.graph(count, width)
            graph.replay(); self.stats['replays'] += 1
            status = self.status.cpu().tolist(); self.stats['host_reads'] += 1
            if status[3]:
                raise RuntimeError(f'resident device step failed validation {status[3]}')
            if status[0] > F or status[1] > W or status[2] or status[7]:
                self.stats['retries'] += 1
                if status[2]:
                    self.refresh_context()
                if status[7]:
                    self.workspace_capacities[(F,W,self.scorer.bank_key)] = (
                        max(F*32,triton.next_power_of_2(status[5])),
                        max(F*16,triton.next_power_of_2(status[6])))
                count, width = status[:2]
                continue
            if status[4] == 0:
                break
            count, width = status[:2]
        counts = self.counters.cpu().tolist()
        return (self.final[:self.wave_n].cpu().numpy(),
                self.anchors[:self.wave_n].cpu().numpy().astype(bool),
                dict(zip(('terminal_rows', 'anchor_rows', 'cutoff_rows', 'forced_rows', 'policy_rows'), counts)),
                int(self.depth.sum().cpu()))

    def close(self):
        torch.cuda.synchronize()
        for graph, _, _ in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.pool = torch.cuda.graph_pool_handle()

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
from env.core.cuda_observation import canonical_misc
from env.search import resident_step_kernels as K


class ResidentStepPipeline:
    # Current anchor/context heads share the device with the unchanged PPO
    # actor. Bound only physical scoring storage; every logical job is retained.
    frame_capacity = 32768

    def __init__(self, scorer, arena, max_depth, capacity=128, *, root_stride=0):
        if 'observe_live' not in arena.functions:
            raise ValueError('resident step graphs require observe_live simulator artifacts')
        self.scorer, self.arena, self.max_depth = scorer, arena, max_depth
        self.n, self.capacity = arena.capacity, capacity
        self.root_stride = root_stride
        self.root_count = len(scorer.root['deck_ids']) if root_stride else 0
        self.completion_host = None
        self.graphs = OrderedDict()
        self.workspace_capacities = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stats = dict(captures=0, replays=0, retries=0, host_reads=0, evictions=0)
        def empty(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=arena.device)
        self.live, self.depth, self.cursor, self.kind, self.forced, self.need, self.prefix, self.ids = (
            empty((self.n,)) for _ in range(8))
        self.anchors = empty((self.n,))
        self.changed_context = empty((self.n,)).zero_()
        self.active_ids, self.active_count = empty((self.n,)), empty((1,))
        self.frame_count = empty((1,))
        self.frame_control = empty((2,)).zero_()
        self.frame_capacity = min(self.n, arena.observation_capacity, type(self).frame_capacity)
        self.frame_ids = empty((self.frame_capacity,))
        self.depth_limit = empty((1,)).fill_(max_depth)
        self.compact_simulator = all(k in arena.functions for k in ('observe_indexed', 'apply_compact'))
        self.final = empty((self.n,), torch.float64)
        self.totals = empty((triton.cdiv(self.n, 1024),))
        self.status, self.enabled, self.counters = empty((10+self.root_count,)), empty(()), empty((5,), torch.int64)
        self.randoms = empty((self.n, max_depth), torch.float64)
        self.expected = empty((self.n, 96), torch.float32)
        self.context = empty((self.n, scorer.context.combat.shape[1]), torch.float32)
        # Root banks may change shape with relic width; the card identity bank
        # has a fixed capacity and address for all waves of this collection.
        self.deck = scorer.metadata_root['deck_ids']
        self.enabled.zero_()

    def begin(self, owners, randoms):
        n = len(owners)
        self.live.zero_(); self.live[:n].fill_(1)
        self.depth.zero_(); self.depth[:n].fill_(1)
        self.cursor.zero_(); self.anchors.zero_(); self.final.zero_(); self.counters.zero_()
        self.changed_context.zero_()
        self.randoms[:n].copy_(randoms)
        self.expected[:n].copy_(self.scorer.root['outer_deck_misc'][owners])
        self.context[:n].copy_(self.scorer.context.combat[owners])
        self.deck = self.scorer.metadata_root['deck_ids']
        self.wave_n, self.owners = n, owners
        self.frame_control.zero_(); self.frame_control[1:2].fill_(1)
        self.enabled.fill_(1)

    def refill(self, slots, owners, world_seeds):
        """Replace completed logical jobs; graph addresses and live siblings stay owned."""
        slots = torch.as_tensor(slots, device=self.arena.device, dtype=torch.long)
        self.completion_host = None
        from env.search.paired_rng import paired_uniforms
        paired_uniforms(world_seeds, slots, self.randoms)
        root_misc=self.scorer.root['outer_deck_misc'];root_context=self.scorer.context.combat
        K.refill[(len(slots),)](slots,owners,self.owners,self.scorer.wave_owners,
            root_misc,root_context,self.expected,self.context,self.depth,self.cursor,
            self.anchors,self.final,self.changed_context,self.live,len(slots),self.context.shape[1],
            root_misc.stride(0),root_context.stride(0),
            triton.next_power_of_2(max(96,self.context.shape[1])),num_warps=4)
        # A refill invalidates the previous compact pass, not its graph pool.
        self.frame_control.zero_()
        self.frame_control[1:2].fill_(1)

    def execute(self, rows, uniforms, width, token_capacity, query_capacity):
        s, a, n, F = self.scorer, self.arena, self.n, len(rows)
        a.selected.fill_(-1)
        nb = triton.cdiv(n, 1024)
        # Port the old pool/frame split: completed lanes immediately admit
        # waiting jobs into a bounded neural frame. More queued chains must
        # not allocate a worst-case neural workspace for every waiting job.
        K.compact_live_local[(nb,)](self.live, self.prefix, self.totals, self.frame_control, n, 1024)
        K.compact_live_scatter[(nb,)](self.live, self.prefix, self.totals,
            self.active_ids, self.active_count, self.frame_control, n, 1024, nb, triton.next_power_of_2(nb))
        K.frame_prepare[(triton.cdiv(self.frame_capacity,256),)](self.active_ids,
            self.active_count,self.frame_ids,self.frame_count,self.frame_control,self.frame_capacity,256)
        if self.compact_simulator:
            observation_args = a.observe_args+[self.frame_ids, self.frame_count, s.wave_offset]
            if 'observe_search' in a.functions:
                a._launch('observe_search', observation_args+[self.depth, self.depth_limit], n,
                          dispatch_n=self.frame_capacity)
            else:
                a._launch('observe_indexed', observation_args, n, dispatch_n=self.frame_capacity)
        else:
            # Old artifacts lack indexed dispatch; the observer still sees all
            # live lanes, but classification/commit owns only the current frame.
            a._launch('observe_live', a.observe_args+[self.live, s.wave_offset], n)
        self.status.zero_()
        frame_n=self.frame_capacity; frame_nb=triton.cdiv(frame_n,1024)
        K.classify[(triton.cdiv(frame_n, 32),)](a.outcome, a.mask, a.categorical, a.misc,
            self.expected, self.deck, s.wave_owners, self.frame_ids, self.frame_count, self.depth,
            self.kind, self.forced, self.need, self.changed_context, self.status, s.wave_offset,
            frame_n, self.max_depth, MISC_SCALE, 32, num_warps=4)
        K.compact_local[(frame_nb,)](self.need, self.prefix, self.totals, frame_n, 1024)
        K.compact_scatter[(frame_nb,)](self.need, self.prefix, self.totals, self.ids,
            self.status, frame_n, 1024, frame_nb, triton.next_power_of_2(frame_nb))
        K.prepare[(triton.cdiv(F, 256),)](self.ids, self.frame_ids, rows, self.randoms, self.cursor,
            uniforms, self.status, F, self.max_depth, 256)
        logits, values = s.step_forward({'rows': rows}, width, contexts=self.context,
            row_count=self.status, workspace=(token_capacity, query_capacity, self.status))
        sampled = torch.empty(F, dtype=torch.int32, device=a.device)
        K.sample_fp64[(F,)](logits, a.mask, rows, s.wave_offset, uniforms, sampled,
                            self.status, a.actions, triton.next_power_of_2(a.actions), num_warps=4)
        K.validate[(triton.cdiv(F, 256),)](values, self.status, F, 256)
        K.commit[(triton.cdiv(frame_n, 256),)](self.kind, self.forced, self.live, self.depth,
            self.cursor, self.anchors, self.final, a.selected, self.status,
            self.enabled, self.counters, s.wave_offset, self.frame_ids, self.frame_count,
            frame_n, F, width, 256)
        K.commit_scores[(triton.cdiv(F, 256),)](rows, sampled, values, self.kind,
            self.final, a.selected, self.status, self.enabled, s.wave_offset, F, width, 256)
        if self.compact_simulator:
            a._launch('apply_compact', a.apply_args+[
                self.frame_ids, self.frame_count, s.wave_offset], n, dispatch_n=self.frame_capacity)
        else:
            a._launch('apply', a.apply_args, n)
        K.frame_finish[(1,)](self.active_count,self.frame_count,self.frame_control,
            self.status,self.enabled,F,width)
        self.status[4:5].zero_()
        if self.root_stride:
            K.remaining_roots[(self.root_count,)](self.live,self.status,self.root_stride,
                                                 triton.next_power_of_2(self.root_stride))
        else:
            K.remaining_live[(nb,)](self.live, self.status, n, 1024)

    def graph(self, count, width):
        # Persistent kernels consume device counts. Bounded physical buckets
        # also keep the current head adapter from projecting a huge PAD tail.
        # Ragged blocks skip the remaining padded rows/tokens on device.
        F = max(32, triton.next_power_of_2(max(1, min(count, self.frame_capacity))))
        width = next(w for w in (1, 4, 16, 96) if width <= w)
        base = F, width, self.scorer.bank_key
        ct,cq = self.workspace_capacities.get(base, (F*32, F*16))
        key = base, ct, cq
        if key not in self.graphs:
            # Original StepPipeline owns one frame graph. A larger inventory
            # or workspace retires it before capturing its replacement.
            if self.graphs:
                torch.cuda.synchronize()
                for old, _, _ in self.graphs.values():
                    old.reset()
                self.stats['evictions'] += len(self.graphs)
                self.graphs.clear()
                del old
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
            # The PPO actor owns another CUDA process. Eager warmup scratch
            # cannot stay reserved here after the graph owns its live buffers;
            # that process cannot reclaim our idle caching-allocator blocks.
            torch.cuda.empty_cache()
        self.graphs.move_to_end(key)
        return self.graphs[key][0], F, width

    def refresh_context(self):
        s, a = self.scorer, self.arena
        changed = self.changed_context.nonzero().flatten()
        if not len(changed):
            raise RuntimeError('device context retry did not identify a changed row')
        offset = int(s.wave_offset.cpu())
        owners = self.owners[changed]
        misc = a.misc[offset+changed]
        original = s.root['outer_deck_misc'][owners]
        projected = torch.where((misc >= 0) & (s.root['deck_ids'][owners] != 0),
                                canonical_misc(misc), original)
        refreshed = s.resolve_changed_context(owners, misc, projected,
                                             torch.arange(len(changed),device=a.device), False)
        self.context.index_copy_(0, changed, refreshed)
        self.expected.index_copy_(0, changed, projected)
        self.changed_context[changed] = 0

    def tick(self, remaining, deadline):
        """Legacy StepPipeline tick: one complete pass, bounded readonly retries."""
        if not remaining:
            return 0
        width = getattr(self, 'stream_width', 1)
        frames = max(1, triton.cdiv(remaining, self.frame_capacity))
        for _ in range(8):
            if time.monotonic() > deadline:
                raise TimeoutError('resident stream budget incomplete; no partial teacher accepted')
            graph, F, W = self.graph(self.frame_capacity, width)
            for _ in range(frames):
                graph.replay()
            self.stats['replays'] += frames
            snapshot = self.status.cpu().numpy()
            status = snapshot[:10].tolist()
            self.stats['host_reads'] += 1
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
                width = max(width, status[1])
                # Some earlier frames may already have committed. The device
                # cursor owns that prefix; retry only its unfinished suffix.
                frames = max(1, triton.cdiv(status[9]-status[8], self.frame_capacity))
                continue
            if status[8] != 0:
                raise RuntimeError('stream tick returned inside an unfinished pass')
            self.stream_width = max(width, status[1])
            self.completion_host = snapshot[10:] if self.root_count else None
            return status[4]
        raise RuntimeError('resident stream overflow retry budget exhausted')

    def run(self, deadline):
        count, width = self.wave_n, 1
        remaining = self.wave_n
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('resident paired budget incomplete; no partial teacher accepted')
            graph, F, W = self.graph(count, width)
            # As in the original StepPipeline, submit a bounded group of
            # frames before rendezvous. A failed transaction cannot advance
            # state or RNG, so later submissions retain the same retry/error.
            replays = min(4, max(1, triton.cdiv(remaining, self.frame_capacity)))
            for _ in range(replays):
                graph.replay()
            self.stats['replays'] += replays
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
                count, width = max(count,status[0]), max(width,status[1])
                continue
            if status[4] == 0:
                break
            remaining = status[4]
            count = min(count,remaining)
            width = max(width,status[1])
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

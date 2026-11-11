"""Benchmark-only phase scheduler for deck-latent FP32 PPO over the existing ring."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import torch


def gradient_buckets(policy):
    core = policy.core
    def members(*modules):
        return {id(p) for m in modules for p in m.parameters()}
    early = members(core.upper_tokenizer, core.overworld) - members(core.context, core.combat, core.anchor)
    params = [p for p in policy.parameters() if p.requires_grad]
    return tuple([p for p in params if (id(p) in early) == upper] for upper in (True, False))


def bucket_partitions(policy, groups, world):
    """Keep each element on its original four-rank FP32 addition path."""
    params = [p for p in policy.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    edges = np.cumsum([0] + [(total+world-i-1)//world for i in range(world)]).tolist()
    offsets = {}
    start = 0
    for p in params:
        offsets[id(p)] = (start, start+p.numel())
        start += p.numel()
    return [[sum(max(0, min(offsets[id(p)][1], hi)-max(offsets[id(p)][0], lo))
                 for p in group) for lo, hi in zip(edges[:-1], edges[1:])] for group in groups]


class UpperFirstSchedule:
    """Compute upper first; replay its shared gradient additions after combat."""
    def __init__(self, model):
        self.model = model
        self.groups = gradient_buckets(model.policy)

    def after_physical_chunk(self, phase, count, total):
        if phase == 'combat':
            return
        partial = []
        for parameter in self.groups[1]:
            grad = parameter.grad
            if grad is not None:
                # A long OOM subdivision must not retain all its new partials
                # in VRAM. CPU copies preserve the FP32 values and ordering.
                partial.append((parameter, grad.cpu() if count < total else grad))
                parameter.grad = None
        self.deferred.append(partial)

    def schedule(self, chunks):
        self.deferred = []
        try:
            for chunk in chunks:
                if chunk[3] == 'combat':
                    continue
                before = len(self.deferred)
                committed = self.model._ppo_microbatch_stats['chunks']
                yield chunk
                if len(self.deferred) == before:
                    # Older benchmark callers only supplied schedule_chunks.
                    if self.model._ppo_microbatch_stats['chunks'] != committed + 1:
                        raise RuntimeError('subdivided upper chunk requires the commit hook')
                    rows = len(chunk[0].advantages)
                    self.after_physical_chunk(chunk[3], rows, rows)
            self.upper_ready()
            for chunk in chunks:
                if chunk[3] == 'combat':
                    yield chunk
            self.combat_done()
            for partial in self.deferred:
                dest, src = [], []
                for parameter, grad in partial:
                    grad = grad.to(parameter.device)
                    if parameter.grad is None:
                        parameter.grad = grad
                    else:
                        dest.append(parameter.grad)
                        src.append(grad)
                if dest:
                    torch._foreach_add_(dest, src)
                partial.clear()
        finally:
            self.deferred.clear()

    def upper_ready(self):
        pass

    def combat_done(self):
        pass


class EarlyUpperReduction(UpperFirstSchedule):
    """Only the frozen experiment uses this. One outstanding ring call at a time."""
    def __init__(self, model, ring, *, exact_ring=False):
        cfg = model.ppo_update_config
        if (model.device.type != 'cuda' or cfg['precision'] != 'fp32'
                or not cfg.get('upper_screen_max_rows') or cfg.get('context_gradient_replay')
                or model.ppo_update_loss_scale != 1):
            raise ValueError('requires CUDA FP32 per-screen chunks without latent replay')
        super().__init__(model)
        self.partitions = bucket_partitions(model.policy, self.groups, ring.world_size) if exact_ring else None
        if any(not group for group in self.groups):
            raise ValueError('expected both upper-exclusive and remaining gradients')
        if any(p.dtype != torch.float32 for group in self.groups for p in group):
            raise TypeError('requires FP32 gradients')
        self.ring = ring
        self.buffers = [torch.empty(sum(p.numel() for p in group), pin_memory=True)
                        for group in self.groups]
        self.copy = torch.cuda.Stream(device=model.device)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.steps = []
        self.pending = None

    def stage(self, index):
        params, host = self.groups[index], self.buffers[index]
        parts = [(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                 for p in params]
        flat = torch.cat(parts)
        self.copy.wait_stream(torch.cuda.current_stream(flat.device))
        with torch.cuda.stream(self.copy):
            begin = self.copy.record_event(torch.cuda.Event(enable_timing=True))
            host.copy_(flat, non_blocking=True)
            ready = self.copy.record_event(torch.cuda.Event(enable_timing=True))
            flat.record_stream(self.copy)
        return begin, ready

    def reduce(self, index, events):
        begin, ready = events
        ready.synchronize()
        started = perf_counter()
        kwargs = {} if self.partitions is None else dict(chunk_sizes=self.partitions[index])
        averaged, sent = self.ring.average(self.buffers[index], **kwargs)
        ended = perf_counter()
        if not np.isfinite(averaged.numpy()).all():
            raise FloatingPointError('nonfinite ring average; Adam was not advanced')
        return averaged, sent, ended-started, begin.elapsed_time(ready)/1000, started, ended

    def upper_ready(self):
        if self.pending is not None:
            raise RuntimeError('previous reduction still pending')
        self.started = perf_counter()
        self.pending = self.worker.submit(self.reduce, 0, self.stage(0))

    def combat_done(self):
        finished = perf_counter()
        self.early = self.pending.result()
        self.pending = None
        self.record = dict(ready_to_combat_done_sec=finished-self.started,
                           upper_wait_sec=perf_counter()-finished,
                           upper_ring_sec=self.early[2], upper_d2h_sec=self.early[3],
                           upper_ring_before_combat_done_sec=max(0., min(finished,self.early[5])-self.early[4]))

    def synchronize(self, _step):
        # Called only after the existing rank-wide local-loss/gradient finite vote.
        started = perf_counter()
        late = self.reduce(1, self.stage(1))
        for group, (host, *_) in zip(self.groups, (self.early, late), strict=True):
            device = host.to(group[0].device)
            for p, part in zip(group, device.split([p.numel() for p in group]), strict=True):
                p.grad = part.view_as(p)
        self.record.update(tail_ring_sec=late[2], tail_d2h_sec=late[3], late_wall_sec=perf_counter()-started)
        self.steps.append(self.record)
        seconds = self.early[2]+late[2]
        return dict(gradient_pack_sec=0., sync_sec=self.record['late_wall_sec'],
                    sync_mbps=(self.early[1]+late[1]) / 2**20 / seconds,
                    grad_bytes=sum(b.numel()*b.element_size() for b in self.buffers))

    def close(self):
        self.worker.shutdown(wait=True)

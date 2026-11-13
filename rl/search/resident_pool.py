"""Accepted streaming pool lifecycle with current model/state/target adapters.

The ledger, page allocator and submit/receive/reap ownership are migrated from
devicestepfull480141. Logical job addresses point into nonmoving, demand-grown
state pages; no root or sibling is relocated when another root is admitted.
"""
import numpy as np
import torch

from rl.policies.deck_latent import SearchContext
from rl.search.resident_step import ResidentStepPipeline


class ResidentPool:
    def __init__(self, arena, scorer, capacity, budget_cap, max_depth, timeout):
        self.arena, self.scorer = arena, scorer
        self.capacity, self.budget_cap = capacity, budget_cap
        self.max_depth, self.timeout = max_depth, timeout
        self.ledger = arena.host.RootLedger(capacity)
        self.bank, self.page_leases, self.jobs = {}, {}, {}
        self.closed, self.pipeline = False, None
        self.n = arena.capacity-capacity
        self.root_stride = ((budget_cap+15)//16)*16
        if self.n < capacity*self.root_stride:
            raise ValueError('arena cannot hold the declared root budgets')
        if not arena.paged:
            raise ValueError('streaming pool requires the legacy paged-state ABI')
        self.roots = torch.empty((capacity,arena.battle_bytes),dtype=torch.uint8,device=arena.device)
        arena.arena[:capacity].copy_(self.roots.data_ptr()+
            torch.arange(capacity,dtype=torch.int64,device=arena.device)*arena.battle_bytes)
        def allocate(count):
            return torch.empty((count,arena.battle_bytes),dtype=torch.uint8,device=arena.device)
        self.state_pages = arena.host.DevicePages(capacity, budget_cap, arena.battle_bytes,
            allocate, growth_pages=min(256, max(1, self.n//16)))
        self.remaining = 0
        self.counts = dict(chains=0, atomic_steps=0, budget_floor_overage=0,
                          max_active=0, submitted_roots=0, retired_roots=0,
                          submissions_while_pending=0, ticks=0)

    @property
    def pending(self):
        return bool(len(self.ledger))

    @torch.no_grad()
    def submit(self, rows):
        if self.closed:
            raise RuntimeError('streaming pool closed')
        if not rows:
            return {}
        for row in rows.values():
            if row['root'].get('native_fallback') or not row['root']['snapshot']['portable']:
                raise ValueError('nonportable root belongs to native fallback')
            if not 0 < len(row['worlds'])*len(row['root']['actions']) <= self.budget_cap:
                raise ValueError('paired budget exceeds the state-page contract')
        was_pending = self.pending
        handles = self.ledger.claim(rows)
        self.ledger.set_deadlines(handles,np.asarray([rows[h.key]['submitted_at']+self.timeout
                                                     for h in handles],np.float64))
        try:
            leases, pointers = self.state_pages.acquire([
                ((h.slot,h.generation),len(rows[h.key]['worlds'])*len(rows[h.key]['root']['actions']))
                for h in handles])
        except BaseException:
            for handle in handles:
                self.ledger.release(handle)
            raise
        slots = [h.slot for h in handles]
        for handle in handles:
            self.page_leases[handle.slot]=leases[(handle.slot,handle.generation)]
        payloads = [rows[h.key]['root'] for h in handles]
        self.arena.load_slots([r['snapshot'] for r in payloads], [r['prefix'] for r in payloads], slots)
        for step in range(max((len(r.get('replay',())) for r in payloads), default=0)):
            selected = [j for j,r in enumerate(payloads) if step < len(r.get('replay',()))]
            self.arena.replay_slots([slots[j] for j in selected],
                                   [payloads[j]['replay'][step] for j in selected])
        context = SearchContext(*(np.stack([rows[h.key]['context'][k]
            for h in handles]) for k in (0,1)))
        self.scorer.update_roots(slots,[rows[h.key]['observation'] for h in handles],context,self.capacity)
        if self.pipeline is None:
            owners = torch.zeros(self.n,dtype=torch.long,device=self.arena.device)
            self.scorer.bind_wave(self.arena,owners,self.capacity)
            self.pipeline = ResidentStepPipeline(self.scorer,self.arena,self.max_depth,
                                                root_stride=self.root_stride)
            self.scorer.step_pipeline = self.pipeline
            self.pipeline.begin_empty(owners)
        source,destination,world_seeds,initial_bits,draw,action_indices,counts = self.arena.host.pack_jobs(
            payloads,[rows[h.key]['worlds'] for h in handles],np.asarray(slots,np.int64),
            self.root_stride,self.capacity)
        page_addresses = []
        at = 0
        for handle,count in zip(handles,counts,strict=True):
            row = rows[handle.key]
            identity = handle.slot,handle.generation
            addresses = pointers[identity]
            page_addresses.append(addresses.view(np.int64))
            self.bank[handle.slot] = row
            self.jobs[handle.slot] = destination[at:at+count]-self.capacity,action_indices[at:at+count]
            self.counts['chains'] += count
            self.counts['budget_floor_overage'] += max(0,count-row['budget'])
            at += count
        # Old prepare_pages uploads the whole admission together. Root-by-root
        # copies serialize the host and GPU despite independent page ownership.
        self.arena.arena.index_copy_(0,torch.as_tensor(destination,device=self.arena.device),
            torch.as_tensor(np.concatenate(page_addresses),device=self.arena.device))
        owners,physical,seeds = self.arena.start_selected(source,destination,world_seeds,initial_bits,draw)
        self.pipeline.refill(physical.long()-self.capacity,owners,seeds)
        self.pipeline.deck = self.scorer.metadata_root['deck_ids']
        self.remaining += len(destination)
        self.counts['max_active'] = max(self.counts['max_active'],self.remaining)
        self.counts['submitted_roots'] += len(rows)
        self.counts['submissions_while_pending'] += int(was_pending)
        return {h.key:h for h in handles}

    @torch.no_grad()
    def reap(self):
        if not self.pending:
            return {}
        pipeline = self.pipeline
        # Jobs already occupy disjoint, fixed root segments. Reducing those
        # segments avoids atomically adding thousands of inactive zeros into
        # owner zero at every receive; no root/chain identity changes.
        left = pipeline.completion_host
        if left is None:
            left = pipeline.live[:self.capacity*self.root_stride].view(
                self.capacity,self.root_stride).sum(1,dtype=torch.int32).cpu().numpy()
        pipeline.completion_host = None
        handles = self.ledger.completed(left)
        if not handles:
            return {}
        all_jobs = self.arena.host.completed_jobs(handles,self.jobs)
        completed = pipeline.completed(all_jobs)
        anchors = completed[:,1].astype(bool)
        values = completed[:,0].copy()
        if anchors.any():
            selected = all_jobs[anchors]
            scores=[]
            batch_size=min(self.arena.observation_capacity,self.scorer.anchor_batch_size)
            for start in range(0,len(selected),batch_size):
                jobs=selected[start:start+batch_size]
                raw,physical,compact=self.arena.observe_anchor_selected(self.capacity+jobs)
                _,score=self.scorer._score_anchor(raw,pipeline.owners[jobs],physical,compact,return_host=True)
                if not np.isfinite(score).all():
                    raise FloatingPointError('nonfinite streaming anchor value')
                scores.append(score)
            values[anchors] = np.concatenate(scores)
        self.counts['atomic_steps'] += int(completed[:,2].sum())
        out = self.arena.host.complete_results(handles,self.bank,self.jobs,values)
        # Old lifecycle: destruction on the owner stream precedes page release,
        # and generation release precedes admission of the next logical root.
        self.arena.release_slots(np.r_[self.capacity+all_jobs,[h.slot for h in handles]])
        self.arena.arena[self.capacity+all_jobs] = 0
        self.state_pages.release([self.page_leases[h.slot] for h in handles])
        for handle in handles:
            self.ledger.release(handle)
            del self.bank[handle.slot],self.jobs[handle.slot],self.page_leases[handle.slot]
        self.counts['retired_roots'] += len(out)
        return out

    @torch.no_grad()
    def receive(self, steps=1):
        if steps < 0:
            raise ValueError('negative stream tick count')
        if steps == 0:
            return self.reap()
        # Every preceding tick already reaped all completed roots. Admission
        # only adds live jobs; repeating the same host rendezvous does no work.
        out = {}
        for _ in range(steps):
            if not self.remaining:
                break
            deadline = self.ledger.deadline()
            self.remaining = self.pipeline.tick(self.remaining,deadline)
            self.counts['ticks'] += 1
            out = self.reap()
            if out:
                return out
        if not out and not self.remaining and self.pending:
            raise RuntimeError('unfinished root without an active job')
        return out

    def statistics(self):
        result = dict(self.counts,state_pages=self.state_pages.statistics())
        if self.pipeline is not None:
            result.update(zip(('terminal_rows','anchor_rows','cutoff_rows','forced_rows','policy_rows'),
                              self.pipeline.counters.cpu().tolist()))
            result['graph'] = dict(self.pipeline.stats)
        return result

    def close(self, cancel=False):
        if self.closed:
            return
        if self.pending and not cancel:
            raise RuntimeError('normal close cannot abandon streaming roots')
        if self.pending:
            self.arena.release_slots(np.flatnonzero(self.arena.loaded))
            self.state_pages.release(list(self.page_leases.values()))
            for handle in list(self.ledger.pending.values()):
                self.ledger.release(handle)
            self.bank.clear(); self.page_leases.clear(); self.jobs.clear()
        self.scorer.close()
        self.arena.close()
        self.state_pages.close()
        self.closed = True

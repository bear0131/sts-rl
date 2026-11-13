"""Accepted streaming pool lifecycle with current model/state/target adapters.

The ledger, page allocator and submit/receive/reap ownership are migrated from
devicestepfull480141. Logical job addresses point into nonmoving, demand-grown
state pages; no root or sibling is relocated when another root is admitted.
"""
import time

import numpy as np
import torch

from env.search.root_ledger import RootLedger
from env.search.state_pages import DevicePages
from rl.policies.deck_latent import SearchContext
from rl.search.resident_step import ResidentStepPipeline


class ResidentPool:
    def __init__(self, arena, scorer, capacity, budget_cap, max_depth, timeout):
        self.arena, self.scorer = arena, scorer
        self.capacity, self.budget_cap = capacity, budget_cap
        self.max_depth, self.timeout = max_depth, timeout
        self.ledger = RootLedger(capacity)
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
        self.state_pages = DevicePages(capacity, budget_cap, arena.battle_bytes,
            allocate, growth_pages=min(256, max(1, self.n//16)))
        self.remaining = 0
        self.root_remaining = torch.zeros(capacity, dtype=torch.int32, device=arena.device)
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
        context = SearchContext(*(torch.as_tensor(np.stack([rows[h.key]['context'][k]
            for h in handles]), device=self.arena.device) for k in (0,1)))
        self.scorer.update_roots(slots,[rows[h.key]['observation'] for h in handles],context,self.capacity)
        if self.pipeline is None:
            owners = torch.zeros(self.n,dtype=torch.long,device=self.arena.device)
            self.scorer.bind_wave(self.arena,owners,self.capacity)
            self.pipeline = ResidentStepPipeline(self.scorer,self.arena,self.max_depth)
            self.scorer.step_pipeline = self.pipeline
            self.pipeline.begin(owners,torch.zeros((self.n,self.max_depth),
                dtype=torch.float64,device=self.arena.device))
            self.pipeline.live.zero_()
            self.pipeline.depth.zero_()
        source, destination, world_seeds, initial_bits, draw = [],[],[],[],[]
        page_addresses = []
        for handle in handles:
            row = rows[handle.key]
            root, worlds = row['root'], row['worlds']
            identity = handle.slot,handle.generation
            addresses = pointers[identity]
            physical = self.capacity+handle.slot*((self.budget_cap+15)//16)*16+np.arange(len(addresses))
            page_addresses.append(addresses.view(np.int64))
            actions = np.tile(np.arange(len(root['actions'])), len(worlds))
            seeds = np.repeat(worlds,len(root['actions']))
            source.extend([handle.slot]*len(physical)); destination.extend(physical)
            world_seeds.extend(seeds); initial_bits.extend(root['bits'][actions])
            draw.extend([root['draw_selection']]*len(physical))
            self.bank[handle.slot] = row
            self.jobs[handle.slot] = physical-self.capacity,actions
            self.counts['chains'] += len(physical)
            self.counts['budget_floor_overage'] += max(0,len(physical)-row['budget'])
        # Old prepare_pages uploads the whole admission together. Root-by-root
        # copies serialize the host and GPU despite independent page ownership.
        self.arena.arena.index_copy_(0,torch.as_tensor(destination,device=self.arena.device),
            torch.as_tensor(np.concatenate(page_addresses),device=self.arena.device))
        self.arena.fork(source,destination,world_seeds)
        self.arena.first_selected(source,destination,initial_bits,draw)
        self.pipeline.refill(np.asarray(destination)-self.capacity,
            torch.as_tensor(source,device=self.arena.device),
            torch.as_tensor(world_seeds,device=self.arena.device))
        self.pipeline.deck[:self.capacity].copy_(self.scorer.root['deck_ids'])
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
        self.root_remaining.copy_(pipeline.live[:self.capacity*self.root_stride]
            .view(self.capacity,self.root_stride).sum(1,dtype=torch.int32))
        left = self.root_remaining.cpu().numpy()
        handles = [h for h in self.ledger.pending.values() if left[h.slot] == 0]
        if not handles:
            return {}
        all_jobs = np.concatenate([self.jobs[h.slot][0] for h in handles])
        anchors = pipeline.anchors[all_jobs].cpu().numpy().astype(bool)
        values = pipeline.final[all_jobs].cpu().numpy()
        if anchors.any():
            selected = all_jobs[anchors]
            scores=[]
            batch_size=min(self.arena.observation_capacity,self.scorer.anchor_batch_size)
            for start in range(0,len(selected),batch_size):
                jobs=selected[start:start+batch_size]
                physical=torch.as_tensor(self.capacity+jobs,device=self.arena.device)
                raw=self.arena.observe_selected(self.capacity+jobs)
                compact=self.arena.observation_slots[physical]
                raw={k:v[compact if k in ('scalars','upgrades') else physical] for k,v in raw.items()}
                _, score=self.scorer.score(raw,pipeline.owners[jobs],np.arange(len(jobs)),anchor=True)
                if not torch.isfinite(score).all():
                    raise FloatingPointError('nonfinite streaming anchor value')
                scores.append(score.cpu().numpy())
            values[anchors] = np.concatenate(scores)
        self.counts['atomic_steps'] += int(pipeline.depth[all_jobs].sum().item())
        out, offset = {},0
        for handle in handles:
            row = self.bank[handle.slot]
            jobs, actions = self.jobs[handle.slot]
            n = len(row['root']['actions'])
            sums, visits = np.zeros(n,np.float64),np.zeros(n,np.int64)
            np.add.at(sums,actions,values[offset:offset+len(jobs)])
            np.add.at(visits,actions,1)
            if not (visits == len(row['worlds'])).all():
                raise RuntimeError('incomplete paired streaming teacher')
            out[handle.key] = dict(actions=row['root']['actions'],values=sums/visits,visits=visits)
            offset += len(jobs)
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
        out = self.reap()
        if out:
            return out
        for _ in range(steps):
            if not self.remaining:
                break
            deadline = min(row['submitted_at']+self.timeout for row in self.bank.values())
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

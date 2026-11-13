"""Native exceptional roots progress beside GPU roots, using legacy async RPCs.

A busy native worker group is held until its own frozen search completes; all
other groups continue. Pipe replies have one owner, including on cancellation.
"""
import time
import numpy as np
from rl.training.deck_search_worker import search_options


class NativeFrontierStream:
    def __init__(self, vec, service, core, config, generation):
        self.vec, self.service, self.config, self.generation = vec, service, config, generation
        self.static_keys = (*core.context.keys, 'phase')
        self.fields = tuple(sorted(set(self.static_keys) | set(core.combat.search_input_keys)
            | {'player_scalars', 'potions', *core.anchor.dynamic_keys, 'draw_counts',
               'discard_counts', 'exhaust_counts', 'pile_upgraded_counts', 'pile_upgrade_sums', 'relic_counters'}))
        self.queued, self.jobs = {}, {}

    def blocked(self):
        return {i for group in self.queued.keys() | self.jobs.keys() for i in self.vec._groups[group]}

    def submit(self, item):
        group = item['groups'][0]
        self.queued.setdefault(group, []).append(item)

    def poll(self):
        completed = []
        for group in list(self.queued):
            if group in self.jobs: continue
            item = self.queued[group].pop(0)
            if not self.queued[group]: del self.queued[group]
            key = (item['positions'][0], int(item['indices'][0]))
            context = tuple(np.asarray(x) for x in (item['context'].combat, item['context'].anchor))
            ident = self.service.begin_call('fallback_begin', (key, item['observations'], context))
            self.jobs[group] = dict(item=item, key=key, rpc=ident, stage='begin', started=time.monotonic(),
                totals=dict(scoring_seconds=0., worker_and_transport_seconds=0., frontier_batches=0.,
                            frontier_rows=0., transport_bytes=0.))
        for group, job in list(self.jobs.items()):
            remote = self.vec.remotes[group]
            if job['stage'] == 'native':
                if not self.vec.processes[group].is_alive(): self.vec._raise_worker_failure(group)
                if not remote.poll(): continue
                response = self.vec._receive_groups({remote: group})[group]
                job['totals']['worker_and_transport_seconds'] += time.monotonic()-job['sent']
                if response['done']:
                    job['result'], stats = response['result']
                    for key, value in stats.items(): job['totals'][key] = job['totals'].get(key, 0.)+value
                    job.update(stage='end', rpc=self.service.begin_call('fallback_end', job['key']))
                else:
                    job['sequence'] = response['sequence']
                    job['totals']['frontier_batches'] += 1
                    job['totals']['frontier_rows'] += len(response['rows'])
                    job['totals']['transport_bytes'] += response['observation_bytes']
                    job.update(stage='score', rpc=self.service.begin_call('fallback', (job['key'],response['rows'])),
                               sent=time.monotonic())
                continue
            ready, value = self.service.take_reply(job['rpc'])
            if not ready: continue
            if job['stage'] == 'end':
                job['totals'].update(value)
                job['totals']['native_fallback_seconds'] = time.monotonic()-job['started']
                completed.append((job['item'], job['result'], job['totals']))
                del self.jobs[group]
                continue
            if job['stage'] == 'begin':
                item = job['item']
                call = dict(indices=[self.vec._groups[group].index(int(item['indices'][0]))],
                    budgets=item['budgets'], observation_keys=self.fields, static_keys=self.static_keys,
                    seed=int(np.random.SeedSequence([item['seeds'][0],group]).generate_state(1,dtype=np.uint64)[0]),
                    generation=self.generation, options=search_options(self.config))
                remote.send(('deck_frontier_start', call))
            else:
                job['totals']['scoring_seconds'] += time.monotonic()-job['sent']
                remote.send(('deck_frontier_advance', dict(generation=self.generation,
                    sequence=job['sequence'], outputs=value)))
            job.update(stage='native', sent=time.monotonic())
        return completed

    def close(self):
        # Read an outstanding response before sending close, so another caller
        # cannot mistake a frontier reply for an environment transition.
        for group, job in self.jobs.items():
            remote = self.vec.remotes[group]
            if not self.vec.processes[group].is_alive(): continue
            if job['stage'] == 'native': self.vec._receive_groups({remote: group})
            remote.send(('deck_frontier_close', None))
            self.vec._receive_groups({remote: group})
        self.jobs.clear(); self.queued.clear()

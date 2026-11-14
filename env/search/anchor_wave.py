"""WaveDiag's native lanes with the current paired-world/anchor boundary.

The simulator advances every selected and forced action on resident C++
threads. Python assembles only scored frontiers; one GPU RPC may be in flight
while the caller continues advancing unrelated real environments.
"""
import time
import numpy as np
import slaythespire as sts

from env.core.native_observation import NativeRootObservationBatch


class NativeAnchorFrontier:
    """Model-free native lanes, usable in the environment's owning process."""
    def __init__(self, sources, roots, worlds, *, max_active, max_depth, threads,
                 fields, static):
        if not hasattr(sts, 'AnchorWave'):
            raise RuntimeError('legacy native fallback requires the pinned AnchorWave build')
        self.sources, self.roots = sources, roots
        self.fields, self.static = tuple(fields), frozenset(static)
        uniforms = [np.stack([np.random.default_rng(int(w)^0x504F4C494359).random(max_depth)
                              for w in row]) for row in worlds]
        self.wave = sts.AnchorWave([e.thick.bc for e in self.sources],
            [sum(1 << j for j in (e._multi or ())) for e in self.sources],
            worlds, uniforms, max_active, min(8, threads), max_depth)
        self.encoder = NativeRootObservationBatch(self.sources[0])
        self.views = {}
        self.complete = False

    def _frontier(self, rows):
        views, live, observations = [], [], [None] * len(rows)
        kinds = ('combat', 'anchor', 'cutoff')
        for j,(slot, root, job, prefix, depth, kind) in enumerate(rows):
            slot, root, kind = int(slot), int(root), int(kind)
            previous = self.views.get(slot)
            if previous is None or previous[0] != root:
                if previous is not None:
                    previous[1].close()
                view = self.sources[root].fork_run()
                view.thick.bc = self.wave.state(slot)
                self.views[slot] = root, view
            else:
                view = previous[1]
            view._multi = {i for i in range(10) if int(prefix) & (1 << i)} or None
            view.thick.combat_exit_anchor_pending = kind == 1
            view._invalidate_backend_cache()
            views.append(view)
            if kind == 1:
                observations[j] = view._obs()
            else:
                live.append(j)
        if live:
            encoded = self.encoder.encode([views[j] for j in live])
            for row,j in enumerate(live):
                observations[j] = {k: v[row] for k,v in encoded.items()}
        result = []
        for row,view,obs in zip(rows, views, observations, strict=True):
            _,root,job,_,_,kind = map(int,row)
            # Exact static comparison includes real permanent card/relic changes.
            delta = {k: obs[k] for k in self.fields if k not in self.static
                     or not np.array_equal(obs[k], self.roots[root][k])}
            result.append((root,job,kinds[kind],view.action_masks(),delta))
        return result

    def advance(self, outputs=None):
        if self.complete:
            raise RuntimeError('native fallback already consumed')
        if outputs is not None:
            logits,values = zip(*outputs)
            width = self.sources[0].action_space.n
            scores = np.stack([np.zeros(width) if v is None else v for v in logits])
            self.wave.advance(scores.astype(np.float64),np.asarray(values,np.float64))
        rows = self.wave.ready()
        if not len(rows):
            self.complete = True
            return dict(done=True, results=self.wave.results(), stats=self.wave.statistics())
        return dict(done=False, rows=self._frontier(rows))

    def close(self):
        for _,view in self.views.values():
            view.close()
        self.views.clear()
        self.wave = None


class CpuFallback:
    def __init__(self, keys, metadata, environments, service, config, *, threads=8):
        self.keys, self.service, self.config = list(keys), service, config
        self.environments, self.threads = environments, threads
        self.slots = [metadata[k]['slot'] for k in keys]
        self.roots = [metadata[k]['row']['observation'] for k in keys]
        self.worlds = [metadata[k]['row']['worlds'] for k in keys]
        self.frontier, self.actors = None, None
        self.complete = False
        self.started = time.monotonic()
        self.token = tuple(keys)
        self.stats = dict(budget_floor_overage=sum(max(0, len(self.worlds[j]) *
            len(metadata[k]['row']['root']['actions'])-metadata[k]['row']['budget'])
            for j,k in enumerate(keys)), context_refreshes=0, seconds=0.)
        obs = {k: np.stack([o[k] for o in self.roots]) for k in self.roots[0]}
        masks = np.stack([metadata[k]['mask'] for k in keys])
        capacity = getattr(environments, 'num_envs', None)
        if capacity is None:
            capacity = len(environments)
        self.pending = ('begin', self.service.begin_call('fallback_begin',
            (self.token, obs, masks, np.asarray(self.slots, np.intp), capacity)))

    def step(self):
        if self.complete:
            raise RuntimeError('native fallback already consumed')
        if time.monotonic()-self.started > self.config.timeout_seconds:
            raise TimeoutError('native paired budget incomplete; no partial result accepted')
        kind, ident = self.pending
        done, answer = self.service.take_reply(ident)
        if not done:
            return {}
        if kind == 'begin':
            self.actors = answer['actor']
            options = dict(max_active=self.config.max_active, max_depth=self.config.max_depth,
                           threads=self.threads, fields=answer['fields'], static=answer['static'])
            if hasattr(self.environments, 'anchor_frontier'):
                self.frontier = self.environments.anchor_frontier(self.slots, self.roots, self.worlds, options)
            else:
                sources = [self.environments[i].unwrapped for i in self.slots]
                self.frontier = NativeAnchorFrontier(sources, self.roots, self.worlds, **options)
            packet = self.frontier.advance()
        else:
            packet = self.frontier.advance(answer)
        if packet['done']:
            self.complete = True
            results = packet['results']
            for j, result in enumerate(results):
                result['actor'] = dict(logits=self.actors['logprobs'][j],
                    probabilities=self.actors['probabilities'][j], value=self.actors['values'][j])
            self.stats.update(packet['stats'])
            self.stats['seconds'] = time.monotonic()-self.started
            self.stats.update(self.service.call('fallback_end', self.token))
            self.actors = None
            return dict(zip(self.keys, results, strict=True))
        self.pending = ('score', self.service.begin_call('fallback', (self.token, packet['rows'])))
        return {}

    def close(self):
        if self.frontier is not None:
            self.frontier.close()
            self.frontier = None

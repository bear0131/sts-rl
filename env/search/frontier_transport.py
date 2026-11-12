"""Model-free, bounded frontier exchange for a frozen external scorer.

Only dynamic head inputs cross the pipe on every frontier. Static context
fields are included when different from the root, including permanent combat
changes. No GPU tensor, model, or branch-local inference cache is serialized.
"""
import numpy as np

from env.core.native_observation import NativeRootObservationBatch
from env.search.anchored_chains import paired_anchor_frontiers


class FrontierSession:
    def __init__(self, envs):
        self.envs = envs
        self.encoder = NativeRootObservationBatch(envs[0].unwrapped)
        self.session = None

    def start(self, request):
        if self.session is not None:
            raise RuntimeError('a shadow frontier session is already active')
        roots = [self.envs[i].unwrapped for i in request['indices']]
        self.roots = [root._obs() for root in roots]
        self.fields = tuple(request['observation_keys'])
        self.static = frozenset(request['static_keys'])
        if not self.static.issubset(self.fields):
            raise ValueError('static search fields must belong to the observation contract')
        self.generation = request['generation']
        self.sequence = 0
        self.session = paired_anchor_frontiers(roots, request['budgets'], seed=request['seed'],
            observation_batch=self.encoder, **request['options'])
        return self._advance(None)

    def advance(self, request):
        if (self.session is None or request['generation'] != self.generation
                or request['sequence'] != self.sequence):
            raise RuntimeError('stale or out-of-order search frontier response')
        return self._advance(request['outputs'])

    def _advance(self, outputs):
        try:
            frontier = next(self.session) if outputs is None else self.session.send(outputs)
            self.sequence += 1
            rows, nbytes = [], 0
            for row in frontier:
                observation = {k: row.observation[k] for k in self.fields
                    if k not in self.static or not np.array_equal(row.observation[k], self.roots[row.root][k])}
                nbytes += sum(v.nbytes for v in observation.values()) + row.action_mask.nbytes
                rows.append((row.root, row.lane, row.kind, row.action_mask, observation))
            return dict(done=False, sequence=self.sequence, rows=rows, observation_bytes=nbytes)
        except StopIteration as complete:
            self.close()
            return dict(done=True, result=complete.value)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.roots = []

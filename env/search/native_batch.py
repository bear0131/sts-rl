"""Model-free adapter for same-process native search workers."""
import numpy as np
import slaythespire as sts

from env.core import encode
from env.core.run_env import _nosl_compact_lut, _nosl_compact_empty, _nosl_card_stats


class NativeSearchBatch:
    def __init__(self, envs, threads):
        self.envs = [env.unwrapped for env in envs]
        if not hasattr(sts, 'NeuralSearchBatch'):
            raise RuntimeError('native vector environment requires the NeuralSearchBatch extension')
        signatures = {(e._card_language, e._combat_language, e._joint_actions,
                       e._autoregressive_actions, e.action_space.n) for e in self.envs}
        if len(signatures) != 1 or any(e._combat_language for e in self.envs):
            raise ValueError('native search requires a common compact observation contract')
        self.pool = sts.NeuralSearchBatch(threads)
        self.language = (_nosl_compact_lut()['language_card_index']
                         if self.envs[0]._card_language else None)
        first = self.envs[0]
        shapes = dict(_nosl_compact_empty(card_language=first._card_language))
        if first._joint_actions:
            shapes['combat_legal'] = np.zeros(first._neural_width, dtype=bool)
        self.decoder = sts.NeuralObservationDecoder(shapes, _nosl_compact_lut(), _nosl_card_stats(),
            {key: getattr(encode, key+'_SCALE') for key in
                ('HP','BLOCK','ENERGY','TURN','BUFF','COUNTER','DMG','HITS','UPGRADE','PICK')},
            dict(counters=encode.COMBAT_COUNTERS, moves=encode.V.MOVE_VOCAB_CAPACITY,
                 intents=len(encode.V.INTENT_CATEGORY), contexts=encode.V.SELECT_CONTEXT_VOCAB_CAPACITY),
            first._policy_offset, first._neural_width if first._joint_actions else 0,
            first._autoregressive_actions)
        self.indices = []
        self.payload = None
        self.global_fields = None
        self.leaves = None

    def begin(self, calls):
        if not calls:
            raise ValueError('native search requires at least one tree')
        self.indices = list(calls)
        self.leaves = np.zeros(len(self.indices), dtype=np.int64)
        self.native_seconds = self.pack_seconds = 0.0
        sessions, fields = [], []
        try:
            for index in self.indices:
                args, kwargs = calls[index]
                if kwargs.get('leaf_batch_size', 1) != 1 or not kwargs.get('compact_observation'):
                    raise ValueError('native cross-tree batching requires compact observations and one leaf per tree')
                env = self.envs[index]
                sessions.append(env.nosl_is_az_cpp_begin(*args, **kwargs,
                                                        _defer_leaf_selection=True))
                fields.append(env._native_packed_global_fields())
            self.global_fields = {key: np.stack([f[key] for f in fields]) for key in fields[0]}
            self.payload = self.pool.begin(sessions, self.language)
            self._record_timings()
        except BaseException:
            self.clear()
            raise

    @property
    def done(self):
        return bool(self.payload['done'])

    def observations(self):
        return self.decoder.decode(self.payload, self.global_fields)

    def advance(self, logits, values):
        self.leaves[np.asarray(self.payload['owners'], dtype=np.intp)] += 1
        self.payload = self.pool.advance(values, logits, self.language)
        self._record_timings()

    def _record_timings(self):
        self.native_seconds += self.payload['native_seconds']
        self.pack_seconds += self.payload['pack_seconds']

    def results(self):
        if not self.done:
            raise RuntimeError('native search results requested before completion')
        results = {}
        for index in self.indices:
            # Existing finish adapter owns native/thin action mapping, pending
            # actions, diagnostics and session lifetime. No leaves remain.
            response = self.envs[index].nosl_is_az_cpp_leaf_batch(1)
            if not response['done']:
                raise RuntimeError('native batch finished with an outstanding tree')
            results[index] = response['result']
        return results

    def clear(self):
        self.pool.clear()
        for index in self.indices:
            self.envs[index].nosl_is_az_cpp_abort()
        self.indices = []
        self.payload = self.global_fields = None

    def close(self):
        if self.pool is not None:
            self.clear()
            self.pool = None

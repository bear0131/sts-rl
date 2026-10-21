"""Model-free adapter for same-process native search workers."""
import numpy as np
import slaythespire as sts
import time

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
        self.outer_return = first.outer_return_context
        if any(e.outer_return_context != self.outer_return for e in self.envs):
            raise ValueError('native batch cannot mix outer-return and legacy environments')
        shapes = dict(_nosl_compact_empty(card_language=first._card_language))
        if first._joint_actions:
            shapes['combat_legal'] = np.zeros(first._neural_width, dtype=bool)
        # The decoder must preserve the public observation-space dtypes.  In
        # particular categorical observations are int64; narrowing just these
        # two fields made batched native leaves differ byte-for-byte from the
        # serial bridge path.
        self._prototypes = dict(shapes)
        self._decoder_keys = frozenset(shapes)
        if self.outer_return:
            from env.core.outer_return import CONTEXT_KEYS
            self._prototypes.update({key: np.zeros(first.observation_space[key].shape,
                dtype=first.observation_space[key].dtype) for key in CONTEXT_KEYS})
        self._layouts = {}
        # Grow printed-card data only when a real leaf needs a higher upgrade
        # count, so Searing Blow remains exact without reserving the complete
        # int16 range in every search worker.
        self._card_stats = _nosl_card_stats()
        self.decoder = sts.NeuralObservationDecoder(
            shapes, _nosl_compact_lut(), self._card_stats,
            {key: getattr(encode, key+'_SCALE') for key in
                ('HP','BLOCK','ENERGY','TURN','BUFF','COUNTER','DMG','HITS','UPGRADE','PICK','GOLD')},
            dict(counters=encode.COMBAT_COUNTERS, moves=encode.V.MOVE_VOCAB_CAPACITY,
                 intents=len(encode.V.INTENT_CATEGORY), contexts=encode.V.SELECT_CONTEXT_VOCAB_CAPACITY),
            first._policy_offset, first._neural_width if first._joint_actions else 0,
            first._autoregressive_actions)
        self.indices = []
        self.payload = None
        self.global_fields = None
        self.leaves = None
        self.native_seconds = self.pack_seconds = self.decode_seconds = 0.0

    def begin(self, calls):
        if not calls:
            raise ValueError('native search requires at least one tree')
        self.indices = list(calls)
        self.leaves = np.zeros(len(self.indices), dtype=np.int64)
        self.native_seconds = self.pack_seconds = 0.0
        self.decode_seconds = 0.0
        sessions, fields = [], []
        try:
            for index in self.indices:
                args, kwargs = calls[index]
                if bool(kwargs.get('outer_return')) != self.outer_return:
                    raise ValueError('search objective and environment context disagree')
                if kwargs.get('leaf_batch_size', 1) != 1 or not kwargs.get('compact_observation'):
                    raise ValueError('native cross-tree batching requires compact observations and one leaf per tree')
                env = self.envs[index]
                sessions.append(env.nosl_is_az_cpp_begin(*args, **kwargs,
                                                        _defer_leaf_selection=True))
                fields.append(env._native_packed_global_fields())
            self.global_fields = {key: np.stack([f[key] for f in fields]) for key in fields[0]}
            self._decoder_globals = {key: value for key, value in self.global_fields.items()
                                     if key in self._decoder_keys}
            relics = self.global_fields['relics']
            self._relic_widths = np.maximum(1, np.max(
                (relics != 0) * np.arange(1, relics.shape[1]+1), axis=1))
            self.payload = self.pool.begin(sessions, self.language)
            self._record_timings()
        except BaseException:
            self.clear()
            raise

    @property
    def done(self):
        return bool(self.payload['done'])

    def observations(self):
        started = time.perf_counter()
        self._ensure_card_stats()
        result = self.decoder.decode(self.payload, self._decoder_globals)
        if self.outer_return:
            from env.core.outer_return import CONTEXT_KEYS, MISC_SCALE
            if self.payload.get('outer_return_schema') != 1:
                raise RuntimeError('native outer-return leaf contract is missing')
            owners = np.asarray(self.payload['owners'], dtype=np.intp)
            result.update({key: self.global_fields[key][owners].copy() for key in CONTEXT_KEYS})
            hp = result['player_scalars']
            result['run_hp'][:, :2] = hp[:, :2]
            result['run_hp'][:, 2] = np.divide(hp[:, 0], hp[:, 1],
                out=np.zeros(len(hp), dtype=np.float32), where=hp[:, 1] != 0)
            for row, changes in self.payload['permanent_misc'].items():
                for index, value in changes.items():
                    if 0 <= index < result['deck_ids'].shape[1] and result['deck_ids'][row, index] != 0:
                        result['outer_deck_misc'][row, index] = value / MISC_SCALE
        self.decode_seconds += time.perf_counter()-started
        return result

    def combat_exits(self):
        if not self.outer_return:
            return {}
        return {row: self.envs[self.indices[self.payload['owners'][row]]].settle_search_combat_exit(battle)
                for row, battle in self.payload['combat_exits'].items()}

    @property
    def rows(self):
        return len(self.payload['owners'])

    @property
    def active_relic_width(self):
        return int(max(self._relic_widths[owner] for owner in self.payload['owners']))

    def observation_layout(self):
        """Shape/dtype views only; their contents are not model observations."""
        count = self.rows
        if count not in self._layouts:
            self._layouts[count] = {key: np.broadcast_to(value, (count, *value.shape))
                                    for key, value in self._prototypes.items()}
        return self._layouts[count]

    def write_observations(self, destination):
        """Fill caller-owned contiguous arrays before their next device upload."""
        if self.outer_return:
            for key, value in self.observations().items():
                np.copyto(destination[key][:len(value)], value)
            return
        started = time.perf_counter()
        self._ensure_card_stats()
        self.decoder.decode(self.payload, self._decoder_globals, destination)
        self.decode_seconds += time.perf_counter()-started

    def _ensure_card_stats(self):
        layout = _nosl_compact_lut()['layout']
        hand_start = (6 + int(layout['player_status_count'])
                      + encode.COMBAT_COUNTERS + 1 + 30
                      + 5 * int(layout['monster_status_count']))
        scalars = np.asarray(self.payload['scalars'])
        if scalars.ndim != 2 or scalars.shape[1] < hand_start + 20:
            raise ValueError('native observation layout omits hand upgrade data')
        upgrades = scalars[:, hand_start:hand_start+20:2]
        if (not np.all(np.isfinite(upgrades)) or np.any(upgrades < 0)
                or np.any(upgrades != np.rint(upgrades))):
            raise ValueError('native hand upgrade count is invalid')
        needed = int(upgrades.max(initial=0)) + 1
        if needed > self._card_stats['damage'].shape[1]:
            self._card_stats = _nosl_card_stats(needed)
            self.decoder.set_card_stats(self._card_stats)

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

    def rulebase_turn_decisions(self, calls):
        """Search independent combats in the existing GIL-free worker pool."""
        from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig

        results, indices, battles, nonces, relic_lists = {}, [], [], [], []
        config = None
        for index, (args, kwargs) in calls.items():
            if args:
                raise ValueError('rulebase batch accepts keyword configuration only')
            env = self.envs[index]
            if env.thick is None or env._phase() != 'combat' or env.thick.bc is None:
                raise RuntimeError('rulebase turn policy requires a live combat state')
            current = TurnPlannerConfig(**{k: v for k, v in kwargs.items() if k != 'state_nonce'})
            if config is not None and config != current:
                raise ValueError('batched rule search requires a common configuration')
            config = current
            relics = tuple(item['id'] for item in env.thick.gc.relic_state())
            if 'Runic Dome' in relics or env._rulebase_pending_multi is not None:
                results[index] = env.rulebase_turn_decision(**kwargs)
                continue
            indices.append(index)
            battles.append(env.thick.bc)
            nonces.append(kwargs['state_nonce'])
            relic_lists.append(relics)
        if indices:
            plans = NativeTurnPlanner(config).choose_batch(battles, state_nonces=nonces,
                public_relics=relic_lists, pool=self.pool)
            for index, plan in zip(indices, plans, strict=True):
                results[index] = self.envs[index]._apply_rulebase_turn_decision(
                    plan.action_bits, dict(plan.diagnostics, planner_dome_fallback=0))
        return {index: results[index] for index in calls}

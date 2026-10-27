"""Model-free adapter for same-process native search workers."""
import numpy as np
import slaythespire as sts
import time

from env.core import encode
from env.core.run_env import (
    _nosl_compact_lut, _nosl_compact_empty, _nosl_card_stats,
    _NOSL_ROOT_GLOBAL_FIELDS,
)


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
        self.public_context = first.outer_return_context
        self.outer_return = first.outer_return_objective
        if any((e.outer_return_context, e.outer_return_objective) !=
               (self.public_context, self.outer_return) for e in self.envs):
            raise ValueError('native batch cannot mix outer-return and legacy environments')
        shapes = dict(_nosl_compact_empty(card_language=first._card_language))
        if first._joint_actions:
            shapes['combat_legal'] = np.zeros(first._neural_width, dtype=bool)
        # The decoder must preserve the public observation-space dtypes.  In
        # particular categorical observations are int64; narrowing just these
        # two fields made batched native leaves differ byte-for-byte from the
        # serial bridge path.
        self._prototypes = dict(shapes)
        if self.public_context:
            from env.core.outer_return import CONTEXT_KEYS
            self._prototypes.update({key: np.zeros(first.observation_space[key].shape,
                dtype=first.observation_space[key].dtype) for key in CONTEXT_KEYS})
        self._decoder_keys = frozenset(self._prototypes)
        self._layouts = {}
        # Grow printed-card data only when a real leaf needs a higher upgrade
        # count, so Searing Blow remains exact without reserving the complete
        # int16 range in every search worker.
        self._card_stats = _nosl_card_stats()
        self.decoder = sts.NeuralObservationDecoder(
            self._prototypes, _nosl_compact_lut(), self._card_stats,
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

    def begin(self, calls, *, root_observations=None):
        if not calls:
            raise ValueError('native search requires at least one tree')
        context_keys = ()
        global_keys = ()
        if root_observations is not None:
            from env.core.outer_return import CONTEXT_KEYS
            if not self.public_context:
                raise ValueError('root context transport requires the outer-return objective')
            context_keys = CONTEXT_KEYS
            if all(key in root_observations for key in _NOSL_ROOT_GLOBAL_FIELDS):
                global_keys = _NOSL_ROOT_GLOBAL_FIELDS
            for key in set(context_keys).union(global_keys):
                space = self.envs[0].observation_space[key]
                value = root_observations.get(key)
                if (not isinstance(value, np.ndarray) or
                        value.shape != (len(self.envs), *space.shape) or value.dtype != space.dtype):
                    raise ValueError(f'root context batch disagrees with observation contract: {key}')
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
                globals = ({key: root_observations[key][index] for key in global_keys}
                           if global_keys else None)
                sessions.append(env.nosl_is_az_cpp_begin(*args, **kwargs,
                    _defer_leaf_selection=True, _root_global_fields=globals))
                # Collector already encoded these exact current roots. Read
                # the supplied arrays only during this call; stack below owns
                # the search copy before the next environment transition.
                context = ({key: root_observations[key][index] for key in context_keys}
                           if root_observations is not None else None)
                fields.append(env._native_packed_global_fields(context))
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

    def observations(self, rows=None):
        return self._decode_observations(rows=rows)

    def _decode_observations(self, destination=None, rows=None):
        started = time.perf_counter()
        self._ensure_card_stats(validate_all=rows is not None)
        payload = self.payload
        if rows is not None:
            # Select the few packed arrays before decoding, not every expanded
            # observation field afterwards. Owners retain original root IDs.
            payload = dict(payload)
            for key in ('categorical', 'scalars', 'owners', 'selection_prefix',
                        'legal_policy_ids', 'language_ids', 'language_features'):
                if key in payload:
                    payload[key] = np.asarray(payload[key])[rows]
        result = self.decoder.decode(payload, self._decoder_globals, destination)
        if self.public_context:
            from env.core.outer_return import MISC_SCALE
            if self.payload.get('outer_return_schema') != 1:
                raise RuntimeError('native outer-return leaf contract is missing')
            count = len(payload['owners'])
            hp = result['player_scalars'][:count]
            result['run_hp'][:count, :2] = hp[:, :2]
            result['run_hp'][:count, 2] = np.divide(hp[:, 0], hp[:, 1],
                out=np.zeros(len(hp), dtype=np.float32), where=hp[:, 1] != 0)
            remap = None if rows is None else {original: index for index, original in enumerate(rows)}
            for row, changes in self.payload['permanent_misc'].items():
                if remap is not None:
                    if row not in remap:
                        continue
                    row = remap[row]
                for index, value in changes.items():
                    if 0 <= index < result['deck_ids'].shape[1] and result['deck_ids'][row, index] != 0:
                        result['outer_deck_misc'][row, index] = value / MISC_SCALE
        self.decode_seconds += time.perf_counter()-started
        return result

    def combat_exits(self, *, upper_value_only=False):
        if not self.outer_return:
            return {}
        return {row: self.envs[self.indices[self.payload['owners'][row]]].settle_search_combat_exit(
                    battle, upper_value_only=upper_value_only)
                for row, battle in self.payload['combat_exits'].items()}

    @property
    def combat_exit_rows(self):
        return self.payload['combat_exits'].keys() if self.outer_return else ()

    @property
    def rows(self):
        return len(self.payload['owners'])

    @property
    def active_relic_width(self):
        return self.relic_width()

    def relic_width(self, rows=None):
        owners = self.payload['owners']
        if rows is not None:
            owners = [owners[i] for i in rows]
        return int(self._relic_widths[owners].max())

    def observation_layout(self, rows=None):
        """Shape/dtype views only; their contents are not model observations."""
        count = self.rows if rows is None else len(rows)
        if count not in self._layouts:
            self._layouts[count] = {key: np.broadcast_to(value, (count, *value.shape))
                                    for key, value in self._prototypes.items()}
        return self._layouts[count]

    def write_observations(self, destination, rows=None):
        """Fill caller-owned contiguous arrays before their next device upload."""
        self._decode_observations(destination, rows)

    def _ensure_card_stats(self, *, validate_all=False):
        layout = _nosl_compact_lut()['layout']
        hand_start = (6 + int(layout['player_status_count'])
                      + encode.COMBAT_COUNTERS + 1 + 30
                      + 5 * int(layout['monster_status_count']))
        scalars = np.asarray(self.payload['scalars'])
        if scalars.ndim != 2 or scalars.shape[1] < hand_start + 20:
            raise ValueError('native observation layout omits hand upgrade data')
        upgrades = scalars[:, hand_start:hand_start+20:2]
        # A subset decoder will not see the omitted rows. Keep the original
        # whole-packet validation in that path, including empty selections.
        if validate_all and (not np.all(np.isfinite(upgrades)) or np.any(upgrades < 0)
                or np.any(upgrades != np.rint(upgrades))):
            raise ValueError('native hand upgrade count is invalid')
        highest = float(upgrades.max(initial=0))
        # The native decoder checks every hand entry for finite, nonnegative
        # integer upgrades, including entries below this maximum. Here we only
        # need to grow the printed-card table before decoding, not duplicate
        # those checks with temporary arrays on every search wave.
        if highest < self._card_stats['damage'].shape[1]:
            return
        if not np.isfinite(highest) or highest != int(highest):
            raise ValueError('native hand upgrade count is invalid')
        self._card_stats = _nosl_card_stats(int(highest) + 1)
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

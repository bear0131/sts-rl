"""Batch encoding of settled native combat roots using the search leaf codec.

Overworld/terminal observations still use the canonical encoder. This adapter
owns no game state or persistent observation cache: every call reads the live
roots, including mutable master-deck metadata, and returns fresh arrays.
"""
import numpy as np

from . import encode
from .outer_return import CONTEXT_KEYS, permanent_misc
from .run_env import _nosl_compact_lut, _nosl_card_stats
from schemas.combat_state import normalize_sim_global
import slaythespire as sts


class NativeRootObservationBatch:
    def __init__(self, first):
        if not hasattr(sts, 'pack_neural_roots'):
            raise RuntimeError('native root observations require pack_neural_roots')
        if not first._autoregressive_actions or first._combat_language:
            raise ValueError('native root observations require AR actions without combat language')
        self.prototypes = {key: np.zeros(space.shape, dtype=space.dtype)
                           for key, space in first.observation_space.spaces.items()}
        self.language = (_nosl_compact_lut()['language_card_index']
                         if first._card_language else None)
        # These are exactly the fields written by _encode_global. Other
        # (phase-gated) groups remain PAD, as in encode.encode on a combat root.
        names = {'phase', 'arg_step', 'post_enemy_turn_boundary', 'combat_is_boss',
                 'character', 'ascension', 'act', 'floor', 'next_boss', 'gold',
                 'run_hp', 'run_keys', 'relics', 'relic_counters', 'potions',
                 'deck_ids', 'deck_upgrades', 'deck_size', 'outer_deck_misc'}
        names.update(key for key in self.prototypes if key.startswith('deck_card_'))
        self.global_keys = frozenset(names.intersection(self.prototypes))
        self.stats = _nosl_card_stats()
        self.decoder = sts.NeuralObservationDecoder(
            self.prototypes, _nosl_compact_lut(), self.stats,
            {key: getattr(encode, key+'_SCALE') for key in
             ('HP', 'BLOCK', 'ENERGY', 'TURN', 'BUFF', 'COUNTER', 'DMG', 'HITS', 'UPGRADE', 'PICK', 'GOLD')},
            dict(counters=encode.COMBAT_COUNTERS, moves=encode.V.MOVE_VOCAB_CAPACITY,
                 intents=len(encode.V.INTENT_CATEGORY), contexts=encode.V.SELECT_CONTEXT_VOCAB_CAPACITY),
            first._policy_offset, first._neural_width, True)

    def encode(self, envs):
        if not envs:
            raise ValueError('native root observation batch must not be empty')
        for env in envs:
            if env.thick.bc is None or env._pending is not None:
                raise ValueError('native root observation requires a settled combat state')
            env._multi_desc()
        payload = sts.pack_neural_roots(
            [env.thick.bc for env in envs],
            [sum(1 << i for i in (env._multi or ())) for env in envs], self.language)
        layout = _nosl_compact_lut()['layout']
        hand_start = 6 + layout['player_status_count'] + encode.COMBAT_COUNTERS + 1 + 30 + 5*layout['monster_status_count']
        upgrades = payload['scalars'][:, hand_start:hand_start+20:2]
        if (not np.all(np.isfinite(upgrades)) or np.any(upgrades < 0)
                or np.any(upgrades != np.rint(upgrades))):
            raise ValueError('native root hand upgrade count is invalid')
        needed = int(upgrades.max(initial=0)) + 1
        if needed > self.stats['damage'].shape[1]:
            self.stats = _nosl_card_stats(needed)
            self.decoder.set_card_stats(self.stats)
        globals = {key: np.zeros((len(envs), *self.prototypes[key].shape),
                                 dtype=self.prototypes[key].dtype) for key in self.global_keys}
        for row, env in enumerate(envs):
            scalar = payload['scalars'][row]
            # Use the canonical global encoder, including float64 division
            # followed by float32 storage for run_hp, not ratios of scaled HP.
            canon = {'phase': 'combat', 'arg_step': env._arg_step(),
                     'global': normalize_sim_global(env.thick.gc, combat=True, for_encoding=True),
                     'player': dict(hp=int(scalar[0]), max_hp=int(scalar[1]), gold=int(scalar[5]))}
            fields = {key: value[row] for key, value in globals.items()}
            encode._encode_global(fields, canon)
            if env.outer_return_context:
                fields['outer_deck_misc'][:] = permanent_misc(env.thick.gc, env.thick.bc)
        result = self.decoder.decode(payload, globals)
        # The public canonical schema hides Exploder's explosion damage, as
        # CommunicationMod does. Keep search leaves' existing projection and
        # actual simulator damage unchanged, including on dead retained slots.
        explosions = result['monster_move'] == encode.V.MOVE_VOCAB_INDEX['EXPLODER_EXPLODE']
        result['monster_intent_cat'][explosions] = 0
        result['monster_scalars'][explosions, 4:6] = 0
        # The full Gym codec projects the native curse sentinel -3 to the
        # public unplayable sentinel -2. Search leaves retain their existing
        # transport convention; this root-only adapter must preserve Gym bytes.
        for key in ('hand_costs', 'select_card_costs'):
            result[key][result[key] == np.float32(-3 / encode.ENERGY_SCALE)] = -2 / encode.ENERGY_SCALE
        select_start = hand_start + 20 + 3 * layout['card_count']
        for row, env in enumerate(envs):
            if env._multi is not None:
                # Match integer subtraction BEFORE normalization in _obs;
                # subtracting from an already-rounded float32 differs by 1 ULP.
                count = max(0, int(payload['scalars'][row, select_start]) - len(env._multi))
                result['pick_remaining'][row, 0] = count / encode.PICK_SCALE
            result['combat_legal'][row] = env.action_masks()[env._policy_offset:]
            if env.outer_return_context:
                env._nosl_current_context = {key: result[key][row].copy() for key in CONTEXT_KEYS}
        return result

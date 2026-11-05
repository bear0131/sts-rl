"""Native projections of combat roots and run-global/settled reward inputs.

Screen-specific overworld observations still use the canonical encoder. This
adapter owns no game state or persistent observation cache: every call reads the live
roots, including mutable master-deck metadata, and returns fresh arrays.
"""
import numpy as np
from functools import lru_cache

from . import encode
from .outer_return import CONTEXT_KEYS, permanent_misc
from .run_env import _nosl_compact_lut, _nosl_card_stats
from schemas.combat_state import normalize_sim_global
import slaythespire as sts


_GLOBAL_KEYS = frozenset({'phase', 'arg_step', 'post_enemy_turn_boundary', 'combat_is_boss',
    'character', 'ascension', 'act', 'floor', 'next_boss', 'gold', 'run_hp', 'run_keys',
    'relics', 'relic_counters', 'potions', 'deck_ids', 'deck_upgrades', 'deck_size',
    'outer_deck_misc', 'combat_mc_exit'})


def upper_value_runtime(teacher, max_bytes, *, pad):
    """Collection-owned batching/cache over the frozen checkpoint's schema.

    The native path accepts owned native rows or encoded float32/int64 arrays,
    including strided views. It preserves phase, physical token width and full input bytes as
    cache identity. Other backends retain the Python transport.
    """
    if not hasattr(sts, 'UpperValueRuntime'):
        return None
    schemas = {}
    for name, phase in encode.V.PHASE_VOCAB_INDEX.items():
        if name == 'combat':
            continue
        keys = (encode.OVERWORLD_OBSERVATION_KEYS |
                encode.OVERWORLD_PHASE_OBSERVATION_KEYS.get(phase, frozenset()))
        schemas[phase] = {key: np.empty(space.shape, dtype=space.dtype)
                          for key, space in teacher.observation_space.spaces.items()
                          if key in keys}
    return sts.UpperValueRuntime(schemas, encode.V.PHASE_VOCAB_INDEX['combat'], max_bytes, pad)


def _global_projection_schema(phase):
    """One canonical vocabulary/scale source for native run-global projections."""
    from schemas.combat_state import _sim_card_id
    layout = _nosl_compact_lut()['layout']
    cards = {str(name): encode._card_id_idx(_sim_card_id(str(name)))
             for name in layout['card_names'] if str(name) != 'INVALID'}
    vocab = dict(card=cards, relic=encode.V.RELIC_VOCAB_INDEX,
                 potion=encode.V.POTION_VOCAB_INDEX,
                 boss=encode.V.BOSS_VOCAB_INDEX,
                 character=encode.V.CHARACTER_VOCAB_INDEX,
                 language_cards=[encode._card_id_idx(name) for name in encode.LANGUAGE_CARDS],
                 phase=phase, arg_step=encode.V.ARG_STEP_VOCAB_INDEX['root'])
    scales = {key: getattr(encode, key+'_SCALE') for key in
              ('HP', 'GOLD', 'ASCENSION', 'ACT', 'FLOOR', 'COUNTER', 'UPGRADE', 'DECK_SIZE')}
    return vocab, scales


@lru_cache(maxsize=2)
def upper_reward_encoder(deck_size_observation):
    """Native projection of settled rewards; schema only, no game-state cache."""
    phase = encode.V.PHASE_VOCAB_INDEX['combat_reward']
    keys = (encode.OVERWORLD_OBSERVATION_KEYS |
            encode.OVERWORLD_PHASE_OBSERVATION_KEYS[phase])
    prototypes = {key: value for key, value in
                  encode._empty(deck_size_observation=deck_size_observation).items()
                  if key in keys}
    return sts.UpperRewardObservationEncoder(prototypes, *_global_projection_schema(phase))


@lru_cache(maxsize=4)
def overworld_global_encoder(card_language, deck_size_observation):
    """Cache immutable schema only; each encode reads the complete live inventory."""
    prototypes = {key: value for key, value in encode._empty(
        card_language=card_language, deck_size_observation=deck_size_observation).items()
        if key in _GLOBAL_KEYS or key.startswith('deck_card_')}
    return sts.OverworldGlobalObservationEncoder(prototypes,
        *_global_projection_schema(encode.V.PHASE_VOCAB_INDEX['none']))


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
        names = set(_GLOBAL_KEYS)
        names.update(key for key in self.prototypes if key.startswith('deck_card_'))
        self.global_keys = frozenset(names.intersection(self.prototypes))
        self.global_encoder = (sts.CombatGlobalObservationEncoder(
            {key: self.prototypes[key] for key in self.global_keys},
            *_global_projection_schema(encode.V.PHASE_VOCAB_INDEX['combat']))
            if hasattr(sts, 'CombatGlobalObservationEncoder') else None)
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
        if self.global_encoder is not None:
            globals = self.global_encoder.encode(
                [env.thick.gc for env in envs], payload['scalars'],
                [encode.V.ARG_STEP_VOCAB_INDEX[env._arg_step()] for env in envs])
        else:
            # Retained native builds still use the canonical numeric projection.
            globals = {key: np.zeros((len(envs), *self.prototypes[key].shape),
                                     dtype=self.prototypes[key].dtype) for key in self.global_keys}
            for row, env in enumerate(envs):
                scalar = payload['scalars'][row]
                canon = {'phase': 'combat', 'arg_step': env._arg_step(),
                         'global': normalize_sim_global(env.thick.gc, combat=True, for_encoding=True),
                         'player': dict(hp=int(scalar[0]), max_hp=int(scalar[1]), gold=int(scalar[5]))}
                encode._encode_global({key: value[row] for key, value in globals.items()}, canon)
        for row, env in enumerate(envs):
            if env.outer_return_context:
                globals['outer_deck_misc'][row] = permanent_misc(env.thick.gc, env.thick.bc)
        result = self.decoder.decode(payload, globals)
        if 'upper_public_context' in result:
            result['upper_public_context'].fill(1.)
        # The canonical encoder leaves untargetable monsters' power rows at
        # positive zero. Multiplying a negative native power by a zero mask
        # instead produces -0.0, which is numerically equal but not byte-equal.
        result['monster_buffs'][result['monster_scalars'][:, :, 0] == 0] = 0
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

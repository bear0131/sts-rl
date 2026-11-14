"""Native projections of combat roots and run-global/settled reward inputs.

Screen-specific overworld observations still use the canonical encoder. This
adapter owns no game state or persistent observation cache: every call reads the live
roots, including mutable master-deck metadata, and returns fresh arrays.
"""
import numpy as np
from functools import lru_cache

from . import encode
from .outer_return import CONTEXT_KEYS, permanent_misc
from .run_env import _nosl_compact_lut, _nosl_card_stats, _nosl_compact_empty
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
    def __init__(self, first, *, fields=None):
        if not hasattr(sts, 'pack_neural_roots'):
            raise RuntimeError('native root observations require pack_neural_roots')
        if not first._autoregressive_actions or first._combat_language:
            raise ValueError('native root observations require AR actions without combat language')
        self.prototypes = {key: np.zeros(space.shape, dtype=space.dtype)
                           for key, space in first.observation_space.spaces.items()}
        if fields is not None:
            requested = frozenset(fields)
            if not requested <= self.prototypes.keys():
                raise ValueError('unknown native observation consumer field')
            # Keep the compact decoder's established mandatory fields and
            # global projection. Never materialize unrelated upper-screen rows.
            required = set(_nosl_compact_empty(card_language=first._card_language))
            required.update(_GLOBAL_KEYS)
            required.update(key for key in self.prototypes if key.startswith('deck_card_'))
            required.update(requested)
            self.prototypes = {key: value for key, value in self.prototypes.items() if key in required}
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
        self.deck_encoder = None
        if first.deck_latent_context and hasattr(sts, 'DeckContextEncoder'):
            from schemas.combat_state import _sim_card_id
            from .outer_return import MISC_SCALE
            cards = [-1 if str(name) == 'INVALID' else encode._card_id_idx(_sim_card_id(str(name)))
                     for name in _nosl_compact_lut()['layout']['card_names']]
            self.deck_encoder = sts.DeckContextEncoder(cards, encode.N_CARDS,
                encode.DECK_SLOTS, encode.UPGRADE_SCALE, MISC_SCALE)
        self.stats = _nosl_card_stats()
        self.decoder = sts.NeuralObservationDecoder(
            self.prototypes, _nosl_compact_lut(), self.stats,
            {key: getattr(encode, key+'_SCALE') for key in
             ('HP', 'BLOCK', 'ENERGY', 'TURN', 'BUFF', 'COUNTER', 'DMG', 'HITS', 'UPGRADE', 'PICK', 'GOLD')},
            dict(counters=encode.COMBAT_COUNTERS, moves=encode.V.MOVE_VOCAB_CAPACITY,
                 intents=len(encode.V.INTENT_CATEGORY), contexts=encode.V.SELECT_CONTEXT_VOCAB_CAPACITY),
            first._policy_offset, first._neural_width, True)

    def encode_progress(self, envs):
        """Only the live fields strategic credit consumes inside a fight.

        This is an internal collector projection, not a policy observation.
        Preserve selection bookkeeping performed by the complete root codec.
        """
        if not envs:
            raise ValueError('native root observation batch must not be empty')
        result = {key: np.empty((len(envs), *self.prototypes[key].shape),
                                dtype=self.prototypes[key].dtype)
                  for key in ('phase', 'act', 'floor')}
        result['phase'].fill(encode.V.PHASE_VOCAB_INDEX['combat'])
        for row, env in enumerate(envs):
            if env.thick.bc is None or env._pending is not None:
                raise ValueError('native root observation requires a settled combat state')
            env._multi_desc()
            result['act'][row, 0] = env.thick.gc.act / encode.ACT_SCALE
            result['floor'][row, 0] = env.thick.gc.floor_num / encode.FLOOR_SCALE
        return result

    def encode(self, envs, *, destination=None):
        """Encode owned rows, or fill an internal caller's exact-sized buffers."""
        if destination is not None and any(len(value) != len(envs) for value in destination.values()):
            raise ValueError('native root destination must have exactly one row per environment')
        if not envs:
            raise ValueError('native root observation batch must not be empty')
        for env in envs:
            if env.thick.bc is None or env._pending is not None:
                raise ValueError('native root observation requires a settled combat state')
            env._multi_desc()
        return self._encode(envs, [env.thick.bc for env in envs],
            [sum(1 << i for i in (env._multi or ())) for env in envs], destination=destination)

    def encode_search(self, sources, battles, prefixes, roots):
        """Read native lanes without copying a run or touching its real root.

        AnchorWave mutates only BattleContext. GameContext and map are the
        frozen root's; permanent battle card changes still come from the live
        battle overlay below. Anchors retain their separate canonical codec.
        """
        if (not sources or not len(sources) == len(battles) == len(prefixes) == len(roots)
                or any(not e._joint_actions or not e._autoregressive_actions
                       or e.outer_return_context for e in sources)):
            raise ValueError('native lane projection requires current autoregressive deck roots')
        return self._encode(sources, battles, prefixes, search_roots=roots)

    def _encode(self, envs, battles, prefixes, *, destination=None, search_roots=None):
        payload = sts.pack_neural_roots(
            battles, prefixes, self.language)
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
                ([encode.V.ARG_STEP_VOCAB_INDEX['root']] * len(envs) if search_roots is not None
                 else [encode.V.ARG_STEP_VOCAB_INDEX[env._arg_step()] for env in envs]))
        else:
            # Retained native builds still use the canonical numeric projection.
            globals = {key: np.zeros((len(envs), *self.prototypes[key].shape),
                                     dtype=self.prototypes[key].dtype) for key in self.global_keys}
            for row, env in enumerate(envs):
                scalar = payload['scalars'][row]
                canon = {'phase': 'combat', 'arg_step': 'root' if search_roots is not None else env._arg_step(),
                         'global': normalize_sim_global(env.thick.gc, combat=True, for_encoding=True),
                         'player': dict(hp=int(scalar[0]), max_hp=int(scalar[1]), gold=int(scalar[5]))}
                encode._encode_global({key: value[row] for key, value in globals.items()}, canon)
        for row, env in enumerate(envs):
            if env.outer_return_context:
                globals['outer_deck_misc'][row] = permanent_misc(env.thick.gc, env.thick.bc)
        result = self.decoder.decode(payload, globals, destination)
        if self.deck_encoder is not None:
            extension = self.deck_encoder.encode([e.thick.gc for e in envs], battles)
            if 'deck_latent_context' in result:
                result['deck_latent_context'].fill(1.)
            for key, values in extension.items():
                if key in result:
                    result[key][...] = values
        if 'combat_exit_anchor' in result:
            # Anchor rows use the canonical codec; all native live rows are zero.
            result['combat_exit_anchor'].fill(0.)
        if 'reward_stage' in result:
            result['reward_stage'].fill(0.)
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
            if search_roots is not None or env._multi is not None:
                # Match integer subtraction BEFORE normalization in _obs;
                # subtracting from an already-rounded float32 differs by 1 ULP.
                selected = int(prefixes[row]).bit_count() if search_roots is not None else len(env._multi)
                count = max(0, int(payload['scalars'][row, select_start]) - selected)
                result['pick_remaining'][row, 0] = count / encode.PICK_SCALE
            if search_roots is None:
                result['combat_legal'][row] = env.action_masks()[env._policy_offset:]
            else:
                # pack_neural_roots carries observations, not action IDs.
                # Read the same native AR codec as RunEnv on the live lane.
                ids = [aid for aid, _ in battles[row].autoregressive_policy_actions(int(prefixes[row]))]
                result['combat_legal'][row, ids] = True
            if env.deck_latent_context:
                if self.deck_encoder is None:
                    if search_roots is not None:
                        raise RuntimeError('native lane projection requires the pinned deck encoder')
                    from .deck_context import project_native_root
                    project_native_root(env, {key: values[row] for key, values in result.items()})
                else:
                    view = ({key: value for key, value in search_roots[row].items() if key.startswith('map_')}
                            if search_roots is not None else env.thick.encoding_map_observation(dict(phase='map')))
                    for key, values in view.items():
                        if key in result:
                            result[key][row] = values
            if env.outer_return_context:
                env._nosl_current_context = {key: result[key][row].copy() for key in CONTEXT_KEYS}
        return result

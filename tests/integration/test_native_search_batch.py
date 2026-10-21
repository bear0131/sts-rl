"""Thread scheduling must preserve every tree's leaf and completed decision."""
from dataclasses import asdict

import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch
from rl.search.native import NativeSearchConfig

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('threads', [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize('fixed_worldline,end_turn_as_leaf', [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize('contract,language,selection', [
    ('legacy', False, False), ('joint-v2', True, False), ('autoregressive-v2', True, False),
    pytest.param('joint-v2', True, True, id='joint-selection'),
    pytest.param('autoregressive-v2', True, True, id='autoregressive-selection')])
def test_native_batch_preserves_each_leaf_and_result(combat_factory, threads, fixed_worldline, end_turn_as_leaf, contract, language, selection):
    import slaythespire as sts

    deck = ['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 4 + ['BASH', 'BATTLE_TRANCE', 'TRUE_GRIT']
    def environments():
        result = []
        for i, encounter in enumerate(('Jaw Worm', 'Two Louse', 'Automaton', 'Collector')):
            card = 'PURITY' if i % 2 == 0 else 'SECRET_TECHNIQUE'
            relics = (['Anchor'] if i < 2 else ['Burning Blood']) if selection else (['Burning Blood'] if i % 2 else [])
            env = combat_factory([card] * 5 if selection else deck,
                    'Two Louse' if selection else encounter, seed=42+i, relics=relics,
                    combat_action_contract=contract, card_language=language)
            env.thick.bc.apply_custom_state({
                'player': {'powers': [('STRENGTH', -2+i), ('ARTIFACT', i)],
                           'counters': {'happy_flower': i, 'pen_nib': 3}},
                'draw': [('STRIKE_RED', 1), ('DEFEND_RED', 0), ('STRIKE_RED', 2), ('BASH', 0)],
                'discard': [('BASH', 1), ('DEFEND_RED', 2), ('BASH', 0)]})
            if selection:
                env.thick.gc.gold = 37 + i * 101
                env.thick.bc.apply_custom_state({
                    'player': {'gold': 37 + i * 101},
                    'hand': [card, 'STRIKE_RED', 'DEFEND_RED', 'BASH', 'ANGER', 'DEFEND_RED'],
                    'draw': ['STRIKE_RED'] * 12 + ['SHRUG_IT_OFF', 'DEFEND_RED'],
                    'monsters': [{'hp': 0}, {'hp': 13}],
                })
                env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
            env._invalidate_backend_cache()
            result.append(env)
        return result
    native, serial = environments(), environments()
    roots = [env._obs() for env in native]
    if selection:
        for i, (env, root) in enumerate(zip(native, roots, strict=True)):
            descriptor = env.thick.bc.select_info()
            assert descriptor['zone'] == ('hand' if i % 2 == 0 else 'draw')
            assert root['select_zone'][0] == (1 if i % 2 == 0 else 2)
            handles = [candidate['native_index'] + 1 for candidate in descriptor['candidates']]
            np.testing.assert_array_equal(root['select_native_indices'][:len(handles)], handles)
            if i % 2:
                assert min(handles) > 10, 'draw candidates must retain non-hand execution handles'
            assert root['monster_id'][0] != 0 and root['monster_id'][1] != 0
            np.testing.assert_array_equal(root['monster_scalars'][:2, 0], [0, 1])
        assert len({root['gold'].tobytes() for root in roots}) == 4
        assert roots[0]['relics'].tobytes() != roots[2]['relics'].tobytes()
    pool = NativeSearchBatch(native, threads)
    config = NativeSearchConfig(simulations=128, leaf_batch_size=1,
                               fixed_worldline=fixed_worldline, end_turn_as_leaf=end_turn_as_leaf)
    calls = {i: ((np.linspace(-.3, .7, e.action_space.n),), dict(
        **asdict(config), root_value=.5, state_nonce=1729+i, compact_observation=True))
        for i, e in enumerate(native)}
    try:
        # Two roots on unchanged states exercise reuse and release of old sessions.
        for _ in range(2):
            pool.begin(calls)
            responses = {i: e.nosl_is_az_cpp_begin(*calls[i][0], **calls[i][1])
                         for i, e in enumerate(serial)}
            total = 0
            prefixes = set()
            while not pool.done:
                owners = pool.payload['owners']
                assert owners == [i for i, r in responses.items() if not r['done']]
                obs = pool.observations()
                for row, i in enumerate(owners):
                    reference = responses[i]['observation_batch']
                    assert obs.keys() == reference.keys()
                    for key in obs:
                        np.testing.assert_array_equal(obs[key][row], reference[key][0], err_msg=f'{i}: {key}')
                        assert obs[key].dtype == reference[key].dtype, key
                        assert obs[key][row].tobytes() == reference[key][0].tobytes(), key
                    assert bool(pool.payload['needs_policy'][row]) == responses[i]['leaf_meta'][0]['needs_policy']
                    if selection:
                        for key in ('gold', 'relics', 'monster_id'):
                            np.testing.assert_array_equal(obs[key][row], roots[i][key], err_msg=f'{i}: {key}')
                        assert obs['monster_scalars'][row, 0, 0] == 0, 'dead target slot must remain absent'
                        if obs['select_hand_mask'][row].any():
                            prefixes.add(i)
                # The pinned-buffer path must have the same bytes while leaving
                # padding untouched; reuse the same storage as row counts shrink.
                if total == 0:
                    target = {key: np.full((len(native)+2, *value.shape[1:]), 1, dtype=value.dtype)
                              for key, value in obs.items()}
                padding = {key: value[len(owners):].tobytes() for key, value in target.items()}
                pool.write_observations(target)
                for key, value in obs.items():
                    assert target[key][:len(owners)].tobytes() == value.tobytes(), key
                    assert target[key][len(owners):].tobytes() == padding[key], key
                active = np.flatnonzero(obs['relics'].any(axis=0))
                assert pool.active_relic_width == (int(active[-1])+1 if active.size else 1)
                values = np.asarray(.2 + .6 * obs['player_scalars'][:, 0], dtype=np.float32)
                logits = np.broadcast_to(np.linspace(-.2, .8, native[0].action_space.n, dtype=np.float32),
                                          (len(owners), native[0].action_space.n)).copy()
                for row, i in enumerate(owners):
                    leaf = responses[i]['leaf_meta'][0]
                    policy = logits[row] if leaf['needs_policy'] else np.empty(0)
                    responses[i] = serial[i].nosl_is_az_cpp_backprop_and_leaf_batch(
                        [(leaf['request_id'], float(values[row]), policy)], max_leaves=1)
                total += len(owners)
                pool.advance(logits, values)
            assert total > 0
            assert sum(pool.leaves) == total
            if selection and contract == 'autoregressive-v2':
                assert {0, 2} <= prefixes, 'both hand-selection roots must expose real prefixes'
            actual = pool.results()
            for i in calls:
                assert responses[i]['done']
                for key in ('action', 'native_action_bits', 'root_value', 'simulations',
                            'visit_policy', 'root_visit_policy', 'root_visits'):
                    np.testing.assert_array_equal(actual[i][key], responses[i]['result'][key])
                if selection:
                    np.testing.assert_equal(native[i]._obs(), roots[i])
                    np.testing.assert_equal(serial[i]._obs(), roots[i])
    finally:
        pool.close()


def test_native_decoder_keeps_autoregressive_prefix(combat_factory):
    import slaythespire as sts
    from env.core import native_actions as C
    env = combat_factory(['PURITY'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    env.thick.bc.apply_custom_state({'hand': ['PURITY'] + ['STRIKE_RED'] * 5})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    pool = NativeSearchBatch([env], 2)
    logits = np.zeros(C.AUTOREGRESSIVE_SIZE, np.float32)
    logits[C.AUTOREGRESSIVE_MULTI] = 30
    calls = {0: ((logits,), dict(simulations=16, state_nonce=42, depth=1,
        compact_observation=True, leaf_batch_size=1, root_search_mode=2,
        paper_full_gumbel=True, gumbel_scale=0, gumbel_rescale_q=False, gumbel_value_scale=1))}
    seen = False
    try:
        pool.begin(calls)
        while not pool.done:
            obs = pool.observations()
            expected = env._native_packed_observation_batch(pool.payload)
            seen |= bool(obs['select_hand_mask'].any())
            for key in obs:
                assert obs[key].dtype == expected[key].dtype
                assert obs[key].tobytes() == expected[key].tobytes(), key
            pool.advance(logits[None], np.array([.3], np.float32))
        assert seen
        assert pool.results()[0]['action'] == C.AUTOREGRESSIVE_MULTI
    finally:
        pool.close()


def test_native_batch_invalid_reply_can_abort_and_reuse(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 5)
    pool = NativeSearchBatch([env], 2)
    calls = {0: ((np.zeros(env.action_space.n),), dict(simulations=8,
               compact_observation=True, leaf_batch_size=1, state_nonce=1729))}
    try:
        pool.begin(calls)
        with pytest.raises(ValueError, match='rows must match'):
            pool.advance(np.zeros((0, env.action_space.n), np.float32), np.zeros(0, np.float32))
        pool.clear()
        pool.begin(calls)
        while not pool.done:
            n = len(pool.payload['owners'])
            pool.advance(np.zeros((n, env.action_space.n), np.float32), np.full(n, .5, np.float32))
        assert pool.results()[0]['simulations'] == 8
    finally:
        pool.close()


@pytest.mark.parametrize('invalid', ['dtype', 'shape', 'rows', 'stride', 'readonly'])
def test_native_decoder_rejects_invalid_destination(combat_factory, invalid):
    env = combat_factory(['STRIKE_RED'] * 5)
    pool = NativeSearchBatch([env], 1)
    try:
        pool.begin({0: ((np.zeros(env.action_space.n),), dict(simulations=8,
            compact_observation=True, leaf_batch_size=1, state_nonce=1729))})
        target = pool.observations()
        value = target['player_scalars']
        if invalid == 'dtype': value = value.astype(np.float64)
        if invalid == 'shape': value = value[:, :-1].copy()
        if invalid == 'rows': value = value[:0]
        if invalid == 'stride': value = value[:, ::-1]
        if invalid == 'readonly': value.setflags(write=False)
        target['player_scalars'] = value
        with pytest.raises(ValueError, match='destination'):
            pool.write_observations(target)
    finally:
        pool.close()


@pytest.mark.parametrize('language', [False, True])
def test_native_large_batch_conversion_matches_serial_and_recovers(combat_factory, language):
    """Large ordered batches preserve bytes, padding and decoder recovery."""
    def environments():
        return [combat_factory(['STRIKE_RED'] * 5 + ['DEFEND_RED'] * 4 + ['BASH'],
                    ('Jaw Worm', 'Two Louse')[i % 2], seed=300+i,
                    relics=['Anchor'] if i % 2 else [],
                    combat_action_contract='autoregressive-v2', card_language=language)
                for i in range(65)]
    native, serial = environments(), environments()
    pool = NativeSearchBatch(native, 16)
    calls = {i: ((np.linspace(-.3, .7, env.action_space.n),), dict(
        simulations=8+i % 3, leaf_batch_size=1, state_nonce=900+i,
        compact_observation=True, root_value=.5)) for i, env in enumerate(native)}
    try:
        pool.begin(calls)
        responses = {i: env.nosl_is_az_cpp_begin(*calls[i][0], **calls[i][1])
                     for i, env in enumerate(serial)}
        assert pool.rows == 65
        target = None
        while not pool.done:
            owners = pool.payload['owners']
            assert owners == [i for i, response in responses.items() if not response['done']]
            observed = pool.observations()
            if target is None:
                # An invalid row must not poison subsequent decoder calls.
                for field, message in [('owners', 'owner out of range'),
                                       ('categorical', 'selection card enum is outside')]:
                    invalid = dict(pool.payload)
                    value = np.asarray(invalid[field]).copy()
                    if field == 'owners': value[-1] = 65
                    else: value[-1, 31] = -1
                    invalid[field] = value
                    with pytest.raises(ValueError, match=message):
                        pool.decoder.decode(invalid, pool.global_fields)
                target = {key: np.full((67, *value.shape[1:]), 1, dtype=value.dtype)
                          for key, value in observed.items()}
            padding = {key: value[len(owners):].tobytes() for key, value in target.items()}
            pool.write_observations(target)
            for key, value in observed.items():
                assert target[key][:len(owners)].tobytes() == value.tobytes(), key
                assert target[key][len(owners):].tobytes() == padding[key], key
                for row, owner in enumerate(owners):
                    reference = responses[owner]['observation_batch'][key][0]
                    assert value.dtype == reference.dtype
                    assert value[row].tobytes() == reference.tobytes(), (owner, key)
            values = np.asarray(.2 + .6 * observed['player_scalars'][:, 0], np.float32)
            logits = np.broadcast_to(np.linspace(-.2, .8, native[0].action_space.n, dtype=np.float32),
                                     (len(owners), native[0].action_space.n)).copy()
            for row, owner in enumerate(owners):
                leaf = responses[owner]['leaf_meta'][0]
                responses[owner] = serial[owner].nosl_is_az_cpp_backprop_and_leaf_batch(
                    [(leaf['request_id'], float(values[row]), logits[row] if leaf['needs_policy'] else np.empty(0))],
                    max_leaves=1)
            pool.advance(logits, values)
        results = pool.results()
        for owner in calls:
            for key in ('action', 'native_action_bits', 'root_value', 'simulations',
                        'visit_policy', 'root_visit_policy', 'root_visits'):
                np.testing.assert_array_equal(results[owner][key], responses[owner]['result'][key])
    finally:
        pool.close()
@pytest.mark.native
@pytest.mark.parametrize(('column', 'bad'), [(0, -1), (10, 9999), (15, 9999),
                                           (20, 9), (25, -1), (30, 9999),
                                           (31, -1), (127, 65537), (223, 7)])
def test_compact_decoders_reject_unmapped_identity_instead_of_clamping(
    combat_factory, column, bad,
):
    from env.search.native_batch import NativeSearchBatch
    env = combat_factory(['STRIKE_RED'] * 5)
    pool = NativeSearchBatch([env], 1)
    try:
        packed = {key: value[None, :].copy()
                  for key, value in env.thick.bc.compact_leaf_arrays().items()}
        packed.update(owners=[0], selection_prefix=[0],
                      legal_policy_ids=np.empty((1, 0), np.int32))
        packed['categorical'][0, column] = bad
        fields = {key: value[None, :] for key, value
                  in env._native_packed_global_fields().items()}
        with pytest.raises(RuntimeError):
            env._native_packed_observation_batch(packed)
        with pytest.raises(ValueError):
            pool.decoder.decode(packed, fields)
    finally:
        pool.close()


@pytest.mark.native
def test_packed_searing_blow_above_old_upgrade_cache_remains_exact(combat_factory):
    from env.search.native_batch import NativeSearchBatch
    from env.core import encode
    env = combat_factory(['SEARING_BLOW'] * 5)
    env.thick.bc.apply_custom_state({'hand': [('SEARING_BLOW', 130, 2)]})
    env._invalidate_backend_cache()
    pool = NativeSearchBatch([env], 1)
    try:
        pool.payload = {key: value[None, :]
                        for key, value in env.thick.bc.compact_leaf_arrays().items()}
        pool.payload.update(owners=[0], selection_prefix=[0],
                            legal_policy_ids=np.empty((1, 0), np.int32))
        pool.global_fields = {key: value[None, :] for key, value
                              in env._native_packed_global_fields().items()}
        # This fixture injects packed leaves without begin(); initialize its
        # legacy decoder projection as the real transport does at begin().
        pool._decoder_globals = {key: value for key, value in pool.global_fields.items()
                                 if key in pool._decoder_keys}
        actual = pool.observations()
        expected = env._native_leaf_observation(env.thick.bc.to_dict(), combat_only=True)
        assert actual['hand_damage'][0, 0] * encode.DMG_SCALE == 8917
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key][0], value, err_msg=key)
    finally:
        pool.close()

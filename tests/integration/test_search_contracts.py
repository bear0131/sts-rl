import numpy as np
import pytest

from env.core.run_env import unpack_nosl_observation_buffer
from schemas.combat_state import normalize_sim_global
from env.search.nosl_is_az import NOSLISAZConfig, NOSLISAZSession, NOSLISAZSearch
from env.search.nosl_sampler import NOSLSampler
from tests.unit.test_nosl_is_az import _Branch, _DeepBranch


@pytest.mark.native
@pytest.mark.parametrize('contract', ['autoregressive-v2', 'joint-v2'])
@pytest.mark.parametrize('leaf_batch_size', [1, 4])
def test_grouped_packed_leaf_encoding_preserves_each_root(
    combat_factory, contract, leaf_batch_size):
    import pickle
    from rl.training.cached_mask_vec_env import (
        _materialize_grouped_packed_leaves, _pack_leaf_responses,
        _unpack_leaf_responses,
    )

    envs = [combat_factory(['STRIKE_RED'] * 5, 'Jaw Worm', seed=70 + i,
                           combat_action_contract=contract, card_language=True)
            for i in range(2)]
    responses, expected = {}, {}
    for i, env in enumerate(envs):
        env.thick.gc.gold = 37 + 11 * i
        response = env.nosl_is_az_cpp_begin(
            np.zeros(env.action_space.n), state_nonce=700 + i,
            simulations=16, depth=0, leaf_batch_size=leaf_batch_size,
            compact_observation=True,
            _defer_packed_observation=True)
        assert not response['done']
        raw = response['_native_packed_response']
        expected[i] = (
            env._native_packed_observation_batch(raw),
            [env._leaf_policy_mask(np.asarray(row)[np.asarray(row) >= 0])
             for row in raw['legal_policy_ids']],
            list(raw['request_ids']),
        )
        responses[i] = response
    _materialize_grouped_packed_leaves(envs, responses)
    wire = _pack_leaf_responses(responses)
    restored = _unpack_leaf_responses(pickle.loads(pickle.dumps(wire)))
    for i, response in restored.items():
        observation, masks, request_ids = expected[i]
        assert observation.keys() == response['observation_batch'].keys()
        for key, value in observation.items():
            np.testing.assert_array_equal(value, response['observation_batch'][key],
                                          err_msg=f'{i}:{key}')
        for meta, mask, request_id in zip(response['leaf_meta'], masks, request_ids):
            np.testing.assert_array_equal(meta['action_mask'], mask)
            assert meta['request_id'] == request_id


@pytest.mark.native
def test_grouped_leaf_encoding_keeps_full_search_tree(combat_factory):
    from rl.training.cached_mask_vec_env import _materialize_grouped_packed_leaves

    old = [combat_factory(['STRIKE_RED'] * 5, 'Jaw Worm', seed=90 + i,
                          combat_action_contract='autoregressive-v2',
                          card_language=True) for i in range(2)]
    new = [combat_factory(['STRIKE_RED'] * 5, 'Jaw Worm', seed=90 + i,
                          combat_action_contract='autoregressive-v2',
                          card_language=True) for i in range(2)]
    args = dict(simulations=128, depth=1, compact_observation=True,
                root_search_mode=2, paper_full_gumbel=True,
                gumbel_value_scale=1.0, gumbel_rescale_q=False,
                fixed_worldline=True)
    reference = {i: env.nosl_is_az_cpp_begin(
        np.linspace(-0.5, 0.5, env.action_space.n),
        state_nonce=170 + i, **args) for i, env in enumerate(old)}
    candidate = {i: env.nosl_is_az_cpp_begin(
        np.linspace(-0.5, 0.5, env.action_space.n),
        state_nonce=170 + i, _defer_packed_observation=True, **args)
        for i, env in enumerate(new)}
    for _ in range(256):
        _materialize_grouped_packed_leaves(new, candidate)
        for i in range(2):
            left, right = reference[i], candidate[i]
            assert left['done'] == right['done']
            if left['done']:
                for key in ('action', 'native_action_bits', 'simulations',
                            'root_value', 'used_completed_policy'):
                    assert left['result'][key] == right['result'][key], key
                for key in ('visit_policy', 'root_visit_policy',
                            'improved_policy', 'root_visits'):
                    np.testing.assert_array_equal(left['result'][key], right['result'][key])
                assert (old[i]._nosl_cpp_last_session.tree_snapshot((), 64)
                        == new[i]._nosl_cpp_last_session.tree_snapshot((), 64))
                continue
            assert len(left['leaf_meta']) == len(right['leaf_meta'])
            for key, value in left['observation_batch'].items():
                np.testing.assert_array_equal(value, right['observation_batch'][key],
                                              err_msg=f'{i}:{key}')
            updates = []
            for a, b in zip(left['leaf_meta'], right['leaf_meta']):
                assert a['request_id'] == b['request_id']
                assert a['needs_policy'] == b['needs_policy']
                np.testing.assert_array_equal(a['action_mask'], b['action_mask'])
                updates.append((a['request_id'], .4, np.zeros(old[i].action_space.n)))
            reference[i] = old[i].nosl_is_az_cpp_backprop_and_leaf_batch(
                updates, max_leaves=1)
            candidate[i] = new[i].nosl_is_az_cpp_backprop_and_leaf_batch(
                updates, max_leaves=1, _defer_packed_observation=True)
        if all(response['done'] for response in reference.values()):
            return
    pytest.fail('fixed-budget paired searches did not finish')


@pytest.mark.native
@pytest.mark.parametrize('contract', ['legacy', 'joint-v2'])
def test_language_leaf_group_transport_matches_historical_bytes(combat_factory, contract):
    import pickle
    from rl.search.native import decode_leaves
    from rl.training.cached_mask_vec_env import _pack_leaf_responses, _unpack_leaf_responses

    # Exercise canonical encoding with distinct hands, a multi-target encounter,
    # piles, and a selectable card. Comparing all fields protects candidate IDs,
    # slot order, language programs, masks, and value-only leaf metadata together.
    envs = [combat_factory(['ARMAMENTS', 'BASH', 'DEFEND_RED', 'STRIKE_RED', 'ANGER'] * 2,
                          'Two Louse', seed=seed, combat_action_contract=contract,
                          card_language=True) for seed in (70, 71)]
    responses = {i: env.nosl_is_az_cpp_begin(
        np.zeros(env.action_space.n), state_nonce=700 + i, simulations=16, depth=0,
        compact_observation=True) for i, env in enumerate(envs)}
    seen = 0
    for _ in range(32):
        active = {i: r for i, r in responses.items() if not r['done']}
        if not active:
            break
        legacy = {}
        for i, response in active.items():
            batch = response['observation_batch']
            assert batch and response['observation_buffer'] is None
            chunks, spec, offset = [], [], 0
            for key, array in batch.items():
                payload = array.tobytes()
                spec.append((key, array.dtype.str, array.shape, offset, len(payload)))
                chunks.append(payload)
                offset += len(payload)
            legacy[i] = dict(response, observation_batch=None,
                             observation_buffer=b''.join(chunks), observation_spec=spec)
        packet = _pack_leaf_responses(active)
        if len(active) > 1:
            assert not isinstance(packet, dict), 'language leaves must use grouped arrays'
        actual = _unpack_leaf_responses(pickle.loads(pickle.dumps(packet)))
        expected_records, expected = decode_leaves(legacy)
        records, observed = decode_leaves(actual)
        assert observed.keys() == expected.keys()
        for (i, leaf), (j, reference) in zip(records, expected_records, strict=True):
            assert (i, leaf['request_id'], leaf['needs_policy']) == (
                j, reference['request_id'], reference['needs_policy'])
            np.testing.assert_array_equal(leaf['action_mask'], reference['action_mask'])
        for key, array in observed.items():
            assert array.shape == expected[key].shape and array.dtype == expected[key].dtype
            assert array.flags.c_contiguous
            assert array.tobytes() == expected[key].tobytes(), key
        seen += len(records)
        for i, response in active.items():
            responses[i] = envs[i].nosl_is_az_cpp_backprop_and_leaf_batch([
                (leaf['request_id'], .4, np.zeros(envs[i].action_space.n))
                for leaf in response['leaf_meta']], max_leaves=1)
    assert seen > 8
    assert all(r['done'] and r['result']['simulations'] == 16 for r in responses.values())


def test_native_combat_fork_victory_does_not_settle_parent(combat_factory):
    import json
    env = combat_factory(['STRIKE_RED']*5, encounter='Cultist', relics=('Burning Blood',))
    env.thick.gc.cur_hp = 30
    env.thick.bc.apply_custom_state({'player': {'hp': 30}})
    env.thick.bc.set_player_strength(999)
    env._invalidate_backend_cache()
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    stats = dict(env.thick.run_stats())
    branch = env.fork_combat()
    # Native CARD source0 target0 is bit pattern 0, not the thin End Turn id.
    _, _, terminated, info = branch._backend_step_native_combat(0)
    assert terminated and branch.thick.bc is not None
    assert branch.thick.bc.to_dict()['player']['hp'] == 30
    assert info['nosl_value_terminal_observation']['player_scalars'][0] == pytest.approx(30/200)
    assert env.thick.run_stats() == stats
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before


def test_boss_victory_metadata_carries_transition_target(
    combat_factory, monkeypatch
):
    env = combat_factory(
        ["STRIKE_RED"] * 5,
        encounter="The Guardian",
        relics=("Burning Blood",),
    )
    env.thick._nosl_is_az_boss_transition_target = True
    env.thick.gc.cur_room = type(env.thick.gc.cur_room).BOSS
    env.thick.gc.boss = env.thick.gc.encounter
    env.thick.bc.set_player_strength(999)
    # The combat_factory starts a standalone battle without the boss-room
    # regain-control callback. The target is captured before settlement, so
    # suppress that unrelated callback only for this metadata-path fixture.
    monkeypatch.setattr(env.thick, "_settle", lambda: None)

    _obs, _reward, _terminated, info = env._backend_step_native_combat(0)

    assert info["nosl_value_terminal_target"] == pytest.approx(
        env.thick.bc.to_dict()["player"]["max_hp"] / 200.0
    )


@pytest.mark.parametrize("branch_type", [_Branch, _DeepBranch])
def test_python_unbounded_depth_advances_beyond_root(branch_type):
    env = branch_type()
    session = NOSLISAZSession(env, env.action_masks(), prior_logits=np.zeros(3),
                             sampler=NOSLSampler(1), state_nonce=2, decision_index=0,
                             config=NOSLISAZConfig(simulations=8, depth=0))
    response = session.next_request()
    steps = []
    for _ in range(16):
        if response["done"]:
            break
        steps.append(session.pending[0].thick.bc.steps)
        response = session.backprop(0.5, np.zeros(3))
    assert response["done"]
    assert steps and max(steps) > 0


def test_synchronous_unbounded_search_expands_one_new_edge_per_simulation():
    env = _DeepBranch()
    steps = []
    def value(branch):
        steps.append(branch.thick.bc.steps)
        return 0.5
    result = NOSLISAZSearch(NOSLISAZConfig(simulations=8, depth=0)).choose(
        env, env.action_masks(), prior_logits=np.zeros(3), sampler=NOSLSampler(1),
        state_nonce=2, decision_index=0, value_fn=value,
    )
    assert steps == list(range(1, 9))
    assert result.simulations == 8
    assert env.thick.bc.steps == 0


@pytest.mark.native
def test_native_terminal_excludes_post_combat_healing(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Burning Blood",))
    env.thick.gc.cur_hp = 50
    import slaythespire as sts
    battle = sts.create_battle(env.thick.gc, "Jaw Worm")
    battle.set_player_strength(999)
    logits = np.full(181, -100.0)
    logits[1:6] = 100.0
    session = battle.nosl_neural_search_begin(logits, simulations=1, depth=0, nonce=5)
    assert session.next_batch(1) == []
    assert session.done()
    assert session.result()["root_value"] == pytest.approx(50 / 200)
    from env.search.nosl_is_az import _terminal_value
    from types import SimpleNamespace
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    branch = SimpleNamespace(thick=SimpleNamespace(bc=battle, gc=env.thick.gc))
    assert battle.projected_exit_hp(env.thick.gc) == 56
    assert _terminal_value(branch) == pytest.approx(50 / 200)


@pytest.mark.native
def test_model_victory_leaf_uses_value_only_and_preserves_default(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, "Jaw Worm", seed=42)
    env.thick.bc.set_player_strength(999)
    logits = np.full(181, -100.0)
    logits[1:6] = 100.0
    kwargs = dict(simulations=1, depth=0, nonce=5)

    default = env.thick.bc.nosl_neural_search_begin(logits, **kwargs)
    assert default.next_batch(1) == []
    assert default.result()["root_value"] == pytest.approx(
        env.thick.bc.to_dict()["player"]["hp"] / 200
    )

    experimental = env.thick.bc.nosl_neural_search_begin(
        logits, model_victory_terminal=True, **kwargs
    )
    leaf = experimental.next_batch(1)
    assert len(leaf) == 1
    assert leaf[0]["needs_policy"] is False
    assert leaf[0]["state"]["outcome"] == 1
    experimental.backprop(leaf[0]["request_id"], 0.73, np.empty(0))
    assert experimental.done()
    assert experimental.result()["root_value"] == pytest.approx(0.73)

    packed = env.thick.bc.nosl_neural_search_begin(
        logits, model_victory_terminal=True, **kwargs
    )
    batch = packed.next_batch_packed(1)
    assert batch["done"] is False
    assert list(batch["needs_policy"]) == [False]


@pytest.mark.native
@pytest.mark.parametrize("encounter", ["Jaw Worm", "Two Louse", "Gremlin Leader", "Automaton", "Collector"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("card_language", [False, True])
def test_packed_leaf_matches_canonical_bitwise(combat_factory, encounter, batch_size, card_language):
    from env.core.run_env import _nosl_compact_lut

    env = combat_factory(["STRIKE_RED"] * 5, encounter, relics=("Akabeko",),
                         card_language=card_language)
    env.thick.bc.apply_custom_state({
        "player": {"powers": [("STRENGTH", 3), ("ARTIFACT", 2)],
                   "counters": {"happy_flower": 2, "pen_nib": 3}},
        "monsters": [{"powers": [("STRENGTH", 4), ("VULNERABLE", 2)]}],
        "draw": [("STRIKE_RED", 1), ("DEFEND_RED", 0),
                 ("STRIKE_RED", 2), ("BASH", 0)],
        "discard": [("BASH", 1), ("DEFEND_RED", 2),
                    ("BASH", 0)],
    })
    env._nosl_cpp_global_template = normalize_sim_global(env.thick.gc)
    logits = np.zeros(181)
    logits[1] = 50
    config = dict(simulations=1, depth=1, nonce=123)
    raw, payloads = [], []
    for row in range(batch_size):
        env.thick.bc.apply_custom_state({"player": {"hp": 70 - 3 * row,
                                                    "block": 2 * row}})
        raw_session = env.thick.bc.nosl_neural_search_begin(logits, **config)
        packed_session = env.thick.bc.nosl_neural_search_begin(logits, **config)
        raw.append(raw_session.next_batch(1)[0])
        payloads.append(packed_session.next_batch_packed(
            1, _nosl_compact_lut()["language_card_index"])
            if card_language else packed_session.next_batch_packed(1))
    packed = env._native_packed_observation_batch({
        key: np.concatenate([payload[key] for payload in payloads])
        for key in (("categorical", "scalars", "language_ids", "language_features")
                    if card_language else ("categorical", "scalars"))
    })
    for row, leaf in enumerate(raw):
        expected = env._native_leaf_observation(leaf["state"], combat_only=True)
        assert expected.keys() == packed.keys()
        for key in expected:
            np.testing.assert_array_equal(expected[key], packed[key][row], err_msg=key)


@pytest.mark.native
def test_packed_leaf_global_cache_is_root_scoped(combat_factory, monkeypatch):
    from env.core import encode

    env = combat_factory(["STRIKE_RED"] * 5)
    original_encode = encode._encode_combat_global
    encoded_roots = []

    def record_encode(observation, canonical):
        encoded_roots.append(canonical["global"]["gold"])
        return original_encode(observation, canonical)

    monkeypatch.setattr(encode, "_encode_combat_global", record_encode)

    def begin(gold):
        env.thick.gc.gold = gold
        response = env.nosl_is_az_cpp_begin(
            np.zeros(181), state_nonce=123, simulations=3, depth=1,
            leaf_batch_size=1, compact_observation=True)
        assert not response["done"]
        np.testing.assert_array_equal(
            response["observation_batch"]["gold"],
            np.asarray([[gold / encode.GOLD_SCALE]], dtype=np.float32))
        return response

    def advance(response):
        return env.nosl_is_az_cpp_backprop_and_leaf_batch([
            (leaf["request_id"], 0.4, np.zeros(181))
            for leaf in response["leaf_meta"]
        ], max_leaves=1)

    response = begin(123)
    # Returned observations belong to the caller; their mutation must not
    # corrupt the persistent root snapshot used by later leaves.
    response["observation_batch"]["gold"].fill(-1)
    response = advance(response)
    assert not response["done"]
    np.testing.assert_array_equal(
        response["observation_batch"]["gold"],
        np.asarray([[123 / encode.GOLD_SCALE]], dtype=np.float32))
    assert encoded_roots == [123]

    branch = env.fork_combat()
    try:
        assert branch._nosl_cpp_encoded_global_template is None
        assert env._nosl_cpp_encoded_global_template is not None
    finally:
        branch.close()

    # Replacing an unfinished session and explicitly aborting a later one must
    # both make the next root read the changed live metadata.
    response = begin(731)
    assert encoded_roots == [123, 731]
    env.nosl_is_az_cpp_abort()
    assert env._nosl_cpp_encoded_global_template is None
    response = begin(415)
    for _ in range(3):
        response = advance(response)
        if response["done"]:
            break
    assert response["done"]
    assert env._nosl_cpp_encoded_global_template is None
    begin(617)
    env.reset(seed=7)
    assert env._nosl_cpp_encoded_global_template is None


@pytest.mark.native
@pytest.mark.parametrize("mode,strict", [(0, False), (2, False), (2, True)])
def test_native_search_budget_and_live_state_contract(combat_factory, mode, strict):
    env = combat_factory(["STRIKE_RED"] * 5, "Two Louse")
    before = env.thick.bc.to_dict()
    response = env.nosl_is_az_cpp_begin(np.zeros(181), state_nonce=123, simulations=8,
                                      depth=0, root_search_mode=mode,
                                      paper_full_gumbel=strict, leaf_batch_size=1,
                                      compact_observation=True)
    for _ in range(16):
        if response["done"]:
            break
        batch = unpack_nosl_observation_buffer(response)
        assert batch is not None and response["leaves"]
        updates = [(leaf["request_id"], 0.4, np.zeros(181))
                   for leaf in response["leaves"]]
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(updates, max_leaves=1)
    assert response["done"]
    assert response["result"]["simulations"] == 8
    assert response["result"]["visit_policy"].sum() == pytest.approx(1.0)
    assert env.thick.bc.to_dict() == before
    action = response["result"]["action"]
    assert env.action_masks()[action]
    assert not env.step(action)[4].get("illegal", False)


@pytest.mark.native
def test_native_search_supports_piles_larger_than_fixed_list_capacity(combat_factory):
    # Production builds use vector-backed piles. Long runs and generated cards
    # can therefore exceed CardManager::MAX_GROUP_SIZE (64), and state-key
    # canonicalization must use the actual pile size rather than that optional
    # fixed-list build limit.
    env = combat_factory(["STRIKE_RED"] * 70, "Jaw Worm", seed=7)
    session = env.thick.bc.nosl_neural_search_begin(
        np.zeros(181),
        simulations=8,
        depth=0,
        nonce=7,
        root_search_mode=2,
        paper_full_gumbel=True,
    )

    while not session.done():
        leaves = session.next_batch(1)
        assert leaves
        leaf = leaves[0]
        session.backprop(leaf["request_id"], 0.4, np.zeros(181))

    assert session.result()["simulations"] == 8


@pytest.mark.native
def test_packed_leaf_preserves_more_than_128_legal_actions(combat_factory):
    env = combat_factory(["PURITY"] * 5, "Jaw Worm", seed=19)
    env.thick.bc.apply_custom_state({
        "player": {"energy": 3},
        "hand": ["PURITY"] + ["STRIKE_RED"] * 9,
        "draw": [],
        "discard": [],
        "exhaust": [],
    })
    root_logits = np.full(181, -100.0)
    root_logits[1] = 100.0
    kwargs = dict(simulations=1, depth=2, nonce=19)
    raw_session = env.thick.bc.nosl_neural_search_begin(root_logits, **kwargs)
    packed_session = env.thick.bc.nosl_neural_search_begin(root_logits, **kwargs)

    raw = raw_session.next_batch(1)[0]
    packed = packed_session.next_batch_packed(1)
    expected = np.asarray(raw["legal_action_bits"], dtype=np.uint32)
    counts = np.asarray(packed["legal_action_counts"], dtype=np.uint32)
    matrix = np.asarray(packed["legal_action_bits"], dtype=np.uint32)

    assert expected.size == 130  # choose up to three of nine, including empty
    assert counts.tolist() == [expected.size]
    assert matrix.shape == (1, expected.size)
    np.testing.assert_array_equal(matrix[0], expected)


@pytest.mark.native
def test_native_batch_backprop_matches_scalar_order(combat_factory):
    env_batch = combat_factory(["STRIKE_RED"] * 5, "Jaw Worm", seed=91)
    env_scalar = combat_factory(["STRIKE_RED"] * 5, "Jaw Worm", seed=91)
    root_logits = np.linspace(-1.0, 1.0, 181, dtype=np.float64)
    kwargs = dict(
        simulations=24,
        depth=2,
        nonce=173,
        root_search_mode=2,
        paper_full_gumbel=False,
    )
    batch_session = env_batch.thick.bc.nosl_neural_search_begin(
        root_logits, **kwargs
    )
    scalar_session = env_scalar.thick.bc.nosl_neural_search_begin(
        root_logits, **kwargs
    )

    while not batch_session.done():
        batch = batch_session.next_batch(8)
        scalar_batch = scalar_session.next_batch(8)
        assert len(batch) == len(scalar_batch)
        assert batch
        ids = np.asarray([leaf["request_id"] for leaf in batch], dtype=np.uint64)
        values = np.asarray(
            [0.2 + 0.03 * (index % 5) for index in range(len(batch))],
            dtype=np.float64,
        )
        logits = np.stack(
            [root_logits + 0.01 * index for index in range(len(batch))], axis=0
        )
        batch_session.backprop_batch_array(ids, values, logits)
        for leaf, value, policy in zip(scalar_batch, values, logits):
            scalar_session.backprop(leaf["request_id"], float(value), policy)

    assert scalar_session.done()
    batch_result = batch_session.result()
    scalar_result = scalar_session.result()
    assert batch_result["action_bits"] == scalar_result["action_bits"]
    assert batch_result["total_root_visits"] == scalar_result["total_root_visits"]
    assert batch_result["root_visits"] == scalar_result["root_visits"]
    assert batch_result["root_value"] == pytest.approx(
        scalar_result["root_value"], rel=0, abs=1e-12
    )

    batch_tree = batch_session.tree_snapshot((), 64)
    scalar_tree = scalar_session.tree_snapshot((), 64)
    assert batch_tree["visits"] == scalar_tree["visits"]
    assert batch_tree["value_visits"] == scalar_tree["value_visits"]
    assert batch_tree["children"] == scalar_tree["children"]


@pytest.mark.native
def test_strict_prior_temperature_reaches_native_tree(combat_factory):
    def root_priors(temperature):
        env = combat_factory(["STRIKE_RED"] * 5, "Jaw Worm", seed=42)
        response = env.nosl_is_az_cpp_begin(
            np.linspace(-2.0, 2.0, 181),
            state_nonce=123,
            simulations=1,
            depth=0,
            prior_temperature=temperature,
            root_search_mode=2,
            paper_full_gumbel=True,
            leaf_batch_size=1,
            compact_observation=False,
        )
        while not response["done"]:
            response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
                [(leaf["request_id"], 0.4, np.zeros(181))
                 for leaf in response["leaves"]],
                max_leaves=1,
            )
        return [child["prior"] for child in
                env.nosl_is_az_cpp_tree_snapshot((), 64)["children"]]

    low = root_priors(1.0)
    high = root_priors(5.0)
    assert len(low) == len(high) > 0
    assert not np.allclose(low, high, rtol=0.0, atol=1e-10)


@pytest.mark.native
@pytest.mark.parametrize(("rescale_q", "value_scale"), [(True, 0.1), (False, 1.0)])
def test_strict_gumbel_matches_reference_schedule_transform_and_target(
    combat_factory, rescale_q, value_scale,
):
    env = combat_factory(["STRIKE_RED"] * 5, "Gremlin Leader")
    response = env.nosl_is_az_cpp_begin(
        np.zeros(181),
        root_value=0.35,
        state_nonce=123,
        simulations=32,
        depth=1,
        root_search_mode=2,
        paper_full_gumbel=True,
        leaf_batch_size=1,
        compact_observation=True,
        gumbel_rescale_q=rescale_q,
        gumbel_value_scale=value_scale,
    )
    leaf_index = 0
    while not response["done"]:
        updates = [
            (leaf["request_id"], 0.2 + (leaf_index % 5) * 0.1, np.zeros(181))
            for leaf in response["leaves"]
        ]
        leaf_index += len(updates)
        response = env.nosl_is_az_cpp_backprop_and_leaf_batch(
            updates, max_leaves=1
        )

    children = env.nosl_is_az_cpp_tree_snapshot((), 64)["children"]
    assert len(children) == 16
    assert sum(child["candidate"] for child in children) == 16
    assert sorted((child["visits"] for child in children), reverse=True) == (
        [4] * 4 + [2] * 4 + [1] * 8
    )

    raw_q = np.asarray([child["raw_q"] for child in children])
    expected_normalized = (raw_q - raw_q.min()) / max(
        raw_q.max() - raw_q.min(), 1e-8
    )
    expected_completed = (expected_normalized if rescale_q else raw_q) * (50 + 4) * value_scale
    np.testing.assert_allclose(
        [child["normalized_q"] for child in children],
        expected_normalized,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [child["completed_q"] for child in children],
        expected_completed,
        rtol=0,
        atol=1e-12,
    )

    complete_weights = np.asarray([
        child["prior"] * np.exp(child["completed_q"])
        for child in children
    ])
    complete_weights /= complete_weights.sum()
    expected_target = np.zeros(181)
    for child, weight in zip(children, complete_weights):
        action_type = (child["action_bits"] >> 29) & 0xF
        thin_action = 0 if action_type == 4 else 1 + child["source_idx"]
        expected_target[thin_action] += weight
    np.testing.assert_allclose(
        response["result"]["visit_policy"], expected_target, rtol=0, atol=1e-7
    )

    max_visits = max(child["visits"] for child in children)
    expected_action = max(
        (child for child in children if child["visits"] == max_visits),
        key=lambda child: (child["completed_score"], -child["action_bits"]),
    )
    assert response["result"]["native_action_bits"] == expected_action["action_bits"]

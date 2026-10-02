"""Public entities, execution handles and native leaf transport must agree."""
import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A, encode

pytestmark = pytest.mark.native


@pytest.mark.parametrize("card,zone", [
    ("HEADBUTT", "discard"), ("EXHUME", "exhaust"),
    ("SECRET_TECHNIQUE", "draw"), ("DISCOVERY", "generated"),
    ("PURITY", "hand"),
])
def test_selection_entities_match_packed_leaf(combat_factory, card, zone):
    env = combat_factory([card] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({"player": {"energy": 3},
        "hand": [card, "STRIKE_RED", "DEFEND_RED"],
        "draw": ["STRIKE_RED"] * 12 + ["SHRUG_IT_OFF", "DEFEND_RED"],
        "discard": ["BASH", "DEFEND_RED"], "exhaust": ["STRIKE_RED", "DEFEND_RED"]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    desc = battle.select_info()
    assert desc["zone"] == zone and len(desc["candidates"]) >= 2
    packed = env._native_packed_observation_batch({
        k: v[None, :] for k, v in battle.compact_leaf_arrays().items()})
    expected = env._native_leaf_observation(battle.to_dict(), combat_only=True)
    assert expected.keys() == packed.keys()
    for key in expected:
        np.testing.assert_array_equal(expected[key], packed[key][0], err_msg=key)
    obs = env._obs()
    for i, candidate in enumerate(desc["candidates"]):
        assert obs["select_native_indices"][i] == candidate["native_index"] + 1
        assert obs["select_card_ids"][i] != 0
    if not desc["multi"]:
        native_index = desc["candidates"][-1]["native_index"]
        expected_battle = battle.clone()
        expected_battle.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, native_index))
        env.step(A._PICK + len(desc["candidates"]) - 1)
        assert battle.to_dict() == expected_battle.to_dict()


def test_draw_candidate_features_do_not_reveal_pile_order(combat_factory):
    env = combat_factory(["SECRET_TECHNIQUE"] * 5)
    observations = []
    for cards in (["SHRUG_IT_OFF", "DEFEND_RED"], ["DEFEND_RED", "SHRUG_IT_OFF"]):
        battle = env.thick.bc.clone()
        battle.apply_custom_state({"hand": ["SECRET_TECHNIQUE"], "draw": cards})
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
        observations.append(env._native_leaf_observation(battle.to_dict(), combat_only=True))
    assert not np.array_equal(observations[0]["select_native_indices"], observations[1]["select_native_indices"])
    for key in observations[0].keys() - {"select_native_indices"}:
        np.testing.assert_array_equal(observations[0][key], observations[1][key], err_msg=key)


@pytest.mark.parametrize('card,zone', [('EXHUME', 'exhaust'), ('HEADBUTT', 'discard'),
                                     ('SECRET_TECHNIQUE', 'draw')])
@pytest.mark.parametrize('contract', ['legacy', 'joint-v2', 'autoregressive-v2'])
def test_overflow_keeps_distinct_choices_and_real_execution_handles(combat_factory, card, zone, contract):
    from env.core import native_actions as C
    env = combat_factory([card] * 5, combat_action_contract=contract)
    battle = env.thick.bc
    # Last option must survive; it is beyond both the hand and policy slot range.
    battle.apply_custom_state({'hand': [card], zone: ['DEFEND_RED'] * 100 + [('DEFEND_RED', 1)]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._invalidate_backend_cache()
    desc = battle.select_info()
    assert len(desc['candidates']) == 2
    handles = [c['native_index'] for c in desc['candidates']]
    assert handles == [0, 100]
    if contract == 'legacy':
        # This diagnostic endpoint returns the base packed layout. Neural
        # language/action extensions are covered by the native batch test.
        packed = env._native_packed_observation_batch({
            k: v[None, :] for k, v in battle.compact_leaf_arrays().items()})
        expected = env._native_leaf_observation(battle.to_dict(), combat_only=True)
        for key in expected:
            np.testing.assert_array_equal(expected[key], packed[key][0], err_msg=key)
    action = (A._PICK if contract == 'legacy' else
              C.JOINT_PICK - (C.JOINT_END if contract == 'autoregressive-v2' else 0)) + 1
    assert env.action_masks()[action]
    reference = battle.clone()
    reference.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, 100))
    env.step(action)
    assert battle.to_dict() == reference.to_dict()


def test_existing_candidate_slots_are_unchanged(combat_factory):
    env = combat_factory(['EXHUME'] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': ['EXHUME'], 'exhaust': ['DAZED'] * 96})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert [c['native_index'] for c in battle.select_info()['candidates']] == list(range(96))


def test_exhume_overflow_runs_native_search_and_policy(combat_factory):
    from env.search.native_batch import NativeSearchBatch
    from rl.policies.entity_combat import EntityCombatPolicy
    env = combat_factory(['EXHUME'] * 5, combat_action_contract='autoregressive-v2', card_language=True)
    env.thick.bc.apply_custom_state({'hand': ['EXHUME'],
        'exhaust': ['DAZED'] * 99 + ['DEFEND_RED', 'STRIKE_RED']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    policy = EntityCombatPolicy(env.observation_space, env.action_space, lambda _: 0.,
        entity_config=dict(width=32, layers=1, heads=4, quantiles=8))
    policy.set_training_mode(False)
    action, _ = policy.predict(env._obs(), deterministic=True, action_masks=env.action_masks())
    assert env.action_masks()[int(action)]
    pool = NativeSearchBatch([env], 2)
    logits = np.zeros(env.action_space.n, np.float32)
    try:
        pool.begin({0: ((logits,), dict(simulations=128, depth=1, state_nonce=42,
            root_search_mode=2, paper_full_gumbel=True, gumbel_scale=0,
            compact_observation=True, leaf_batch_size=1, fixed_worldline=True))})
        while not pool.done:
            observed = pool.observations()
            reference = env._native_packed_observation_batch(pool.payload)
            for key in observed:
                np.testing.assert_array_equal(observed[key], reference[key], err_msg=key)
            pool.advance(logits[None].repeat(pool.rows, axis=0), np.full(pool.rows, .5, np.float32))
        result = pool.results()[0]
        assert result['simulations'] == 128
        assert env.action_masks()[result['action']]
        assert np.isfinite(result['visit_policy']).all()
        assert result['visit_policy'].sum() == pytest.approx(1)
        env.step(result['action'])
    finally:
        pool.close()


def test_distinct_candidate_overflow_still_fails_explicitly(combat_factory):
    env = combat_factory(["EXHUME"] * 5)
    battle = env.thick.bc
    # Permanent instance effects must never be silently merged or discarded.
    battle.apply_custom_state({"hand": ["EXHUME"], "exhaust": ["RITUAL_DAGGER"] * 97})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    with pytest.raises(RuntimeError, match="97 distinct of 97; task=EXHUME"):
        battle.compact_leaf_arrays()


def test_codex_skip_is_a_public_candidate(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Nilry's Codex",))
    battle = env.thick.bc
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    env._invalidate_backend_cache()
    desc = battle.select_info()
    assert desc["context"] == "CODEX"
    skip = next(i for i, c in enumerate(desc["candidates"]) if c["skip"])
    obs = env._obs()
    assert obs["select_card_ids"][skip] == 0
    assert obs["select_native_indices"][skip] != 0
    assert env.action_masks()[A._PICK + skip]

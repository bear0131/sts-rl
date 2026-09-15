import numpy as np
import pytest

from env.core.run_env import unpack_nosl_observation_buffer
from schemas.combat_state import normalize_sim_global
from env.search.nosl_is_az import NOSLISAZConfig, NOSLISAZSession, NOSLISAZSearch
from env.search.nosl_sampler import NOSLSampler
from tests.unit.test_nosl_is_az import _Branch, _DeepBranch


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
@pytest.mark.xfail(reason="F05: native terminal value is pre-exit, learner post-exit HP",
                   strict=True, raises=AssertionError)
def test_native_terminal_matches_post_exit_training_value(combat_factory):
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
    assert session.result()["root_value"] == pytest.approx(56 / 200)


@pytest.mark.native
@pytest.mark.parametrize("encounter", ["Jaw Worm", "Two Louse", "Gremlin Leader"])
def test_packed_leaf_matches_canonical_bitwise(combat_factory, encounter):
    env = combat_factory(["STRIKE_RED"] * 5, encounter, relics=("Akabeko",))
    env._nosl_cpp_global_template = normalize_sim_global(env.thick.gc)
    logits = np.zeros(181)
    logits[1] = 50
    config = dict(simulations=1, depth=1, nonce=123)
    raw_session = env.thick.bc.nosl_neural_search_begin(logits, **config)
    packed_session = env.thick.bc.nosl_neural_search_begin(logits, **config)
    raw = raw_session.next_batch(1)[0]
    packed = env._native_packed_observation_batch(packed_session.next_batch_packed(1))
    expected = env._native_leaf_observation(raw["state"], combat_only=True)
    assert expected.keys() == packed.keys()
    for key in expected:
        np.testing.assert_array_equal(expected[key], packed[key][0], err_msg=key)


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
def test_strict_gumbel_matches_reference_schedule_transform_and_target(
    combat_factory,
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
    expected_completed = expected_normalized * (50 + 4) * 0.1
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

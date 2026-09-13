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

"""OBS02: compact leaves must preserve canonical live relic observations."""
from dataclasses import asdict

import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch
from bridge.schemas.combat_state import normalize_sim, normalize_sim_global
from bridge.schemas.entity_ids import native_relic_counter
from rl.search.native import NativeSearchConfig

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


class LiveRelicObservationMismatch(AssertionError):
    """Only the known OBS02 field mismatch is an expected failure."""


@pytest.mark.parametrize("relics", [
    ["Centennial Puzzle"], ["Lizard Tail"], ["Happy Flower", "Pen Nib", "InkBottle"],
])
def test_leaf_live_relic_counters_match_canonical(combat_factory, relics):
    def make():
        return combat_factory(["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"],
            "Cultist", seed=20261016, relics=relics,
            combat_action_contract="autoregressive-v2", card_language=False)

    native, reference = make(), make()
    config = NativeSearchConfig(simulations=128, leaf_batch_size=1,
        fixed_worldline=True, merge_equivalent_actions=False)
    logits = np.linspace(-.3, .7, native.action_space.n, dtype=np.float32)
    kwargs = dict(**asdict(config), root_value=.5, state_nonce=1729,
                  compact_observation=True)
    pool = NativeSearchBatch([native], 1)
    try:
        pool.begin({0: ((logits,), kwargs)})
        session = reference.nosl_is_az_cpp_begin(logits, **kwargs, _defer_leaf_selection=True)
        comparisons = []
        while not pool.done:
            requests = session.next_batch(1)
            assert len(requests) == 1
            request = requests[0]
            canonical = reference._native_leaf_observation(request["state"], combat_only=True)
            actual = pool.observations()
            comparisons.append((actual["relic_counters"][0].copy(), canonical["relic_counters"].copy()))
            session.backprop(request["request_id"], .5,
                logits.astype(np.float64) if request.get("needs_policy", True) else np.empty(0))
            pool.advance(logits[None], np.array([.5], dtype=np.float32))
        assert len(comparisons) == 128 and session.done()
        # Restrict expected failure to the actual observation contract assertion.
        for actual, expected in comparisons:
            try:
                np.testing.assert_array_equal(actual, expected,
                    err_msg="OBS02: native leaf must encode current branch relic state")
            except AssertionError as exc:
                raise LiveRelicObservationMismatch(str(exc)) from exc
    finally:
        pool.close()
        reference.nosl_is_az_cpp_abort()


@pytest.mark.parametrize("active", [0, 1])
def test_live_relic_flags_survive_import_and_both_leaf_decoders(combat_factory, active):
    env = combat_factory(["STRIKE_RED"] * 5, relics=["Centennial Puzzle", "Lizard Tail"])
    battle = env.thick.bc
    battle.apply_custom_state({"player": {"counters": {
        "centennial_puzzle_active": active, "lizard_tail_active": active}}})
    env._invalidate_backend_cache()
    canonical = normalize_sim(battle.to_dict())["player"]["combat_counters"]
    assert canonical["centennial_puzzle_active"] == active
    assert canonical["lizard_tail_active"] == active
    raw = battle.compact_leaf_arrays()
    packed = {"categorical": np.asarray(raw["categorical"])[None],
              "scalars": np.asarray(raw["scalars"])[None]}
    observed = env._native_packed_observation_batch(packed)
    np.testing.assert_array_equal(observed["relic_counters"][0, :2], [active, active])
    np.testing.assert_array_equal(observed["relic_counters"][0], env._obs()["relic_counters"])


@pytest.mark.parametrize("java_counter,expected", [(-1, 1), (-2, 0)])
def test_real_tail_sentinel_does_not_resurrect_consumed_relic(combat_factory, java_counter, expected):
    import slaythespire as sts
    env = combat_factory(["STRIKE_RED"] * 5)
    env.thick.gc.set_relic_state([{"id": "Lizard Tail", "counter": native_relic_counter("Lizard Tail", java_counter)}])
    battle = sts.create_battle(env.thick.gc, "Cultist")
    assert battle.to_dict()["player"]["combat_counters"]["lizard_tail_active"] == bool(expected)


def test_pen_nib_armed_native_counter_is_importable(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=["Pen Nib"])
    env.thick.bc.apply_custom_state({"player": {"counters": {"pen_nib": -1}}})
    assert normalize_sim(env.thick.bc.to_dict())["player"]["combat_counters"]["pen_nib"] == -1


def test_java_relic_sentinel_word_table_matches_native_save_conversion():
    import slaythespire as sts
    from bridge.schemas.entity_ids import RELIC_COUNTER_SENTINELS
    for relic_id, values in RELIC_COUNTER_SENTINELS.items():
        for raw in range(-3, 11):
            if raw in values:
                assert sts.java_relic_counter(relic_id, raw) == native_relic_counter(relic_id, raw) == values[raw]
            else:
                with pytest.raises(ValueError, match="counter"):
                    native_relic_counter(relic_id, raw)
                with pytest.raises(ValueError, match="counter"):
                    sts.java_relic_counter(relic_id, raw)


def test_tea_set_pending_charge_is_spent_in_combat_observation(combat_factory):
    import slaythespire as sts
    env = combat_factory(["STRIKE_RED"] * 5)
    env.thick.gc.set_relic_state([{"id": "Ancient Tea Set", "counter": 1}])
    assert normalize_sim_global(env.thick.gc, combat=False)["relic_counters"] == [1]
    env.thick.bc = sts.create_battle(env.thick.gc, "Cultist")
    env._invalidate_backend_cache()
    assert env.thick.bc.to_dict()["player"]["energy"] == 5
    assert env.thick.obs_canonical()["global"]["relic_counters"] == [0]
    assert env._obs()["relic_counters"][0] == 0
    raw = env.thick.bc.compact_leaf_arrays()
    packed = {"categorical": np.asarray(raw["categorical"])[None],
              "scalars": np.asarray(raw["scalars"])[None]}
    assert env._native_packed_observation_batch(packed)["relic_counters"][0, 0] == 0

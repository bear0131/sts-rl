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
    ["Kunai", "Letter Opener", "Ornamental Fan", "Pocketwatch", "Shuriken"],
    ["CaptainsWheel", "HornCleat", "StoneCalendar", "Velvet Choker"],
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


def test_every_relic_numeric_word_table_matches_native_save_converter():
    import slaythespire as sts
    from env.core.vocab import RELIC_VOCAB
    for relic_id in RELIC_VOCAB[1:]:
        for raw in (*range(-3, 11), 2147483647):
            try:
                expected = native_relic_counter(relic_id, raw)
            except ValueError:
                with pytest.raises(ValueError, match="counter"):
                    sts.java_relic_counter(relic_id, raw)
            else:
                assert sts.java_relic_counter(relic_id, raw) == expected, (relic_id, raw)


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


def test_du_vu_observation_and_strength_derive_from_all_master_deck_curses(combat_factory):
    import slaythespire as sts
    env = combat_factory(["STRIKE_RED"], relics=["Du-Vu Doll"])
    for cards, expected in [(["ASCENDERS_BANE", "CURSE_OF_THE_BELL", "PAIN"], 3),
                            (["ASCENDERS_BANE", "CURSE_OF_THE_BELL"], 2),
                            (["STRIKE_RED"], 0)]:
        env.thick.gc.set_deck([(card, 0) for card in cards])
        env.thick.bc = sts.create_battle(env.thick.gc, "Cultist")
        env._invalidate_backend_cache()
        assert env.thick.gc.relic_state()[0]["counter"] == expected
        assert dict(env.thick.bc.to_dict()["player"]["powers"]).get("Strength", 0) == expected
        assert env._obs()["relic_counters"][0] == np.float32(expected / 10)


@pytest.mark.parametrize("played,drawn", [(3, 8), (4, 5)])
def test_pocketwatch_boundary_retains_next_turn_draw_condition_in_both_decoders(combat_factory, played, drawn):
    import slaythespire as sts
    env = combat_factory(["STRIKE_RED"] * 20, "Cultist", relics=["Pocketwatch"])
    original = env.thick.bc
    original.apply_custom_state({"player": {"counters": {"cards_played": played}}})
    boundary = original.clone()
    raw = boundary.execute_to_enemy_boundary(4 << 29)
    assert raw["post_enemy_turn_boundary"]
    assert raw["player"]["combat_counters"]["cards_played"] == 0
    assert raw["pocketwatch_cards_at_boundary"] == played
    canonical = normalize_sim(raw)
    assert canonical["boundary_relic_counters"] == {"Pocketwatch": played}
    original.execute(sts.Action(sts.ActionType.END_TURN))
    assert len(original.to_dict()["hand"]) == drawn
    env.thick.bc = boundary
    env._invalidate_backend_cache()
    raw_packed = boundary.compact_leaf_arrays()
    packed = {"categorical": np.asarray(raw_packed["categorical"])[None],
              "scalars": np.asarray(raw_packed["scalars"])[None],
              "owners": np.array([0], dtype=np.int64),
              "selection_prefix": np.array([0], dtype=np.int32),
              "legal_policy_ids": np.empty((1, 0), dtype=np.int32)}
    expected = np.float32(played / 10)
    assert env._obs()["relic_counters"][0] == expected
    assert env._native_packed_observation_batch(packed)["relic_counters"][0, 0] == expected
    pool = NativeSearchBatch([env], 1)
    try:
        globals = {key: value[None] for key, value in env._native_packed_global_fields().items()}
        assert pool.decoder.decode(packed, globals)["relic_counters"][0, 0] == expected
    finally:
        pool.close()


def test_pocketwatch_future_key_keeps_three_and_four_card_boundaries_distinct(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 20, "Cultist", relics=["Pocketwatch"])
    keys = []
    for played in (3, 4):
        branch = env.thick.bc.clone()
        branch.apply_custom_state({"player": {"counters": {"cards_played": played}}})
        branch.execute_to_enemy_boundary(4 << 29)
        assert branch.to_dict()["player"]["combat_counters"]["cards_played"] == 0
        keys.append(branch.nosl_future_key())
    assert keys[0] != keys[1], "Pocketwatch's future extra draw must not be merged"

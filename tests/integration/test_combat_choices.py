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


def test_candidate_overflow_fails_explicitly(combat_factory):
    env = combat_factory(["EXHUME"] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({"hand": ["EXHUME"], "exhaust": ["STRIKE_RED"] * 97})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    with pytest.raises(RuntimeError, match="capacity 96 exceeded"):
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

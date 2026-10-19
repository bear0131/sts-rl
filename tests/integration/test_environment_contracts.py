"""Counterexamples that a refactor must satisfy, not implementation snapshots."""
import numpy as np
import pytest

from env.core import action_space as A, encode, vocab as V
from env.core.run_env import SlayTheSpireEnv
from schemas.combat_state import normalize_sim
import slaythespire as sts

pytestmark = pytest.mark.native


def test_auto_reset_reproduces_episode_seed_stream():
    streams = []
    for _ in range(2):
        env = SlayTheSpireEnv()
        try:
            stream = []
            for seed in (42, None, None, None):
                env.reset(seed=seed)
                stream.append(env.thick.gc.seed)
            streams.append(stream)
        finally:
            env.close()
    assert len(set(streams[0])) == len(streams[0])
    assert streams[0] == streams[1]


@pytest.mark.xfail(reason="F01: canonical monster compaction changes target slots",
                   strict=True, raises=AssertionError)
def test_gremlin_leader_observation_matches_native_target_slot(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, "Gremlin Leader")
    raw = env.thick.bc.to_dict()
    leader_slot = next(i for i, m in enumerate(raw["monsters"])
                       if m["id"] == "GREMLIN_LEADER")
    assert env._obs()["monster_id"][leader_slot] == V.MONSTER_VOCAB_INDEX["GremlinLeader"]


@pytest.mark.xfail(reason="F02: ordinary target observations omit pending card",
                   strict=True, raises=AssertionError)
def test_target_observation_distinguishes_selected_card(combat_factory):
    env = combat_factory(["BASH", "STRIKE_RED", "STRIKE_RED", "DEFEND_RED", "DEFEND_RED"],
                         "Two Louse")
    hand = env.thick.bc.to_dict()["hand"]
    observations = []
    for name in ("BASH", "STRIKE_RED"):
        branch = env.fork_combat()
        slot = next(i for i, card in enumerate(hand) if card[0] == name)
        observations.append(branch.step(A._PLAY + slot)[0])
    assert any(not np.array_equal(observations[0][key], observations[1][key])
               for key in observations[0])


@pytest.mark.xfail(reason="F03: Headbutt discard candidates absent from observation",
                   strict=True, raises=AssertionError)
def test_headbutt_candidates_are_discard_cards(combat_factory):
    env = combat_factory(["HEADBUTT", "STRIKE_RED", "DEFEND_RED", "DEFEND_RED", "BASH"])
    battle = env.thick.bc
    battle.set_player_energy(10)
    for name in ("DEFEND_RED", "STRIKE_RED", "HEADBUTT"):
        slot = next(i for i, c in enumerate(battle.to_dict()["hand"]) if c[0] == name)
        battle.execute(sts.Action(sts.ActionType.CARD, slot, 0))
    env._invalidate_backend_cache()
    expected = [encode.CARD_INDEX["Defend_R"], encode.CARD_INDEX["Strike_R"]]
    np.testing.assert_array_equal(env._obs()["select_card_ids"][:2], expected)


@pytest.mark.parametrize("card, filler, choices", [
    ("SECRET_TECHNIQUE", "STRIKE_RED", ["DEFEND_RED", "SHRUG_IT_OFF"]),
    ("SECRET_WEAPON", "DEFEND_RED", ["STRIKE_RED", "BASH"]),
])
@pytest.mark.parametrize("padding", [
    0,
    pytest.param(10, marks=pytest.mark.xfail(
        reason="F03: legal draw-pile choices at native indices >=10 vanish from the thin mask",
        strict=True, raises=pytest.fail.Exception)),
])
def test_draw_selection_remains_actionable_beyond_hand_slots(combat_factory, card, filler, choices, padding):
    env = combat_factory([card] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({
        "player": {"energy": 3}, "hand": [card],
        "draw": [filler] * padding + choices, "discard": [], "exhaust": [],
    })
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()
    native_choices = battle.legal_actions()
    assert [a.source_idx for a in native_choices] == [padding, padding + 1]
    assert all(a.action_type == sts.ActionType.SINGLE_CARD_SELECT for a in native_choices)

    mask = env.action_masks()
    if np.count_nonzero(mask) != 2:
        pytest.fail("Native draw choices must remain selectable through the thin API")
    _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[0]))
    assert not terminated and not truncated
    assert env.thick.select_descriptor() is None
    assert battle.to_dict()["hand"][0][0] == choices[0]


def test_gauntlet_preserves_feed_max_hp(gauntlet_factory):
    env = gauntlet_factory(["FEED"] * 5)
    env.bc.set_player_strength(999)
    env.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    won = env.bc.to_dict()
    assert won["outcome"] == 1
    env._settle()
    next_player = env.bc.to_dict()["player"]
    assert next_player["hp"] <= next_player["max_hp"]
    assert next_player["max_hp"] == won["player"]["max_hp"] == 83


@pytest.mark.parametrize("has_attack", [False, True])
def test_impatience_draws_only_without_attacks(combat_factory, has_attack):
    env = combat_factory(["IMPATIENCE"] + ["STRIKE_RED" if has_attack else "DEFEND_RED"] * 9, seed=0)
    battle = env.thick.bc
    before = battle.to_dict()
    slot = next(i for i, c in enumerate(before["hand"]) if c[0] == "IMPATIENCE")
    battle.execute(sts.Action(sts.ActionType.CARD, slot, 0))
    after = battle.to_dict()
    drawn = 0 if has_attack else 2
    assert len(after["hand"]) == len(before["hand"]) - 1 + drawn
    assert len(after["draw"]) == len(before["draw"]) - drawn


@pytest.mark.parametrize("hint", [-1, 0, 4, 100])
def test_exhaust_uses_card_identity_when_index_hint_is_stale(combat_factory, hint):
    env = combat_factory(["STRIKE_RED", "DEFEND_RED", "BASH", "ANGER", "FEED"])
    battle = env.thick.bc
    before = battle.to_dict()
    expected = before["hand"][2]
    battle.exhaust_hand_card_with_hint(2, hint)
    after = battle.to_dict()
    assert after["hand"] == before["hand"][:2] + before["hand"][3:]
    assert after["exhaust"] == before["exhaust"] + [expected[:2]]


@pytest.mark.parametrize("icecream", [False, True])
def test_end_turn_boundary_keeps_only_persistent_energy(combat_factory, icecream):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Ice Cream",) if icecream else ())
    observations, energy = [], []
    for available in (1, 3):
        battle = env.thick.bc.clone()
        battle.set_player_energy(available)
        boundary = battle.preview_end_turn_boundary(4 << 29)
        observations.append(encode.encode_combat(normalize_sim(boundary)))
        battle.execute(sts.Action(sts.ActionType.END_TURN))
        energy.append(battle.to_dict()["player"]["energy"])
    assert energy == ([4, 6] if icecream else [3, 3])
    different = any(not np.array_equal(observations[0][key], observations[1][key])
                    for key in observations[0])
    assert different == icecream


def test_native_fork_is_read_only_until_committed(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5)
    before = env.thick.bc.to_dict()
    fork = env.fork_combat()
    fork.step(A.END_TURN)
    assert env.thick.bc.to_dict() == before
    assert fork.thick.bc.to_dict() != before


@pytest.mark.parametrize("relic", ["Burning Blood", "Black Blood", "Meat on the Bone"])
def test_gauntlet_commits_counters_without_victory_healing(gauntlet_factory, relic):
    env = gauntlet_factory(["STRIKE_RED"] * 5, relics=(relic, "Pen Nib"), hp=30)
    for count in (1, 2):
        env.bc.set_player_strength(999)
        env.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        env._settle()
        assert env.combats_won == count and not env.is_over
        assert env.bc.to_dict()["player"]["hp"] == env.gc.cur_hp == 30
        assert next(r["counter"] for r in env.gc.relic_state() if r["id"] == "Pen Nib") == count


def test_gauntlet_potions_obtained_and_discarded_persist(gauntlet_factory):
    env = gauntlet_factory(["STRIKE_RED"] * 5, seed=0)
    assert env.gc.get_neow_options()[1]["bonus"] == "Obtain three potions."
    env.gc.choose_neow_option(1)
    for _ in range(3):
        env.gc.claim_potion(0)
    env.bc = sts.create_battle(env.gc, "Jaw Worm")
    potions = env.bc.potions
    assert env.bc.legal_potions()
    env.bc.set_player_strength(999)
    env.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._settle()
    assert env.bc.potions == env.gc.overworld_potions == potions
    env.bc.discard_potion(0)
    empty = env.bc.potions
    assert empty != potions
    slot = next(i for i, c in enumerate(env.bc.to_dict()["hand"]) if c[0] == "STRIKE_RED")
    env.bc.set_player_strength(999)
    env.bc.execute(sts.Action(sts.ActionType.CARD, slot, 0))
    env._settle()
    assert env.bc.potions == env.gc.overworld_potions == empty


def test_gauntlet_preserves_ritual_dagger_growth(gauntlet_factory):
    env = gauntlet_factory(["RITUAL_DAGGER"] * 5)
    assert [card.misc for card in env.gc.deck] == [15] * 5
    env.bc.set_player_strength(999)
    env.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env._settle()
    assert sorted(card.misc for card in env.gc.deck) == [15, 15, 15, 15, 18]

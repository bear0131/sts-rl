"""Counterexamples that a refactor must satisfy, not implementation snapshots."""
import numpy as np
import pytest

from env.core import action_space as A, encode, vocab as V
from env.core.run_env import SlayTheSpireEnv
from env.core.combat_backend import SimCombatBackend
from schemas.combat_state import normalize_sim
import slaythespire as sts

pytestmark = pytest.mark.native


@pytest.mark.xfail(reason="F08: backend recreation loses the seeded episode stream",
                   strict=True, raises=AssertionError)
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


@pytest.mark.xfail(reason="F07: gauntlet does not commit Feed max HP",
                   strict=True, raises=AssertionError)
def test_gauntlet_preserves_feed_max_hp():
    def factory(_rng):
        def make():
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
            gc.floor_num = 1
            gc.set_relics([])
            gc.set_deck([("FEED", 0)] * 5)
            return gc
        return make, lambda _gc: "Jaw Worm"

    env = SimCombatBackend(factory)
    env.reset(seed=42)
    env.bc.set_player_strength(999)
    env.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    won = env.bc.to_dict()
    assert won["outcome"] == 1
    env._settle()
    next_player = env.bc.to_dict()["player"]
    assert next_player["hp"] <= next_player["max_hp"]
    assert next_player["max_hp"] == won["player"]["max_hp"] == 83


@pytest.mark.xfail(reason="F10: Impatience draws even when attacks remain",
                   strict=True, raises=AssertionError)
def test_impatience_does_not_draw_with_attack_in_hand(combat_factory):
    env = combat_factory(["IMPATIENCE"] + ["STRIKE_RED"] * 9, seed=0)
    battle = env.thick.bc
    before = battle.to_dict()
    slot = next(i for i, c in enumerate(before["hand"]) if c[0] == "IMPATIENCE")
    battle.execute(sts.Action(sts.ActionType.CARD, slot, 0))
    after = battle.to_dict()
    assert len(after["hand"]) == len(before["hand"]) - 1
    assert after["draw"] == before["draw"]


@pytest.mark.xfail(reason="F19: value boundary erases Ice Cream carryover",
                   strict=True, raises=AssertionError)
def test_end_turn_boundary_keeps_icecream_future_resource(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5, relics=("Ice Cream",))
    observations, energy = [], []
    for available in (1, 3):
        battle = env.thick.bc.clone()
        battle.set_player_energy(available)
        boundary = battle.preview_end_turn_boundary(4 << 29)
        observations.append(encode.encode_combat(normalize_sim(boundary)))
        battle.execute(sts.Action(sts.ActionType.END_TURN))
        energy.append(battle.to_dict()["player"]["energy"])
    assert energy == [4, 6]
    assert any(not np.array_equal(observations[0][key], observations[1][key])
               for key in observations[0])


def test_native_fork_is_read_only_until_committed(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5)
    before = env.thick.bc.to_dict()
    fork = env.fork_combat()
    fork.step(A.END_TURN)
    assert env.thick.bc.to_dict() == before
    assert fork.thick.bc.to_dict() != before

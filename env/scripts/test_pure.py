"""Smoke test for the v2 backend-agnostic obs layer (vocab + encode + intents +
actions + render). Runs WITHOUT the sim built: it feeds hand-made canonical dicts
(the shape bridge.schemas produces) through the pure functions.

Run from the repo root:  python -m env.scripts.test_pure
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from env.core import encode
from env.core import actions
from env.core import render
from env.core import intents
from env.core import vocab as V


# ------------------------------ fixtures ------------------------------

def _fake_combat():
    """A combat-block canonical (normalize_sim shape) + phase/global/options."""
    return {
        "phase": "combat",
        "turn": 2,
        "player": {
            "hp": 68, "max_hp": 80, "block": 5, "energy": 3,
            "powers": {"Strength": 2, "Vulnerable": 0, "Frail": 1},
        },
        "monsters": [
            {"id": "JawWorm", "hp": 40, "max_hp": 44, "block": 0, "gone": False,
             "base_damage": 11, "hits": 1, "powers": {"Strength": 3},
             "move": "JAW_WORM_CHOMP", "move_id": 87},
            {"id": "Cultist", "hp": 0, "max_hp": 48, "block": 0, "gone": True,
             "base_damage": 0, "hits": 0, "powers": {},
             "move": "INVALID", "move_id": 0},
        ],
        "hand": [{"id": "Strike_R", "up": 0}, {"id": "Bash", "up": 0},
                 {"id": "Defend_R", "up": 1}],
        "draw": [{"id": "Strike_R", "up": 0}, {"id": "Strike_R", "up": 0}],
        "discard": [{"id": "Defend_R", "up": 0}],
        "exhaust": [],
        "global": {
            "character": "IRONCLAD", "ascension": 3, "act": 1, "floor": 6,
            "boss": "HEXAGHOST", "relics": ["Burning Blood", "Akabeko"],
            "potions": ["Fire Potion", "INVALID"], "gold": 99,
            "deck": [("Strike_R", 0), ("Strike_R", 0), ("Bash", 0),
                     ("Defend_R", 1)],
            "boss_relics": [],
        },
    }


def _fake_map():
    return {
        "phase": "map",
        "options": {"nodes": [(2, "MONSTER"), (3, "EVENT"), (4, "REST")]},
        "global": {
            "character": "IRONCLAD", "ascension": 0, "act": 1, "floor": 5,
            "boss": "HEXAGHOST", "relics": ["Burning Blood"], "potions": [],
            "gold": 50, "deck": [("Strike_R", 0)], "boss_relics": [],
        },
    }


# ------------------------------ vocab ------------------------------

def test_vocab_invariants():
    # id vocabs: index 0 is the PAD sentinel; no duplicates; index map consistent.
    for name in ("MONSTER_VOCAB", "MOVE_VOCAB", "RELIC_VOCAB", "POTION_VOCAB",
                 "EVENT_VOCAB", "BOSS_VOCAB"):
        voc = getattr(V, name)
        idx = getattr(V, name + "_INDEX")
        assert voc[0] == "PAD", (name, voc[0])
        assert len(voc) == len(set(voc)), f"{name} has duplicates"
        assert idx[voc[7]] == 7
        cap = getattr(V, name + "_CAPACITY")
        assert len(voc) <= cap, (name, len(voc), cap)
    # buff vocabs: dense, no PAD sentinel, fit in their reserved vector length.
    assert len(V.PLAYER_BUFF_VOCAB) <= V.PLAYER_BUFF_VEC
    assert len(V.MONSTER_BUFF_VOCAB) <= V.MONSTER_BUFF_VEC
    assert V.PLAYER_BUFF_VOCAB[0] == "Strength"
    assert V.PHASE_VOCAB[0] == "none" and V.CHARACTER_VOCAB[0] == "INVALID"
    assert V.INTENT_CATEGORY[0] == "unknown"
    print(f"vocab OK: monsters={len(V.MONSTER_VOCAB)} moves={len(V.MOVE_VOCAB)} "
          f"relics={len(V.RELIC_VOCAB)} buffs P{len(V.PLAYER_BUFF_VOCAB)}/"
          f"M{len(V.MONSTER_BUFF_VOCAB)}")


# ------------------------------ intents ------------------------------

def test_intent_derivation():
    assert intents.derive("JAW_WORM_CHOMP", 11) == "attack"   # base_damage wins
    assert intents.derive("CENTURION_DEFEND", 0) == "defend"
    assert intents.derive("CULTIST_INCANTATION", 0) == "buff"
    assert intents.derive("THE_COLLECTOR_MEGA_DEBUFF", 0) == "strong_debuff"
    assert intents.derive("SNAKE_PLANT_ENFEEBLING_SPORES", 0) == "debuff"
    assert intents.derive("LAGAVULIN_SLEEP", 0) == "sleep"
    assert intents.derive("LOOTER_ESCAPE", 0) == "escape"
    assert intents.derive("BYRD_STUNNED", 0) == "stun"
    assert intents.derive(None, 0) == "unknown"
    assert intents.derive("INVALID", 0) == "unknown"
    print("intent derivation OK")


# ------------------------------ encode ------------------------------

def test_obs_matches_space():
    space = encode.obs_space()
    for canon in (_fake_combat(), _fake_map(), {}):  # combat / overworld / empty
        obs = encode.encode(canon)
        assert set(obs) == set(space.spaces), (set(obs) ^ set(space.spaces))
        for k, v in obs.items():
            sub = space.spaces[k]
            assert v.shape == sub.shape, (k, v.shape, sub.shape)
            assert v.dtype == sub.dtype, (k, v.dtype, sub.dtype)
            assert sub.contains(v), f"{k} out of declared bounds"
    print(f"obs matches space (OBS_VERSION={encode.OBS_VERSION}); "
          f"{len(space.spaces)} groups")


def test_combat_fields_land():
    from env.core.cards_vocab import CARD_INDEX
    o = encode.encode(_fake_combat())
    assert abs(o["player_scalars"][0] - 68 / encode.HP_SCALE) < 1e-6
    assert abs(o["player_scalars"][5] - 68 / 80) < 1e-6
    assert abs(o["player_buffs"][V.PLAYER_BUFF_VOCAB_INDEX["Strength"]]
               - 2 / encode.BUFF_SCALE) < 1e-6
    assert o["monster_id"][0] == V.MONSTER_VOCAB_INDEX["JawWorm"]
    assert o["monster_scalars"][0][0] == 1.0
    assert o["monster_move"][0] == 87
    assert o["monster_intent_cat"][0] == V.INTENT_CATEGORY_INDEX["attack"]
    assert o["hand_ids"][1] == CARD_INDEX["Bash"]
    assert o["draw_counts"][CARD_INDEX["Strike_R"]] == 2.0
    print("combat fields land OK")


def test_fixed_monster_slots_no_drift():
    """Dead monster must STAY in its slot with present=0 (no slot shift)."""
    o = encode.encode(_fake_combat())
    assert o["monster_scalars"][0][0] == 1.0       # JawWorm present
    assert o["monster_id"][1] == V.MONSTER_VOCAB_INDEX["Cultist"]
    assert o["monster_scalars"][1][0] == 0.0       # Cultist absent, slot kept
    print("fixed monster slots OK (no drift)")


def test_global_and_options():
    o = encode.encode(_fake_combat())
    assert o["phase"][0] == V.PHASE_VOCAB_INDEX["combat"]
    assert o["character"][0] == V.CHARACTER_VOCAB_INDEX["IRONCLAD"]
    assert o["next_boss"][0] == V.BOSS_VOCAB_INDEX["HEXAGHOST"]
    assert o["relics"][0] == V.RELIC_VOCAB_INDEX["Burning Blood"]
    assert o["potions"][0] == V.POTION_VOCAB_INDEX["Fire Potion"]
    assert o["potions"][1] == 0                    # 'INVALID' -> PAD
    assert abs(o["gold"][0] - 99 / encode.GOLD_SCALE) < 1e-6

    m = encode.encode(_fake_map())
    assert m["phase"][0] == V.PHASE_VOCAB_INDEX["map"]
    assert m["map_node_rooms"][0] == V.ROOM_VOCAB_INDEX["MONSTER"]
    assert m["map_node_rooms"][1] == V.ROOM_VOCAB_INDEX["EVENT"]
    assert abs(m["map_node_x"][0] - 2 / encode.MAP_X_SCALE) < 1e-6
    assert m["monster_id"].sum() == 0 and m["hand_ids"].sum() == 0
    print("global + options OK")


# --------------------------- actions / render ---------------------------

def test_action_codec_roundtrip():
    seen = set()
    for aid in range(actions.N_ACTIONS):
        kind = actions.decode(aid)
        seen.add(aid)
        if kind[0] == actions.KIND_PLAY:
            assert actions.encode_play(kind[1], kind[2]) == aid
        elif kind[0] == actions.KIND_SELECT:
            assert actions.encode_select(kind[1]) == aid
    assert len(seen) == actions.N_ACTIONS
    print(f"N_ACTIONS = {actions.N_ACTIONS}  (version {actions.ACTION_VERSION})")


def test_render():
    canon = _fake_combat()
    s = render.render_state(canon)
    assert "JawWorm" in s and "68/80" in s and "Strike_R" in s, s
    assert "Cultist" in s and "gone" in s, s
    assert render.render_action(actions.END_TURN) == "end turn"
    label = render.render_action(actions.encode_play(1, 0), canon)
    assert "Bash" in label and "JawWorm" in label, label
    print("render OK")


if __name__ == "__main__":
    test_vocab_invariants()
    test_intent_derivation()
    test_obs_matches_space()
    test_combat_fields_land()
    test_fixed_monster_slots_no_drift()
    test_global_and_options()
    test_action_codec_roundtrip()
    test_render()
    print("\nALL OK")

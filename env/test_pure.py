"""Smoke test for the backend-agnostic env layer (encode + actions).

Runs WITHOUT the sim built: it feeds a hand-made canonical combat dict (the same
shape bridge.schemas produces) through encode(), and round-trips the action
codec. This is what we can verify on a machine where slaythespire isn't compiled.

Run: python env/test_pure.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import encode
import actions


def _fake_canon():
    return {
        "turn": 2,
        "player": {
            "hp": 68, "max_hp": 80, "block": 5, "energy": 3,
            "powers": {"Strength": 2, "Vulnerable": 0, "Frail": 1},
        },
        "monsters": [
            {"id": "JawWorm", "hp": 40, "max_hp": 44, "block": 0, "gone": False,
             "base_damage": 11, "hits": 1, "powers": {"Strength": 3}},
            {"id": "Cultist", "hp": 0, "max_hp": 48, "block": 0, "gone": True,
             "base_damage": 0, "hits": 0, "powers": {}},
        ],
        "hand": [{"id": "Strike_R", "up": 0}, {"id": "Bash", "up": 0},
                 {"id": "Defend_R", "up": 1}],
        "draw": [{"id": "Strike_R", "up": 0}, {"id": "Strike_R", "up": 0}],
        "discard": [{"id": "Defend_R", "up": 0}],
        "exhaust": [],
    }


def test_encode_shape_and_stability():
    obs = encode.encode(_fake_canon())
    assert obs.shape == (encode.OBS_SIZE,), obs.shape
    assert obs.dtype == np.float32
    # encoder is deterministic
    assert np.array_equal(obs, encode.encode(_fake_canon()))
    # a known field landed: player hp / HP_SCALE at offset 0
    assert abs(obs[0] - 68 / encode.HP_SCALE) < 1e-6
    print(f"OBS_SIZE = {encode.OBS_SIZE}  (version {encode.OBS_VERSION})")


def test_dead_monster_padding():
    # the gone Cultist must not occupy a monster row; only the live JawWorm does
    canon = _fake_canon()
    obs = encode.encode(canon)
    m0_present = obs[encode.PLAYER_BLOCK + 0 * encode.MON_BLOCK + 0]
    m1_present = obs[encode.PLAYER_BLOCK + 1 * encode.MON_BLOCK + 0]
    assert m0_present == 1.0 and m1_present == 0.0, (m0_present, m1_present)


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


if __name__ == "__main__":
    test_encode_shape_and_stability()
    test_dead_monster_padding()
    test_action_codec_roundtrip()
    print("OK")

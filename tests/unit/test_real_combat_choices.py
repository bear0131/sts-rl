import numpy as np

from bridge.real_game_env import RealGameAdapter
from env.core import action_space as A, game_actions as ga, encode


def test_real_grid_candidates_and_command_preserve_indices():
    adapter = RealGameAdapter()
    cards = [{"id": "Strike_R", "upgrades": 0}] * 12 + [{"id": "Demon Form", "upgrades": 1}]
    adapter.update({"in_game": True, "available_commands": ["choose"], "game_state": {
        "room_phase": "COMBAT", "screen_type": "GRID",
        "combat_state": {"hand": []},
        "screen_state": {"cards": cards, "context": "HEADBUTT"},
    }})
    legal = adapter.legal_semantic_actions()
    assert legal == [ga.select(i) for i in range(13)]
    semantic, pending = A.resolve(A._PICK + 12, legal, None, "combat")
    assert pending is None and adapter.command_for_semantic(semantic) == "CHOOSE 12"
    desc = adapter.select_descriptor()
    obs = encode.encode_combat({"options": {"select": desc}})
    assert obs["select_card_ids"][12] == encode.CARD_INDEX["Demon Form"]
    assert obs["select_card_upgrades"][12] == 1 / encode.UPGRADE_SCALE
    assert obs["select_native_indices"][12] == 13


def test_real_pending_sources_are_always_encoded(monkeypatch):
    adapter = RealGameAdapter()
    monkeypatch.setattr(adapter, "_base_canonical", lambda: {"phase": "combat"})
    first = adapter.observation(("play", 2))
    second = adapter.observation(("potion", 1))
    assert first["pending_card_slot"].tolist() == [3]
    assert first["pending_potion_slot"].tolist() == [0]
    assert second["pending_potion_slot"].tolist() == [2]
    assert not np.array_equal(first["pending_card_slot"], second["pending_card_slot"])

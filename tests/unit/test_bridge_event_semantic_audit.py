"""BRIDGE06: native event semantics must not be UI button ordinals.

Fixtures describe public option contracts observed in BRIDGE05 real frames.
No live game or checkpoint is needed for this static adapter audit.
"""
import pytest

from bridge.real_game_env import RealGameAdapter
from env.core import game_actions as ga


@pytest.fixture
def event_adapter():
    def make(event, options, floor=19):
        adapter = RealGameAdapter()
        adapter.update({
            "in_game": True, "ready_for_command": True,
            "available_commands": ["choose", "state"],
            "game_state": {
                "seed": "200205", "floor": floor, "act": 2,
                "screen_type": "EVENT", "room_phase": "EVENT",
                "choice_list": [o[0] for o in options if not o[1]],
                "screen_state": {"event_id": event, "options": [
                    {"text": text, "label": text, "disabled": disabled,
                     "choice_index": None if disabled else sum(not x[1] for x in options[:i])}
                    for i, (text, disabled) in enumerate(options)
                ]},
                "deck": [], "relics": [], "potions": [],
                "current_hp": 60, "max_hp": 80, "gold": 14,
            },
        })
        return adapter
    return make


@pytest.mark.xfail(strict=True, reason="BRIDGE06: Knowing Skull native potion action2 incorrectly emits UI2/card")
def test_skull_potion_semantic_targets_potion_button(event_adapter):
    a = event_adapter("Knowing Skull", [("[来点喝的？] 获得一瓶药水。", False),
        ("[财富？] 获得90金币。", False), ("[成功？] 获得一张无色牌。", False),
        ("[我要怎么离开？] 失去6生命。", False)])
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 0"


@pytest.mark.xfail(strict=True, reason="BRIDGE06: absent Blood Vial must mask native action0, not renumber accept/refuse")
def test_vampires_without_vial_has_accept_and_refuse_native_ids(event_adapter):
    a = event_adapter("Vampires", [("[接受] 失去24最大生命。", False), ("[拒绝]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(1)) == "CHOOSE 0"


@pytest.mark.xfail(strict=True, reason="BRIDGE06: disabled Red Mask option compresses transport indices, not native semantics")
def test_red_mask_disabled_option_keeps_native_ids(event_adapter):
    a = event_adapter("Tomb of Lord Red Mask", [("[锁定] 需要红面具。", True),
        ("[献上14金币] 获得遗物。", False), ("[离开]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 1"


@pytest.mark.xfail(strict=True, reason="BRIDGE06: late Mind Bloom heal is native action3, not wealth action2")
def test_late_mind_bloom_heal_is_not_wealth(event_adapter):
    a = event_adapter("MindBloom", [("[我必凯旋] 战斗。", False),
        ("[我已觉醒] 升级所有牌。", False), ("[我无伤痛] 回复所有生命。", False)], floor=45)
    assert a._legal_event_actions() == [ga.event(0), ga.event(1), ga.event(3)]
    assert a.command_for_semantic(ga.event(3)) == "CHOOSE 2"

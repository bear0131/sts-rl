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


def test_skull_potion_semantic_targets_potion_button(event_adapter):
    a = event_adapter("Knowing Skull", [("[来点喝的？] 获得一瓶药水。", False),
        ("[财富？] 获得90金币。", False), ("[成功？] 获得一张无色牌。", False),
        ("[我要怎么离开？] 失去6生命。", False)])
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 0"


def test_vampires_without_vial_has_accept_and_refuse_native_ids(event_adapter):
    a = event_adapter("Vampires", [("[接受] 失去24最大生命。", False), ("[拒绝]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(1)) == "CHOOSE 0"


def test_red_mask_disabled_option_keeps_native_ids(event_adapter):
    a = event_adapter("Tomb of Lord Red Mask", [("[锁定] 需要红面具。", True),
        ("[献上14金币] 获得遗物。", False), ("[离开]", False)])
    assert a._legal_event_actions() == [ga.event(1), ga.event(2)]
    assert a.command_for_semantic(ga.event(2)) == "CHOOSE 1"


def test_late_mind_bloom_heal_is_not_wealth(event_adapter):
    a = event_adapter("MindBloom", [("[我必凯旋] 战斗。", False),
        ("[我已觉醒] 升级所有牌。", False), ("[我无伤痛] 回复所有生命。", False)], floor=45)
    assert a._legal_event_actions() == [ga.event(0), ga.event(1), ga.event(3)]
    assert a.command_for_semantic(ga.event(3)) == "CHOOSE 2"


def test_unknown_event_does_not_fall_back_to_number(event_adapter):
    a = event_adapter("Cursed Tome", [("[继续阅读]", False)])
    with pytest.raises(ValueError, match="unsupported event"):
        a._legal_event_actions()


def test_dialogue_bypasses_policy_without_claiming_resource_effect(event_adapter):
    a = event_adapter("Knowing Skull", [("[继续]", False)])
    assert a.auto_command() == ("CHOOSE 0", "event_dialogue_transport")


def test_unknown_label_is_not_guessed(event_adapter):
    a = event_adapter("Knowing Skull", [("[未知选项]", False)])
    with pytest.raises(ValueError, match="unrecognized"):
        a._legal_event_actions()


def test_event_potion_effect_rejects_card_gain(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter("Knowing Skull", [("[来点喝的？] 获得一瓶药水。", False)])
    a.gs['potions'] = [{'id': 'Potion Slot'}]
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 0)
    assert ack.waiting(a.frame, 1)
    bad = deepcopy(a.frame)
    bad['game_state']['deck'] = [{'id': 'Mind Blast', 'upgrades': 0}]
    bad['game_state']['current_hp'] -= 6
    with pytest.raises(RuntimeError, match="effect type mismatch"):
        ack.waiting(bad, 2)
    good = deepcopy(a.frame)
    good['game_state']['potions'] = [{'id': 'Strength Potion'}]
    good['game_state']['current_hp'] -= 6
    assert not ack.waiting(good, 2)
    assert ack.pending is None


def test_event_effect_timeout_and_grid_wait(event_adapter):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter("Purifier", [("[祈祷]", False)])
    a.gs['deck'] = [{'id': 'Strike_R', 'upgrades': 0}]
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 0)), 0)
    grid = deepcopy(a.frame)
    grid['game_state']['screen_type'] = 'GRID'
    assert not ack.waiting(grid, 1)
    assert ack.pending is not None
    with pytest.raises(RuntimeError, match="timeout"):
        ack.waiting(a.frame, 31)


@pytest.mark.parametrize('sozu', [False, True])
def test_skull_blocked_potion_requires_no_gain_and_completed_effect(event_adapter, sozu):
    from bridge.event_contract import EventEffectAck
    from copy import deepcopy
    from types import SimpleNamespace
    a = event_adapter('Knowing Skull', [('[来点喝的？] 获得一瓶药水。失去 9 生命。', False)])
    a.gs['potions'] = [{'id': 'Potion Slot' if sozu else 'Strength Potion'}]
    a.gs['relics'] = [{'id': 'Sozu'}] if sozu else []
    ack = EventEffectAck()
    ack.arm(a.frame, SimpleNamespace(semantic=('event', 2)), 0)
    assert ack.waiting(a.frame, 1)
    after = deepcopy(a.frame)
    after['game_state']['current_hp'] -= 9
    assert ack.waiting(after, 2)  # HP-only intermediate frame is not completion.
    after['game_state']['screen_state']['options'][0]['text'] = '[来点喝的？] 失去 10 生命。'
    assert not ack.waiting(after, 3)

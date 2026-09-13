from __future__ import annotations

from rl.evaluation.rules.explain_card_play import (
    SandboxConfig,
    build_report,
    explain_effect,
    resolve_card_id,
)


def test_resolve_card_id_accepts_canonical_and_enum_style_names():
    assert resolve_card_id("Dropkick") == "Dropkick"
    assert resolve_card_id("dropkick") == "Dropkick"
    assert resolve_card_id("飞身踢") == "Dropkick"
    assert resolve_card_id("Strike_R") == "Strike_R"


def test_explain_effect_translates_energy_and_draw_events():
    assert explain_effect({
        "type": "PLAYER_VALUE_CHANGE", "atomic": 0, "field": "energy",
        "before": 3, "delta": -1, "after": 2,
    }) == "[原子步骤 0] 玩家能量：3 -1 -> 2。"
    assert "抽牌堆移动到手牌" in explain_effect({
        "type": "CARD_MOVE", "atomic": 3, "source": "draw",
        "destination": "hand", "card": "Dropkick", "up": 0,
    })


def test_dropkick_report_contains_rule_formula_branch_and_tokens():
    config = SandboxConfig(
        card_id="Dropkick", player_strength=2,
        enemy_block=3, enemy_vulnerable=2,
    )
    before = {
        "player": {"energy": 3},
        "monsters": [{"id": "Cultist", "hp": 52, "block": 3}],
    }
    after = {
        "player": {"energy": 3},
        "monsters": [{"id": "Cultist", "hp": 45, "block": 0}],
    }
    effects = [
        {"type": "PLAYER_VALUE_CHANGE", "atomic": 0, "field": "energy",
         "before": 3, "delta": -1, "after": 2},
        {"type": "MONSTER_VALUE_CHANGE", "atomic": 1, "slot": 0,
         "field": "hp", "before": 52, "delta": -7, "after": 45},
        {"type": "MONSTER_VALUE_CHANGE", "atomic": 1, "slot": 0,
         "field": "block", "before": 3, "delta": -3, "after": 0},
        {"type": "PLAYER_VALUE_CHANGE", "atomic": 2, "field": "energy",
         "before": 2, "delta": 1, "after": 3},
        {"type": "ACTION_COMPLETE", "atomic_steps": 3},
    ]
    result = {
        "config": config,
        "before": before,
        "after": after,
        "action": {"card": {"id": "Dropkick", "up": 0, "cost": 1}},
        "effects": effects,
        "calculations": [],
        "rule_steps": [],
    }
    report = build_report(result)
    assert "CARD_BEGIN TYPE_ATTACK RARITY_UNCOMMON" in report
    assert "floor(7 x 3 / 2) = 10" in report
    assert "TARGET_HAS_VULNERABLE：2 > 0，为真" in report
    assert "条件奖励回能 1：能量 2 -> 3；随后抽 1 张牌" in report
    assert "FIELD_atomic" not in report
    assert "STEP_INDEX" not in report
    assert "EFFECT_ACTION_COMPLETE" not in report
    assert "<CALCULATIONS_BEGIN> <CALCULATIONS_END>" not in report

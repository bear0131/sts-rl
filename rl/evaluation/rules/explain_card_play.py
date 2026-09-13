"""Simulate one card against one enemy and explain the resulting transition.

The command uses the rule-trace build of sts_lightspeed.  It does not implement
card behavior in Python: the simulator remains authoritative, while this module
turns its atomic snapshots into rule-model tokens and Chinese explanations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_ZH = {
    "hp": "生命",
    "max_hp": "最大生命",
    "block": "格挡",
    "energy": "能量",
    "gold": "金币",
}
ZONE_ZH = {
    "hand": "手牌",
    "draw": "抽牌堆",
    "discard": "弃牌堆",
    "exhaust": "消耗堆",
}
COUNTER_ZH = {
    "attacks_played": "本回合已打出的攻击牌数",
    "cards_played": "本回合已打出的卡牌数",
    "skills_played": "本回合已打出的技能牌数",
    "pen_nib": "钢笔尖计数",
}


@dataclass(frozen=True)
class SandboxConfig:
    card_id: str
    upgraded: bool = False
    enemy: str = "Cultist"
    seed: int = 42
    floor: int = 1
    player_energy: int = 3
    player_strength: int = 0
    enemy_block: int = 0
    enemy_vulnerable: int = 0
    player_block: int = 0
    deck: tuple[tuple[str, int], ...] | None = None
    opening_hand_attempts: int = 1


class OpeningHandError(RuntimeError):
    """The requested card version was not playable in sampled opening hands."""


def _configure_sim_build() -> None:
    """Select a compatible local rule-trace build when none was specified."""
    if "STS_LIGHTSPEED_BUILD_DIR" not in os.environ:
        major, minor = sys.version_info[:2]
        if os.name == "nt":
            suffix = f"cp{major}{minor}-win_amd64.pyd"
        else:
            suffix = f"cpython-{major}{minor}*.so"
        candidates = (
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1-cp311",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-combat-language",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build",
        )
        for candidate in candidates:
            if list(candidate.glob(f"slaythespire.*{suffix}")):
                os.environ["STS_LIGHTSPEED_BUILD_DIR"] = str(candidate)
                break

    if os.name == "nt" and "STS_LIGHTSPEED_DLL_DIRS" not in os.environ:
        runtime_dirs = []
        for candidate in (
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1-cp311",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build-rule-v1",
            REPO_ROOT / "sim" / "sts_lightspeed" / "build",
        ):
            if candidate.is_dir():
                runtime_dirs.append(str(candidate))
        if runtime_dirs:
            os.environ["STS_LIGHTSPEED_DLL_DIRS"] = os.pathsep.join(runtime_dirs)


# env.core caches the selected build directory at import time, so bootstrap the
# trace build before importing any of its otherwise pure rule-language modules.
_configure_sim_build()

from env.core.card_language import card_program  # noqa: E402
from env.core.ironclad_cards import ATTACK, IRONCLAD_CARDS  # noqa: E402
from env.core.rule_language import (  # noqa: E402
    card_rule_sentence,
    deck_rule_sentence,
    relic_loadout_sentence,
)
from rl.rules.rule_data import (  # noqa: E402
    CALCULATIONS_BEGIN,
    CALCULATIONS_END,
    EFFECTS_BEGIN,
    EFFECTS_END,
    NEXT_STATE_BEGIN,
    NEXT_STATE_END,
    OUTPUT_BEGIN,
    OUTPUT_END,
    RuleTokenizer,
    derive_calculations,
    derive_local_calculations,
    derive_rule_steps,
    events_from_snapshots,
    sample_tokens,
    serialize_effects,
)


def resolve_card_id(value: str) -> str:
    """Resolve a canonical id, simulator enum spelling, or Chinese card name."""
    normalized = value.strip().replace("_", " ").casefold()
    matches = [
        card_id for card_id, info in IRONCLAD_CARDS.items()
        if card_id.casefold() == normalized
        or card_id.replace("_", " ").casefold() == normalized
        or info.name_zh.casefold() == normalized
    ]
    if not matches:
        raise ValueError(
            f"unknown Ironclad card {value!r}; use a canonical id such as "
            "Dropkick, Bash, or Strike_R"
        )
    return matches[0]


def _number_after(tokens: Sequence[str], marker: str) -> int | None:
    try:
        position = tokens.index(marker) + 1
    except ValueError:
        return None
    if position >= len(tokens) or tokens[position] != "NUM_BEGIN":
        return None
    digits = []
    for token in tokens[position + 1:]:
        if token == "NUM_END":
            break
        if not token.startswith("DIGIT_"):
            return None
        digits.append(token.removeprefix("DIGIT_"))
    return int("".join(digits)) if digits else None


def direct_attack_formula(
    config: SandboxConfig,
    before: dict[str, Any],
    after: dict[str, Any],
    effects: Sequence[dict[str, Any]],
) -> list[str]:
    """Explain the ordinary direct-attack arithmetic when the DSL exposes it."""
    info = IRONCLAD_CARDS[config.card_id]
    if info.kind != ATTACK:
        return []
    program = list(card_program(config.card_id, config.upgraded))
    if any(marker in program for marker in ("PLUS", "PER", "TIMES", "RANDOM_ENEMY")):
        return ["该牌使用动态或多段伤害公式，数值以原子影响记录为准。"]
    base_damage = _number_after(program, "DEAL_DAMAGE")
    if base_damage is None:
        return ["该牌的伤害不是 DSL 中的固定直接伤害，数值以原子影响记录为准。"]

    multiplier = _number_after(program, "SET_STRENGTH_MULTIPLIER") or 1
    strengthened = base_damage + config.player_strength * multiplier
    vulnerable = config.enemy_vulnerable
    calculated = strengthened * 3 // 2 if vulnerable else strengthened
    monster_before = before["monsters"][0]
    monster_after = after["monsters"][0]
    block_loss = sum(
        max(0, -int(effect["delta"]))
        for effect in effects
        if effect["type"] == "MONSTER_VALUE_CHANGE"
        and effect.get("slot") == 0
        and effect.get("field") == "block"
    )
    hp_loss = sum(
        max(0, -int(effect["delta"]))
        for effect in effects
        if effect["type"] == "MONSTER_VALUE_CHANGE"
        and effect.get("slot") == 0
        and effect.get("field") == "hp"
    )

    lines = [f"读取牌面基础伤害：{base_damage}。"]
    if multiplier == 1:
        lines.append(
            f"加入力量：{base_damage} + {config.player_strength} = {strengthened}。"
        )
    else:
        lines.append(
            f"加入 {multiplier} 倍力量：{base_damage} + "
            f"{config.player_strength} x {multiplier} = {strengthened}。"
        )
    if vulnerable:
        lines.append(
            f"目标有 {vulnerable} 层易伤：floor({strengthened} x 3 / 2) "
            f"= {calculated} 点攻击伤害。"
        )
    else:
        lines.append(f"目标无易伤：攻击伤害保持为 {calculated}。")
    lines.extend((
        f"格挡吸收 min({monster_before['block']}, {calculated}) = {block_loss}。",
        f"穿透格挡的生命伤害为 {hp_loss}；敌人生命 "
        f"{monster_before['hp']} -> {monster_after['hp']}。",
    ))
    if block_loss + hp_loss < calculated:
        lines.append("剩余伤害因敌人生命归零而成为过量伤害，不再显示为状态差值。")
    return lines


def explain_effect(effect: dict[str, Any]) -> str:
    event_type = effect["type"]
    atomic = effect.get("atomic")
    prefix = f"[原子步骤 {atomic}] " if atomic is not None else ""
    if event_type == "PLAYER_VALUE_CHANGE":
        field = FIELD_ZH.get(effect["field"], effect["field"])
        return (
            f"{prefix}玩家{field}：{effect['before']} "
            f"{int(effect['delta']):+d} -> {effect['after']}。"
        )
    if event_type == "MONSTER_VALUE_CHANGE":
        field = FIELD_ZH.get(effect["field"], effect["field"])
        return (
            f"{prefix}敌人{field}：{effect['before']} "
            f"{int(effect['delta']):+d} -> {effect['after']}。"
        )
    if event_type == "COUNTER_CHANGE":
        counter = COUNTER_ZH.get(effect["counter"], effect["counter"])
        return (
            f"{prefix}{counter}：{effect['before']} "
            f"{int(effect['delta']):+d} -> {effect['after']}。"
        )
    if event_type == "STATUS_CHANGE":
        owner = "玩家" if effect["owner"] == "player" else "敌人"
        return (
            f"{prefix}{owner}状态 {effect['status']}：{effect['before']} "
            f"{int(effect['delta']):+d} -> {effect['after']}。"
        )
    if event_type == "CARD_MOVE":
        source = ZONE_ZH.get(effect["source"], effect["source"])
        destination = ZONE_ZH.get(effect["destination"], effect["destination"])
        return f"{prefix}{effect['card']}：从{source}移动到{destination}。"
    if event_type == "CARD_REMOVE":
        source = ZONE_ZH.get(effect["source"], effect["source"])
        return f"{prefix}{effect['card']}：从{source}移出，进入出牌结算。"
    if event_type == "CARD_ADD":
        destination = ZONE_ZH.get(effect["destination"], effect["destination"])
        return f"{prefix}{effect['card']}：加入{destination}。"
    if event_type == "MONSTER_MOVE_CHANGE":
        return f"{prefix}敌人意图：{effect['before']} -> {effect['after']}。"
    if event_type == "MONSTER_GONE_CHANGE":
        return f"{prefix}敌人离场状态：{effect['before']} -> {effect['after']}。"
    if event_type == "TURN_CHANGE":
        return f"{prefix}回合：{effect['before']} -> {effect['after']}。"
    if event_type == "ACTION_COMPLETE":
        return f"动作结束，共 {effect['atomic_steps']} 个产生状态变化的原子步骤。"
    return f"{prefix}{event_type}：{effect}"


def _first_mismatch(expected: Sequence[str], actual: Sequence[str]) -> int | None:
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def _section_at(tokens: Sequence[str], position: int) -> str:
    sections = (
        ("CALCULATIONS", CALCULATIONS_BEGIN, CALCULATIONS_END),
        ("RULE_STEPS", "<RULE_STEPS_BEGIN>", "<RULE_STEPS_END>"),
        ("EFFECTS", EFFECTS_BEGIN, EFFECTS_END),
        ("NEXT_STATE", NEXT_STATE_BEGIN, NEXT_STATE_END),
    )
    for name, begin, end in sections:
        cursor = 0
        while cursor < len(tokens):
            try:
                left = tokens.index(begin, cursor)
                right = tokens.index(end, left)
            except ValueError:
                break
            if left <= position <= right:
                return name
            cursor = right + 1
    return "OUTPUT"


def _mismatch_record(
    expected: Sequence[str], actual: Sequence[str], position: int | None,
    *, window: int = 8,
) -> dict[str, Any]:
    if position is None:
        return {
            "position": None,
            "section": None,
            "expected": None,
            "actual": None,
            "expected_length": len(expected),
            "actual_length": len(actual),
        }
    lo = max(0, position - window)
    hi = position + window + 1
    return {
        "position": position,
        "section": _section_at(expected, position),
        "expected": expected[position] if position < len(expected) else "<MISSING>",
        "actual": actual[position] if position < len(actual) else "<MISSING>",
        "expected_length": len(expected),
        "actual_length": len(actual),
        "expected_window": list(expected[lo:hi]),
        "actual_window": list(actual[lo:hi]),
    }


def _load_compatible_tokenizer(path: Path) -> RuleTokenizer:
    """Load current or legacy rule vocabularies without changing their ids."""
    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("version", -1))
    if version not in (1, 2, 3, 4, 5):
        raise ValueError(f"unsupported rule tokenizer version {version} in {path}")
    return RuleTokenizer(tuple(data["tokens"]))


def _strip_unsupported_target_sections(
    target: Sequence[str], tokenizer: RuleTokenizer,
) -> list[str]:
    """Adapt a current target to a legacy model's section vocabulary."""
    result = list(target)
    unsupported = []
    if "<RULE_STEPS_BEGIN>" not in tokenizer.index:
        unsupported.append(("<RULE_STEPS_BEGIN>", "<RULE_STEPS_END>"))
    elif "STEP_BEGIN" not in tokenizer.index:
        # Same compatibility pattern as the old empty calculations section.
        unsupported.append(("<RULE_STEPS_BEGIN>", "<RULE_STEPS_END>"))
    # Early rule checkpoints reserved the calculations markers but trained no
    # calculation records. Detect that schema from the vocabulary itself; keep
    # the empty markers because those are part of the old decoder contract.
    if "CALCULATION_BEGIN" not in tokenizer.index:
        unsupported.append((CALCULATIONS_BEGIN, CALCULATIONS_END))
    for begin_marker, end_marker in unsupported:
        preserve_empty = (
            begin_marker in tokenizer.index
            and (
                begin_marker == CALCULATIONS_BEGIN
                or begin_marker == "<RULE_STEPS_BEGIN>"
            )
        )
        cursor = 0
        while cursor < len(result):
            try:
                left = result.index(begin_marker, cursor)
                right = result.index(end_marker, left) + 1
            except ValueError:
                break
            if preserve_empty:
                del result[left + 1:right - 1]
                cursor = left + 2
            else:
                del result[left:right]
                cursor = left
    return result


def compare_model_outputs(
    sample: dict[str, Any], run: Path, checkpoint_name: str = "best.pt",
    *, device_name: str = "cpu", max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Run teacher-forced and autoregressive decoding on one simulated sample.

    ``teacher_target`` is the simulator-produced target.  ``teacher_forced``
    contains the model's argmax at each position while feeding the gold prefix;
    ``autoregressive`` feeds back its own previous argmax.  Keeping all three
    makes exposure-bias errors distinguishable from ordinary per-token errors.
    """
    try:
        import torch
        from rl.rules.rule_model import RuleWorldModel
    except ImportError as exc:
        raise RuntimeError(
            "model comparison requires PyTorch; use the Python environment used "
            "to train the rule checkpoint"
        ) from exc

    run = Path(run)
    tokenizer_path = run / "tokenizer.json"
    tokenizer_version = int(json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    ).get("version", -1))
    tokenizer = _load_compatible_tokenizer(tokenizer_path)
    model = RuleWorldModel.from_config(run / "model_config.json")
    checkpoint_path = run / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"rule checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(device_name)
    model.to(device).eval()

    prefix, full_target = sample_tokens(
        sample, output_version=tokenizer_version,
    )
    target = _strip_unsupported_target_sections(full_target, tokenizer)
    prefix_ids = tokenizer.encode(prefix)
    target_ids = tokenizer.encode(target)
    unknown_id = tokenizer.index["<UNK>"]
    unknowns = [
        ("prefix", index, token)
        for index, (token, token_id) in enumerate(zip(prefix, prefix_ids))
        if token_id == unknown_id
    ] + [
        ("target", index, token)
        for index, (token, token_id) in enumerate(zip(target, target_ids))
        if token_id == unknown_id
    ]
    if unknowns:
        preview = ", ".join(f"{part}[{index}]={token}" for part, index, token in unknowns[:8])
        raise ValueError(
            f"checkpoint tokenizer cannot represent the constructed sample: {preview}"
        )

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        memory, padding = model.encode(prefix_tensor)
        teacher_logits = model.decode(target_tensor[:, :-1], memory, padding)
        teacher_next = teacher_logits.argmax(dim=-1)[0].tolist()
        teacher_forced_ids = [target_ids[0], *teacher_next]
        generated_ids = model.generate(
            prefix_tensor,
            output_begin_id=tokenizer.index[OUTPUT_BEGIN],
            output_end_id=tokenizer.index[OUTPUT_END],
            max_new_tokens=max_new_tokens,
        )[0].tolist()

    teacher_target = target
    teacher_forced = tokenizer.decode(teacher_forced_ids)
    autoregressive = tokenizer.decode(generated_ids)
    teacher_forced = teacher_forced[:teacher_forced.index(OUTPUT_END) + 1] \
        if OUTPUT_END in teacher_forced else teacher_forced
    autoregressive = autoregressive[:autoregressive.index(OUTPUT_END) + 1] \
        if OUTPUT_END in autoregressive else autoregressive
    tf_mismatch = _first_mismatch(teacher_target, teacher_forced)
    ar_mismatch = _first_mismatch(teacher_target, autoregressive)
    cross_mismatch = _first_mismatch(teacher_forced, autoregressive)
    return {
        "run": str(run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "teacher_target": teacher_target,
        "teacher_forced": teacher_forced,
        "autoregressive": autoregressive,
        "teacher_forced_first_mismatch": _mismatch_record(
            teacher_target, teacher_forced, tf_mismatch,
        ),
        "autoregressive_first_mismatch": _mismatch_record(
            teacher_target, autoregressive, ar_mismatch,
        ),
        "autoregressive_vs_teacher_forced_first_mismatch": _mismatch_record(
            teacher_forced, autoregressive, cross_mismatch,
        ),
    }


def _format_sequence_window(name: str, tokens: Sequence[str], position: int | None) -> list[str]:
    if position is None:
        return [f"{name}：与对照序列完全一致。"]
    lo = max(0, position - 8)
    hi = min(len(tokens), position + 9)
    rows = [f"{name}（位置 {position}）:"]
    for index in range(lo, hi):
        marker = "  <<<" if index == position else ""
        rows.append(f"  {index:04d}: {tokens[index]}{marker}")
    return rows


def _format_sync_window(comparison: dict[str, Any]) -> list[str]:
    target = comparison["teacher_target"]
    teacher_forced = comparison["teacher_forced"]
    autoregressive = comparison["autoregressive"]
    positions = [
        item["position"]
        for key in (
            "teacher_forced_first_mismatch",
            "autoregressive_first_mismatch",
            "autoregressive_vs_teacher_forced_first_mismatch",
        )
        if (item := comparison[key])["position"] is not None
    ]
    if not positions:
        return ["三条序列完全一致。"]
    center = min(positions)
    lo = max(0, center - 8)
    hi = min(max(len(target), len(teacher_forced), len(autoregressive)), center + 17)
    lines = ["位置 | 教师目标 | 教师强制 | 自回归 | 状态"]
    for index in range(lo, hi):
        want = target[index] if index < len(target) else "<MISSING>"
        forced = teacher_forced[index] if index < len(teacher_forced) else "<MISSING>"
        generated = autoregressive[index] if index < len(autoregressive) else "<MISSING>"
        states = []
        if forced != want:
            states.append("TF!=目标")
        if generated != want:
            states.append("AR!=目标")
        if generated != forced:
            states.append("AR!=TF")
        marker = " <<<" if states else ""
        lines.append(
            f"{index:04d} | {want} | {forced} | {generated} | "
            f"{','.join(states) if states else '一致'}{marker}"
        )
    return lines


def format_model_comparison(comparison: dict[str, Any]) -> str:
    """Render synchronized teacher/autoregressive output and first divergences."""
    target = comparison["teacher_target"]
    teacher_forced = comparison["teacher_forced"]
    autoregressive = comparison["autoregressive"]
    tf = comparison["teacher_forced_first_mismatch"]
    ar = comparison["autoregressive_first_mismatch"]
    cross = comparison["autoregressive_vs_teacher_forced_first_mismatch"]
    lines = [
        "=== 模型教师强制 / 自回归同步对照 ===",
        f"checkpoint：{comparison['checkpoint']}（epoch {comparison['checkpoint_epoch']}）",
        "说明：教师目标是真实模拟器词元；教师强制模型每一步读取真实前缀；"
        "自回归模型每一步读取自己的前一词元。",
        "",
        "[教师目标]",
        " ".join(target),
        "",
        "[教师强制模型输出]",
        " ".join(teacher_forced),
        "",
        "[自回归模型输出]",
        " ".join(autoregressive),
        "",
        "[首个分歧]",
        (
            f"教师强制 vs 教师目标：位置 {tf['position']}，段落 {tf['section']}，"
            f"期望 {tf['expected']!r}，模型 {tf['actual']!r}。"
            if tf["position"] is not None
            else "教师强制 vs 教师目标：完全一致。"
        ),
        (
            f"自回归 vs 教师目标：位置 {ar['position']}，段落 {ar['section']}，"
            f"期望 {ar['expected']!r}，模型 {ar['actual']!r}。"
            if ar["position"] is not None
            else "自回归 vs 教师目标：完全一致。"
        ),
        (
            f"自回归 vs 教师强制：位置 {cross['position']}，段落 {cross['section']}，"
            f"教师强制 {cross['expected']!r}，自回归 {cross['actual']!r}。"
            if cross["position"] is not None
            else "自回归 vs 教师强制：完全一致。"
        ),
        "",
    ]
    lines.extend(("", "[逐位置同步窗口]"))
    lines.extend(_format_sync_window(comparison))
    lines.extend(("", "[自回归分歧窗口]"))
    lines.extend(_format_sequence_window("教师目标", target, ar["position"]))
    lines.extend(_format_sequence_window("自回归模型", autoregressive, ar["position"]))
    return "\n".join(lines)


def simulate_card_play(config: SandboxConfig) -> dict[str, Any]:
    _configure_sim_build()
    from env.core import _BUILD  # noqa: F401
    import slaythespire as sts
    from schemas.combat_state import (
        normalize_sim,
        normalize_sim_global,
        real_to_sim_card_enum,
    )

    if not hasattr(sts.BattleContext, "execute_with_trace"):
        raise RuntimeError(
            "the selected slaythespire build has no rule tracing; build with "
            "-DSTS_ENABLE_RULE_TRACE=ON or set STS_LIGHTSPEED_BUILD_DIR"
        )
    if (
        config.player_energy < 0
        or config.player_strength < 0
        or config.player_block < 0
    ):
        raise ValueError(
            "player energy, Strength, and Block must be non-negative"
        )
    if config.enemy_block < 0 or config.enemy_vulnerable < 0:
        raise ValueError("enemy Block and Vulnerable must be non-negative")

    if config.floor < 1:
        raise ValueError("floor must be positive")
    if config.opening_hand_attempts < 1:
        raise ValueError("opening_hand_attempts must be positive")

    canonical_deck = config.deck or (
        (config.card_id, int(config.upgraded)),
    ) * 10
    sim_deck = [
        (real_to_sim_card_enum(card_id), int(upgrades))
        for card_id, upgrades in canonical_deck
    ]
    gc = bc = initial = source = None
    actual_seed = config.seed
    for attempt in range(config.opening_hand_attempts):
        actual_seed = config.seed + attempt
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, actual_seed, 0)
        gc.floor_num = config.floor
        gc.act = 1 if config.floor <= 16 else 2 if config.floor <= 33 else 3
        gc.set_deck(sim_deck)
        gc.set_relics([])
        bc = sts.create_battle(gc, config.enemy)
        if hasattr(bc, "set_player_energy"):
            bc.set_player_energy(config.player_energy)
        initial = normalize_sim(bc.to_dict())
        source = next(
            (
                index for index, card in enumerate(initial["hand"])
                if card["id"] == config.card_id
                and int(card.get("up", 0)) == int(config.upgraded)
            ),
            None,
        )
        playable = source is not None and any(
            str(action.action_type).split(".")[-1] == "CARD"
            and int(action.source_idx) == source
            for action in bc.rule_actions()
        )
        if playable:
            break
        source = None
    if source is None:
        raise OpeningHandError(
            f"{config.card_id} {'+' if config.upgraded else 'base'} did not "
            f"reach a playable opening-hand state after "
            f"{config.opening_hand_attempts} deterministic shuffle attempts"
        )
    assert gc is not None and bc is not None and initial is not None
    if len(initial["monsters"]) != 1:
        raise ValueError(
            f"encounter {config.enemy!r} created {len(initial['monsters'])} enemies; "
            "choose a single-enemy encounter"
        )

    if not hasattr(bc, "set_player_energy"):
        if config.player_energy != int(initial["player"]["energy"]):
            raise RuntimeError(
                "the selected rule-trace build lacks set_player_energy; rebuild it "
                "or use the combat's initial Energy"
            )
    else:
        bc.set_player_energy(config.player_energy)
    if not hasattr(bc, "set_player_block"):
        if config.player_block != int(initial["player"]["block"]):
            raise RuntimeError(
                "the selected rule-trace build lacks set_player_block; rebuild it "
                "or use the combat's initial Block"
            )
    else:
        bc.set_player_block(config.player_block)
    bc.set_player_strength(config.player_strength)
    bc.set_monster_block(0, config.enemy_block)
    bc.set_monster_vulnerable(0, config.enemy_vulnerable)

    before = normalize_sim(bc.to_dict())
    source = next(
        (
            index for index, card in enumerate(before["hand"])
            if card["id"] == config.card_id
            and int(card.get("up", 0)) == int(config.upgraded)
        ),
        None,
    )
    if source is None:
        raise RuntimeError(
            f"the constructed opening hand did not contain the requested "
            f"{config.card_id} version"
        )
    actions = [
        action for action in bc.rule_actions()
        if str(action.action_type).split(".")[-1] == "CARD"
        and int(action.source_idx) == source
    ]
    if not actions:
        raise RuntimeError(
            f"{config.card_id} is not playable in the constructed state "
            f"(cost={before['hand'][source].get('cost')}, "
            f"energy={before['player']['energy']})"
        )
    action = actions[0]
    action_record = {
        "type": "CARD",
        "source": source,
        "target": int(action.target_idx),
        "card": dict(before["hand"][source]),
    }
    branch = bc.clone()
    trace = branch.execute_with_trace(action)
    snapshots = [normalize_sim(raw) for raw in (trace["before"], *trace["steps"])]
    after = normalize_sim(trace["after"])
    effects = events_from_snapshots(snapshots)
    global_state = normalize_sim_global(gc)
    global_state.update({
        "hp": before["player"]["hp"],
        "max_hp": before["player"]["max_hp"],
        "gold": before["player"].get("gold", global_state["gold"]),
    })
    before_state = {"combat": before, "global": dict(global_state)}
    after_global = dict(global_state)
    after_global.update({
        "hp": after["player"]["hp"],
        "max_hp": after["player"]["max_hp"],
        "gold": after["player"].get("gold", after_global["gold"]),
    })
    after_state = {"combat": after, "global": after_global}
    sample = {
        "state": before_state,
        "action": action_record,
        "rules": {
            "deck": deck_rule_sentence(before_state["global"]["deck"]),
            "relics": relic_loadout_sentence(before_state["global"]["relics"]),
        },
        "effects": effects,
        "next_state": after_state,
    }
    calculations = derive_calculations(sample)
    local_calculations = derive_local_calculations(sample)
    rule_steps = derive_rule_steps(sample)
    return {
        "config": config,
        "actual_seed": actual_seed,
        "deck": list(canonical_deck),
        "before": before,
        "after": after,
        "sample": sample,
        "action": action_record,
        "effects": effects,
        "calculations": calculations,
        "local_calculations": local_calculations,
        "rule_steps": rule_steps,
    }


def build_report(
    result: dict[str, Any], *, include_process_tokens: bool = True,
    model_comparison: dict[str, Any] | None = None,
) -> str:
    config: SandboxConfig = result["config"]
    info = IRONCLAD_CARDS[config.card_id]
    before = result["before"]
    after = result["after"]
    monster_before = before["monsters"][0]
    monster_after = after["monsters"][0]
    version = "升级版" if config.upgraded else "基础版"
    rule = card_rule_sentence(config.card_id, int(config.upgraded))

    lines = [
        "=== 单卡单怪结算解释 ===",
        f"卡牌：{info.name_zh} / {config.card_id}（{version}）",
        f"敌人：{monster_before['id']}",
        (
            f"初始：玩家能量 {before['player']['energy']}、力量 "
            f"{config.player_strength}；敌人生命 {monster_before['hp']}、格挡 "
            f"{monster_before['block']}、易伤 {config.enemy_vulnerable}。"
        ),
        "",
        "=== 原始规则词元 ===",
        rule,
        "",
        "=== 规则自然语言 ===",
        info.effect_zh,
        "",
        "=== 实际计算 ===",
    ]
    cost = int(result["action"]["card"].get("cost", 0))
    energy_changes = [
        effect for effect in result["effects"]
        if effect["type"] == "PLAYER_VALUE_CHANGE" and effect.get("field") == "energy"
    ]
    if energy_changes:
        paid = energy_changes[0]
        lines.append(
            f"1. 支付费用 {cost}：能量 {paid['before']} -> {paid['after']}。"
        )
    else:
        lines.append(f"1. 本次没有可见的能量变化（当前牌面费用 {cost}）。")
    formula = direct_attack_formula(
        config, before, after, result["effects"]
    )
    lines.extend(f"{index}. {text}" for index, text in enumerate(formula, start=2))
    next_index = 2 + len(formula)
    program = set(card_program(config.card_id, config.upgraded))
    if "TARGET_HAS_VULNERABLE" in program:
        passed = config.enemy_vulnerable > 0
        lines.append(
            f"{next_index}. 条件 TARGET_HAS_VULNERABLE："
            f"{config.enemy_vulnerable} > 0，为{'真' if passed else '假'}。"
        )
        next_index += 1
        if passed:
            gain = next((e for e in energy_changes if int(e["delta"]) > 0), None)
            if gain is not None:
                lines.append(
                    f"{next_index}. 条件奖励回能 1：能量 "
                    f"{gain['before']} -> {gain['after']}；随后抽 1 张牌。"
                )
            else:
                lines.append(f"{next_index}. 条件成立；随后执行回能和抽牌分支。")
            next_index += 1
    lines.extend((
        f"{next_index}. 结算后：玩家能量 {after['player']['energy']}；敌人生命 "
        f"{monster_after['hp']}、格挡 {monster_after['block']}。",
        "",
        "=== 原子影响（模拟器真实 trace） ===",
    ))
    lines.extend(explain_effect(effect) for effect in result["effects"])

    if include_process_tokens:
        rule_steps = result.get("rule_steps")
        if rule_steps is not None and len(rule_steps) != len(result["effects"]):
            rule_steps = None
        local_calculations = result.get("local_calculations")
        if (
            local_calculations is None
            or len(local_calculations) != len(result["effects"])
        ):
            local_calculations = [[] for _effect in result["effects"]]
        lines.extend((
            "",
            "=== 当前规则训练实际序列化的过程词元 ===",
            "[EFFECTS_WITH_LOCAL_CALCULATIONS]",
            " ".join(serialize_effects(
                result["effects"], rule_steps=rule_steps,
                local_calculations=local_calculations,
                least_significant_first=True,
                compact=True,
            )),
        ))
    if model_comparison is not None:
        lines.extend(("", format_model_comparison(model_comparison)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", default="Dropkick", help="canonical card id")
    parser.add_argument("--upgraded", action="store_true")
    parser.add_argument("--enemy", default="Cultist", help="single-enemy encounter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--player-energy", type=int, default=3)
    parser.add_argument("--player-strength", type=int, default=0)
    parser.add_argument("--player-block", type=int, default=0)
    parser.add_argument("--enemy-block", type=int, default=0)
    parser.add_argument("--enemy-vulnerable", type=int, default=0)
    parser.add_argument(
        "--no-process-tokens", action="store_true",
        help="omit raw event-local calculation/effect tokens",
    )
    parser.add_argument(
        "--run", type=Path,
        help="rule-model run directory containing tokenizer.json/model_config.json",
    )
    parser.add_argument(
        "--checkpoint", default="best.pt",
        help="checkpoint filename inside --run (default: best.pt)",
    )
    parser.add_argument(
        "--model-device", default="cpu",
        help="torch device for model comparison, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--model-max-new-tokens", type=int,
        help="maximum autoregressive tokens; defaults to model capacity",
    )
    parser.add_argument("--out", type=Path, help="write UTF-8 report to this path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    card_id = resolve_card_id(args.card)
    config = SandboxConfig(
        card_id=card_id,
        upgraded=args.upgraded,
        enemy=args.enemy,
        seed=args.seed,
        player_energy=args.player_energy,
        player_strength=args.player_strength,
        player_block=args.player_block,
        enemy_block=args.enemy_block,
        enemy_vulnerable=args.enemy_vulnerable,
    )
    result = simulate_card_play(config)
    comparison = None
    if args.run is not None:
        comparison = compare_model_outputs(
            result["sample"], args.run, args.checkpoint,
            device_name=args.model_device,
            max_new_tokens=args.model_max_new_tokens,
        )
    report = build_report(
        result,
        include_process_tokens=not args.no_process_tokens,
        model_comparison=comparison,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(args.out.resolve())
    else:
        print(report)


if __name__ == "__main__":
    main()

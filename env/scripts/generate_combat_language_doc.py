from pathlib import Path

from env.core import combat_language as L
from env.core import vocab as V


def main():
    lines = [
        "# Combat Entity Language / 战斗实体语言", "",
        "This file is generated from `env/core/combat_language.py`.",
        "本文由 `env/core/combat_language.py` 生成。", "",
        "Numbers use `NUM_BEGIN`, optional `SIGN_NEGATIVE`, decimal digit tokens, "
        "and `NUM_END`. Runtime-dependent intent amounts use `VALUE_DYNAMIC`; "
        "current attack damage and hit count are supplied from the live state.",
        "数字使用起止 token、可选负号和逐位十进制 token。依赖进阶或内部状态的意图数值用 "
        "`VALUE_DYNAMIC`；当前攻击伤害和段数由实时状态提供。", "",
        "## Status Bar Grammar / 状态栏语法", "", "```text",
        "ENTITY_BEGIN ENTITY_PLAYER|ENTITY_MONSTER",
        "FIELD HP number STMT_END",
        "FIELD MAX_HP number STMT_END",
        "FIELD BLOCK number STMT_END",
        "(STATUS STATUS_<power> number STMT_END)*",
        "ENTITY_END", "```", "",
        "## Token Dictionary / Token 词典", "",
        "| Token family | Meaning | 含义 |", "|---|---|---|",
        "| `MOVE_*` | Exact simulator move ID | 精确的模拟器招式 ID |",
        "| `INTENT_KIND_*` | Coarse intent category | 意图大类 |",
        "| `DO ... STMT_END` | One effect statement | 一条效果语句 |",
        "| `DEAL_DAMAGE n TARGET_PLAYER TIMES k` | Deal adjusted damage n, k times | 对玩家造成 n 点调整后伤害，共 k 段 |",
        "| `APPLY/GAIN/REDUCE/HEAL` | Change a status, resource, or HP | 改变状态、资源或生命 |",
        "| `VALUE_DYNAMIC` | Amount resolved from ascension/internal state | 由进阶等级或内部状态决定的数值 |",
        "| `NUM_BEGIN DIGIT_* NUM_END` | Digit-level integer | 逐位编码的整数 |",
        "| `STATUS_*` | Exact active power ID | 精确的已有状态 ID |", "",
        "## Monster Intents / 怪物意图", "",
        "| Move | Attack | Program |", "|---|---:|---|",
    ]
    for move in V.MOVE_VOCAB[1:]:
        lines.append(f"| `{move}` | {'是' if move in L.ATTACK_MOVES else '否'} | `{' '.join(L.intent_program(move))}` |")
    lines += ["", "## Status Tokens / 状态 Token", ""]
    for power in sorted(set(V.PLAYER_BUFF_VOCAB) | set(V.MONSTER_BUFF_VOCAB)):
        lines.append(f"- `{power}` -> `STATUS_{power}`")
    out = Path("docs/archive/reference/combat_entity_language.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

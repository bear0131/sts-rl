# Akabeko/Vigor 数值不归零 —— obs 失真 bug

发现于 2026-06-26（写了 `rl/play_gauntlet.py` 手玩车轮战时注意到的）。

## 现象
开局抽到 Akabeko 的 loadout，`Vigor 8` 这个 buff 在打出攻击牌后仍显示 8，不消失。

## 定位（已用 probe 验证）
- **机制是对的**：第一刀 STRIKE 打 14 伤害（6 基础 + 8 Vigor），第二刀只打 6 —— Vigor 确实只加成了第一刀、已被消耗。
- **坏的是序列化**：C++ sim 的 `bc.to_dict()` 里 `player.powers` 的 Vigor amount 永远停在 8，不归零。
- 不是 render / thin-env 的问题：`game_env.render()` 每帧重读 `normalize_sim(bc.to_dict())`，无缓存。
- **危害**：假数值进了模型 obs —— `encode.py:251` 把 player powers 写进 `player_buffs`，Vigor 在 `PLAYER_BUFF_VOCAB_INDEX` 占 idx 81。所以 combat 预训练里只要 loadout 有 Akabeko，模型整局都看到一个永不消失的 +8 Vigor 幻象。
- 玩家 debuff（虚弱等）衰减/消失是正常的：单次施加的 Weak 1 下个玩家回合就归零；之前以为"卡住"的是多怪场景每回合重新施加（如 Gremlin Gang 胖地精）的错觉。

## 修复选项
- **A（止血，1 行）**：`bridge/schemas/combat_state.py:32` 把 `"Vigor"` 加进 `_POWER_DROP`，render+obs 都不再显示。代价：模型也看不到"第一刀 +8"这个真实信息，但好过看到错的常驻 8。
- **B（根治）**：C++ sim 攻击消耗 Vigor 后在 to_dict 里把 amount 清 0。成本高（动 C++ + 重编）。

注意：方案 A 动到 obs 语义，按 OBS_VERSION 纪律可能要 bump 版本 + 报废 ckpt（和 obs-restructure 批次同类）。还没改，待定。

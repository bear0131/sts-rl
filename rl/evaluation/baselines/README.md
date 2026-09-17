# 战斗规则基线

`CombatRuleStrategy` 通过 `DecisionStrategy` 接口决定战斗动作；`PhaseStrategy` 将所有
`combat` 请求（含目标、药水、选牌子步骤）交给它，其余请求仍交给指定的上层 checkpoint。
它是可解释的工程启发式，不代表人类平均水平，也不把分数解释成 Q 或真实终局 HP。

## 支持与信息边界

当前支持 simulator whole-run Ironclad、现有 thin action codec。75 张铁甲战士卡牌复用
`env/core/ironclad_cards.py` 的基础注释，组合价值由本模块手写。常见无色/事件牌有近似规则，
未知牌会记录 `unknown_card`/`rule_unknown_choices`；不宣称全卡牌、遗物或其他角色规则完整。
规则没有网络调用、训练参数、模拟器前瞻、随机搜索或未来敌人动作访问。

只使用当前手牌/费用、HP/格挡/能量、buff、公开敌人意图、可见牌堆内容、药水、遗物与合法动作。
普通抽牌堆进入策略前按牌名/升级排序，不能利用未来抽牌顺序；检索牌堆的选牌屏幕才保留
候选索引以正确提交选择。`Havoc` 按一般收益估计，不查看牌堆顶。原生怪物槽位保留空槽，
避免 canonical 旧路径的槽压缩问题 F01；这不表示全项目 F01 已修复。

`HEADBUTT`/Liquid Memories 用弃牌堆，`EXHUME` 用消耗堆，Secret Weapon/Technique
用公开检索候选，Discovery/Codex 用屏幕生成的选项；不统一当作手牌。新 native build 的
`select_info.generated_cards` 是只读附加字段，仅在生成选牌屏幕读取，不改变神经观测或
动作版本。旧扩展缺少该字段时明确失败。MULTI 由现有 codec 单调选牌并确认。

## 决策方法

每个合法根动作在其合法目标中取最高分；目标子步骤使用显式 pending 来源重新评分，
相同输入以 action ID 打破平局。分项随 `--trace` 写入 JSONL：

- 估计伤害考虑力量、连击、虚弱、易伤、格挡及部分遗物；奖励击杀、取消敌人攻击和整场致死。
  削减敌人格挡也有价值，不能因为一张牌不扣 HP 就停止攻击。
- 只重点奖励覆盖当前预计伤害的格挡；低血致死风险提高防御权重，Barricade 允许持久格挡收益。
- 对抽牌、能量、易伤/虚弱、力量、消耗、回血、能力牌和可立即利用的组合加入分项。
  Flex 等免费铺垫尽量先于受益攻击，花费 HP 的效果计入代价。
- 药水有保留成本，紧急情况降低成本。选牌按所属区域和任务区分保留、消耗、升级、回收；
  End Turn/Confirm 为显式候选。

这些是近似值，不是第二套完整 simulator：例如动态 Rampage 数值、随机连击分配、飞行中途
解除、完整遗物/药水相互作用、无限循环和多回合资源规划尚未精确建模。评测中的差异应从
trace 对照真实动作/结果验证，不能直接把分项当作游戏规则证明。

## 可复现对照

从仓库根目录，使用 tests/README.md 的环境和显式 `STS_LIGHTSPEED_BUILD_DIR`。
位置参数始终是上层；`--combat-checkpoint` 指定下层，仅在 combat 阶段使用。

```bash
python -m rl.evaluation.eval UPPER.zip --solver rules --episodes 16 --seed 20000 --device cpu --num-envs 1 --max-steps 6000 --timeout 600 --output runs/local/rules-holdout.json
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver policy --episodes 16 --seed 20000 --device cpu --num-envs 1 --max-steps 6000 --timeout 600 --output runs/local/policy-holdout.json
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver native --search-mode strict --simulations 128 --depth 0 --episodes 16 --seed 20000 --device cpu --num-envs 1 --max-steps 6000 --timeout 3600 --output runs/local/search-holdout.json
```

固定上层权重并不固定其之后的选择：战斗造成不同 HP/药水/奖励状态，上层可能改变路线和抓牌。
同 seed 整局比较因此衡量整个组合效果。为隔离战斗，沿明确的参考策略轨迹抽取前 N 个战斗开局，
从**同一完整初态和 RNG** 分别 fork 给规则、下层直接 policy 和 native 搜索。fork/推进只在
评测层，规则本身从不持有分支。现有 `runner` 负责三者步数、超时和结果，未复制 episode loop。

```bash
python -m rl.evaluation.eval UPPER.zip --combat-checkpoint COMBAT.zip --solver rules --episodes 2 --seed 10000 --compare-combats 12 --search-mode strict --simulations 128 --depth 0 --device cpu --max-steps 6000 --timeout 1800 --trace runs/local/paired-development.jsonl --output runs/local/paired-development.json
```

这里 `--solver rules` 是参考轨迹；用 policy/native 可采集另一方访问的状态。参考选择必须写入
报告，它会影响样本分布。每场各策略拿到同一隐藏随机状态，但行动不同可消耗不同随机数，
因此不是相同未来抽牌序列保证。当前复现方式是固定 checkpoint/native hash、seed、参考策略、
配置和版本重新运行；JSONL 保存可见状态和动作，不是可恢复完整 RNG 的跨版本快照语料。

JSON 记录上下层/native 文件哈希、逐局结果和 `combat_rows`，包括初始/结束 HP、药水、
搜索量和用时；`combat_comparison` 在双方都赢的配对战斗上统计规则减对手的损血差，同时单列
双方胜场。最终 HP 是**胜利结算回血前**的战斗 HP，不与整局回血后的 HP 混用。
`step_limit`/超时必须单列，未完成组不能伪装成完整样本。CLI 超时是合作式；正式批次还需
外部进程监督硬超时。trace 拒绝覆盖现有文件。

开发种子固定为 10000–10003；留出从 20000 开始。只针对开发 bad case 调整规则，冻结版本后
跑留出；若查看留出并修改，就把它标为新的开发集并另选留出。报告样本数和不确定性，不能从
几局/单个怪物声称总体胜率优越，也不能拿 W&B 滚动训练指标代替固定 checkpoint 评测。

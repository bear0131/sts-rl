# NOSL 出牌顺序敏感性测试

使用 checkpoint `artifacts/latest-nosl-az-s512-20260905/nosl_az_614400_steps.zip`，
并使用最新 `sim/sts_lightspeed/build-bindings-check` C++ 绑定。搜索是训练同款的
C++ NOSL-IS-AZ：根部输入当前模型 policy logits，叶子使用当前模型 value/policy，
不使用 Dirichlet root noise（对应目标参数 `random0`）。原始数据见
`.tmp-nosl-order-sensitivity-614400-latestbuild.json`。另有两组先验消融：
`.tmp-nosl-order-sensitivity-614400-uniform-allnodes.json` 和
`.tmp-nosl-order-sensitivity-614400-none-allnodes.json`。

## 场景

- 遭遇：Cultist，floor 1，seed 36，玩家满血。
- 两个 layout 只交换目标牌在牌堆中的位置，因此抽牌堆组成、RNG、敌人状态相同。
- 初始手牌均包含目标牌、另一张目标牌、1 张 Defend、Warcry、Shrug It Off。
- `bash_plus_strike`：Bash+ + Strike。
- `armaments_plus_strike`：Armaments+ + Strike。
- 牌堆剩余 15 张，包含 Iron Wave、Flex、True Grit、Headbutt、Thunderclap、
  Twin Strike、Cleave、Pommel Strike、Defend 等多种正常牌；下回合抽牌不是唯一组合，
  而且不同组合会产生不同的伤害/防御潜力。

每种 layout 测试 `depth=1,2,3,4` 且 `simulations=128`，以及
`depth=0`（无限展开）且 `simulations=512`。

## 结果摘要

| 场景 | depth | layout | 模型目标动作 | 搜索目标动作 | 目标牌访问 | Strike 访问 | 根 value |
|---|---:|---|---|---|---:|---:|---:|
| Bash+ / Strike | 1-4 | 两种 | Bash+ | Bash+ | 128/128 | 0/0 | 0.2584-0.2594 |
| Bash+ / Strike | 0 | 两种 | Bash+ | Bash+ | 512/512 | 0/0 | 0.2442-0.2444 |
| Armaments+ / Strike | 1-4 | base | Strike | Strike | 4 | 124 | 0.2538-0.2543 |
| Armaments+ / Strike | 1-4 | swapped | Strike | Strike | 2 | 126 | 0.2537-0.2542 |
| Armaments+ / Strike | 0 | base | Strike | Strike | 17 | 488 | 0.2509 |
| Armaments+ / Strike | 0 | swapped | Strike | Strike | 10 | 497 | 0.2506 |

这里“目标牌访问”在第一行是 Bash+，在后两行是 Armaments+；`Strike 访问`只统计
Strike 这个语义动作。换位后的 slot 编号会变化（例如 Bash+@2 -> Bash+@3），
但语义动作和最终搜索选择不变。

## 解释

1. **没有观察到明显的手牌槽位敏感性。** base 与 swapped 的语义选择完全一致，根
   value 也只在 1e-4 量级变化。因此当前模型/搜索对这两个牌槽交换是稳定的。
2. **搜索没有真正改变强先验。** Bash+ 的模型根概率约 99.9986%，所以所有 simulation
   都走 Bash+；Armaments+ 的根概率只有 2.0%-3.6%，因此 128 次中只得到 2-4 次
   Armaments+。即使 depth=0、512 次，也只有 10-17 次。这个结果说明当前 PUCT 的
   访问分布主要由 policy prior 决定，不能据此断言 Armaments+ 的长期 HP 价值更低。
3. **depth 边界在本实验中被根先验遮蔽。** 虽然 `depth=1/2/3/4` 和 `depth=0` 都执行
   了对应搜索配置，但没有足够的 Strike/Armaments 分支访问，因而看不到 endturn 后
   敌人行动、洗牌和下回合抽牌对 Q 的影响。要专门验证 endturn 截断，需要增加均匀或
   温度化 root prior 的对照，或者构造模型对两动作概率接近的场景；否则深度比较会被
   先验锁死。
4. **当前代码的 `depth=0` 确实走无限节点展开**（直到模拟次数耗尽或战斗终止），而
   不是退化成 `depth=1`。本次 512 simulation 的叶评估数也是 512。

## 结论

在目标 checkpoint 上，交换 Bash+/Strike 或 Armaments+/Strike 的手牌槽位不会导致
语义决策改变；但是模型对 Armaments+ 的低先验和对 Bash+ 的极强先验，使搜索几乎
没有探索第二张牌，更无法在当前矩阵中测出 endturn/下回合抽牌的真实长期奖励差异。
下一步应增加一个“同一模型 value、均匀 root prior”的诊断组，或寻找 Armaments+ 与
Strike 根先验接近的状态，再评估 depth=2/3/4 是否跨过 endturn 后产生不同 Q。

## 关闭 prior 的追加测试

这里区分两个含义：

- `uniform`：根节点和所有叶节点都不使用模型 logits，合法动作使用均匀先验，
  但保留 `c_puct=1.5`。这是有效的“无模型 prior”对照。
- `none`：均匀先验并设置 `c_puct=0`，尝试完全移除 PUCT prior 项。

`uniform` 的结果如下：

- Bash+ / Strike：128 次搜索大致分配为 Bash+ 25-26、Strike 21、END_TURN 20-21、
  Warcry 20-21；512 次时 Bash+ 100-101、Strike 83-84、END_TURN 77、Warcry 85-87。
- Armaments+ / Strike：128 次时 Armaments+ 22-23、Strike 22、END_TURN 18-19、
  Warcry 21-22；512 次时 Armaments+ 84-85、Strike 85、END_TURN 70-71、Warcry
  96-97。`depth=3` 的 swapped layout 在一次 128-simulation 测试中选择了
  Shrug It Off，说明当价值差距很小时，采样噪声/平局规则会影响最终动作。
- 两种 layout 的访问分布和 root value 基本一致，仍未发现牌槽位导致的语义敏感性。

`none` 的结果完全不同：所有 simulation 都选择 action bits 最小的 Warcry，关键牌、
END_TURN 和其他动作均为 0 次访问。这不是价值网络得出 Warcry 最优，而是搜索实现的
标准退化行为：未访问边的 Q=0；`c_puct=0` 后没有探索项，第一次被选中的边一旦得到
非零 Q，后续就一直重复。为允许这个实验，我修正了 C++ 配置校验，使 `c_puct=0`
不再被强制改回 1.5；但要得到有意义的“无 prior 纯 Q 搜索”，还需要额外加入未访问边
优先展开、progressive widening 或 UCB fallback，不能只把 `c_puct` 设为零。

因此本轮真正有解释价值的是 `uniform`：它移除了模型 policy 的方向性，同时仍保留
必要的树探索机制。它显示 depth=1 到 depth=4 的 root value 变化仍很小（约 1e-3），
但已经能够访问 END_TURN 以及多种后续抽牌分支；相比原始 model-prior 结果，这才是
继续分析 endturn 边界和隐藏抽牌长期 HP 影响的合适基线。

# NOSL value 健康检查与调优基线

记录时间：2026-09-15 09:45 Asia/Shanghai。代码基线 origin/main `cdb3fa3`；
生产部署 `88ab5a9`。这是本次诊断快照，后续进展见
[工作日志](../../worklog/2026-09-15.md)，复用协议见
[评估 README](../../rl/evaluation/README.md#固定血量敏感性协议)。

## 判断

**训练的实际通关表现仍在改善，暂不停止或盲目提高更新次数；value 的低血量高估值得单独修。**
当前 run/W&B ID `fgs128u5replayfresh10hpmon2`，四 rank 保持同一运行，
strict paper / 128 simulations / updates 5 / batch 24576 / replay 131072。
未进行生产代码部署、超参数修改或重启。

## 观测证据

以下对比使用 rank0 保存的全局日志，窗口为连续 collection；HP 坏事件按
`sum(matches)/sum(count)` 重新计算。胜率是各 rank 的滚动训练 episode 统计，
窗口相互重叠，不是固定种子独立评测，也不能据此估计独立样本置信区间。

| 指标 | 基线 442368–552960（10 行） | 约一小时后 860160–983040（11 行） | 当前 3526656–3637248（10 行） |
|---|---:|---:|---:|
| 预测过高坏事件 | 12.93% | 18.70% | 14.33% |
| 第 5 层通过率 | 97.65% | 99.00% | 99.63% |
| 第 10 层通过率 | 78.20% | 91.18% | 96.13% |
| 第 15 层通过率 | 67.28% | 83.98% | 91.27% |
| Act 1 胜率 | 24.85% | 45.39% | 73.75% |
| Act 2 胜率 | 2.15% | 4.16% | 16.67% |
| 平均到达楼层 | 16.90 | 21.49 | 28.07 |
| replay 更新内 value MAE（×200 HP） | 12.50 | 11.88 | 9.19 |

一小时窗口是事后从完整日志回看，并非声称 03:00 做过现场介入决定。
记录显示当时胜率仍在改善，因此不满足“没有继续变好就停止”的整体条件。
最新四机现场核对 step 3637248、全部 PID 正常且身份匹配、无非有限值，
约 110 秒/collection，五次 optimizer step，约 2.4 GiB 设备显存。
最紧张的 rank1 约 10 GiB 空闲磁盘，checkpoint 保留三代，replay 已满。

## 指标为什么抖

1. 它是 `estimated_value > current_hp >= future_hp` 的**坏事件比例**。分母包含
   会结算回血的战斗 root，并非条件在“不回血”上的违规概率。样本混合改变会让
   曲线改变，不能将 1−此值称为模型的血量边界准确率。
2. 只纳入刚完成的真实战斗 roots；长战斗跨 collection 后才进入分母。实际进入的
   怪物、楼层、HP、终局和战斗长度不断变化，同场 roots 也高度相关。
3. 预测取自 collection，W&B 行在随后的 optimizer update 后上报。replay MAE
   是更新内拟合指标，不能与这条新轨迹、更新前统计当成同一数据集直接比较。
4. 原实现平均 rank fraction；正确全局比例应为 sum/sum。修复已加入单元测试
   与真实生产 main → W&B payload → checkpoint → resume 回归，尚未部署。
   历史 matches 和 count 都是 rank mean，相除仍能精确恢复全局比例。
5. 已知 [F05](../2026-09-13/codebase-audit.md#f05) 仍存在：搜索 terminal
   用结算前 HP，learner 用可能带结算回血的 HP。新增 terminal/boundary 样本
   没有自动消除目标定义差异。本次不暗中重定义训练目标。

## 固定 checkpoint 敏感性

比较 checkpoint **552960** 与 **3649536**，使用同一个已保存 corpus，SHA-256
`2324cb135abb258b53225bed56ee27575d5a08b378e7e5fe3bd803e589f044da`。
五类 encounter × 三 seed × root/boundary × player/enemy，合计 60 条曲线，
包含 AcidSlime_L 和 SpikeSlime_L 的自然生成实例。每条在合法数值 HP 范围内
逐整数扫描；仅做 observation 干预，不随 HP 重算 intent/分裂。

| 结果 | 552960 | 3649536 |
|---|---:|---:|
| 我方 HP 局部反向步数 / 2370 | 0 | 0 |
| 敌方 HP 局部反向步数 / 1856 | 0 | 62 |
| 存在局部反向的曲线 / 60 | 0 | 7 |
| 最大局部反向幅度（HP/相邻一步） | 0 | 0.055 |
| 我方低 HP 扫描最大预测高估（HP） | 17.60 | 21.75 |

反向容差为 normalized value `1e-5`（0.002 HP）。当前反向主要见 Cultist
boundary、JawWorm root、AcidSlime_L boundary 和 Shelled Parasite root；
幅度很小，不等于过去 worklog 中 Acid 在 48→34 HP 的明显倒置已经重现。
最大的低血量高估见 SpikeSlime_L root：当前 HP=1 时预测约 22.75 HP。
语料无遗物且牌组不含回血，但这种固定其它条件的低 HP 干预仍可能是训练分布外状态。
它证明的是函数的异常外推形状，不是训练总体发生率，也没有实测真实后续 return。

校验过程中捕获并修正了探针自身的字段错误：`monster_scalars[0]` 是 presence，
不能用于 max HP。初版结果已废弃。正式测试通过 native 重新编码逐字段比对
玩家和敌人干预，并检查 corpus 序列化逐字节一致，避免自写公式互相验证。

## 后续实验顺序与预算

- 继续正常训练和半小时检查；固定 corpus 每次只读新 checkpoint，CPU 单线程，
  60 曲线实测约 9 秒（不含模型载入），不抢生产 GPU。
- 先补 fresh held-out 的校准/血量余量分布，并分开 root、boundary、terminal、
  怪物和回血来源。当前 replay 未保留完整战斗分组身份，随机拆 replay rows
  会发生同场近邻泄漏，不能当独立验证集。
- 真正需要参数实验时，从四 rank 完整保存点做对照。第一组是 updates 5/10/20
  固定 batch；第二组是 5×24576 对 20×6144，固定每轮抽样总量而改变 optimizer
  步数。microbatch 仅控制显存，不等于 effective batch。每组预算和停止条件
  必须写入日志，不直接把 40 次更新当成正确答案。
- 首轮验收应同时看固定轨迹 held-out error、低 HP 上界偏差、policy KL/熵、
  固定 corpus 局部反向幅度与真实环境胜率。仅训练 loss 降低不能接受变更。
- 有证据后再考虑 value 参数化/边界正则或根部搜索参数。全局强制
  `value<=current_hp` 会误伤 Feed、Reaper、遗物等真实回血情形；应先统一
  objective 和可回血上界。不能用硬裁剪掩盖标签不一致。

正式运行若需停止，使用 STOP 等待四 rank 同步 checkpoint，逐文件 SHA 验证，
保留干预前 generation，再以原 ID/replay/Adam/RNG 恢复。活模拟器战斗仍不能
逐位恢复，会重新开始；报告中必须注明。

原始日志、两 checkpoint、corpus 和完整曲线在忽略的 `runs/local/value-health/`。
初始四机完整恢复包另保留在各机当前 run 的 `diagnostics/baseline-552960/rankN`。

# Inner update / batch 对 HP 敏感性的受控对照

结论：`batch6144 / updates40` 是敌方 HP 单调性上最值得继续验证的候选，
但没有同时证明低血量校准更正确、方差更小。保持线上参数，不能把截图里
update40 的胜率优势直接归因于 update 数，也不能把本次筛选当成上线验收。

## 实验身份与限制

- 来源 run/W&B：`fgs128u5replayfresh10hpmon2`，训练副本源 step4018176 rank0。
- 每个分支从同一个 model、Adam、131072-row replay 重新加载；无新采样，均匀 replay。
- CUDA FP32，禁用 TF32，microbatch2048；batch6144/24576；观察 update0/5/10/20/40。
- learner seeds：0、5311、17、29；共8条训练分支、320次 optimizer update。
- audit：step4681728 的后续 rank0 replay；固定 audit seed5311，四个 HP 区间各64个 policy roots。
  只选 sample ID>=909475，排除源 replay 样本。该验证仍有时间/策略分布变化，不是独立游戏 seed 测试。
- 主评测：5种 encounter ×3个 fixture seeds ×root/boundary ×player/enemy，共60条完整 HP 曲线。
  corpus digest `2324cb135abb258b53225bed56ee27575d5a08b378e7e5fe3bd803e589f044da`。
- 另用60个原生致死攻击后的无回血终局，验证 HP1/5/20/80 的精确标签。
  corpus digest `65f718c9fa3cedb236602ab58b4c19a544d932023bb09ad933d9e3822c73b565`。
- 用户提供的 `value-health.zip` 是 step4521984 的评测资料，其 corpus 与上述完全一致。

这里的 batch 是单个实验 learner 的有效 batch。生产为四 rank，各 rank 使用自己的
replay 并汇总梯度，且包含 fresh replay 约束、BF16 和不断变化的采样分布。
因此不能把单 rank 的噪声水平直接等同于同数值的生产 per-rank batch。

## 主要结果

表内为四个 learner seeds 的均值，括号是最小–最大范围，不是置信区间。
所有配置的我方 HP 曲线均保持单调；但单调不代表预测值正确。

| batch / updates | 敌方反向相邻步数 /1856 | 我方最大超出当前 HP | 后续 replay HP1–5 MAE | HP>40 MAE |
| --- | --- | --- | --- | --- |
| 6144 /5 | 168（132–207） | 20.85 HP | 6.14（5.98–6.32） | 9.76（9.32–10.12） |
| 6144 /10 | 258（191–284） | 22.91 HP | 7.08（6.81–7.26） | 8.89（8.76–9.05） |
| 6144 /20 | 126（36–385） | 21.74 HP | 6.46（5.79–7.19） | 10.64（8.75–11.54） |
| 6144 /40 | 84（13–127） | 20.19 HP | 6.09（5.31–6.48） | 9.48（9.02–9.90） |
| 24576 /5 | 181（138–208） | 21.00 HP | 6.15（6.03–6.33） | 9.71（9.48–10.03） |
| 24576 /10 | 197（158–218） | 22.08 HP | 6.69（6.59–6.85） | 9.09（8.84–9.37） |
| 24576 /20 | 147（44–266） | 22.06 HP | 6.51（5.80–7.14） | 9.94（8.89–11.34） |
| 24576 /40 | 140（0–288） | 20.87 HP | 6.48（5.56–7.70） | 9.79（8.75–12.02） |

`6144/40` 相比 `24576/5`，敌方反向步数均值减少约54%，且四个 seeds
的区间不重叠。相同处理行数下，`6144/40` 相比 `24576/10` 也更好。
但其低HP MAE范围扩大到5.31–6.48，policy argmax变化约9.1%，高于
`24576/5` 的6.2%。因此是部分指标改善，不能宣称 Pareto 改进或整体方差下降。

最大问题仍未解决：精确终局HP1，在 `6144/40` 下平均预测18.22 HP
（seed范围16.80–19.60），对照 `24576/5` 为19.75 HP。改善有限，仍明显错误。
另外原始模型在部分低HP观测上已有约−14.15 HP预测；`6144/40` 最低约
−14.24至−15.67 HP。增加更新没有让 value 自动满足基本范围。
这些受控构造状态不代表真实采样频率；原生搜索在终局直接取真实HP，
并不会使用上述错误网络输出做终局回传。

## 性能和线上影响

每次并行运行两个 seed，两个批次的完整40-update曲线约127–129秒完成。
共享GPU时，6144的40次训练更新累计约22秒，24576约88秒；有效样本量也相差4倍。
峰值 Torch allocated约6.48GiB/实验进程；含生产的设备总显存观测约20.98GiB。
没有 OOM。生产原PID继续前进到4730880；实验期间部分 backward从约4秒变为7–9秒，
即并行存在短暂资源竞争，不是完全无影响，也不能把这些耗时当作独占性能。
所有实验进程正常退出，未改线上配置、run ID、replay或恢复点。

## 可复用入口与交接

使用 `python -m tools.analysis.probe_nosl_updates --help`。示例参数：

```sh
python -m tools.analysis.probe_nosl_updates SOURCE_BUNDLE \
  --device cuda --micro-batch-size 2048 --batch-sizes 6144 24576 \
  --updates 40 --eval-updates 5 10 20 40 --seed 0 --audit-seed 5311 \
  --audit-bundle LATER_BUNDLE --hp-corpus HP_CORPUS.npz \
  --terminal-corpus TERMINAL_CORPUS.npz --max-seconds 600 \
  --output runs/local/matrix-seed0.json
```

给外层进程另设720秒超时；每个分支输出拒绝覆盖，超时的部分结果不能当完整矩阵。
代码和测试归正式工具/测试目录；机器凭据、临时SSH调度代码及大恢复包不入Git。
原始本地结果：`runs/local/value-health/update-matrix-seed{0,5311,17,29}.json`。
远程保留 source/audit SHA 和实验输出于当前 run 的
`diagnostics/update-fit-4018176/`、`diagnostics/update-batch-4018176/`。

下一阶段优先验证6144/40在生产BF16和四rank梯度平均下是否保留优势，
并继续用双方完整HP曲线、负值/超界、真实后续样本误差及policy漂移联合筛选。
不得只因反向步数下降就替换线上配置。若这些仍不改善低HP范围，应单独审查
value目标、低HP状态覆盖和输出参数化；与本次纯update/batch实验分开。

## BF16 训练复核

随后保持源包、audit、四个seeds、batch和microbatch不变，将训练前向改为CUDA
BF16（`--amp bf16`）；所有评测仍是FP32。初始audit与两个corpus逐项完全相同，
排除了评测精度切换。完整8条分支全部完成，每对并行任务约69–74秒。
单个6144-update冒烟约0.61秒、Torch峰值allocated3.85GiB。

| BF16 batch / updates | 敌方反向步数 | HP1–5 MAE | HP>40 MAE | Policy argmax变化 |
| --- | --- | --- | --- | --- |
| 24576 /5 | 183（140–210） | 6.16（6.05–6.36） | 9.69（9.46–10.02） | 6.2% |
| 6144 /40 | 80（0–133） | 6.23（5.62–6.76） | 9.49（8.97–10.39） | 10.1% |
| 24576 /40 | 75（20–154） | 6.21（5.12–7.53） | 10.08（8.93–11.82） | 8.3% |

BF16保留了较多更新改善敌方局部单调性的趋势，但6144/40低HP误差均值并未改善，
seed范围仍扩大。精确终局HP1平均预测18.21，负值问题也仍存在。
24576/40的敌方指标在BF16下好于其FP32版本，但低/高HP误差波动仍大。
因此不根据单调性一项上线，不声称6144/40在所有指标上优于24576/40。

结果保存在 `runs/local/value-health/update-matrix-bf16-seed{0,5311,17,29}.json`，
原始执行代码与输出另存远端 `diagnostics/update-batch-bf16-4018176/`。
四rank验证尚未完成：已保留共同step4866048的四个完整模型/Adam/replay包，
路径为各rank主机上的 `diagnostics/four-rank-step-000004866048/rankN/`。
四份policy全部215个tensor逐字节哈希一致；这只证明该恢复点模型同步，
不等于候选参数已通过四rank/fresh-replay试验。线上仍未改动。

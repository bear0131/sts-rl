# Inner update / batch 对 HP 敏感性的受控对照

冻结 replay 的筛选曾把 `batch6144 / updates40` 列为敌方 HP 单调性候选；
后续真实四 rank、fresh replay 对照没有证明它稳定地改善所有指标。
用户已授权停止原训练，完整恢复点保留于 step4988928，正在做有预算的参数实验。
不能把截图里 update40 的胜率优势直接归因于 update 数，也不能把短程筛选当成上线验收。

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
当时四rank验证尚未完成：曾保留共同step4866048的四个完整模型/Adam/replay包，
路径为各rank主机上的 `diagnostics/four-rank-step-000004866048/rankN/`。
四份policy全部215个tensor逐字节哈希一致；这只证明该恢复点模型同步，
不等于候选参数已通过四rank/fresh-replay试验。该阶段未改线上；后续见下节。

## 真实四 rank / fresh replay 对照

原训练通过 STOP 在四 rank 共同完成 step4988928 后正常退出。保留原 run/W&B
身份与四份完整 model/Adam/replay/RNG，副本路径为各主机当前 run 的
`diagnostics/four-rank-step-000004988928/rankN/`。原生环境中的未完成战斗不在
恢复包内，因此实验从新环境开始；游戏胜率窗口也重置，不能直接与旧窗口相比。

运行使用既有部署，对应88ab5a9；已核对四主机trainer/learner/policy文件哈希一致，
trainer/learner/replay与该提交逐行相同（部署文件有混合换行，不直接以Git元数据认定）。
真实四 rank 梯度同步、BF16、microbatch2048、
fresh10%、每rank24env×128steps、strict paper128次搜索均保持不变。
每组只跑6个collection（73728个global steps），终点5062656，外层1800秒预算。
两组并行，W&B关闭，日志与恢复包分开，未向原run曲线混入实验数据。
原run完整argv及环境只保存于主机的私有恢复文件，不进入Git。

首次 collection 两组的 replay observations/arrays 在每个rank逐字节相同，
实际fresh比例约10%，所以第一轮更新后有较干净的配对比较。
后续各自策略产生不同采样；这是一条配对轨迹，不是多seed方差估计。
两组全部完成，最终模型的215个tensor及Adam在各自四rank之间逐字节一致。

下表是相同固定60条曲线、60个精确终局的FP32评测；反向边数越少越好，
低HP正超额指我方HP1–5的 `max(predicted_HP-current_HP,0)` 均值。
只在本无回血构造语料内解释超额，不能在所有游戏状态强制 value≤当前HP。

| 参数 / 新collection数 | 敌方反向边 /1856 | 终局MAE（HP） | HP1终局平均预测 | 低HP正超额（HP） |
| --- | --- | --- | --- | --- |
| 原始模型 /0 | 128 | 8.70 | 14.34 | 1.95 |
| 24576×5 /1 | 61 | 9.35 | 15.43 | 1.97 |
| 6144×40 /1 | 155 | 10.87 | 16.96 | 3.17 |
| 24576×5 /3 | 45 | 7.70 | 13.01 | 1.10 |
| 6144×40 /3 | 65 | 7.97 | 13.57 | 1.77 |
| 24576×5 /6 | 97 | 10.51 | 17.14 | 2.35 |
| 6144×40 /6 | 131 | 8.70 | 14.18 | 1.33 |

我方全部曲线仍单调，但最终最低预测分别−16.32HP和−14.56HP。
敌方反向主要在Cultist、Jaw Worm、Large Slime；Gremlin Nob与Shell Parasite
在本语料中均无反向。6144×40改善最后一次低HP超估，却未消除波动或负值。
终局网络预测误差不等于原生搜索终局回传错误，后者直接取真实HP。

共同源rank0 replay按HP1–5、6–20、21–40、>40各固定抽256个policy roots，
seed5311，另检查真实状态校准。**这是训练源样本上的检查，不是独立验证集。**
最终6144×40的低HP MAE4.39，对照24576×5为5.02、源模型4.68；高HP分别
9.73/9.64/9.76。Policy argmax改变11.72%/9.77%。这些结果支持“有取舍”，
不支持“整体方差下降”。样本ID哈希和完整分层结果保存在实验目录的
`source-replay-audit-2-branches.json`。

并行时每轮平均115.01秒（24576×5）和128.63秒（6144×40），候选慢11.85%。
这些是同硬件并行条件下的wall time，不是独占吞吐基准；主要时间仍用于搜索采样。
设备内存实测约4.7GiB采样态，部分更新采样时约8.7GiB；无OOM。
本阶段没有连续采集CPU/RSS/Torch峰值，因此不声称这几项改善。

等工作量补充控制：12288×10与6144×20，每轮各122880个样本/rank，
与24576×5相同；6144×40是其两倍。两组同源、同预算、同评测，运行中。
最新进度及恢复操作见 `worklog/2026-09-15.md`。

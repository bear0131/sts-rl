# COMBAT49：512×8更新压缩到50秒的交接

对应唯一任务 `TODO.md::DECK-LATENT-UPDATE512`。用户要求保留512×8，先优化更新时间，并将代码及记录推送Git，由其他会话接续。之前的256×4/COMBAT50重启方案已暂缓；没有启动COMBAT50。

## 当前暂停点与原型复现

正式 COMBAT49 已在第21轮/global16515072正常停止，四个 rank 的恢复包均已校验；模型和 Adam 跨机一致。按用户要求，达到优化门槛前保持 STOP。恢复位置是每机训练 root 下的 `runs/combat49-deck-latent512-t64x2-ppo-search-r1/recovery/rankN/step-000016515072`。冻结实验产生的权重不得用于恢复。

`tools.benchmarks.benchmark_ppo_overlap` 是实验入口。四机各用自身的第21轮恢复包、同一生产运行时、同一未占用的8个连续端口和相同的全新输出目录；例：`python -m tools.benchmarks.benchmark_ppo_overlap --plan training-plan.json --checkpoint-dir <本rank恢复包> --output <四机相同的新目录> --ring-port 31182 --seed 20261111`。每机由外层设置1150秒超时。四个方案共享本机同一次采样，逐步比较确定性模式的16次归约后梯度和最终参数/Adam；速度另在非确定性生产精度下倒序测试，不复制梯度快照。计时含取批、前后向、SSH、等待、裁剪与16次Adam，不含采样、推理图释放、全局优势预计算或保存，不能替代生产 `update_sec` 验收。

原型 `fed93e8b` 比较以下四种同步。在同设备、同保存点、同数据、Torch 2.12.0+cu132、FP32 highest、开启确定性算法的资格测试中，D 的每一步归约梯度和最终模型/Adam 都与 A 完全相同。B 也完全相同；C 的16步梯度和最终状态通过既有容差，但不逐位相同（Adam矩相对L2 7.60e-5）。这项判断不外推到另一个设备或任意数据。简单把上层块放前、不恢复共享参数的本地加法顺序则失败（Adam矩相对L2 2.87e-4）。

- A：生产物理块顺序与单个平坦串行 ring。
- B：上层先算，其对共享参数的逐块贡献在战斗后按原顺序相加，仍用原串行 ring。
- C：B 的加法顺序，上层独占桶提前走 SSH，桶内均分给各 rank。
- D：C 的异步时序，额外保留每个元素在原 ring 上的归约分片，保留各 rank 相加的顺序。对本原型，若上层物理块发生显存退避拆分则中止，不会以近似的加法顺序继续训练。

同一第21轮恢复包的四机无快照测速（非确定性模式、FP32 highest，单个冻结 rollout，实际 SSH 环）：

| 四机冻结实验（每机196608行，16 Adam） | 最慢 rank / 秒 |
| --- | ---: |
| A | 86.339 |
| B | 85.690 |
| C | 80.031 |
| D | 81.235 |

D 对原序串行 A 减少 5.104s（5.9%），对相同本地顺序的串行 B 减少 4.454s。D 最慢rank的上层桶ring 8.383s中有7.555s位于战斗反向窗口内；战斗结束等上层桶0.845s，尾桶ring仍需10.862s。计时包含packing和传输方式变化，不能把端到端差全算成裸链路加速。无OOM。D距离50秒仍差31.24s，未部署。仅同一冻结rollout/每方案一次纯测速；其他数据分布和生产退避不在证明范围。远端各机训练root下 `overlap-proto2/abcd/result.json`、`overlap-proto2/log`、`overlap-proto2/exit`，四机全套回执 `overlap-proto2/receipts.tgz`。

## 构筑计算与 Adam 误差复现

四机 `global-fp32-1/def/result.json` 保存了相同 rollout 上的计算候选。最慢 rank 的冻结更新时间为严格 overlap 79.37s、只裁剪上层末层 query 79.62s、再合并相同卡牌 token 77.50s。后两项的最终 Adam 矩相对 L2 为 2.86e-4、3.11e-4，基线与各候选分别在四个 rank 上一致。精确上下文编号/同 screen 派发的另一组实验在 `global-fp32-2/dg/result.json`，16 步全部梯度和最终状态与基线逐位相同，候选/基线/候选为 81.08/80.99/79.93s。

只裁剪 query 的单步重放可从每机 `global-fp32-diag/src/tools/benchmarks/diagnose.py` 复现。四机各自的 `global-fp32-diag/causes/diag.json` 含样本行、ReLU 跨零的层/实体 token/单元/上下游数值、PPO clip 分支、每步全梯度差、Adam 两个矩和逐参数归因。第12/16步从同一原版状态重放，只裁剪带来的直接梯度 max 差分别为 5.82e-10/6.98e-10；候选轨迹的步前权重即使用原版代码也产生 1.56e-6/1.92e-6 的跳变。还原候选后向的战斗 ReLU 门控，分别消除 99.79%/99.30% 的梯度 L2 差。细节和单样本干预见 [11月12日工作日志](../../worklog/2026-11-12.md)。此单项同批耗时略高于基线，保留其作为可复现的数值诊断样例；重复卡牌单项在 `global-fp32-cards` 的交错计时为81.46/79.79/80.43s（候选/原版/候选）。保留原 attention 槽位顺序的修法，将第12步同状态梯度最大差从3.45e-7降至5.82e-10，整轮交错计时81.34/79.75/80.83s。该步只减少5.23%的构筑token槽，两种实现的计时均不支持合入。正式训练保持第21轮暂停，当前总目标为四机最慢完整 `update_sec < 50s`。

以后验证数值变通可直接进入每个训练机的COMBAT49根目录，使用 `global-fp32-repair/src` 运行 `python -m tools.benchmarks.single --plan training-plan.json --packet global-fp32-repair/causes/step12-frozen.pt --output <新的四机同名目录> --ring-port <未占用端口段的起点>`。四机各用自己的packet和既有运行时，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；外部硬限140秒。新进程已逐位复现参考梯度。跨机回执与脚本在每机 `global-fp32-repair/global-fp32-receipts.tgz`，含各rank大包的SHA；大包保留在原机。

## 第12/13轮的生产结果

已从受保护的第11轮恢复原 COMBAT49，生产源 `d8c2e39d`。第12/13轮四机最慢完整 PPO 更新时间为 **92.19/90.66 秒**。每机每轮196608真实行、16次Adam，四机模型及Adam分别一致。≤50秒目标仍开放。四机的新交接是同一训练 root 下的 `handoffs/update3-d8c2e39d/README.md`，完整分rank计时、manifest验收及实验脚本保存在对应机器的 `update3/`。下文保留第11轮交接时的原始结果。

确定性更新对照中，rank0和3旧基线重复16步的最终权重及Adam矩完全一致；提交版空位压紧和末层全局query对旧基线的Adam矩相对误差为6.46e-6与9.21e-6。原常规模式的控制组也能超过1e-4门槛，具体触发的算子尚未单独隔离。四机新分块候选额外限制每块320个不同构筑、按有效长度分档，同一rank同buffer完整16Adam新旧对照（秒）：[103.685, 110.534, 106.428, 106.242] → [69.808, 72.577, 71.417, 70.784]。旧的100.02→79.51记录仍独立保留。

GPU算子采样在战斗物理块中看到FP32矩阵乘为最大类目。rank1首个12288行Adam组显式mm/addmm/bmm计约111.73 TFLOP，形状含512×2048和512×1536的乘法分别约66.13及33.98 TFLOP。这是单组实际工作量，不能由它推断四机物理极限。下一次实验应分别给这两类乘法的前向/输入梯度/权重梯度计时，并对标准FP32库算法、形状和真实同步预算做同批对照。

## 第11轮交接时的目标与状态

目标为**四机最慢的完整 `update_sec` ≤50秒**，包含取批、前后向、Adam、梯度同步及rank等待；不是单独CUDA反向或某台机器的最好成绩。保持构筑512×8、外层512×2、内层2×64，每机256环境×768步、全局786432真实样本、1遍/16Adam、有效global49152、LR5e-5、FP32 highest、原奖励与λ1 whole-run预热、rollout级优势统计。正式搜索未启用。

现役源 `a3ec30f8` 已含原生观察、GPU图、地图缓存、采样合批以及按真实上层screen的更新分块。停止前第8/9轮采样27.84–32.06秒、完整更新116.25–125.94秒。不可把缩模型、减样本/Adam或改统计目标计入本次等价加速目标。

COMBAT49已在第11轮/global8650752正常停止。四台实验进程均已退出，正式训练保持暂停，便于接续会话测试；本会话不部署候选、不恢复训练、不继续增加实验。

## 保留结果：100.02秒 → 79.51秒

这项结果按用户明确要求保留，供后续继续使用：

| 条件/结果 | 数值 |
| --- | --- |
| 原型与模型 | 当前512×8，38641205参数；`compact_cls` |
| 数据 | rank2第11轮完整状态恢复后采集一次；每臂同196608行、同原始置换与Adam组成员 |
| 更新量 | 16次Adam、12288本地样本/Adam；无跨机梯度同步 |
| 精度 | FP32 highest，AMP/TF32关闭 |
| 基线 | 100.0216018297秒 |
| 候选 | **79.5141584408秒** |
| 原基线复测 | 102.0848837309秒 |
| 减少时间 | 相对第一次基线20.503%；相对前后基线均值约21.31% |
| 物理块/样本/OOM | 各728块、196608行、0 OOM重试；combat cap384、upper screen cap192 |
| 首步梯度 | 相对L2误差1.245997e-6，最大绝对误差1.193985e-8；逐参数rtol3e-4/atol2e-6通过 |
| 16Adam后参数 | 最大绝对误差5.066395e-7（门槛5e-5） |
| Adam矩与step | 矩相对L2误差3.747207e-5（门槛1e-4）；全部step相同 |
| 进程 | exit0，总340.086秒，未达到1200秒超时 |

组合做了两件事：

1. 构筑编码器训练时，按既有attention mask稳定压紧空token；不删除PAD身份但带有效特征的地图节点，不改变保留token顺序和公开坐标。
2. 最后一层仍读取所有有效key/value，只计算最终实际使用的全局query及其FFN输出。权重名称、模型容量与目标不变。

输入token累计17144502→11727768，约减少31.6%。前面各层仍完整计算所有有效token；推理/采样路径保持原形状及现有GPU图，不在本次手动改变推理内核。

硬件为rank2的25核额度Xeon Platinum8470Q、RTX5090、Torch2.12.0+cu132；native SHA `b40edcf73e671332dc8d5d27287264d8294376605ca6bad865aea36596156103`。基线/候选/复测CPU树100.62/80.00/102.63核秒、约100.4–100.5%（一核）；RSS峰11.08/11.86/12.35GB（共享页可能重复），系统可用最低734.71/733.88/733.10GB（非容器限额）。Torch真实allocated峰23.208/22.222/24.827GB，reserved峰23.964/22.995/25.581GB；设备VRAM采样峰23674/22750/25216MiB，设备GPU均值72.99/63.19/72.46%。1秒采样，设备指标可能含其他进程。初始化13.475秒、一次采样34.389秒单列；此100→79.5结果不包含它们及四机同步，也不是生产更新时间已达79.5秒。

## 其他实验及失败，不能混作全部通过

| rank | 候选 | 首基线→候选（秒） | 完整三臂状态 |
| --- | --- | --- | --- |
| 0 | 只压紧空token | 103.265→82.767 | 候选全部数值/16Adam通过；最后基线复测在参数/Adam阈值断言失败，exit1 |
| 1 | 只保留最后全局query | 103.119→100.496 | 基线复测97.522，exit0；候选约等于前后基线均值，没有稳定单项收益 |
| 2 | 两项组合 | 100.022→79.514 | 基线复测102.085，完整exit0；上面重点保留结果 |
| 3 | 组合，上层块192→384 | 101.142→73.587 | 候选全部数值/16Adam通过、0 OOM；最后基线复测断言失败，exit1 |

各rank数据不同，仅作各自内部配对。rank3候选Torch allocated峰30.505GB、reserved31.506GB，显存余量较小；上层384尚未写入生产配置，不能因短窗口成功便认定后期安全。

rank0/3均在最后基线复测完成16Adam后，于 `update2/bench.py:112` 的 `final_comparison['max_abs']<5e-5 and adam_comparison['relative_l2']<1e-4` 失败；脚本没有在断言前落盘误差具体值，因此目前不能判断哪一项超限或原因。后继应先保存失败指标、隔离独立进程/随机状态/缓存和算子确定性再复测；不要放宽阈值掩盖问题。两次基线复测16步进度用时约103.532/102.291秒，不作为已完成资格的时间。

## 已提交候选与尚缺的验收

- `rl/policies/deck_latent.py::RunContextEncoder._pool_tokens`：实现两项组合，只在训练模式启用；没有改变权重形状、观察、奖励或推理图。
- `tests/integration/test_deck_latent.py::test_context_padding_compaction_preserves_output_and_gradients`：洞状mask、仅全局token、全部有效token；两层reference输出、输入/参数梯度、推理固定宽度。显式原生Windows CPU验证3 passed/1.71秒。
- CUDA参数化测试已写入，**此提交尚未执行**；上述完整GPU数字来自原型，不能冒充整理后提交代码的CUDA/完整生产验收。源码整理后还须对齐原型、跑原生CPU/CUDA回归、旧checkpoint加载/预测和两轮真实PPO保存恢复。
- 新候选**未部署**。远端训练仍使用 `a3ec30f8`，不得直接把此提交记为已经生效的生产优化。

## 接续位置、恢复点和证据

**接续不依赖Windows本机。** 候选固定提交为 `ea044ba0aaad52508c2db753266e18efb92ab294`，Git接续分支为 `origin/codex/rollout-200s` 和 `origin/codex/upper-unified-420s`。四台远端都保存同一套交接资料，连接用户为root、主机 `connect.bjb2.seetacloud.com`：

| rank | SSH端口 | 资料目录 |
| --- | --- | --- |
| 0 | 53841 | 下述统一目录 |
| 1 | 13208 | 下述统一目录 |
| 2 | 15409 | 下述统一目录；100→79.5秒原实验机 |
| 3 | 26162 | 下述统一目录 |

统一入口：`/root/autodl-tmp/combat49-deck-latent512-t64x2-20261111/handoffs/update2-ea044ba0/README.md`。

目录内 `source-ea044ba0.tgz` 及解压后的 `source/` 保存该提交的训练、环境、模拟器、测试、配置与工具源码；`candidate.patch` 是生产基线到候选的模型/测试差异；`update2-handoff.tgz` 及 `evidence/` 保存原始脚本、四rank完整结果、日志与退出记录；`inventory.json` 列出四机运行时和恢复点；`SHA256SUMS`、`source-sha256.json` 和 `verification.json` 用于校验。这里的源码是独立副本，未切换正式训练源。凭据及历史大checkpoint不放源码包；当前恢复点和Python/native运行时直接使用各远端已有路径。

进入该目录后执行 `sha256sum -c SHA256SUMS` 可检查交接文件。先读README，再按后续工作顺序接续；原型 `supervisor.py` 会写原 `update2/`，不要直接重跑覆盖证据，应先复制到新的实验目录并修改输出路径。旧Windows工作区有4008bbd7草稿，不作为接续入口。

2026-11-11 22:49（Asia/Shanghai）四机核验通过：每机6项资料文件、2090个解压源码文件、17个解压证据文件、各自manifest中的3个checkpoint文件逐一SHA一致，Python/site-packages/native路径存在，native仍为上文b40。源码归档10598321字节，SHA256 `21145821292f8ebe6d1595d0a7472c401ca959256bdaac4796e50c3d7e0680aa`。这仅是迁移完整性验收，未追加GPU性能或训练验收。其他会话已有 `update3/` 目录，接续前先核对当前运行进程；本次不操作该目录。

四机都使用root `/root/autodl-tmp/combat49-deck-latent512-t64x2-20261111`，run `combat49-deck-latent512-t64x2-ppo-search-r1`；当前机器映射保存在现有launcher/SSH配置，此处不复制凭据。

各机保护目录：`references/before-update2-512/rankN/step-000008650752`，全部manifest核验。必须恢复本rank的模型、Adam、环境和RNG；不得采用实验臂更新后的权重。

- policy：`95049bfe8a966a8eee8964a03a7456f162c695051fda75cc58d85afde7cb49d7`
- Adam：`61f6edc7fecf141219865e7c924e7021a90b77c823b2a51faea67969dfc7866e`
- 各机 `update2-protected.json`、`training-plan.json` 以及保护目录内旧计划足以普通resume；STOP/旧exit需归档后启动，同run保持曲线连续。
- 各机 `update2/bench.py`、`candidates.py`、`supervisor.py`、`result.json`、`process.log`、`exit.json` 保留原型及原始结果，seed20261111、每rank1200秒上限。全部实验均已退出。
- 四机交接目录中的 `evidence/rank2-result.json` 是重点结果，不是跨rank平均。本机原始目录仅作备份，不是接续前置条件。
- 17成员/22769字节归档 `update2-handoff.tgz` 已在四台远端统一交接目录保存；SHA256 `51ddb4679d1f3d35587f92e901d245c28b8f02601373e00ac1fd816486f7b96c`。包含四rank结果/退出/日志和原型脚本，无大checkpoint或凭据。

## 后续工作顺序

1. 修复对照脚本的失败证据落盘，解释rank0/3基线复测异常；对已整理组合代码做CUDA与完整同buffer复验。保留100→79.5为候选证据，不当成最终上线门槛已完成。
2. 在固定512×8/16Adam/全样本下继续找大块开销，区分构筑前后向、上层网络、分块/同步。单项全局query未显示稳定收益，不单独夸大；物理块增大须检验更长卡组/界面分布及显存退避。
3. 确认候选后从原第11轮普通恢复，同run、不重训。四机至少连续两轮全样本/16Adam/全状态SHA和模型Adam一致、监控正常；以四rank完整update_sec最大值验收50秒。所有新实验仍固定seed/步数/超时和资源记录。

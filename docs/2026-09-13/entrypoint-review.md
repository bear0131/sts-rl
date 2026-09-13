# 启动入口盘点与收敛讨论

记录日期：2026-09-13。用户意图：减少入口和重复事实来源，先整理文档、产物和依赖，再讨论下一轮重构；本报告不是新增长期规范或另一份 TODO。

审查对象：`chore/refactor-foundation`，本轮起点 `b90d0bee2e2402df79bfbc2a42a2392b316b4736`。目录整理提交 `4a5b590`，依赖合并提交 `ac88a42`；本报告随其后的收尾提交。盘点 JSON 记录扫描时的 HEAD、未提交文件列表、入口源码 SHA256 和行号，避免把工作树结果误称为纯 HEAD 的结果。

## 结论

上轮主要完成了归位和兼容，尚未减少功能入口；这轮删除了重复的依赖声明和迁移索引，但没有擅自合并不同评估算法。下一轮最值得做的是把“场景、solver、重复/汇总协议”从 eval 脚本拆开，让 CLI 和 dashboard 调用同一个评估实现，再删除被替代的可执行入口。

不能以文件名相似就认定等价。当前 `eval_nosl_fullrun` 仍走 Python NOSL 搜索，禁止 depth=0；你同学的实验是 native strict Full Gumbel、depth=0。并行 fullrun 脚本只是每个 episode 新起进程、重新加载模型。优化它不会自动得到“当前实验的高效评估器”。dashboard 本身也维护搜索和评估循环，不能只清理 `eval_*`。

本轮未更改算法、训练参数、checkpoint 字节或远端实验。性能热点判断来自静态调用链，没有把它写成实测提速。

## 已完成的收拢

| 原来的问题 | 现在的落点 |
|---|---|
| TODO 只凭日期指代 F 问题 | 每个 F 编号链接 [完整审计](codebase-audit.md) 的独立锚点；TODO 仍是唯一状态表 |
| 根 `doc/` 与 `docs/` 并存 | 根 `doc/` 已消除；旧资料在 `docs/archive/`。第三方库自带 doc 目录不动 |
| `refactor-stage1` 被当作长期架构文档 | 原文移至 [package-reorganization-handoff.md](package-reorganization-handoff.md)，明确是工作快照 |
| 多份构建说明/迁移索引 | 维护中的用法收在 `tests/README.md`；三个迁移 JSON 合并成 `tools/validation/migrations.json` |
| 模型、W&B、日志、文档混在 artifacts | 运行输出到 `runs/`，旧文档到 `docs/archive/`；新本地输出到忽略的 `runs/local/` |
| requirements/ci、rl/requirements、pyproject 重复声明 | 删除两个 requirements 文件，只维护根 `pyproject.toml` 和 `uv.lock` |

日期只标记工作记录的时间；引用给具体文件标题、路径和必要锚点。持久用法仍在对应模块 README，进度只追加 worklog。没有创建另一份“永久架构索引”。

已跟踪的历史模型仍在 Git 中，本轮只是保字节移动，没有改写历史。430 个历史路径记录通过哈希核验，其中多代旧路径可以指向同一文件，不能解释成 430 个不同文件。忽略的本地证据标为 local_only，不要求 CI checkout 拥有它们。

唯一暂留项是 `artifacts/archive/root/.tmp-nosl-sync-20260913.tar.gz`，约 288 KB，内容属于历史源码传输包，包含 Python/C++/UI 源码。它既非运行结果也非文档，按用户要求留待讨论；未执行或解包覆盖仓库。建议核对其内容是否已由 Git 保留后再决定删除，不继续扩大 artifacts。

依赖安装只有一个声明来源：CPU/test 使用 `uv sync --locked --extra cpu --group test --no-dev --python 3.11`；GPU 开发选择 `--extra cuda`。两个 extra 互斥，用于 wheel 来源，不是两套 requirements。后续使用环境内 python 或 `uv run --no-sync`；CommunicationMod 的 BAT 也采用后者，避免启动时隐式重新选择安装配置。裸 uv sync 使用 PyPI Torch，复现时应显式选 extra。

## 盘点口径

| 类型 | 数量 | 如何理解 |
|---|---:|---|
| 非测试 Python `__main__` 模块 | 94 | 可执行表面，不代表 94 种不同算法；部分只是生成/汇总/展示 |
| Shell、PowerShell、BAT | 15 | 部分转发 Python/另一个脚本，不再重复计算功能 |
| 原生 C++ executable target | 4 | 不含供 Python 加载的 slaythespire 扩展 |
| 旧 `rl.*` 模块别名 | 100 | 77 个目标有 CLI，23 个是库；主要兼容旧 checkpoint/import |
| 测试里另有 standalone main | 12 | canonical CI 入口仍是 pytest；历史手跑方式尚存 |

方法是对 Git 跟踪的第一方 Python 做 AST 扫描，记录 main guard、parse 调用、直接本地 import、疑似动态模块字符串；Shell 用源码检索，C++ 用 CMake 定义。排除 vendor、docs、runs、兼容转发文件。清单是静态候选，不是“全部入口都成功运行”的证明，也不能推出远端还在使用哪些命令；没有执行部署脚本。

完整逐模块清单、直接依赖、旧名、源码行号和 SHA256 见 [entrypoint-inventory.json](entrypoint-inventory.json)。动态字符串只是线索，不能一律当真实调用边；主要调用链已另行阅读源码确认。附录列出全部入口，便于讨论删并范围。当前没有 `[project.scripts]` console command，主要靠仓库根目录 `python -m 包.模块`；直接执行文件路径不作为可靠接口。

## 主要启动方式与实际模块

### 1. 强化学习训练

| 入口 | 主要链路 | 边界 |
|---|---|---|
| `rl.training.trainer` | task → policy → AccumulatingMaskablePPO → vec env → env/core → native | 单机 SB3 训练，task/模型/参数组合较多 |
| `rl.training.distributed_sb3` | parser/main → make_env → WholeRunTask/向量 worker；更新走 distributed_nosl_is_az_update、replay、ring_allreduce/ssh_transport | 当前多 rank 实验入口；CLI、部署、采集和学习耦在巨型模块 |
| `rl.training.rllib_trainer` | Ray/RLlib → 自己的 module/learner/env adapter | 另一套训练栈及策略格式，不能作为 SB3 的一个等价开关 |
| `rl.training.overfit_test` | 小型 MaskablePPO combat 训练循环 | 实验训练入口，不是单元测试 |
| `rl.benchmarks.train` | 有超时子进程 → 真实 PPO/NOSL 更新 → 资源采样/JSON | 正确性与性能工作负载，不能替代生产训练 |

当前实验链路（来自附件快照与源码，不是远端运行确认）：

```text
experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh
  ├─ configs/train/nosl_paper_replay_fresh10_20260913.args
  └─ rl.distributed_sb3（旧别名）→ rl.training.distributed_sb3
       ├─ make_env → tasks/wholerun → cached_mask_vec_env → env/core/run_env
       │                                                └─ C++ NOSL search
       ├─ policies/split_policy、model
       └─ distributed_nosl_is_az_update
            ├─ algorithms/accumulating_ppo、nosl_replay
            └─ ring_allreduce、ssh_transport
```

核查位置：`distributed_sb3.py` 的 `build_parser:106`、`make_env:538`、`distributed_nosl_is_az_update:1134`、`main:1609`。下一轮应把编排、收集、更新拆出可测试边界；不先做覆盖所有框架的通用 Trainer。

### 2. 策略/搜索评估：目前最值得收敛

| 家族 | 当前入口 | 实际差别 |
|---|---|---|
| 普通策略 | `rl.evaluation.eval`、`eval_combat` | 整局与 combat task，checkpoint/env wrapper/统计口径不同 |
| 混合策略 | `rl.evaluation.eval_hybrid` | RLlib policy_state 与 SB3 按 combat/map 路由，不能泛化成任意 solver 混合 |
| NOSL fullrun | `nosl.eval_nosl_fullrun`、`run_parallel_nosl_eval` | 旧 Python 搜索；parallel 是 episode 子进程调度 |
| native 固定场景 | `eval_nosl_order_sensitivity`、`eval_full_gumbel_bash_clash`、matrix/repeated/trials/sequence | 部分共用私有 helper，但场景、随机流、K、预算和指标有差异 |
| corpus/方法比较 | `collect_nosl_corpus`、`compare_nosl_corpus`、`eval_nosl_methods`、`eval_nosl_single_combat` | 数据采集、汇总、Flat-MC/其他搜索比较，不全是同一种 eval |
| value 探针 | `eval_value_hp_sensitivity`、`eval_value_player_hp_sensitivity` | 改输入 HP 的反事实测量，不是胜率评估 |
| 无模型玩法 | `baselines.eval_leftmost`、`eval_run_leftmost`、`eval_starter_policy` | gauntlet/整局/禁卡奖等玩法约束不同 |

`eval_nosl_fullrun.py:59–103` 使用 `model.policy.core.actor_critic`，调用 `env.nosl_is_az_begin/backprop`，参数校验要求 `depth >= 1`。当前 strict 实验使用 native 接口，depth=0；这两个名字下的算法不能靠换一张 config 就判为等价。

`run_parallel_nosl_eval.py:33–74` 每个 seed 启一个 subprocess；目标脚本 `MaskablePPO.load` 每次重做。ThreadPoolExecutor 只是控制这些进程，模型没有常驻共享或跨环境推理批处理；该 subprocess 调用也没有 timeout。是否加载/IPC 是主要耗时仍需实际 profiling，但这是明确的入口设计问题。

strict 小场景走 `eval_full_gumbel_bash_clash` → `eval_nosl_order_sensitivity._run_search:163` → `env.nosl_is_az_cpp_begin:198` 及 packed leaf/backprop。这条链路是本轮真实 checkpoint bitwise 的对象，不能把其通过推广为旧 fullrun 的正确性或整局胜率。

### 3. GUI/观看也在运行算法

`apps.dashboard` 同时负责 HTTP/UI、checkpoint 加载、session、模型锁/缓存、搜索和评估；`apps.watch` 是 whole-run 终端观看，`apps.play_gauntlet` 是 gauntlet 手玩/模型观看。env/scripts 还保留 combat、厚层 run、薄层 run 的手玩入口。

dashboard 中有 `EvaluationNOSLSearch:539` 与 `WatchSession:704` 两条消费搜索的路径；后者还有 `_evaluate_native_leaves:1236`、`_run_native_search:1284`。`start_evaluation:2185` 和 `_evaluate:2274` 在 GUI 侧运行评估循环，hybrid 另由 `start_hybrid_evaluation:2523` 启动 `rl.evaluation.eval_hybrid` 子进程。

因此 UI README 描述的“只消费 evaluation”是目标边界，当前实现还没有做到。建议先抽共享评估/搜索执行服务，CLI 与 UI 都消费它，保留交互步进所需状态；不再让每增加一个页面就增加一套推理循环。

### 4. 规则模型是另一条产品线

数据生成 7 个入口 → `rule_data` / `vector_rule_data` / `vector_rule_v7_data` / stream → `rule_model` 或 `rule_vector_model` → 4 个 trainer → 12 个 rules eval/解释入口。

训练有 token、匿名向量、v7 effect-tree 和分布式路径；评估有 teacher-forced 错误、自由 autoregressive rollout、简单卡、Strength、表面敏感性、报告分析。这些应先按数据/模型 contract 分组，再把场景和报告切片变为参数。不能仅按 v6/v7 文件名字删除旧 contract 支持。

两处上轮归类我需要纠正：`rl.evaluation.baselines.eval_triple_seven` 实际是合成三段攻击的规则模型 probe，应随 rules 家族处理；`tools.checkpoints.merge_card_representations` 实际合并采样向量 JSON，不是 checkpoint 转换。这里先标明，不继续搬目录掩盖该讨论。

### 5. 真机、维护工具、原生与 CI

- `bridge.client.agent_orchestrator`：CommunicationMod 自动策略，MaskablePPO → 真实帧规范化/动作命令；BAT 是其启动包装。`bridge.client.orchestrator` 是人工参与的 stdio 协议，不是另一个训练器。
- `bridge.replay`：真实日志 → schema → simulator 差分回放。`bridge.schemas.combat_state` 还藏有 standalone 日志诊断 main；库与工具责任尚未分净。
- `env.core._gen_vocab` 是代码生成；env/scripts 有 binding 探针与语言文档导出。两个文档生成器现在默认输出 runs/local，避免覆盖冻结归档。
- `tools.analysis` 21 个入口混合绘图、模型探针、实验小训练、报告分析、W&B 日志回填。不能把这些全部强行塞进 eval；适合保留少数通用数据变换工具、把一次性实验转为可记录的工作负载。
- `tools.benchmarks` 2 个低层 NOSL benchmark；`tools.checkpoints` 2 个工具；`tools.validation` 3 个可复用有界验证/定位入口。它们允许依赖运行时，运行时不应反向依赖工具。
- CMake 有 `main`、`test`、`small-test`、`nosl-combat-train` 四个 executable。最后一个是无 Python/Torch 的原生原型，使用 NoslCombatTrainer 自己的 policy/value 状态，不是当前 SB3 神经网络训练的独立启动方式。
- 唯一第一方 Actions 工作流 `.github/workflows/correctness.yml` 调用同一依赖安装、native build、pytest、compileall/bash -n 和有预算 benchmark。tests 中 12 个历史 main 是额外手跑表面；日常统一推荐 pytest，后续视重叠断言删除重复 runner。

`configs/*.args` 与 `configs/eval/nosl_scenarios.py` 是输入数据，不是启动命令；shell 文件中有 SSH、排队、W&B 和进程控制，仍是部署程序，不能只改后缀叫 config。

## 下一轮怎么减少入口：供讨论的方案

优先顺序建议如下，尚未实施：

1. **先确定支持矩阵。** 以当前 SB3/NOSL strict 训练、纯策略、native strict 小场景与整局、GUI/真机为主要消费者；RLlib 和旧规则 contract 保留为显式 legacy 支持，待确认实际需求后再删。不能从仓库文件推断远端没人使用。
2. **先收敛 eval。** 直接改造现有 `rl.evaluation.eval` 承载明确的场景/solver/运行协议，后端采用共享 service；逐个迁移现有 eval 场景，删除对应 `main` 和独立 parser。不能只新增 `eval_all.py` 再保留全部旧入口。CLI、GUI、验证 harness 共用执行层和结果 schema。
3. **随后拆训练内部。** 从 distributed_sb3 / accumulating_ppo 提取配置校验、rollout、NOSL learner、同步与 checkpoint 边界，先让 trainer/distributed 复用同一段业务更新；暂不为了一个 train 命令把 RLlib/SB3 混成复杂开关矩阵。
4. **实验脚本减少到部署职责。** 已提取数值 preset 的三个 NOSL launcher 可收为同一部署入口加 preset/run 身份；v10 只是转发 v9，版本名已经与实现错位。删除兼容 launcher 前需要确认现有远端自动化调用路径。
5. **区分 CLI 兼容与 checkpoint 兼容。** 100 个别名不能一刀删；77 个可执行旧名可逐步退役，23 个库别名可能被 pickle/checkpoint 使用。目标是减少公开启动方式，同时保留必要的反序列化适配，不让内部继续使用旧名。

建议的 eval 拆分边界是：Scenario 提供初态/种子/重置；Solver 持有模型和一次决策；Runner 控制 episode/重复/超时；Reporter 只汇总结果。名称只是职责描述，不要求马上新建四个框架。先用真实的两三个入口验证边界，避免再增加一层没有减掉旧代码的抽象。

每一批合并必须固定 solver 身份、action/mask/selection 契约、终局 HP 口径、搜索预算和随机流。先保持同设备输出/权重/优化器/replay bitwise，再测加载、采样、编码、推理和 IPC；合并后的速度优化另一个提交。跨 episode batching、不同 RNG 消费顺序未必 bitwise，需要单独定义接受标准。

讨论最关键的两点：是否按“eval 共享执行层 → 训练内部拆分”推进；RLlib、旧规则版本和原生 standalone prototype 是继续正式支持还是归为 legacy。源码同步包的处置是另外一个小决定。

## 验证与交接

- 全量 pytest：**248 passed、8 strict xfailed，78.91 秒**，证据 `runs/local/entry-cleanup-tests.xml`。8 个都是既有 F01/F02/F03/F04/F05/F07，未掩盖为通过。
- CPU 跨版本训练/恢复/搜索：**7208 字段 bitwise pass**。基线 runtime `9d2c6e1`；当前 capture 为 `ac88a42` 加 BAT 工作树改动，四个基线 checkpoint 确实被恢复。之后的改动只涉及文档、文档导出默认路径和盘点；学习/搜索实现未再改变。
- 同一真实 checkpoint 在 baseline/candidate 的 strict 固定场景评估：**8403 字节 JSON 完全相同**。本轮路径从 artifacts 改为 runs，JSON 含 checkpoint 路径，因此字节长度与上轮不同；这是本轮两边逐字节比较，不是拿旧路径 JSON 要求全文件一致。
- checkpoint SHA256：`cf8336a7a01a736d6ce36bff9726cc42b305f2235a0c3f23ed1d1b8e25470dd0`；共用 native SHA256：`fc1f3672104b79280257851a50e851280c9265a170383e1a22501131ecc797e3`。Python 3.11.15 / Torch 2.12.0+cu132，但数值门禁在单线程 CPU 上执行，不代表 GPU 或多机 bitwise。
- uv 0.11.8 锁解析成功，CPU/test、CUDA/test dry-run 分别选择 Torch 2.12.0+cpu / +cu132，仓库 `uv lock --check --python 3.11` 通过；**没有 fresh install、没有修改用户现有环境、没有运行 hosted CI**。测试使用既有已验证 audit-venv，不能等同于新锁文件在所有平台完整安装通过。
- 归档 hash、语法与最终 diff 检查结果随本次提交记录于 `worklog/2026-09-13.md`。没有运行远端脚本、未 push；本报告和 JSON 是工作记录，长期状态继续维护根 TODO。

复验命令集中在 `tests/README.md`，本轮输出在 `runs/local/entry-cleanup-*`。下个执行者先读 AGENTS、TODO、本报告和对应 worklog，再与用户确定入口支持范围；不要从本建议自动获得删除旧算法或操作远端实验的授权。

<!-- GENERATED INVENTORY APPENDIX -->

## 附录：全部静态入口

以下模块用 `python -m` 启动；命令参数与可用性须以其 parser/contract 为准。直接依赖详情在配套 JSON，不复制成第二份维护来源。用途摘录 docstring，仅作为识别线索，正文已纠正容易误读的分类。

### Python（94 个，排除测试及兼容别名）

| 模块 / main guard 行 | 用途线索 |
|---|---|
| `apps.dashboard` L3034 | Local browser workbench for inspecting and evaluating sts-rl checkpoints. |
| `apps.play_gauntlet` L146 | Play the COMBAT GAUNTLET (manually, or watch a checkpoint), model action API.  |
| `apps.watch` L245 | Watch a MaskablePPO checkpoint play whole-run SlayTheSpireEnv, live.  |
| `bridge.client.agent_orchestrator` L365 | Automatic CommunicationMod orchestrator backed by a MaskablePPO checkpoint.  |
| `bridge.client.orchestrator` L276 | Phase-0 human-in-the-loop orchestrator for CommunicationMod.  |
| `bridge.replay` L484 | Replay a whole real run inside the sim and diff at every step.  |
| `bridge.schemas.combat_state` L592 | Canonical combat-state schema + normalizers for differential testing.  |
| `env.core._gen_vocab` L287 | Generator for env/core/vocab.py — the categorical id-space contract for obs.  |
| `env.scripts.check_gamble_probe` L103 | Forced Gamble (combat MULTI card-select) probe.  |
| `env.scripts.check_select_binding` L222 | Verify the card-select bindings + obs contract (obs_design §4.1).  |
| `env.scripts.check_shop_binding` L146 | Verify the shop-info bindings (gc.shop_contents / gc.legal_shop_options).  |
| `env.scripts.generate_card_language_doc` L24 | Regenerate the bilingual Ironclad and Status card-language reference. |
| `env.scripts.generate_combat_language_doc` L48 | 导出战斗实体语言文档 |
| `env.scripts.play` L96 | Interactive combat player — drive a SimCombatEnv by hand.  |
| `env.scripts.play_run` L95 | Interactive whole-run player — drive a SimGameEnv by hand, across screens.  |
| `env.scripts.play_run_v2` L91 | Interactive whole-run player on the THIN SlayTheSpireEnv (action v2).  |
| `rl.benchmarks.train` L243 | CPU/CUDA PPO benchmark with a subprocess wall-time budget and JSON evidence.  |
| `rl.evaluation.baselines.eval_leftmost` L158 | Evaluate a deterministic leftmost-play baseline on the combat gauntlet.  |
| `rl.evaluation.baselines.eval_run_leftmost` L260 | Evaluate a simple full-run Ironclad baseline from floor 1.  |
| `rl.evaluation.baselines.eval_starter_policy` L651 | Evaluate a no-card-reward Ironclad starter-deck heuristic.  |
| `rl.evaluation.baselines.eval_triple_seven` L307 | Probe the latest rule checkpoint with a synthetic three-hit attack card.  |
| `rl.evaluation.eval` L387 | Evaluate a MaskablePPO checkpoint on complete simulated runs.  |
| `rl.evaluation.eval_combat` L292 | Evaluate one or more MaskablePPO checkpoints on the combat task.  |
| `rl.evaluation.eval_hybrid` L518 | Paired full-run evaluation of a native RLlib policy and a hybrid route mix.  |
| `rl.evaluation.nosl.collect_nosl_corpus` L467 | Collect a public combat corpus and run a first NOSL Flat-MC comparison.  |
| `rl.evaluation.nosl.compare_nosl_corpus` L110 | Aggregate the first NOSL-vs-PPO comparison from collector JSONL records. |
| `rl.evaluation.nosl.eval_full_gumbel_bash_clash` L418 | Paired strict Bash/Clash evaluation for Gumbel-AlphaZero variants.  |
| `rl.evaluation.nosl.eval_gumbel_bash_clash_matrix` L115 | Paired Gumbel-budget repetitions for the fixed Bash/Clash fixture.  |
| `rl.evaluation.nosl.eval_gumbel_bash_clash_repeated` L197 | Repeated Bash/Clash search comparison on the 7.68M checkpoint.  |
| `rl.evaluation.nosl.eval_gumbel_bash_clash_trials` L310 | Repeated Bash/Clash trials across public states and NOSL worlds.  |
| `rl.evaluation.nosl.eval_nosl_fullrun` L211 | Evaluate complete runs with NOSL-IS-AZ combat decisions.  |
| `rl.evaluation.nosl.eval_nosl_methods` L381 | Evaluate several combat policies on stratified full-combat corpus states. |
| `rl.evaluation.nosl.eval_nosl_order_sensitivity` L427 | Measure C++ NOSL-IS-AZ sensitivity to hand-slot order and search depth.  |
| `rl.evaluation.nosl.eval_nosl_single_combat` L277 | Evaluate PPO and NOSL Flat-MC through one complete combat.  |
| `rl.evaluation.nosl.eval_strict_bash_clash_sequence` L295 | Strict two-card Bash/Clash order evaluation.  |
| `rl.evaluation.nosl.eval_value_hp_sensitivity` L143 | Check whether the value head responds to enemy current HP.  |
| `rl.evaluation.nosl.eval_value_player_hp_sensitivity` L155 | Measure value-head output as the player's current HP changes.  |
| `rl.evaluation.nosl.run_parallel_nosl_eval` L138 | Run independent NOSL full-run evaluations in parallel CPU workers. |
| `rl.evaluation.rules.eval_attack_damage_accuracy` L558 | Evaluate damage predictions for every Ironclad Attack and upgrade. |
| `rl.evaluation.rules.eval_random_deck_surface_sensitivity` L370 | Counterfactual card-surface sensitivity on paired random decks.  |
| `rl.evaluation.rules.eval_rules` L206 | Evaluate a general combat-rule checkpoint with full autoregressive decoding. |
| `rl.evaluation.rules.eval_strength_rules` L590 | Evaluate rule checkpoints on controlled random-Strength attack samples. |
| `rl.evaluation.rules.eval_vector_rule_autoregressive` L398 | Evaluate a trained v6 vector-rule checkpoint autoregressively.  |
| `rl.evaluation.rules.eval_vector_rule_card_sensitivity` L569 | Evaluate simple Ironclad cards against the v6-wide checkpoint.  |
| `rl.evaluation.rules.eval_vector_rule_v7_cases` L1099 | Autoregressively evaluate the current v7 executor contract on fixed and random cases.  |
| `rl.evaluation.rules.eval_vector_rule_v7_error_positions` L358 | Profile schema-v7 teacher-forced errors by token and delta-vector position. |
| `rl.evaluation.rules.eval_vector_rule_v7_intermediate` L300 | Analyze model-owned calculation results from a v7 evaluation report.  |
| `rl.evaluation.rules.eval_vector_rule_v7_simple_cards` L500 | Evaluate the previously used simple-card probes with the v7 evaluator.  |
| `rl.evaluation.rules.explain_card_play` L928 | Simulate one card against one enemy and explain the resulting transition.  |
| `rl.evaluation.rules.render_rule_vector_examples` L224 | Render count-only vector event examples for the standalone rule model.  |
| `rl.rules.data.build_rule_tokenizer` L62 | Build or merge a shared tokenizer for sharded rule-model data. |
| `rl.rules.data.collect_rule_data` L391 | Collect single-combat state/action/effect/next-state rule samples.  |
| `rl.rules.data.collect_strength_rule_data` L189 | Collect a controlled Strike/Bash data set with random player Strength. |
| `rl.rules.data.convert_rule_data_anonymous` L45 | Convert collected rule samples to the anonymous-card v5 schema. |
| `rl.rules.data.make_vector_rule_smoke_data` L102 | Create a tiny schema-v5 vector-rule dataset for pipeline smoke tests. |
| `rl.rules.data.make_vector_rule_training_data` L844 | Stream fresh, randomized schema-v5 vector-rule records.  |
| `rl.rules.data.make_vector_rule_v7_data` L1245 | Generate schema-v7 effect-tree records and local sequence statistics.  |
| `rl.rules.training.distributed_train_rules` L670 | Synchronous multi-machine training for sharded combat-rule data.  |
| `rl.rules.training.train_rule_vectors` L864 | Train the standalone anonymous vector/event rule model.  |
| `rl.rules.training.train_rule_vectors_v7` L537 | Train the anonymous schema-v7 effect-tree language model.  |
| `rl.rules.training.train_rules` L718 | Train the standalone state+action -> effects+next-state rule model. |
| `rl.rules.visualize_card_vectors` L829 | Visualize learned card vectors from an sts-rl MaskablePPO checkpoint.  |
| `rl.training.distributed_sb3` L2895 | Synchronous multi-machine SB3 PPO probe/trainer.  |
| `rl.training.overfit_test` L205 | Minimal MaskablePPO training loop on SimCombatEnv.  |
| `rl.training.rllib_trainer` L1772 | Distributed RLlib PPO trainer for STS tasks.  |
| `rl.training.trainer` L1390 | Unified MaskablePPO trainer for Slay the Spire.  |
| `tools.analysis.analyze_card_representations` L214 | Compare card representations averaged over random combat contexts.  |
| `tools.analysis.analyze_relic_reward_correlation` L248 | Evaluate relic-count / shaped-return correlation for a whole-run checkpoint. |
| `tools.analysis.audit_vector_rule_v7_contract` L335 | Audit schema-v7 contract-10 local sequences, masks, and delta alignment. |
| `tools.analysis.boss_relic_probe` L200 | 检查 boss relic 与受控选项 |
| `tools.analysis.bughunt` L264 | Run a checkpoint for many whole-run episodes and catch every crash.  |
| `tools.analysis.diagnose_rules` L969 | Diagnose where a combat-rule checkpoint is making prediction errors. |
| `tools.analysis.experiment_continuous_to_number_tokens` L417 | Test exact numeric-token recovery from a normalized continuous state vector.  |
| `tools.analysis.experiment_register_retrieval` L479 | Train and probe an isolated name-to-register-value retrieval task.  |
| `tools.analysis.inspect_vector_rule_v7_self_damage` L188 | Inspect one SELF_DAMAGE local decode without running the full case suite.  |
| `tools.analysis.measure_reasoning_lengths` L642 | Measure lengths of simulator-backed whole-run reasoning transcripts.  |
| `tools.analysis.monitor_vector_v7_wandb` L148 | Backfill and live-stream a running v7 trainer log to Weights & Biases. |
| `tools.analysis.plot_card_representation_projection` L131 | Plot a PCA projection of the merged card representations. |
| `tools.analysis.probe_campfire` L445 | Compare campfire choices from multiple whole-run checkpoints.  |
| `tools.analysis.probe_cultist_bash_order` L294 | Probe whether a checkpoint plays Bash before another Ironclad Attack. |
| `tools.analysis.probe_overworld_upgrade_value` L444 | Measure overworld value-head responses to counterfactual card upgrades. |
| `tools.analysis.probe_two_louse` L301 | Probe a checkpoint on a controlled two-Louse combat observation.  |
| `tools.analysis.probe_vector_rule_v7_memory` L110 | Run one schema-v7 optimizer step at a forced decoder padding width. |
| `tools.analysis.render_card_relic_rules` L63 | Export every currently supported card and combat-relic rule on one line.  |
| `tools.analysis.route_dag_probe` L257 | Independent route-DAG experiment.  |
| `tools.analysis.route_model_probe` L950 | Probe the current whole-run route-planning path on synthetic maps.  |
| `tools.analysis.visualize_vector_rule_card_pca` L255 | Project v6 anonymous Ironclad card-rule vectors to two PCA dimensions.  |
| `tools.benchmarks.benchmark_nosl_paths` L321 | Benchmark repeated NOSL paths with one root redeterminization per path.  |
| `tools.benchmarks.benchmark_nosl_step` L287 | Benchmark one combat transition on real NOSL combat states.  |
| `tools.checkpoints.create_semantic_checkpoint` L115 | Create a fresh semantic whole-run checkpoint with an explicit model width.  |
| `tools.checkpoints.merge_card_representations` L248 | Merge sampled card vectors and compare damage with block representations. |
| `tools.validation.bitwise` L195 | Capture/compare numerical behavior across revisions on one CPU runtime.  |
| `tools.validation.eval_revisions` L59 | Compare a real checkpoint's bounded strict-search evaluation across checkouts.  |
| `tools.validation.locate_legacy` L44 | Locate a historical module/path or verify archived files against their hashes. |

### Shell / PowerShell / BAT（15 个）

| 路径 | 下游 / 职责 |
|---|---|
| `bridge/client/agent_orchestrator.bat` | `bridge.client.agent_orchestrator` |
| `experiments/launchers/launch_distributed_wholerun.ps1` | `rl.distributed_sb3` |
| `experiments/launchers/launch_nosl_fixed_fpu_20260909.sh` | `rl.distributed_sb3` |
| `experiments/launchers/launch_nosl_paper_noetl_noreplay_e288_20260913.sh` | `rl.distributed_sb3` |
| `experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh` | `rl.distributed_sb3` |
| `experiments/launchers/launch_vector_rule_v7_contract_v10.sh` | 转发同目录 launch_vector_rule_v7_contract_v9.sh，文件名不是独立 contract 实现 |
| `experiments/launchers/launch_vector_rule_v7_contract_v4.sh` | `rl.make_vector_rule_v7_data`, `rl.train_rule_vectors_v7` |
| `experiments/launchers/launch_vector_rule_v7_contract_v9.sh` | `rl.make_vector_rule_v7_data`, `rl.train_rule_vectors_v7` |
| `experiments/launchers/queue_nosl_az_unlimited.sh` | `rl.distributed_sb3` |
| `experiments/launchers/run_campfire_longterm.sh` | `rl.probe_campfire` |
| `rl/launch_nosl_fixed_fpu_20260909.sh` | `experiments/launchers/launch_nosl_fixed_fpu_20260909.sh` |
| `rl/launch_nosl_paper_noetl_noreplay_e288_20260913.sh` | `experiments/launchers/launch_nosl_paper_noetl_noreplay_e288_20260913.sh` |
| `rl/launch_nosl_paper_replay_fresh10_20260913.sh` | `experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh` |
| `sim/build_local.sh` | 构建原生扩展 |
| `tools/operations/pull_latest_ckpt.sh` | 下载既有远端 checkpoint 到本地 runs；远端旧目录保持原值 |

### 原生程序（4 个）

| CMake target | 源码 |
|---|---|
| `main` | `sim/sts_lightspeed/apps/main.cpp` |
| `test` | `sim/sts_lightspeed/apps/test.cpp` |
| `small-test` | `sim/sts_lightspeed/apps/small-test.cpp` |
| `nosl-combat-train` | `sim/sts_lightspeed/apps/nosl-combat-train.cpp` |

### 测试中的 standalone main（12 个）

这些是历史手跑表面，不为它们新增 launcher；pytest 已是维护中的统一验证入口。

- `tests.integration.test_combat`
- `tests.integration.test_game`
- `tests.integration.test_nosl_fast_step`
- `tests.integration.test_run_v2`
- `tests.unit.test_card_surface_sensitivity`
- `tests.unit.test_combat_value_search`
- `tests.unit.test_hierarchical_returns`
- `tests.unit.test_nosl_is_az`
- `tests.unit.test_nosl_sampler`
- `tests.unit.test_pure`
- `tests.unit.test_rllib_split_policy`
- `tests.unit.test_split_policy`

旧别名完整表保留在配套 JSON 的 legacy_aliases，运行时唯一维护映射为 tools/validation/migrations.json；报告不增加新的别名注册机制。

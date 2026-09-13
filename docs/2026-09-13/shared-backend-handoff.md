# 共享评估、搜索与训练后端交接

工作仓库：`D:/zxb/my/sts-rl-workbench/sts-rl`，工作分支 `chore/refactor-foundation`，交付目标 `origin/main`。本轮起点 `bc8cccc`，首次运行时验收 commit `1bef7b4`；推送前修正与干净环境验收为 `7ab49c9`，见文末复审。日期是记录时间；持久使用说明在各模块 README，任务状态以根 `TODO.md` 为准。

用户原意：删除已确认不用的 RLlib、overfit 和有继任者的旧版本；允许改变语义，合并 eval/train 重复逻辑，能共享的 eval/rollout 后端一起共享；特别要求面向未来不同策略，而非只为当前实验做窄接口。此次只执行本地工作和 commit。

## 已落地的结构

| 所属模块 | 职责与主要消费者 |
|---|---|
| `rl/evaluation/runner.py` | 环境常驻、seed/reset、批次调度、预算/取消、合法性、结果与汇总；episode CLI 和 dashboard 共用 |
| `rl/evaluation/strategies.py` | `DecisionStrategy` 批量决策协议；policy/native/value/最小合法动作适配；`PhaseStrategy` 组合与谓词路由 |
| `rl/search/native.py` | native 配置、root/leaf 推理、混合 transport 解码、合批回传；rollout、episode eval、固定场景、dashboard watch 共用 |
| `rl/evaluation/nosl/scenarios.py` | 真实固定战斗构造、卡牌/动作表、搜索诊断报告，无 CLI；配置仍在 `configs/eval/nosl_scenarios.py` |
| `rl/evaluation/baselines/routes.py` | 地图 DP 目标，供 baseline 和组合策略复用 |
| `rl/policies/checkpoint.py` | ZIP metadata 预检、外部 card encoder 依赖解析/覆盖、共享加载；预检不再加载完整权重和 Adam |
| `rl/algorithms/ppo.py` | PPO 统计目标及分层 phase 权重 |
| `rl/algorithms/updates.py` | 共享前推/目标、reasoning padding/反传、累积平均→裁剪→optimizer 更新 |
| `rl/training/distributed_sb3.py` | rank/通信/调度，PPO、NOSL、reasoning 复用同一梯度同步与更新原语 |

各模块的简短 README 已同步。`accumulating_ppo.py` 和 distributed main 仍然大；此次抽出了复用边界，并没有声称完成整个 collector/lifecycle 重写。

### 后续怎样增加策略

实现一个带 `name` 和 `decide(requests, check=...)` 的对象即可传给 `evaluate`；每个 request 提供环境、observation、info、episode seed 和 step，每个返回值是合法原子 action、来源名及可累加的数值诊断。策略可以自己持有多个模型或搜索状态，无需继承 SB3。

- 跨 episode 状态按 seed 隔离，step=0 时建立/重置；不要依赖临时批次中的行号恒定。
- 长循环主动调用 `check()`，使超时/取消能在安全边界响应。
- 按战斗、地图或显式条件组合时用 `PhaseStrategy`；无需修改 runner 去识别新的模型名。
- 新初态/任务实现 `Scenario` 的工厂、seeded reset 和 outcome；不要复制 episode loop。
- 只有需要提供 CLI 时才扩展策略工厂表和参数。当前命令行仍以 checkpoint 为输入；程序化 runner 不要求模型。不要为每种参数组合再建 `eval_某策略_某日期.py`。
- 新训练方法可以组合 shared update primitives，或实现自己的目标；采样、探索和监督属于算法。无需把所有新方法伪装成 PPO/NOSL 的布尔开关。

这些边界有任意自定义、非 SB3 策略测试，也测试了多环境复用、batch 顺序、取消、非法动作与环境清理。

## 删除与保留

删除了 RLlib trainer/inference/hybrid eval 和对应 dashboard UI/API，overfit 专用链路，旧 token-rule 模型/采集/训练/评测，已被 v7 接替的旧 vector trainer/eval，相关专属测试和 Ray 依赖。lock 从 103 个包降为 85 个。

旧 Python NOSL fullrun、每 episode 启子进程重载模型的 parallel wrapper、独立 combat eval 由 episode CLI 接管。Bash/Clash matrix/repeated/trials/strict sequence 四个旧版本由现役参数化 Full Gumbel evaluator 接管；其有用 fixture 不再藏在旧 CLI 中。

本轮总计删除 80 个旧 CLI 别名，只保留 20 个 checkpoint 库别名；三个 `rl/launch_nosl_*` shell 包装也删除，部署脚本只从 `experiments/launchers` 进入。对起点的运行时代码及相应测试/说明改动净减少约 1.7 万行。

保留具有不同实验含义的 corpus/Flat-MC/beam/MCTS 比较、卡对/HP 敏感性、starter/leftmost 基准与现役 v7 评测；不同统计问题不因同名 `eval_` 就直接删除。原有 leftmost-card baseline 会优先出牌并处理多选，不能与新增 `--solver leftmost` 的“最小合法 action ID”混称同一个策略。部分 corpus 专用工具仍有旧式私有导入，见 TODO 的 R03。

v7 实际使用的 v6 命名数据/向量基础组件保留；保留的是依赖，并非旧 trainer。原生研究对照实现也未仅凭“看起来旧”删除。唯一未分类的历史源码同步 tar.gz 仍在 artifacts，延续用户此前暂留约定。

旧路径查询统一使用 `python -m tools.validation.locate_legacy OLD_PATH`；registry 给出继任入口或退役说明。历史审计/实验档案保持冻结，不回写历史命令；归档 hash 验证仍通过。

## 主要运行入口

所有命令从仓库根目录运行；先按 `tests/README.md` 安装已选择的 CPU/CUDA 环境并指定原生构建。

```bash
# 单机训练；具体 task/参数以 --help 和 configs 为准
python -m rl.training.trainer --help

# 现役多 rank 训练；本轮未启动任何远端任务
python -m rl.training.distributed_sb3 --help

# Whole-run native 严格搜索，模型/环境常驻并合批
python -m rl.evaluation.eval CHECKPOINT --task wholerun --solver native --search-mode strict --num-envs 4 --episodes 10 --max-steps 20000 --timeout 1800 --output runs/local/eval.json

# Combat gauntlet 的无搜索策略评测
python -m rl.evaluation.eval CHECKPOINT --task combat --solver policy --episodes 10 --output runs/local/combat.json

# 固定真实战斗的严格搜索消融
python -m rl.evaluation.nosl.eval_full_gumbel_bash_clash CHECKPOINT --seeds 219 --repeats 1 --simulations 8 --depth 0 --leaf-batch 1 --modes full --paper-full-gumbel --output runs/local/strict.json

python -m apps.dashboard --help
python -m apps.watch --help
python -m rl.rules.training.train_rule_vectors_v7 --help
```

`--task combat` 当前不支持 native adapter；用 wholerun native 或固定 combat 场景接口。GUI 的 workers 表示常驻环境数量，已不再启动每 episode 的独立模型线程。

## 主动改变的语义

1. 分布式混合 NOSL 的 PPO 与本地一致，排除由 AZ 监督的 combat 样本。此前两种编排目标有分歧。全 combat/未完成宏样本的 phasewise minibatch 使用共享目标的可反传零值，不异常退出。
2. 旧 collector 混用 packed 与 canonical leaf response 时可能只保留 packed 行；共享解码按 worker/request 顺序保留所有行，并检查行数/schema。
3. 达到评测步数上限单独记 `step_limit`，不混入死亡。非法动作或策略异常显式失败并关闭环境。
4. strict native 参数集中归一；本地新模型、分布式 resume、eval 使用同一约束。非 strict 模式仍允许其参数空间。
5. 合批会改变浮点计算形状，随机策略还会改变 RNG 消费顺序。小 CPU 样本一致不能推广为任意 GPU/batch 的 bitwise 保证。

## 验证证据

软件：Python 3.11.15、Torch 2.12.0+cu132，实际计算设备为 CPU、Torch 单线程。使用同一原生扩展 SHA256 `fc1f3672104b79280257851a50e851280c9265a170383e1a22501131ecc797e3`；本轮无 C++ 修改。

| 检查 | 结果 / 本地证据 |
|---|---|
| 全量 pytest | 224 passed / 8 strict xfailed；`runs/local/shared-backend-final-tests.xml` |
| optimizer 同步收尾后全部 training marker | 7 passed；`runs/local/shared-backend-training-final.xml` |
| 基线 `9d2c6e1` → `1bef7b4` | 7208 字段 BITWISE PASS；`runs/local/shared-backend-verified-bitwise.json` 对 `refactor1-baseline.json` |
| 实际恢复四份旧 checkpoint | 纳入上述 capture；比较完整权重、Adam、rollout、有效 replay、RNG、预测和搜索，不比较 ZIP 元数据 |
| 真实 65 MB 实验 checkpoint | strict 固定小场景 8403 JSON 字节与基线相同；`runs/local/shared-backend-real-eval/comparison.json` |
| 新目标语义 | mixed/all-combat × combined/phasewise 四组合梯度回归；本地与单 rank loopback 两次真实 replay 更新权重/Adam 逐位相同 |
| 静态/布局 | compileall、ruff 未定义名、dashboard JS、7 个 Bash 脚本、PowerShell AST、uv lock、历史归档 hash 通过 |

真实 checkpoint SHA256：`cf8336a7a01a736d6ce36bff9726cc42b305f2235a0c3f23ed1d1b8e25470dd0`。跨版本工具会在两个 checkout 中分别选择可用入口，当前仓库无需继续保留旧 CLI。

8 个既有 xfail 对应 F01/F02/F03/F04（三个实例）/F05/F07。它们没有被重构掩盖，也没有因结构更整齐就被宣称修复。总测试数下降主要来自退役功能专属测试与旧别名参数；新增现役合同仍保留。

## 性能与开发便利性

真实 checkpoint、CPU 单线程、4 episodes × 32-step 上限，1 与 4 常驻环境交错各测三次。原始证据 `runs/local/shared-eval-benchmark.json`；这是同一新后端的 batch 消融，不是拿旧 Python 搜索与新 native 算法混比。

| 策略 | 1 env 中位耗时 | 4 env 中位耗时 | 比值 | root 前推次数 |
|---|---:|---:|---:|---:|
| policy | 3.055 s | 1.984 s | 1.54× | 128 → 32 |
| native，4 simulations | 7.298 s | 5.116 s | 1.43× | 128 → 49 |

这些小样本的逐局结果（除计时）hash 一致。RSS 采样约 682–696 MiB，进程树 CPU 99–106%（一个逻辑核=100%，包含遥测辅助进程）。GPU 遥测为设备级，模型实际跑 CPU，因此不能从中推断模型的显存占用或 GPU 吞吐。

训练侧已减少 reasoning token 的逐行 GPU 传输、共用 leaf packing，并避免不记录梯度 norm 的调用方额外取 CPU 标量。未在受控 GPU/多机环境测量训练吞吐收益；CPU/GPU/显存/内存的正式基准工具仍是 `rl.benchmarks.train`，稳定性能验收见 PERF01。

开发上，添加普通策略不再复制 eval loop；修改 native leaf inference 不再同时追踪 GUI、collector 和固定评测的多份实现；修改 PPO 目标不再各改本地和分布式；退役路径有明确可查询去向。代价是共享边界改动影响更多调用方，因此本轮增加了跨后端、批次顺序与真实更新合同。

## 交接边界

后续任务只维护根 TODO：R02 处理大 collector/main 生命周期；R03/F16 处理剩余 corpus 协议、固定评测集、有效 solver manifest、硬超时。当前 deadline/cancel 是 step/leaf batch 边界检查，不能中断卡死的 native 调用。推送前复审已补上 CLI 异常/超时的部分结果保存及失败标记，但硬杀进程仍无法保证落盘。F13 的完整依赖 manifest、F14 的原生能力版本化，以及游戏状态/动作/HP 等 F 项仍需独立设计。

没有执行 remote launcher、远端训练同步、GPU bitwise、多机 ring 或长程胜率实验。单 rank loopback 测试只验证更新与传输适配边界，不代表真实网络容错已经验收。首次交接尚未做 fresh install；推送前独立安装/重建已补验，见下节。托管 CI 未通过当前访问路径确认状态。

提交顺序：`0bff237` 退役旧链路 → `98a9aa0` 共享搜索/评估 → `0210e98` 共享训练更新 → `42b1dfe` 收拢场景与 CLI → `1bef7b4` 梯度 norm 同步收尾 → `1c3c301` 首次交接 → `1bde5eb` 异常边界/新人说明 → `7ab49c9` 构建依赖修正。推送前验收记录另作文档提交。

## 推送前复审与新维护者入口

用户进一步授权检查后推送 `origin/main`。fetch 后远端 main 为 `1325041`，是全部本地改动的祖先；使用正常 fast-forward，不强推。新读者从根 README 的“第一次接手”开始，按 tests/README 完成环境激活/原生编译与无 checkpoint 冒烟，再根据 policies/README 配齐模型依赖；无需先阅读以前的重构过程。

复审补修四项：dashboard 策略初始化失败进入 error 状态；CLI 超时/异常/Ctrl-C 保留完成的评测行且仍以失败退出；metadata 预检支持 SB3 省略 `.zip` 的写法；CMake 4.4.3 与内置 json 3.1 policy 不兼容，依赖约束到已实测的 3.31.10。专属回归保留在 tests/unit。

以 `7ab49c9` 建独立干净工作树，`.venv` 禁用 system-site-packages，执行文档中的 `uv sync --locked --extra cpu --group test --no-dev --python 3.11`；Python 3.11.15、Torch 2.12.0+cpu、CMake 3.31.10、GCC 13.2，原生库从源码重建。新库 SHA256 为 `1b1e377f421d810409b4b6f7bc06b6fe425502a2c87a92d490e0dfd13809a2de`。

- 全量 **228 passed / 8 strict xfailed**，94.12 秒；缺少原生库、额外本地文件或用户全局 Python 包均未被用来跳过测试。
- README 的首次训练冒烟成功：128 个测量 transitions、真实优化更新，端到端约 17.3 秒；仅验证可运行，不当作稳定性能排名。
- 六个主要 CLI 的 `--help`、32 份维护文档的 56 个本地链接、7 个 Bash 脚本、JS/模块语法及完整待推送 diff 空白检查通过。
- 在原固定解释器/原生库上重跑旧 checkpoint 恢复，**7208-field bitwise 通过**；真实 checkpoint 固定评测仍为 **8403 字节一致**。独立安装的 CPU wheel 与新构建用于运行验证，不与原二进制混称跨环境 bitwise。

验证文件统一归 `runs/local/prepush-*`；工作日志记录具体路径。GitHub Actions API 对当前未认证读取返回 404，未因此声称 hosted CI 已通过；仓库工作流会按其 push 触发配置运行。

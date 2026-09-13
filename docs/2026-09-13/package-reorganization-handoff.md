# 重构一阶段：架构与接续指南

本阶段从 `9d2c6e1` 开始，目标是理清职责、隔离配置/测试/产物、保持可执行行为，为后续内部重构和 eval 优化提供基线。工作分支为 `chore/refactor-foundation`。具体进度、验收结果和提交记录只维护在 `worklog/2026-09-13.md`；长期任务只维护在 `TODO.md`。

## 从哪里读代码

| 要理解什么 | 首先阅读 |
|---|---|
| 当前训练的入口与默认值 | `rl/training/distributed_sb3.py::build_parser/main` |
| 单机训练/task 组合 | `rl/training/trainer.py`、`rl/tasks/` |
| PPO/分层/NOSL 收集与更新 | `rl/algorithms/accumulating_ppo.py` |
| fresh/old replay、容量、有效样本 | `rl/algorithms/nosl_replay.py` |
| 共享编码器与 combat/overworld 头 | `rl/policies/model.py`、`split_policy.py` |
| Gym 状态与 native 搜索接口 | `env/core/run_env.py` |
| 无模型的 worker session/nonce | `env/search/` |
| native strict Full Gumbel | `sim/sts_lightspeed/src/sim/search/Nosl.cpp` |
| 真实实验模型的小场景评估 | `rl/evaluation/nosl/eval_full_gumbel_bash_clash.py` |
| whole-run policy 评估 | `rl/evaluation/eval.py` |
| 规则学习与数据生成 | `rl/rules/`；准确率/解释在 `rl/evaluation/rules/` |
| GUI/终端观看 | `apps/dashboard.py`、`apps/watch.py` |
| 真机边界 | `bridge/schemas/`、`bridge/client/` |

这些目录都有简短 README。`rl` 顶层不再放具体 Python 实现；100 个旧 flat module 已归位。巨型 collector/learner/main 尚未拆成更多对象：移动包边界与改变控制流分开验收。

## 已建立的边界

- `env` 不导入 `rl/apps/tools/tests`。原来 worker 从 rl 借用的无模型 NOSL session/nonce 已归 `env/search`；value/policy 驱动的学习侧搜索仍在 `rl/search`。
- 学习运行时不反向导入 GUI 或工作工具。规则评估实际共享的场景解释/渲染保留在其归属包，而非塞进临时脚本目录。
- 测试只由 `tests/` 收集：`unit/` 与 `integration/`。历史断言完整迁移；旧 `overfit_test.py` 实际是训练命令，位于 `rl/training/`。
- `configs/train/*.args` 是原生 argparse response file；`configs/eval/nosl_scenarios.py` 只有纯数据。包含环境构造/搜索/计分的 eval 程序仍是执行代码。
- `experiments/launchers` 管机器地址、rank、run/W&B 身份和启动；`tools` 管工作工具。现有大型模型和 `rl/data` 的序列化路径依赖保留，后续按 F13/F17 单独迁移。
- 当前输出归 `artifacts/local`，历史原始输出归 `artifacts/archive`，日期进度/阶段报告归 `worklog`，可复用协议/接口说明归 `docs`。

边界由 `tests/unit/test_repository_layout.py` 检查，不能只靠 README 约定。

## 兼容方式与限制

旧 `rl.<module>` 从 `rl/_compat/` 惰性加载 canonical 模块。该目录是 `rl.__path__` 的第二个搜索目录，使用普通 Python import 机制；别名与新模块共享一个 `sys.modules` 对象，不复制类或全局状态。旧 `python -m rl.<module>` 也转到新入口。

这用于旧 SB3/cloudpickle checkpoint 和已有实验脚本。新代码必须使用新路径。模块移动不改变 state_dict key、tensor shape 或动作/观测定义。验证覆盖旧 checkpoint 在新代码中加载；不承诺新代码保存的 checkpoint 可以在未迁移的旧 checkout 中反向加载。

三个 `rl/launch_nosl_*.sh` 路径保留薄转发；其他旧 shell 路径查 `docs/launcher-migration.json`。迁移后的整个仓库应一起使用，不能只拷新 launcher 而漏掉其 `.args` 文件。

旧位置查询：

```bash
python -m tools.validation.locate_legacy rl.distributed_sb3
python -m tools.validation.locate_legacy docs/current-wholerun-training.md
python -m tools.validation.locate_legacy --verify
```

三个映射文件分别记录模块、launcher 和历史文件。200 个归档/重定位文件保留原字节及 SHA256。历史文档内的相对链接可能指旧位置，先查映射；归档内容不是当前规范。`.gitattributes` 只对历史工作记录关闭空白规范化检查，以保留原始材料。

## 当前实验如何定位

用户提供的原始材料在 `worklog/archive/input/current-nosl-paper-replay-training.md`。这是 2026-09-13 的快照，不能当作实时状态。此次没有连接或重启四个远端 rank。

当前部署配置为 `experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh`，训练数值参数为 `configs/train/nosl_paper_replay_fresh10_20260913.args`。保持 strict Full Gumbel、depth=0、leaf batch=1、128 simulations、replay capacity 131072、old fraction 上限 0.90、24576 batch、40 updates、2048 microbatch 和每 rank 24 env。

模型冻结范围、boundary 特征与外层模型仍由原 checkpoint 和训练代码共同决定。参数文件不是完整 checkpoint manifest；该问题属于 F13。历史 no-replay/e288 与 fixed-FPU 配置单独保留，没有混成当前默认。

真实快照 `nosl_az_replay_fresh10_1843200_steps.zip` 的 SHA256 为 `cf8336a7a01a736d6ce36bff9726cc42b305f2235a0c3f23ed1d1b8e25470dd0`。验证使用仓库 artifacts 内匹配的副本，不依赖远端状态。

## 怎样验证下一次重构

完整命令见 `docs/development-checks.md`。至少执行：

```bash
python -m pytest -q
python -m tools.validation.bitwise capture --output artifacts/local/after.json --restore ABS_BASELINE/artifacts/local/before-checkpoints
python -m tools.validation.bitwise compare ABS_BASELINE/artifacts/local/before.json artifacts/local/after.json
python -m tools.validation.eval_revisions --baseline-repo ABS_BASELINE --candidate-repo . --checkpoint CHECKPOINT --output-dir artifacts/local/real-eval
```

基线 checkout 可用 `e42a0cd`：它只在 `9d2c6e1` 运行时上加了跨版本 capture。先在基线生成 before.json，再改代码；不要拿两次重构后运行替代前后对照。

数值 capture 比较 7,208 个字段：普通/分层 PPO、PUCT、strict Full Gumbel replay 的权重、Adam、rollout、RNG、有效 replay、旧 checkpoint 恢复后的八步评估，以及 native 搜索。每个模式 64 transitions，单线程 CPU，同软件和同 native binary；用 dtype/shape/原始字节哈希判断相等。短战斗 fixture 使用额外 Strength 缩短运行，不用于评估玩法强度。

真实模型评估固定 seed 219、8 simulations、严格模式；两份完整 JSON 逐字节相等才通过。它是加载/前推/搜索兼容检查，不是完整游戏胜率证明。

本次共享 native 构建来自前一阶段 audit checkout，SHA256 为 `fc1f3672104b79280257851a50e851280c9265a170383e1a22501131ecc797e3`；一阶段没有修改 C++。使用本仓库时按 development-checks 显式编译/选择构建，不能默认认为已有 `build-gumbel-current` 与当前源码相同。

## 接续时不要误判的事项

F01/F02/F03/F04/F05/F07 等已知语义缺陷仍存在，并保留 8 个 strict xfail。bitwise 相同证明本阶段未改变被测行为，不证明原算法/规则本身正确。

没有验证四 rank SSH 的完整同步训练、bf16 GPU 跨版本 bitwise、长程收敛、完整游戏回放或 hosted Actions。Python/依赖、native 编译器或设备变化后，重新建立基线，不套用本机哈希结果。

下一阶段的长期任务位置是 TODO 的 R02/R03 及 F05/F13/F16/F17。接手时先读当日日志和相关模块 README，选择一个 collector/learner/评估协议边界做小提交，再跑对应的前后数值门禁。eval 运行效率按用户意图延期：先固定 solver/终值/场景协议，再测批量前推、编码和 IPC 的代价；此次没有声称吞吐提升。

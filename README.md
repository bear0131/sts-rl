# sts-rl

在 C++ Slay the Spire 模拟器上训练和评估策略，并通过 bridge 接入真实游戏。

项目主要研究 Ironclad 的整局决策和战斗搜索。通常先在模拟器中做实验，不需要启动游戏客户端；真实游戏接入是独立的 bridge 路径。这里有两种任务：`wholerun` 从地图、事件、选牌到战斗推进一整局；`combat` 是随机装备/牌组的连续战斗训练，以能打过多少场衡量表现。

一次训练的流程是：环境给出观测和合法动作 → 策略网络直接选动作，或搜索多个可能后续再选动作 → 环境执行 → 收集轨迹并更新网络。评估只做决策和计分，不更新权重。PPO 是直接从轨迹学习的算法；NOSL 搜索重新采样不可见的未来随机状态；AZ learner 使用搜索给出的动作分布和战斗结果监督网络。本文中的 strict Full Gumbel 是一种具体搜索配置，不是整个项目的唯一策略接口。

## 第一次接手

1. 克隆仓库并进入根目录。准备 Git、uv、支持 C++17 的 GCC/Clang（Windows 用兼容 MinGW）；Python/CMake/Ninja 等依赖按 [构建与测试说明](tests/README.md#runtime-and-native-build) 安装。首次验证选单线程 CPU，无需 GPU、W&B 账号或训练 checkpoint。
2. 编译 Python 原生扩展并设置 `STS_LIGHTSPEED_BUILD_DIR`，运行 `python -m pytest -q`。8 个已知游戏契约缺陷有 strict xfail，解释在 [TODO](TODO.md)；缺少原生扩展是错误，不算测试通过。
3. 跑一份有预算的小训练：`python -m rl.benchmarks.train --device cpu --output runs/local/first-smoke.json --timeout 180`。它从随机小模型开始，输出实际更新、运行时间和资源记录；结果文件不能覆盖已有证据。
4. 评估已有策略时，将 `CHECKPOINT` 换成实际 SB3 `.zip` 路径。部分模型另需训练时配套的 card encoder `.pt`，通过 `--card-vector-checkpoint` 或 `STS_CARD_VECTOR_CHECKPOINT` 指定；不要拿任意同形状文件替代。模型与数据约定见 [policies](rl/policies/README.md)、[runs](runs/README.md)。
5. 修改代码前看 [学习模块职责](rl/README.md) 和 [AGENTS](AGENTS.md)。新增策略先读 [评估接口](rl/evaluation/README.md)，修改规则先读 [环境契约](env/README.md)。部署脚本含既有机器/run 配置，不能直接作为首次启动示例运行。

本仓库是研究代码；结构与回归门禁不代表所有卡牌语义或实验结论已经正确。[共享后端交接](docs/2026-09-13/shared-backend-handoff.md) 记录改动范围与验证证据；[TODO](TODO.md) 才是持续维护的问题状态。

## 代码与文件归属

| 目录 | 职责 |
|---|---|
| [sim](sim/README.md) | 原生游戏规则、搜索树、Python binding |
| [env](env/README.md) | canonical 状态、Gym 环境、动作/观测与 worker 搜索 |
| [rl](rl/README.md) | 算法、策略、训练编排、评估与规则学习 |
| [apps](apps/README.md) | dashboard、终端观看 |
| [bridge](bridge/README.md) | 真机协议、规范化与差分回放 |
| [configs](configs/README.md) | train/eval 数值参数、纯场景配置 |
| [experiments](experiments/README.md) | 机器/路径/run 身份与部署脚本 |
| [tests](tests/README.md) | 单元、原生契约、训练集成测试 |
| [tools](tools/README.md) | 诊断、迁移、验证与性能探针 |
| [docs](docs/README.md) | 按日期保存的审计、方案和交接记录；旧材料在 archive |
| [worklog](worklog/) | 日期进度与历史实验记录；唯一长期任务表是 [TODO](TODO.md) |
| [runs](runs/README.md)、[data](data/README.md) | 运行输出/模型与数据输入 |

从仓库根目录运行。先按 [development checks](tests/README.md) 安装依赖、编译并显式指定原生扩展，随后：

```bash
python -m pytest -q
python -m rl.training.trainer --task wholerun --help
python -m rl.training.distributed_sb3 --help
python -m rl.evaluation.eval CHECKPOINT @configs/eval/policy_smoke.args
python -m apps.watch CHECKPOINT
python -m apps.dashboard --help
```

依赖只维护 pyproject.toml 和 uv.lock：`uv sync --locked --extra cpu --group test --no-dev --python 3.11`；GPU 开发选 `--extra cuda`。具体构建与选择方式见 tests/README。允许既有远端 Python 3.12，但此次仅验证本地 3.11；native 二进制不能跨解释器混用。
实验预设说明见 configs/README.md，原始说明快照见 docs/archive/input；它不是远端实时状态。启动命令统一使用归属包；旧 CLI 已退役，仅保留历史 checkpoint 所需的库模块别名。旧路径用 tools.validation.locate_legacy 查询。

规范：[AGENTS](AGENTS.md)；问题与完整证据：[TODO](TODO.md)。阶段报告从 [docs](docs/README.md) 按文件标题查阅。
四机训练启动、每轮 learner 保存和监测用法见 [运维说明](tools/operations/README.md)；性能测量与恢复边界见 [NOSL 性能/恢复交接](docs/2026-09-14/nosl-performance-recovery-handoff.md)。
仅剩一个未归类源码传输包在 artifacts/，详见该目录 README。

# sts-rl

在 C++ Slay the Spire 模拟器上训练和评估策略，并通过 bridge 接入真实游戏。

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
| [docs](docs/README.md) | 持续复用的协议、接口、构建与迁移说明 |
| [worklog](worklog/) | 日期进度与历史实验记录；唯一长期任务表是 [TODO](TODO.md) |
| [artifacts](artifacts/README.md)、[data](data/README.md) | 输出/模型与数据输入；新本地输出写 artifacts/local |

从仓库根目录运行。先按 [development checks](docs/development-checks.md) 安装依赖、编译并显式指定原生扩展，随后：

```bash
python -m pytest -q
python -m rl.training.trainer --task wholerun --help
python -m rl.training.distributed_sb3 --help
python -m rl.evaluation.eval CHECKPOINT @configs/eval/policy_smoke.args
python -m apps.watch CHECKPOINT
python -m apps.dashboard --help
```

本地 Python 3.11 与 pyproject.toml/uv.lock 对齐；远端历史训练镜像有 Python 3.12，不能混用其 .so 与本地 .pyd。Torch 和 native 构建身份属于复现条件。
当前实验的已知设置见 configs/README.md，原始说明快照见 worklog/archive/input；它不是远端实时状态。旧 `python -m rl.<module>` 和 checkpoint 模块名继续兼容，新代码使用归属包。

一阶段接续入口：[refactor-stage1](docs/refactor-stage1.md)。规范：[AGENTS](AGENTS.md)。

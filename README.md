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
当前实验的已知设置见 configs/README.md，原始说明快照见 worklog/archive/input；它不是远端实时状态。旧 `python -m rl.<module>` 和 checkpoint 模块名继续兼容，新代码使用归属包。

规范：[AGENTS](AGENTS.md)；问题与完整证据：[TODO](TODO.md)。阶段报告从 [docs](docs/README.md) 按文件标题查阅。
仅剩一个未归类源码传输包在 artifacts/，详见该目录 README。

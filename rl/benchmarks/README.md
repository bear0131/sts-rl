# 有预算的训练资源基准

`python -m rl.benchmarks.train --help`：同一入口测 PPO/分层 PPO/native NOSL，支持 dummy/subproc/grouped worker；父进程 wall timeout 杀掉其子进程树并记录 JSON。
`resources.py` 采样进程树 CPU/RSS 与设备 GPU 指标。所有结果包含软件/native 身份、实际 transition 数、warmup/collection/optimization 时间；没有跨硬件 FPS 门槛。
更低层的历史探针在 tools/benchmarks。解释口径与命令见 ../../docs/development-checks.md；不要用短程 GPU util 单独判断算法瓶颈。

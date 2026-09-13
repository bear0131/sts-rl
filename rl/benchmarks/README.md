# 有预算的训练资源基准

`python -m rl.benchmarks.train --help`：同一入口测 PPO/分层 PPO/native NOSL，支持 dummy/subproc/grouped worker；父进程 wall timeout 杀掉其子进程树并记录 JSON。
`resources.py` 采样进程树 CPU/RSS 与设备 GPU 指标。所有结果包含软件/native 身份、实际 transition 数、warmup/collection/optimization 时间；没有跨硬件 FPS 门槛。
更低层的历史探针在 tools/benchmarks。解释口径与命令见 ../../tests/README.md；不要用短程 GPU util 单独判断算法瓶颈。

同一入口的 `--workload leaves --checkpoint MODEL --leaf-repeats 300 --profile`
测固定战斗叶批，导出 JSON、CPU/CUDA 算子表和 Chrome trace。
`--workload rollout --checkpoint MODEL --search-config SEARCH.json` 使用生产
checkpoint 环境工厂与采样器，固定 n-envs/n-steps/updates，无优化器更新；
SEARCH.json 是 `NativeSearchConfig` 字段。可对比 `--leaf-inference eager/cuda-graph`
与 `--vec-env subproc/grouped --envs-per-worker N`。输出 hash 覆盖最后一个
rollout 的观测、动作、奖励、value、log-prob 和完整战斗标签；计时不进入摘要。
采样 workload 包含首次 CUDA graph capture，不应冒称全部是稳定热态。

`--workload leaves --leaf-input FAILURE.npz --checkpoint MODEL` 可直接重放保存的
叶输入；与自动进战斗 fixture 共用后续计时/后端，不需要另写临时推理程序。
当单批可通过而长采样失败时，需继续检查 graph/allocator 生命周期，不能据此
排除运行时错误。GPU 合同包含跨多次捕获/淘汰和释放外部 BLAS workspace 后重放。

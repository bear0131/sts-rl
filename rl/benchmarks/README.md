# 有预算的训练资源基准

`python -m rl.benchmarks.train --help`：同一入口测 PPO/分层 PPO/native NOSL，支持 dummy/subproc/grouped worker；父进程 wall timeout 杀掉其子进程树并记录 JSON。
`rl/training/resource_monitor.py` 为训练和基准共用的进程树 CPU/RSS、cgroup 与设备 GPU 采样器。所有结果包含软件/native 身份、实际 transition 数、warmup/collection/optimization 时间；没有跨硬件 FPS 门槛。
更低层的历史探针在 tools/benchmarks。解释口径与命令见 ../../tests/README.md；不要用短程 GPU util 单独判断算法瓶颈。

同一入口的 `--workload leaves --checkpoint MODEL --leaf-repeats 300 --profile`
测固定战斗叶批，导出 JSON、CPU/CUDA 算子表和 Chrome trace。
`--workload rollout --checkpoint MODEL --search-config SEARCH.json` 使用生产
checkpoint 环境工厂与采样器，固定 n-envs/n-steps/updates，无优化器更新；
SEARCH.json 是 `NativeSearchConfig` 字段。可对比 `--leaf-inference eager/cuda-graph`
与 `--vec-env subproc/grouped --envs-per-worker N`。输出 hash 覆盖最后一个
rollout 的观测、动作、奖励、value、log-prob 和完整战斗标签；计时不进入摘要。
采样 workload 包含首次 CUDA graph capture，不应冒称全部是稳定热态。

Rollout 对照参数：`--inference-schedule fixed_wait|adaptive`、
`--inference-batch-size N`、`--inference-wait-ms N`（0 表示有结果即可发送）、
`--leaf-graph-batch-multiple N`、`--graph-profile-interval N`。固定同一 checkpoint、
native 扩展、seed、env/search 预算，先短测再验证生产长度。分桶或异步调度改变
实际合批形状时，不能要求整场搜索轨迹摘要 bitwise 相同。
Leaves workload 在计时前自动对比 graph 与同形状 eager（逐位），并报告相对
原始未补齐 eager 的 logits/value 最大、均值和 p99 绝对误差；可用实际叶 NPZ
重复验证，不能只靠随机小模型证明生产 checkpoint 的数值误差。

分布式入口 `--rollout-resource-interval 2` 每2秒采样，逐 rank 的 JSON 日志
`rollout_resources` 包含 CPU（100%=一核）、进程树 RSS、容器内存/CPU 配额、
GPU util 与设备显存峰值。RSS 会重复计算共享页，设备采样可能包含其他进程；
缺失指标是 null。关闭采样用0；采样器只覆盖 collection，不混入 backward。

`--workload leaves --leaf-input FAILURE.npz --checkpoint MODEL` 可直接重放保存的
叶输入；与自动进战斗 fixture 共用后续计时/后端，不需要另写临时推理程序。
当单批可通过而长采样失败时，需继续检查 graph/allocator 生命周期，不能据此
排除运行时错误。GPU 合同包含跨多次捕获/淘汰和释放外部 BLAS workspace 后重放。

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
`--warmup-updates 0` 的采样包含首次 CUDA graph capture/编译；设为1则先执行一个
完整collection，分别报告warmup_seconds与测量段时间/动作数，不能隐藏启动成本。

Rollout 对照参数：`--inference-schedule fixed_wait|adaptive`、
`--inference-batch-size N`、`--inference-wait-ms N`（0 表示有结果即可发送）、
`--leaf-graph-batch-multiple N`、`--graph-profile-interval N`。固定同一 checkpoint、
native 扩展、seed、env/search 预算，先短测再验证生产长度。分桶或异步调度改变
实际合批形状时，不能要求整场搜索轨迹摘要 bitwise 相同。
Leaves workload 在计时前自动对比 graph 与同形状 eager（普通graph逐位；Entity
AR BF16融合按logits/utility/合法动作概率容差），并报告相对
原始未补齐 eager 的 logits/value 最大、均值和 p99 绝对误差；可用实际叶 NPZ
重复验证，不能只靠随机小模型证明生产 checkpoint 的数值误差。

本机显存/编译兼容测试使用同一入口的 `--leaf-graph-capacity N`（默认32）、
`--no-leaf-compile`（默认开启编译）和 `--leaf-host-wait spin|block`（默认spin）。
这些参数只支持 leaves/rollout 的 cuda-graph 后端，并写入结果配置；不改变训练入口默认值。
缩小缓存会降低显存需求，也可能增加捕获成本；关闭编译仍执行CUDA图，但不代表生产融合性能。
block让CPU在等待叶批完成时休眠，可能增加唤醒延迟；比较时同时看端到端时间与进程CPU时间。
图生命周期回归覆盖两种等待方式，性能候选还须经过同模型、同seed的生产长度collection。

`--model-workload virtual|shadow` 仅用于 rollout 性能对照：两者向搜索返回相同的
零 logits 与 value=0.5。virtual 跳过神经计算；shadow 执行实际模型但丢弃其结果。
这样可严格比较相同树工作量与完整输出 hash，不把神经输出变化导致的叶数差当加速。
它们不测策略质量。`controlled_model_call_seconds` 包含同步及调用包装，不能当纯 GPU
kernel 时间；graph 的 `replay_ms` 是按 `--graph-profile-interval` 采样的图内设备事件总和，
不再包含Python逐个提交事件的空隙，但仍可能包含驱动/设备调度间隔。开启计时会在图中
增加事件节点，端到端速度同时用interval=0验证。virtual也保留实际H2D/D2H，
仅将网络替换为GPU常量输出；早期直接返回NumPy常量的结果只代表CPU下界。
`--vec-env native --native-search-threads 4` 使用与训练相同的常驻原生搜索路径。
比较必须保持 checkpoint、seed、环境数、搜索预算与 warmup 相同，并报告完整采样、
模型计算、编译/capture、观测转换、真实环境 step 与资源指标。

分布式入口 `--rollout-resource-interval 2` 每2秒采样，逐 rank 的 JSON 日志
`rollout_resources` 包含 CPU（100%=一核）、进程树 RSS、容器内存/CPU 配额、
GPU util 与设备显存峰值。RSS 会重复计算共享页，设备采样可能包含其他进程；
缺失指标是 null。关闭采样用0。分布式入口的同一个采样器在 collection 结束时冻结
`rollout_resources`，继续采集到本轮 learner、同步、周期保存及 checkpoint 发布完成，
输出 `iteration_resources`。两者的 `elapsed_seconds`、`tree_cpu_seconds` 表示各自范围，
不要把两个重叠范围相加；W&B 排队及整个 run 的最终收尾保存不在单轮范围。
独立 rollout benchmark 仍只测 collection，不能代替完整训练轮验收。
分布式资源记录还包含指定 CUDA 设备的进程内 `torch_allocated_bytes`、
`torch_reserved_bytes` 当前值和对应 `torch_*_peak_sampled_bytes`。这些峰值是采样下界，
不是 CUDA allocator 的精确高水位；监控器不重置训练的峰值计数。
CPU、尚未初始化 CUDA 或查询不可用时为 null，原因见 `torch_memory_unavailable_reason`。
采样同时记录可用的GPU核心/显存时钟、功耗、温度、P-state及笔记本是否接电；
比较前核对供电和时钟，`power_plugged_observed`出现两个值表示采样期间供电改变，
桌面机或无电池传感器为null。不能把跨供电条件的耗时差归为代码收益。

`--workload leaves --leaf-input FAILURE.npz --checkpoint MODEL` 可直接重放保存的
叶输入；与自动进战斗 fixture 共用后续计时/后端，不需要另写临时推理程序。
当单批可通过而长采样失败时，需继续检查 graph/allocator 生命周期，不能据此
排除运行时错误。GPU 合同包含跨多次捕获/淘汰和释放外部 BLAS workspace 后重放。

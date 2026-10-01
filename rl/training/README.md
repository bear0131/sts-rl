# 训练编排

`trainer.py` 是单机 task CLI；`distributed_sb3.py` 是 SSH ring 多 rank 训练编排。RLlib 与 overfit 专用训练入口已退役，需要历史实现时从 Git 恢复。

`cached_mask_vec_env.py` 处理 worker 与 mask/packed leaf IPC；`ring_allreduce.py` / `ssh_transport.py` 处理同步；`reward.py` 提供训练 reward wrapper。小预算训练使用 rl.benchmarks.train，与生产共享算法。

当前实验入口为 `python -m rl.training.distributed_sb3`。算法参数、机器地址、结果目录分别见 configs/train、experiments/launchers 和忽略的 runs/local。

PPO、NOSL 与 reasoning 使用同一个梯度同步边界（ring 或 file transport）和 algorithms/updates.py 的 optimizer 更新；编排负责 rank、barrier、批次调度和日志。新的策略/学习方法应显式选择其目标和采样方式，不能通过复制整套 trainer 来添加算法。大 main 的生命周期拆分仍待做。

环境构造与 checkpoint schema 适配在 `environments.py`，采样基准复用同一工厂。
`--vec native --native-search-threads 4` 将环境留在当前进程，原生搜索线程池并行处理
不同树的 select/backup；每棵树仍只有一个待求值叶，模型每波只接收一个数字数组批。
支持 compact legacy/joint-v2/autoregressive-v2（含 card-language），不支持 combat-language
或同树多个在途叶。原生扩展须重建；缺失批处理接口会明确报错。
此模式的环境真实 step 仍由 Python 编排，只有搜索/叶编码走 C++；默认后端保持 subproc。
无叶子 IPC 不等于端到端零开销，`native_tree_seconds` 与 `native_pack_seconds` 是
begin/advance 的内部拆分，不能再次加到 collection 总时间。环境 checkpoint 状态沿用相同合同。
`--leaf-inference cuda-graph` 仅加速 CUDA compact combat 叶节点；其它路径默认 eager。
`--leaf-graph-batch-multiple`、缓存生命周期和数值边界见 [搜索说明](../search/README.md)。
Grouped worker 对同一组 native 数字观测按字段合并 IPC，调用方仍收到原来的
逐环境 RPC 结果；混合终局、不同 batch 长度与请求顺序有真实子进程回归。
`--rollout-resource-interval 2` 在每个 rank 的 JSON 记录中加入 collection 的
CPU/RSS/cgroup/GPU 采样，0 关闭；定义与复现实验入口见 [资源基准](../benchmarks/README.md)。

生产恢复默认使用 `--training-state-dir DIR --save-every-updates 1 --max-checkpoints 3`。
在需要测量 checkpoint 固定成本时，launcher 可通过
`NOSL_SAVE_EVERY_UPDATES=N` 将完整 recovery bundle 改为每 N 个 update 保存一次；
stop-file 和正常结束仍强制发布最后一个 bundle。该选项只改变恢复窗口，不改变采样、搜索、loss、optimizer 或 replay。
每个 rank 各存 `DIR/rankN/step-XXXXXXXXXXXX/{model.zip,learner.pt,manifest.json}`，
支持环境恢复的 schema 2 另存 `environment.pt`。
manifest 最后原子发布，包含配置、run/W&B ID 和文件 SHA256。恢复时四 rank 必须
选同一个已完整保存的 global step，传入各自的 `--resume-state`；不得用 rank 0 的
replay 代替其它 rank。模型/Adam、有效 replay slots、父进程 Python/NumPy/Torch/CUDA
RNG 恢复，global step 连续。完整性或 rank/world 不匹配立即拒绝恢复。

schema 2 恢复原生环境、wrapper 状态、当前观测、episode 窗口和未结束战斗的
pending labels，并校验恢复观测摘要。schema 1 的旧 bundle 只恢复 learner，需重启
环境。显式 `--fork-reset-environments` 也会重启环境并丢弃环境局部进度；改变环境
数量的 fork 必须使用这一选项。恢复与 fork 不重新初始化已训练的 shared embeddings。
新环境不安装已退役的 occupancy 预热 wrapper；旧 schema 2 快照若记录已 release
的 wrapper，恢复时重建其 episode bookkeeping。尚未 release 的旧快照明确拒绝，
需用 `--fork-reset-environments` 重启环境，避免恢复后永久等待。
learner/environment 使用 pickle protocol 5 保存 NumPy buffer，保持 bundle schema
不变；历史 protocol 2 仍可读取。实际恢复须使用对应 rank 的完整 bundle。
同一下一训练 batch 的 learner/Adam/RNG 有逐位回归；异步搜索调度及 CUDA 非确定性
仍可能改变后续轨迹，不能将状态恢复等同于整场训练的逐位重演。
每轮完整快照包含每 rank 的 replay，写盘成本需计入实际生产吞吐。
`dist/iteration_wall_sec` 与吞吐分母包含指标聚合、恢复点写盘和跨 rank 发布等待，
`dist/checkpoint_sec` 单独记录保存/发布等待；不包含随后 W&B 客户端排队的时间。
原 `dist/rank0_total_sec` 保留旧计时口径，避免误当作新端到端口径。

`--stop-file PATH` 在下一次更新/跨 rank 同步后触发保存并协同结束；多机要求 ssh-ring。
正常维护创建所有 rank 的 stop 文件并等待全部退出、核对同一恢复代数，再删除这些
请求文件并恢复。不要在训练中直接 kill 后声称最后一轮已保存。

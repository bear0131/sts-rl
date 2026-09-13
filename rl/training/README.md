# 训练编排

`trainer.py` 是单机 task CLI；`distributed_sb3.py` 是 SSH ring 多 rank 训练编排。RLlib 与 overfit 专用训练入口已退役，需要历史实现时从 Git 恢复。

`cached_mask_vec_env.py` 处理 worker 与 mask/packed leaf IPC；`ring_allreduce.py` / `ssh_transport.py` 处理同步；`reward.py` 提供训练 reward wrapper。小预算训练使用 rl.benchmarks.train，与生产共享算法。

当前实验入口为 `python -m rl.training.distributed_sb3`。算法参数、机器地址、结果目录分别见 configs/train、experiments/launchers 和忽略的 runs/local。

PPO、NOSL 与 reasoning 使用同一个梯度同步边界（ring 或 file transport）和 algorithms/updates.py 的 optimizer 更新；编排负责 rank、barrier、批次调度和日志。新的策略/学习方法应显式选择其目标和采样方式，不能通过复制整套 trainer 来添加算法。大 main 的生命周期拆分仍待做。

环境构造与 checkpoint schema 适配在 `environments.py`，采样基准复用同一工厂。
`--leaf-inference cuda-graph` 仅加速 CUDA compact combat 叶节点；其它路径默认 eager。

生产恢复使用 `--training-state-dir DIR --save-every-updates 1 --max-checkpoints 3`。
每个 rank 各存 `DIR/rankN/step-XXXXXXXXXXXX/{model.zip,learner.pt,manifest.json}`，
manifest 最后原子发布，包含配置、run/W&B ID 和文件 SHA256。恢复时四 rank 必须
选同一个已完整保存的 global step，传入各自的 `--resume-state`；不得用 rank 0 的
replay 代替其它 rank。模型/Adam、有效 replay slots、父进程 Python/NumPy/Torch/CUDA
RNG 恢复，global step 连续。完整性或 rank/world 不匹配立即拒绝恢复。

**原生环境暂不序列化**：恢复会重启游戏，按 global step 选择新的确定性 seed block，
丢弃尚未结束战斗的 pending labels。已完成的 replay 不丢。只有同一下一训练 batch
的 learner/Adam/RNG 有 bitwise 回归，不能称作原生游戏轨迹的逐位续跑。
每轮完整快照包含每 rank 的 replay，写盘成本需计入实际生产吞吐。

`--stop-file PATH` 在下一次更新/跨 rank 同步后触发保存并协同结束；多机要求 ssh-ring。
正常维护创建所有 rank 的 stop 文件并等待全部退出、核对同一恢复代数，再删除这些
请求文件并恢复。不要在训练中直接 kill 后声称最后一轮已保存。

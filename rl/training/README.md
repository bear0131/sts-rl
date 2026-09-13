# 训练编排

`trainer.py` 是单机 task CLI；`distributed_sb3.py` 是 SSH ring 多 rank 训练编排。RLlib 与 overfit 专用训练入口已退役，需要历史实现时从 Git 恢复。

`cached_mask_vec_env.py` 处理 worker 与 mask/packed leaf IPC；`ring_allreduce.py` / `ssh_transport.py` 处理同步；`reward.py` 提供训练 reward wrapper。小预算训练使用 rl.benchmarks.train，与生产共享算法。

当前实验入口为 `python -m rl.training.distributed_sb3`。算法参数、机器地址、结果目录分别见 configs/train、experiments/launchers 和忽略的 runs/local。

PPO、NOSL 与 reasoning 使用同一个梯度同步边界（ring 或 file transport）和 algorithms/updates.py 的 optimizer 更新；编排负责 rank、barrier、批次调度和日志。新的策略/学习方法应显式选择其目标和采样方式，不能通过复制整套 trainer 来添加算法。大 main 的生命周期拆分仍待做。

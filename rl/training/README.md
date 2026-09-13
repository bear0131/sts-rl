# 训练编排

`trainer.py` 是单机 task CLI；`distributed_sb3.py` 是当前 SSH ring 多 rank 训练编排；`rllib_trainer.py` 是历史 Ray/RLlib 路径。三者不是可互换的执行后端。

`cached_mask_vec_env.py` 处理 worker 与 mask/packed leaf IPC；`ring_allreduce.py` / `ssh_transport.py` 处理同步；`reward.py` 提供训练 reward wrapper。`overfit_test.py` 是可运行的训练实验，名字遗留，不由 pytest 收集。

当前实验入口为 `python -m rl.training.distributed_sb3`，旧 `python -m rl.distributed_sb3` 兼容。算法参数、机器地址、结果目录分别见 configs/train、experiments/launchers 和忽略的 runs/local。大 main 的拆分属于下一阶段；本阶段保持解析默认值及 optimizer 顺序。

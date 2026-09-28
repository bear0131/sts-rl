# 更新与存储

`accumulating_ppo.py` 是 SB3 扩展：含 PPO 梯度累积、分层 rollout 和 NOSL combat learner。`hierarchical_returns.py` 定义宏/微回报；`nosl_replay.py` 负责紧凑 FIFO、fresh/old 配额与采样。

输入是 env rollout 和 search root targets，输出是 policy/optimizer 更新。只在完成战斗后提交最终 HP 标签；当前 F05/F11/F12 等语义问题见根 TODO。
`ppo.py` 定义纯 PPO 目标；`updates.py` 共享 PPO 前推、reasoning batch/反传和累积平均→裁剪→optimizer 更新边界。本地与分布式编排都调用这些原语；NOSL 模式下 combat 由 AZ 监督，不再同时进入 PPO 目标。统计加权与 replay 配额属于算法选择，不能由传输层悄悄改变。

新增学习方法可以组合这些原语，也可以独立实现目标；不要求继承 PPO 或伪装成 NOSL。继续拆 collector/learner 时用 tools.validation.bitwise 对照 rollout、有效 replay、Adam 和权重，不能只测最终 action。

NOSL CUDA learner 对至少32768条的有效 batch 用最多4个线程提取独立 replay
字段。采样 RNG、索引顺序和 fresh/old 配额不变；`get_batch` 返回前线程全部结束，
调用期间不可写 replay。CPU 和小 batch 默认串行。
紧凑字段先放入 pinned host memory，再异步传输并在 GPU 转回 observation space
规定的 dtype。每次更新末尾的同步诊断回传完成后才释放 host buffer；不缓存旧权重，
不改变 microbatch、梯度归一化或 Adam 更新顺序。
浮点逐位对照需固定设备、依赖、确定性算法及 CUDA BLAS 配置；默认 CUDA 的末位
差异不能当作跨设备 bitwise 保证。吞吐还须覆盖完整 collection/update/save 周期。

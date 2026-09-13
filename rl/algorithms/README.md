# 更新与存储

`accumulating_ppo.py` 是 SB3 扩展：含 PPO 梯度累积、分层 rollout 和 NOSL combat learner。`hierarchical_returns.py` 定义宏/微回报；`nosl_replay.py` 负责紧凑 FIFO、fresh/old 配额与采样。

输入是 env rollout 和 search root targets，输出是 policy/optimizer 更新。只在完成战斗后提交最终 HP 标签；当前 F05/F11/F12 等语义问题见根 TODO。
`ppo.py` 定义纯 PPO 目标；`updates.py` 共享 PPO 前推、reasoning batch/反传和累积平均→裁剪→optimizer 更新边界。本地与分布式编排都调用这些原语；NOSL 模式下 combat 由 AZ 监督，不再同时进入 PPO 目标。统计加权与 replay 配额属于算法选择，不能由传输层悄悄改变。

新增学习方法可以组合这些原语，也可以独立实现目标；不要求继承 PPO 或伪装成 NOSL。继续拆 collector/learner 时用 tools.validation.bitwise 对照 rollout、有效 replay、Adam 和权重，不能只测最终 action。

# 更新与存储

`accumulating_ppo.py` 是 SB3 扩展：含 PPO 梯度累积、分层 rollout 和 NOSL combat learner。`hierarchical_returns.py` 定义宏/微回报；`nosl_replay.py` 负责紧凑 FIFO、fresh/old 配额与采样。

输入是 env rollout 和 search root targets，输出是 policy/optimizer 更新。只在完成战斗后提交最终 HP 标签；当前 F05/F11/F12 等语义问题见根 TODO。
本阶段搬迁不拆改变算法顺序。后续拆 collector/learner 必须用 tools.validation.bitwise 对照 rollout、有效 replay、Adam 和权重，不能只测最终 action。

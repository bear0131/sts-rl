# Task 组合

`base.py` 定义环境/模型/CLI 组合接口；`combat.py` 提供战斗/gauntlet；`wholerun.py` 提供整局；`wholerun_basic.py` 是对应 reward wrapper 变体。
训练器通过 task 建 env 和 policy。任务组合不负责搜索树、PPO loss 或评估统计；改默认 wrapper 顺序会改变 seed/截断/奖励契约，需训练回归。

# 观测到动作/价值

`model.py` 管 tokenizer、编码器与 `observation_for_policy` additive-field 兼容。`pointer_policy.py` 是单 policy；`split_policy.py` 管共享表征和 combat/overworld 两组 actor-critic。`semantic_reasoning_policy.py` / `wholerun_reasoning.py` 是可选 reasoning 路径；`rllib_inference.py` 是历史 RLlib checkpoint 的推理适配。

参数名称和 tensor shape 是 checkpoint 契约。旧类模块路径由 rl/_compat 保留；不能靠改 key 名“整理”state_dict。mask/动作目标语义归 env 管，策略不自行压缩怪物槽。

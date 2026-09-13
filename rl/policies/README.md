# 观测到动作/价值

`model.py` 管 tokenizer、编码器与 `observation_for_policy` additive-field 兼容。`pointer_policy.py` 是单 policy；`split_policy.py` 管共享表征和 combat/overworld 两组 actor-critic。`semantic_reasoning_policy.py` / `wholerun_reasoning.py` 是可选 reasoning 路径。RLlib checkpoint 不再支持。

参数名称和 tensor shape 是 checkpoint 契约。旧类模块路径由 rl/_compat 保留；不能靠改 key 名“整理”state_dict。mask/动作目标语义归 env 管，策略不自行压缩怪物槽。

`checkpoint.py` 是 SB3 `.zip` 的共享加载入口。策略 checkpoint 与 card encoder `.pt` 是不同文件：某些模型只保存外部 encoder 路径，移动文件后需要显式提供训练时同一份 encoder。eval CLI 用 `--card-vector-checkpoint PATH`，其他消费者可以设置 `STS_CARD_VECTOR_CHECKPOINT`。loader 先读 metadata 解析依赖，再加载模型；支持 SB3 的省略 `.zip` 写法。现役实验模型不保证存在于新 clone 中，先跑不需要 checkpoint 的训练冒烟可验证安装。

自动探测会跳过另一台机器上不存在或无权访问的历史路径，继续尝试本地候选。显式参数或环境变量具有优先级：指定文件不可访问时直接报错，不能悄悄换用另一个 encoder。

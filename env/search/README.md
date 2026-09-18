# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则 v1。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层可用 `--nosl-is-az-upper-strategy rules`（或 `@configs/train/nosl_ruleupper_v1.args`），仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则 v1。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层可用 `--nosl-is-az-upper-strategy rules`（或 `@configs/train/nosl_ruleupper_v1.args`），仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

`turn_planner.py` 提供无模型 native 回合规划器。它在独立 NOSL 样本中搜索完整战斗动作，
每次只提交首个动作并重新观察；`node_budget` 是每个 sampled world 的 transition 预算，
`worlds`、实际 transition、根覆盖下限和 fallback leaf 均进入诊断。人工状态价值不是胜率或
游戏 oracle。`ironclad-turn-v8-awakened-phase` 在既有 slime phase、终局优先级、最短等价
胜利和资源保护逻辑上，补充 Awakened One 可见阶段的剩余生命估值。Runic Dome 不进入
native 规划，adapter 回退 `rules-ironclad-v3`。主要回归入口为
`tests/integration/test_turn_planner*.py`。

# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

## 当前选牌与隐藏未来采样

draw-select 的索引属于当前公开候选实体。`nosl_redeterminize` 会重排 draw，
而执行一次 selection 会继续运行 native 队列；例如 Secret Weapon 的耗尽会
触发 Dark Embrace 抽牌。因此不能先执行真实副本的选择、再重新采样：排队的
抽牌已经读了真实未来。

新 planner 应先 clone/redeterminize，然后调用
`sample.remap_draw_selection_from(source, action_bits)` 得到 sampled clone 中的
完整合法 uint32 动作。该只读 helper 通过 native `CardInstance.uniqueId` 定位，
不按卡名/升级猜测；同名重复牌不合并，持久 specialData 等字段必须保持一致。
源/目标必须是同类型的 live draw-select 界面；非法、丢失或歧义实体明确报错。
随后只在 sampled clone 执行重映射后的动作。不能把 sampled 索引直接提交给
真实环境；真实执行仍用原 root action。

`BattleContext.preview_end_turn_result(action_bits)` 只在内部 clone 上执行一次
合法 END_TURN，返回 post-enemy-turn boundary 或 terminal 的完整 raw state。
非终局但未到边界会抛错；source 不变。新搜索用它避免旧
`preview_end_turn_boundary` 在 terminal 时返回 None、调用方再次 execute 的重复
模拟。旧 API 和旧 NOSL 的重采样/训练语义均保持兼容；采用新 helper 的 planner
需要显式加载含这些绑定的新 native 构建。

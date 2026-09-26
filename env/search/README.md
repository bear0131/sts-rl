# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则 v3。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层可用 `--nosl-is-az-upper-strategy rules`（或 `@configs/train/nosl_ruleupper_v1.args`），仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

启用 `--nosl-is-az-train-shared-embeddings` 会按 run seed 重置并训练 split policy 的共享 card/relic/potion embedding，projection 与整个外层 specialist 仍冻结。共享实体既作为候选 token 直接影响 `USE_POTION`、`PICK_CARD`、card/relic/potion reward、shop 与 boss relic 动作，也作为常驻 deck/relic/potion token 进入每个外层 phase 的 Transformer，因此 map、event、proceed、chest、campfire、Neow 等固定按钮 logits 也可能间接受影响。规则上层只有返回 `None` 的状态会暴露给这条 masked-argmax fallback；v3 在 v2 的确定性事件与 Bonfire Spirits 基础上，让 native canonical state 公开 Falling、N'loth、We Meet Again、Note For Yourself、Designer 按钮上已经显示的具体卡牌、遗物、药水、金额及服务模式，并据此评分。该附加元数据不进入 observation tensor；旧扩展或缺项候选仍整屏返回 `None`，不会按固定 action index 猜测。stock CommunicationMod 目前只给本地化按钮文本，真实 bridge 在收到 action-aligned typed 扩展前仍对这五类 fail closed，不从文字猜实体。

`turn_planner.py` 提供无模型 native 回合规划器。它在独立 NOSL 样本中搜索完整战斗动作，
每次只提交首个动作并重新观察；`node_budget` 是每个 sampled world 的 transition 预算，
`worlds`、实际 transition、根覆盖下限和 fallback leaf 均进入诊断。人工状态价值不是胜率或
游戏 oracle。`ironclad-turn-v8-awakened-phase` 在既有 slime phase、终局优先级、最短等价
胜利和资源保护逻辑上，补充 Awakened One 可见阶段的剩余生命估值。Runic Dome 不进入
native 规划，adapter 回退 `rules-ironclad-v3`。主要回归入口为
`tests/integration/test_turn_planner*.py`。

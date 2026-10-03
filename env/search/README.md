# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；value-guided lookahead 对照搜索归 rl/search/。

`turn_planner.py` 持有原生回合规划的采样、展开、预算和终局排序，不加载模型。
可选 `leaf_evaluator` 只评价已选定的合法分支，不增加 transition 或重采样。
评测侧 `rl/evaluation/baselines/turn_planner.py` 负责请求与策略组合，
`planner_value.py::CheckpointLeafValue` 负责按模型合同调用共享观测编码器和叶推理。
边界、兼容范围及命令参数统一见 [evaluation README](../../rl/evaluation/README.md)。

`native_batch.py` 是常驻环境的无模型跨树适配层：C++ 线程池持有独立 session，
仅 select/backup 在无 GIL 线程运行，Python 在每一波边界调用外部模型。C++ 数字观测
转换从 `env/core/encode.py` 接收词表、卡牌数据、形状与比例，不独立维护规则。
每棵树使用自己的 relic/global 元数据；原 `run_env` 的结果适配保留动作映射及 pending 状态。
单个 batch 只有一个调用方；异常必须 clear，成功 results 后保留待执行动作直到 env.step。

原生观测可直接写入调用方的连续数组，只覆盖有效行；图缓存负责 padding 与输入
生命周期，并保留模型原有 dtype 存储布局。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则，当前 canonical 版本由 `OVERWORLD_RULE_VERSION` 声明为 `ironclad-overworld-v3`。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层使用 `--nosl-is-az-upper-strategy rules`，或 canonical response file `@configs/train/nosl_ruleupper_latest.args`；该入口在代码更新后自动使用当前最新规则版本，不再绑定 `v1` 文件名。仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

启用 `--nosl-is-az-train-shared-embeddings` 会按 run seed 重置并训练 split policy 的共享 card/relic/potion embedding，projection 与整个外层 specialist 仍冻结。共享实体既作为候选 token 直接影响 `USE_POTION`、`PICK_CARD`、card/relic/potion reward、shop 与 boss relic 动作，也作为常驻 deck/relic/potion token 进入每个外层 phase 的 Transformer，因此 map、event、proceed、chest、campfire、Neow 等固定按钮 logits 也可能间接受影响。规则上层只有返回 `None` 的状态会暴露给这条 masked-argmax fallback；v3 在 v2 的确定性事件与 Bonfire Spirits 基础上，让 native canonical state 公开 Falling、N'loth、We Meet Again、Note For Yourself、Designer 按钮上已经显示的具体卡牌、遗物、药水、金额及服务模式，并据此评分。该附加元数据不进入 observation tensor；旧扩展或缺项候选仍整屏返回 `None`，不会按固定 action index 猜测。stock CommunicationMod 目前只给本地化按钮文本，真实 bridge 在收到 action-aligned typed 扩展前仍对这五类 fail closed，不从文字猜实体。

`turn_planner.py` 提供无模型 native 回合规划器。它在独立 NOSL 样本中搜索完整战斗动作，
每次只提交首个动作并重新观察；`node_budget` 是每个 sampled world 的 transition 预算，
`worlds`、实际 transition、根覆盖下限和 fallback leaf 均进入诊断。人工状态价值不是胜率或
游戏 oracle。当前 `ironclad-turn-v10-live-boundary` 保留终局/价值/最短同值胜利的
优先级；仅在这些层级精确平分、且全部平分根均已完成采样边界时，优先以更少原子动作
到达同值的存活边界。含未完成 fallback 的平分组仍按 native bits 选择，不加数值容差。
该规则修复已复现的自替换出牌递延，不保证所有有限预算重规划都无循环。
叶权重仍为 `time-clock-v9`：在 v8 的 slime/Awakened phase 和资源保护逻辑上，
补充 Time Eater 回合边界的公开 Time Warp 计数：计数
超过7时，叶子按下一回合被强制结束前的剩余出牌额度降权；触发第12张后计数重置，不处罚。
`legacy-v8` profile 保留原值用于配对复现。Runic Dome 不进入
native 规划，adapter 回退 `rules-ironclad-v3`。主要回归入口为
`tests/integration/test_turn_planner*.py`。

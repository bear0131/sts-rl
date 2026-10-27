# Worker 搜索基础

`nosl_sampler.py` 派生 opaque nonce；`nosl_is_az.py` 提供只依赖 NumPy/观测编码的 Python tree/session。模型前推由外部 value callback/leaf request 提供，不导入 Torch、训练器或 policy。

env/core/run_env.py 在 worker 中驱动这些 session；学习侧通过同一接口消费。这里解决了过去 env 反向导入 rl 的耦合。原生严格 Full Gumbel 树仍归 sim/；checkpoint value 对照搜索归 rl/search/。

`native_batch.py` 是常驻环境的无模型跨树适配层：C++ 线程池持有独立 session，
仅 select/backup 在无 GIL 线程运行，Python 在每一波边界调用外部模型。C++ 数字观测
转换从 `env/core/encode.py` 接收词表、卡牌数据、形状与比例，不独立维护规则。
每棵树使用自己的 relic/global 元数据；原 `run_env` 的结果适配保留动作映射及 pending 状态。
单个 batch 只有一个调用方；异常必须 clear，成功 results 后保留待执行动作直到 env.step。

原生观测可直接写入调用方的连续数组，只覆盖有效行；图缓存负责 padding 与输入
生命周期，并保留模型原有 dtype 存储布局。

`NativeSearchVecEnv` 使用 `direct_observations=True`：begin/advance 只返回叶子
元数据，decoder 在 batch 的所有权锁内读取 pending 状态，复用 compact codec，
直接填充模型 host buffer。不生成整批 categorical/scalars 中间数组；选择候选的
尾部空槽仍存在并清零，只省去其逐项转换。packed API 和默认的诊断 adapter 路径
继续保留完整数据及校验。该路径需要重新构建含 `decode_native` 的扩展；观测版本、
dtype、搜索顺序不变，恢复已有完整训练状态仍须遵守 native SHA 身份检查。

直接编码复用 batch 的常驻线程池处理独立叶行，持有 batch 所有权，释放 GIL；
全部任务结束后才交还缓冲，异常也等待其他任务退出。调用方若传入重叠的输出字段，
仍按串行顺序转换；不新增线程或改变叶子顺序。

原生 `NeuralSearchBatch.work_counts()` 只读返回当前树的
`simulator_transitions`（实际执行的完整游戏动作，包括每次 simulation 重放的路径）、
`selection_prefix_decisions`（仅修改多选前缀、尚未执行游戏动作）和
`completed_simulations`。新 begin 的计数属于新树，clear 后归零；在等待叶子回复时，
已执行动作计入工作量，尚未完成的 simulation 不计入 completed。
这不是叶子数，也不是伤害/抽牌等引擎内部事件数。此接口用于有界性能诊断，
不改变训练日志或 checkpoint 格式；升级原生构建仍需遵守完整状态的 SHA 校验。

`overworld_rules.py` 是训练 worker 与 evaluation 共用的 Ironclad 公共状态上层规则，当前 canonical 版本由 `OVERWORLD_RULE_VERSION` 声明为 `ironclad-overworld-v3`。只对非战斗合法动作打分；未知事件/选择画面返回 `None`，由消费者显式回退。分层 NOSL 的冻结上层使用 `--nosl-is-az-upper-strategy rules`，或 canonical response file `@configs/train/nosl_ruleupper_latest.args`；该入口在代码更新后自动使用当前最新规则版本，不再绑定 `v1` 文件名。仅支持 inner-only NOSL；未覆盖画面使用冻结模型 masked argmax。默认 `model` 保留随机采样。该设置保存在 checkpoint，CLI 省略时沿用 checkpoint；不会改变战斗搜索或训练目标。

启用 `--nosl-is-az-train-shared-embeddings` 会按 run seed 重置并训练 split policy 的共享 card/relic/potion embedding，projection 与整个外层 specialist 仍冻结。共享实体既作为候选 token 直接影响 `USE_POTION`、`PICK_CARD`、card/relic/potion reward、shop 与 boss relic 动作，也作为常驻 deck/relic/potion token 进入每个外层 phase 的 Transformer，因此 map、event、proceed、chest、campfire、Neow 等固定按钮 logits 也可能间接受影响。规则上层只有返回 `None` 的状态会暴露给这条 masked-argmax fallback；v3 在 v2 的确定性事件与 Bonfire Spirits 基础上，让 native canonical state 公开 Falling、N'loth、We Meet Again、Note For Yourself、Designer 按钮上已经显示的具体卡牌、遗物、药水、金额及服务模式，并据此评分。该附加元数据不进入 observation tensor；旧扩展或缺项候选仍整屏返回 `None`，不会按固定 action index 猜测。真实 bridge 的各事件契约与支持版本见 `bridge/client/README.md`；parity-v35/v36/v37分别补WeMeetAgain/Designer/NoteForYourself实际typed公开目标与来源回执。词表、源码/构建和自然实机覆盖是不同证据，不按压缩的界面编号猜测原生语义。

`turn_planner.py` 提供无模型 native 回合规划器。它在独立 NOSL 样本中搜索完整战斗动作，
每次只提交首个动作并重新观察；`node_budget` 是每个 sampled world 的 transition 预算，
`worlds`、实际 transition、根覆盖下限和 fallback leaf 均进入诊断。人工状态价值不是胜率或
游戏 oracle。`ironclad-turn-v9-time-clock` 在 v8 的 slime/Awakened phase、终局优先级、
最短等价胜利和资源保护逻辑上，只补充 Time Eater 回合边界的公开 Time Warp 计数：计数
超过7时，叶子按下一回合被强制结束前的剩余出牌额度降权；触发第12张后计数重置，不处罚。
`legacy-v8` profile 保留原值用于配对复现。Runic Dome 不进入
native 规划，adapter 回退 `rules-ironclad-v3`。主要回归入口为
`tests/integration/test_turn_planner*.py`。

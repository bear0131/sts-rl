# Worker 搜索基础

`cuda_battle.py` 是无模型 CUDA 常驻战斗组件，通过显式 `cuda-resident` 后端接入影子搜索；生产资格须单独验收。
它要求调用方提供固定 SHA256 的 `deck-resident-battle-v1` 模块；模块包含
`cuda_battle_kernels.cuh`，并使用与当前原生环境一致的规则、compact v9 观测及
逻辑状态序列化器构建。私有模拟器源码、生成文件和编译产物不入 Git。
旧规则构建中的 185 个方法和 99 个动作工厂的 `__noinline__` 处理需要保留；
未应用时，当前原子出牌路径已复现 64 KiB device stack overflow。

状态传输版本2按逻辑快照实际字数分配CPU/GPU导入导出缓冲，支持超过128张的牌堆及超过8192字的状态；canonical编码不变。bridge以`DECK_DYNAMIC_STATE_IMPORT`显式构建，manifest声明`transport_version=2`，与模块layout共同校验。旧版本1仍保留原容量检查。最后一个arena关闭时恢复本组件租借的CUDA栈限制，避免搜索退出后闲置栈挤占PPO显存。

连续搜索的主机端热循环位于 `host_runtime.h`，由同一 `cuda_bridge.cpp` 构建并声明 `host_runtime_version=1`。根代际、非移动页租约、完整字节缓存与 intern、快照/任务打包及配对结果按序求和由 C++ 执行；大数组循环在取好 Python 元数据后释放 GIL。所有对象仅由搜索 owner 使用，不支持多线程同时修改同一实例。Python 参考账本与缓存仍用于差分验收；模型计算和命令编排保留原 PyTorch/FIFO 路径。

重新构建 bridge 必须使用现役 native 模拟器相同版本的 pybind11 头文件和 C++ ABI；不能直接改用 Torch 附带的另一个 pybind11 版本。`STS_ENABLE_RULE_TRACE`、`DECK_DYNAMIC_STATE_IMPORT` 和 native include/state codec 的构建身份必须记录。使用 `-ffp-contract=off`，禁止 fast-math，以保留原 FP64 配对累加顺序。产物 manifest 记录 bridge/cubin SHA 和 host 源码 SHA；更新 host 时可复用逐 SHA 校验的相同 cubin，不覆盖旧构建。

验收入口为 `tests/integration/test_resident_host.py` 和 `test_resident_stream.py`，并与 CUDA battle、paired RNG、真实 PPO/NOSL 两轮保存恢复测试共同运行。通过显式 `STS_CUDA_BATTLE_ARTIFACTS` 指向构建；缺失构建不算通过。新 host ABI 不更改模型/Adam/环境 checkpoint schema，但旧 bridge 不满足新运行时，恢复计划须同时固定源码和产物。

CPU 待执行回调不能跨设备导入。`load` 拒绝这类根；在 GPU 中产生的回调由同一
模块持有，可随独立分支复制，直到状态销毁。调度接入必须维护真实战斗的 GPU
镜像，或明确报告尚不能导入的根，不能跳过 continuation 或静默改变搜索预算。
`fork` 使用当前 `noslRedeterminize`；动作沿当前 autoregressive-v2 执行，选择
前缀、候选槽位、完整逻辑状态和随机状态须与 CPU 对照。它不负责叶评分或策略采样。

单个 arena 只允许一个调用方在创建它的 CUDA stream 上串行操作；观察结果是下次
`observe` 会覆盖的视图。固定状态缓冲区跨图淘汰保留，图按行数有界缓存；关闭时
先释放图，再销毁设备状态，最后卸载模块。不能让带函数指针的状态跨模块存活。
显式 GPU 回归入口为 `tests/integration/test_cuda_battle.py`，设置
`STS_CUDA_BATTLE_ARTIFACTS` 指向带 `artifacts.json`、固定 SHA 的 cubin 和 Python 3.11
桥接扩展的目录；缺少构建或 CUDA 是失败，不是跳过。该组件测试不替代完整
256×768 collection、当前模型评分、两轮训练更新/恢复和端到端速度验收。

含 `deck_observe_live_block` 的模块支持 `resident_step_kernels.py` 的稳定设备压缩与
无副作用重试；只观察活跃分支，已结束的分支保持最后状态。每行64线程协作清零，
共享 `compact_leaf::packInto` 直接写目标存储，不创建大型线程私有临时Row。模型侧整步图在
`rl/search/resident_step.py`，负责观测、动态打包、评分、配对采样、提交及推进。
捕获/预热时提交关闭；容量不足或永久卡牌变化时整步不推进，扩容/批量刷新缓存后
重放。PCG64 流仍按 world 生成，只有真正的策略动作前移 cursor，强制动作及深度
截断不消耗随机数。旧不含此符号的模块继续使用已支持的逐步调度，不虚报整步图。
同时含`deck_apply_compact`与`deck_observe_indexed_block`的构建使用稳定live索引压缩模拟器线程，状态本身不搬移，分支随机流和原始槽位不变。它是可选构建能力，是否有端到端收益仍须同机完整负载实测。
搜索整步将等待分支池与最大16384行评分帧分开；已结束行即时补入等待分支，等待行保持状态/深度/随机cursor。`deck_observe_search_block`支持先生成动作mask/outcome，再只为策略选择及深度截断行编码完整特征；终局锚点在最终评分时仍完整观察。物理波容量只改变汇批，不改变每根预算、真实环境数或搜索目标位置。
`rl/policies/resident_ragged.py` 迁入旧2×64的融合QKV、分桶attention、mix/FFN及
动态计数路径；第二层只查询动作实体及global，牌堆/遗物仍参与两层的key/value。
仅冻结FP32推理使用该路径，训练及checkpoint参数布局不变。

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

GPU 常驻模拟器的 shim vector 使用 `cuda_vector.h` 替换固定容量实现：常见牌堆保留内联存储，超过容量时由该实例独占堆内存并增长；复制/分支不共享可变存储，销毁时释放。该头在 shim namespace 内包含，依赖既有 `vector_capacity`、placement new、malloc/free。编译产物必须与 `cuda_battle.py` 的逻辑状态协议一起逐 SHA 验证；不能在既有 arena 中更换模块。逻辑 checkpoint 字段/顺序不变，物理对象大小由模块 layout 返回。回归包括 `tests/unit/cuda_vector_contract.cpp` 的 ASan/UBSan，以及 `tests/integration/test_cuda_vector.py` 的 CPU/GPU 状态、扩容后 fork 和图释放生命周期。

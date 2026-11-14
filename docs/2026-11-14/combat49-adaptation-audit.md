# Combat49：本次新增与原代码改动清单

用户原话：“你明确标出你自己加了哪些部分的适配哪些部分的改动”；并要求“最小化所有的适配，尽可能只在原本基础上改不加东西”。

当前审查范围为恢复旧 480 自然局后的工作树；下方 JSON 给出固定原件基线和当前逐段改动。正式训练仍停止在 246，首轮完整结果为 126.5 万/秒，后续修正仍在验收。

已撤回我增加的固定 PPO 主循环、战斗根先走 upper、根完成时另造 teacher、CPU fallback 完整观测字典以及独立 add_rows 缓冲接口。e7d757f8 中采样循环 AST 除可选属性读取、回调取消清理外与归档相同。本阶段进一步在原循环中改了 Runic Dome 的无搜索决策分流：送入已有 GPU upper 队列，并补齐该批当前战斗输入，避免原 CPU 特例调用大模型。没有另建推理消息、队列或采样循环。

**仍由我新增的模型适配包括构筑/锚点缓存消费者、当前输入表、当前输出头、原生 raw 数组的当前模型字段，以及结果包里的 64 个 float32 缓存投影。不能将这些也称为旧代码原封不动。**

## 基线与读法

- 旧 GPU 完整主体以实际 `stepfull480-01` 归档为准，SHA256 `3a72e10edb48ae26a44655acfba717f3cc4cec707e3476c2229dd4d03258ebc3`。`d5cd40be` 是原件入库基线；38 个历史源码文件的导入不等于本次新写了 38 个算法。
- 旧 CPU WaveDiag 原件以 `248aa3f8` 的导入为基线；`native/include/zones.h` 是另外核对过的历史头文件，来源及 SHA 在原有 runtime-sources.json 中。
- 原仓库已有的推理、buffer、模型、learner 等公共文件以任务开始 `e507fdde` 为基线。不能统一用 `d5cd40be` 比较，因为缓存和 buffer 的初次改动已经在这个提交之前。
- [逐文件、逐改动段清单](combat49-adaptation-audit.json) 给出每个文件的比较基线、增删行数、当前行号、内容 SHA256、是否尚未提交；行数按文本 diff 计算，不代表语义改动大小。
- 另有忽略目录内的可直接查看补丁 `runs/local/full480-reconnect/adaptation-audit.patch`。未包含被用户拒收的 8318/661b 采样代码。

## 我新增或改变的实际执行部分

| 标号 | 文件与定位 | 我具体新增/改变的内容 | 与旧路径的关系 |
| --- | --- | --- | --- |
| C01 | `rl/algorithms/full480.py` 的 `collect`；`accumulating_ppo.py` 的入口校验 | 恢复原自然局函数，删除固定 PPO 循环、战斗根 upper、GAE 和 add_rows 调用；保留新模型属性兼容与取消回收；Dome 无搜索行进入原 GPU policy 队列 | 原 ready/submit/receive/自然结束/根记录顺序恢复；不接受固定 PPO 配置误用终局诊断 |
| C02 | `full_seeded.py`、`paired_token_rollout.py`、`env/search/small_playout.py`、`training/small_playout.py` | 恢复 batch.prepare / accept_ready；在原 score_roots 替换模型输入与小网；根观测由现有 NativeRootObservationBatch 一次编码复用；当前记录保存 canonical observation、动作映射、visits、Q、prior_logits、推荐动作 | 未另造完成队列或根排序；GPU 已算出的 64 维投影随原结果包返回，避免 CPU 再编码整模型。Dome 改由 C01 的原 GPU 队列评分，其他无 GPU 投影的根仍有缓存消费者 |
| C03 | `rl/search/rollout_inference.py`，`_DeckForward`、`_DeckConsumers`；`UpperPolicyInference` 的 deck 分支 | 新增构筑编码及两份静态投影消费者、动态 head、独立锚点结果缓存；旧 `_upper_graph_inputs` 接回；head 图容量设为 256；缓存权重登记检查和一次返回上下文 | 复用原 `ExactInputCache` 与 `CombatLeafGraphs`，但两个 Deck 消费者类和路由是我写的；未关闭构筑/锚点缓存 |
| C04 | `rl/search/inference.py`，`CombatLeafGraphs._capture` | 增加 `scratch_only`、共享 scratch 时图外独立设备输出区、最后一个图淘汰时重置 scratch handle | 修改旧图执行器的内存所有权，不是仅换模型参数 |
| C05 | `cpu_boundary.py`、`fallback_frame.py`、native `bindings.cpp::pack_raw` | 删除 compact→decoder→observations 字典；原 WaveDiag 原生批量填原始 scalars/category/牌堆升级数、布局、抽样数，进入原单份 pinned 区和融合图 | 当前数字布局由模型输入替换而来，仍是本次改动；没有逐分支 Gym 字典或 Python 观察解码。特征共用 GPU observer 的 token 函数 |
| C06 | `native/bindings.cpp`、`native/wave.cpp`、`native/include/sim/search/WaveDiag.h` | 仍保留当前 AR 前缀和 PCG 状态字段；删除 compact 字典接口、叶子 neural anchor/cutoff 请求；恢复原终局值/未完成 NaN，并允许精确预算的末个不完整 world | 原 WaveDiag 线程池、任务队列、ready/choose 继续使用；新增接口和模型字段不是原件原有 |
| C07 | `worker_devicestepfull480141.py`、`service_devicestepfull480141.py` | 删除根先请求 upper 后回传再提交的绕路；原 submit 内计算/查询构筑缓存，原根结果包附 64 维投影；fallback 沿原命令处理原始数组 | FIFO、64 命令后 tick、begin/take 保留；新数据只服务当前模型输入与缓存，不新增消息通道 |
| C08 | 新文件 `rl/search/full480/deck_inputs.py`、`deck_inputs.cuh`；原 `observer_entry.cu`、`token_builder.cu`、`frame_modules.py`、`frame_pipeline.py` | 新增 `DeckInputTables` 和 CUDA 当前模型特征构造；原 observer/builder 增加 Deck 编译分支、当前物理槽位/token/查询映射；新增构建函数及升级表溢出增长 | 两个 deck_inputs 文件是本次从零新增的模型输入适配，不是旧文件搬迁。执行仍嵌在旧 observer → builder → frame 中 |
| C09 | `rl/search/full480/combo_model.py`、`combo_kernels.py`、`fallback_ieee_model.py`、`fallback_ieee_kernels.py`；`rl/policies/deck_latent.py` | 新增 `DeckWeights`、当前输出头、global/subset/action 融合 kernel；旧 2×64 主体接当前头；从已有 `encode` 提取 `token_inputs` | 旧 attention/FFN、查询子集和融合执行保留，输入与输出头分支是我改写的；当前较宽输出头复用结束使用的 QKV 存储，避免额外 576 MiB（生产帧）的分配 |
| C10 | `rules_entry.cu`、`finite_plan.py`、`flex_runtime.py`、`flex_stream.py`、`frame_pipeline.py`、`step_pipeline.py` | 当前 PCG/AR 与永久变化缓存失效保留；恢复旧 finish/reap 终局 HP/Boss 胜负，不在完成后调用 anchor/cutoff 推理；根上传改回批量。dirty context 只刷新活跃链；删除本次旧终局目标不会调用的 complete 适配及逐链 anchor 副本（480×1000×256×4 字节） | 原 GPU 原地 clone/step、页租约、整步图和回收保留。缓存代码保留，终局目标不调用 anchor 不是禁用缓存优化 |
| C11 | `search_auxiliary.py`、`room_returns.py`、`tests/unit/test_search_auxiliary.py` | 已撤回独立 add_rows/row_positions 及对应新测试、外传 last_values 改动；只保留 full480 后端名称的配置识别 | 原 PPO buffer、returns 与更新实现恢复任务起点；固定 PPO 和普通恢复仍未完成资格 |
| C12 | `run_device.py`、`layout_bindings.py`、`sim/sts_lightspeed/CMakeLists.txt`，各运行模块的 import | 去掉历史固定目录/固定模型加载，改显式产物路径与包内 import；添加原生 full480 构建目标及依赖校验 | 运行绑定和构建改动，不是纯模型数学改动；部分文件只有 import 改动，不能与新增搜索逻辑混算 |

## 原样保留与并未由我修改的文件

原件导入后，`root_ledger.py`、`state_pages.py`、`step_kernels.py`、`frame_kernels.py`、`cooperative_gpu.py` 没有本次行为改动；native 的 `raw_public.h` 保持原件。`native/include/zones.h` 是历史依赖原样导入。

`graph_kernels.py`、`graph_model.py`、`pool_gpu.py`、`ragged_model.py`、`ragged_pack.py` 只改包内 import。`ragged_runtime.py`、`search_runtime.py` 还改历史产物定位；其他 runtime 文件的模型选择、尺寸或生命周期变更均已计入 C08–C12，不归为“未改”。

本轮已实际修改 `paired_token_rollout.py` 与 `env/search/small_playout.py` 的模型边界；此前“没有改动”的说明仅适用于 dbf8e466 审计快照。

## 尚未提交与已撤回的内容

- 已随 e7d757f8 提交的 4 个根观测文件：`env/core/deck_context.py`、`env/core/native_observation.py`、`sim/sts_lightspeed/bindings/slaythespire.cpp`、`tests/integration/test_native_root_observation.py`。它们把当前模型的牌堆升级计数/总量放进已有 `pack_neural_roots`，替代 Python 每牌构造；不是新增采样入口。新增 `deck_context_schema` 参数和能力版本标记也属于我加的接口。明确构建 SHA `4daefadd5b63a056b7e01ca262d4935ac61699a31d0bf5b6f69fbe8589e681f4` 下 20 项测试通过，已完成480自然局，速度仍不合格。
- 已按用户最新约束撤回的未提交尝试：`upper-value.cpp` 的 `forward_keys`/新构造函数，以及 `rollout_inference.py` 的 native scalar cache 分支。未编译验证、未提交、未用于完整测速；现已恢复 HEAD。原先工作的 Python 精确构筑和锚点缓存继续保留。
- 新增测试：`test_full480_pipeline.py`、`test_full480_collection.py`；已有 deck latent、buffer、native root 测试中也新增了回归。测试增量和 fixture 变更单独列入 JSON，不混成运行时实现。

## 传输审计的实测事实

旧版根训练样本一直包含 `state/actions/keys/action_bits/visits/recommended_action/env`，以及 PairedTokenBatch 加入的 `tokens/values/prior_logits/return_units`。它们由 CPU 根状态编码并保存；GPU 搜索返回动作统计和结果。当前完整 PPO 并不是“首次保存每个根的数据”。

同一批正式 246 恢复状态的前 32 个根，在相同当前搜索预算下重建旧 `root_plan + inventory + mirror` 数据包，再对比当前包：

| 指标 | 旧路径结构 | 当前结构 |
| --- | ---: | ---: |
| 单根提交消息平均 pickle 字节 | 16,790 | 29,811 |
| 根训练样本 pickle 平均字节 | 18,737 | 不作异构 pickle 直接比较 |
| 当前根完整观察数组有效字节 | — | 34,150 |

当前根消息平均为旧结构 **1.775 倍**。这是相同根的格式大小比较，不是重跑旧模型的速度对照。旧训练样本与新 PPO 字段不同，不能将 18,737 与 34,150 当作严格倍率。

当前第一份 CPU fallback 消息为 1,627,764 字节/64 行：观察数组占 1,545,856 字节，布局 68,056 字节，共享根上下文 9,108 字节。最大的新增模型输入为两个牌堆升级数组，各 284,928 字节/64 行。原 pinned 区的完整长度实测，旧平均每评分行复制 29,079.5 字节，当前 48,298.8 字节；这是不同实际轨迹/容量工作集的观测，不能当作同根 A/B。

完整旧轮 IPC 76.429 GB/233,897 根，当前轮 IPC 28.502 GB/98,475 根；搜索预算和叶子数不同，不能直接用总量或平均根数证明端到端速度因果。当前 upper 消息 6.838 GB、root submit 2.962 GB、fallback 18.590 GB。

审计运行 seed 20261111、256×16、0 更新、180 秒进程上限，exit 0；4096 真实行、1924 搜索根。短运行含审计开销，不作速度资格。原始 `transport-1794650298811516411.json`、对应日志和 `audit_transport.py` 在忽略的 `runs/local/full480-reconnect/`；完整最新 466.891 秒结果另存 `collection-dbf8e466-{result,complete}.json`，仍未过性能门槛。

上述传输数字是 dbf8e466 的历史测量，不代表本轮 raw 数组布局；当前已收回额外数据流，但完整吞吐仍待本轮 480 自然局回执。

## 本轮已定位的自加慢点

第一次恢复旧目标的完整运行 `collection-1794653592544064387.log` 在121秒时根处理89.784秒，10215根；该轮主动中断，不算完整验收。原因之一是我在 PairedTokenBatch 新模型边界调用完整模型，在 CPU 另算大构筑编码。修正为原结果包携带 GPU 缓存的 64 维投影；原根评分只跑战斗小网，测试把大编码器设为抛错仍通过。后续相同480/相同种子启动的30秒检查中根处理5.719秒/6926根，前一版本19.219秒/882根。早期计数受进度影响，不当作整轮最终倍率。

完整运行 `collection-1794653982223970963.log` 为572.965秒/126.51万有效决策每秒；后续1abef11d为592.053秒/122.36万。两轮480自然结束、0更新，但速度均不合格。

### 已定位到我所加代码的后续证据

- 45 秒提交阶段 profile：19.475 秒中 `dome_policy` 为 15.612 秒，CPU 大编码器 97 次、7.701 秒；当前模型被沿用到旧 CPU 特例而未转入已有 GPU 批量队列，是明确新增慢点。恢复后 90 秒检查提交 28261 根、累计 10.208 秒（约0.36ms/根），未作为最终吞吐。
- 测试回调漏写已有 `_uses_rollout_observations=False`，使旧观察省略优化未启用；已在原测试入口修正。不是减少根记录：完整 e7d757f8 结果111139搜索根、111064多动作根训练记录，75个单动作根按原accept合同不写训练标签。
- 第二轮 collection-1794654968469327901.log 在原状态页扩容时 OOM，退出1，不能称完整通过。90秒状态页已占5733220352字节、峰值333000在途simulation；新增逐链anchor副本和单独宽输出头合计1095499776字节，现已撤除冗余副本/复用原scratch。原缓存、精度、480环境、搜索预算与并发调度不缩减。

### 空槽与重复牌堆观察

- C03根评分的当前模型输入保留了整段遗物PAD尾部；75秒cProfile中`relic_terms`为8.164秒。直接在原`PairedTokenBatch.score_roots`裁尾，根训练观察仍保存完整数组，物理中间空槽不删除。C04原GPU小头使用已有宽度分桶裁尾。
- C08的`deck_input::prepare`调用canonical pack生成并复制三份稠密牌堆直方图，而后续原地GPU特征直接读取同一BattleContext牌堆。canonical pack新增默认开启的模板参数，只在该GPU消费者关闭这三个不用的直方图；CPU默认输出合同不变。未增加采样或传输路径。
- 完整逐段位置与SHA见同名JSON；本次观察修改的GPU数值/生命周期及自然局回归17项通过，完整速度结论仍待新480局。

C07原CPU回退现直接复用已经随原回复返回的root_contexts，未变永久卡牌数值的分支不再展开完整构筑输入/查询缓存；永久变化仍以精确内容重新查询。同一根多帧测试要求只查询一次，成长后重复根仍要求命中。C04权重检查改用批量注册身份比较，仍检查新增/删除buffer、替换子模块/参数和Tensor版本；原地2000次同模型对照从约0.690秒降至0.599秒，只能说明此检查成本下降约13%。

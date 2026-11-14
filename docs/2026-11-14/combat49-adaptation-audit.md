# Combat49：本次新增与原代码改动清单

用户原话：“你明确标出你自己加了哪些部分的适配哪些部分的改动”；并要求“最小化所有的适配，尽可能只在原本基础上改不加东西”。

当前审查范围包含原480主体内的固定PPO消费者；下方JSON给出固定原件基线和当前逐段改动。PPO增量逐项见[交接文档第10节](combat49-old-sampling-handoff.md#10-固定-ppo-接线改动清单)。P18/P19调度阶段四机满预算固定256×768第二慢408.808→346.956秒（减少15.13%）；原480自然局379.176秒/190.78万有效非强制决策每秒，180万自然局门槛通过。25%预算四机第二慢251.474→198.292秒（减少21.15%）；当前代码两轮完整PPO资格待验收；先前b711的两轮100%覆盖资格不能替代本阶段。正式246保持停止；原shadow 5%对照尚未按新覆盖率重新对齐。

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
| C01 | `rl/algorithms/full480.py` 的 `collect`；`accumulating_ppo.py` 的入口校验 | 原自然局循环内增加固定每槽768停止与原buffer直接写入；Dome及固定PPO未搜索/强制/锚点行使用原GPU upper队列 | 同一ready/submit/receive，不新增循环或add_rows；P01–P05/P08–P09列明新增成本 |
| C02 | `full_seeded.py`、`paired_token_rollout.py`、`env/search/small_playout.py`、`training/small_playout.py` | 恢复 batch.prepare / accept_ready；在原 score_roots 替换模型输入与小网；根观测由现有 NativeRootObservationBatch 一次编码复用；当前记录保存 canonical observation、动作映射、visits、Q、prior_logits、推荐动作 | 未另造完成队列或根排序；GPU 已算出的 64 维投影随原结果包返回，避免 CPU 再编码整模型。Dome 改由 C01 的原 GPU 队列评分，其他无 GPU 投影的根仍有缓存消费者 |
| C03 | `rl/search/rollout_inference.py`，`_DeckForward`、`_DeckConsumers`；`UpperPolicyInference` 的 deck 分支 | 新增构筑编码及两份静态投影消费者、动态 head、独立锚点结果缓存；旧 `_upper_graph_inputs` 接回；head 图容量设为 256；缓存权重登记检查和一次返回上下文 | 复用原 `ExactInputCache` 与 `CombatLeafGraphs`，但两个 Deck 消费者类和路由是我写的；未关闭构筑/锚点缓存 |
| C04 | `rl/search/inference.py`，`CombatLeafGraphs._capture` | 增加 `scratch_only`、共享 scratch 时图外独立设备输出区、最后一个图淘汰时重置 scratch handle | 修改旧图执行器的内存所有权，不是仅换模型参数 |
| C05 | `cpu_boundary.py`、`fallback_frame.py`、native `bindings.cpp::pack_raw` | 删除 compact→decoder→observations 字典；原 WaveDiag 原生批量填原始 scalars/category/牌堆升级数、布局、抽样数，进入原单份 pinned 区和融合图 | 当前数字布局由模型输入替换而来，仍是本次改动；没有逐分支 Gym 字典或 Python 观察解码。特征共用 GPU observer 的 token 函数 |
| C06 | `native/bindings.cpp`、`native/wave.cpp`、`native/include/sim/search/WaveDiag.h` | 仍保留当前 AR 前缀和 PCG 状态字段；删除 compact 字典接口、叶子 neural anchor/cutoff 请求；恢复原终局值/未完成 NaN，并允许精确预算的末个不完整 world | 原 WaveDiag 线程池、任务队列、ready/choose 继续使用；新增接口和模型字段不是原件原有 |
| C07 | `worker_devicestepfull480141.py`、`service_devicestepfull480141.py` | 删除根先请求 upper 后回传再提交的绕路；原 submit 内计算/查询构筑缓存，原根结果包附 64 维投影；fallback 沿原命令处理原始数组 | FIFO、64 命令后 tick、begin/take 保留；新数据只服务当前模型输入与缓存，不新增消息通道 |
| C08 | 新文件 `rl/search/full480/deck_inputs.py`、`deck_inputs.cuh`；原 `observer_entry.cu`、`token_builder.cu`、`frame_modules.py`、`frame_pipeline.py` | 新增 `DeckInputTables` 和 CUDA 当前模型特征构造；原 observer/builder 增加 Deck 编译分支、当前物理槽位/token/查询映射；新增构建函数及升级表溢出增长 | 两个 deck_inputs 文件是本次从零新增的模型输入适配，不是旧文件搬迁。执行仍嵌在旧 observer → builder → frame 中 |
| C09 | `rl/search/full480/combo_model.py`、`combo_kernels.py`、`fallback_ieee_model.py`、`fallback_ieee_kernels.py`；`rl/policies/deck_latent.py` | 新增 `DeckWeights`、当前输出头、global/subset/action 融合 kernel；旧 2×64 主体接当前头；从已有 `encode` 提取 `token_inputs` | 旧 attention/FFN、查询子集和融合执行保留，输入与输出头分支是我改写的；当前较宽输出头复用结束使用的 QKV 存储，避免额外 576 MiB（生产帧）的分配 |
| C10 | `rules_entry.cu`、`finite_plan.py`、`flex_runtime.py`、`flex_stream.py`、`frame_pipeline.py`、`step_pipeline.py` | 自然局保留旧终局HP/Boss目标；固定PPO在原reap接当前anchor/cutoff值，复用已有观察kernel和缓存。永久变化时刷新对应context | 原GPU原地clone/step、页租约、整步图和回收保留；P06新增完成值函数必须计入适配，不再增加逐链anchor副本 |
| C11 | `search_auxiliary.py`、`room_returns.py`、`tests/unit/test_search_auxiliary.py` | 已撤回独立add_rows/row_positions及外传last_values；仅保留full480名称识别；新collector调用原teacher、room returns和PPO | 既有CPU/CUDA真实更新及普通恢复保留；b711两轮四rank100%覆盖/16Adam/恢复已过，P18/P19当前代码资格另验 |
| C12 | `run_device.py`、`layout_bindings.py`、`sim/sts_lightspeed/CMakeLists.txt`，各运行模块的 import | 去掉历史固定目录/固定模型加载，改显式产物路径与包内 import；添加原生 full480 构建目标及依赖校验 | 运行绑定和构建改动，不是纯模型数学改动；部分文件只有 import 改动，不能与新增搜索逻辑混算 |
| C13 | `native/cpu_bridge.cpp`、`native/include/state_codec.h`、模拟器`ActionQueue/Actions/BattleContext`及原arena导入校验 | 原逻辑状态codec加入带类型回调；原动作函数执行，保留队列顺序/clearOnVictory，支持CARD_SELECT；显式新增bridge构建目标 | 这些文件及回调表示是本次改动；未登记回调仍拒绝，完整256×768实测0CPU回退。新增回调包装的成本也属于性能审查范围 |

## 原样保留与并未由我修改的文件

原件导入后，`root_ledger.py`、`state_pages.py`、`frame_kernels.py`、`cooperative_gpu.py` 没有本次行为改动；native 的 `raw_public.h` 保持原件。`native/include/zones.h` 是历史依赖原样导入。`step_kernels.py` 已由P19修改，不再归为原样保留。

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

C08的原强制动作跳过边界现前移至`deck_input::prepare`：普通单动作保留原动作与合法mask，免去无消费者的观测/实体打包；value_only仍打包全部输入并沿同一图估值。17项测试通过，新增单动作估值与eager数值断言；本改动尚不能当作完整速度改善证据。

C04接漏历史合批：worker有效剖析624个上层批次被新增消费者拆成3206次小头图。现在直接调用11月9日3881a367已有的MERGED_UPPER_PHASE；三个实际使用的模型方法与8bac9752逐AST相同，未复用11月13–14日新采样代码。普通上层合批，card_select/combat/anchor合同与旧队列不变。正式246固定64行上层ABBA各100次约2.05→1.07秒（输出通过容差），17项回归通过；完整480收益另验收。当前最新完整f5c048f3为547.726秒/131.89万，仍未达标。

C08删除GPU临时完整Row及复制：canonical packInto直接写原帧，CPU pack仍用同一函数、原默认输出。17项语义/生命周期通过，固定5472状态评分图ABBA约244.4→208.1ms，只有局部14.8%收益；完整轮另验收。C03单阶段继续原phase头，仅混合普通阶段走历史合批；此前强制单阶段合批实测慢15%，修正后100次约0.267秒，与原0.265秒相当。当前最新完整0689b0b6为535.988秒/134.97万，仍未达标。详细失败回执、原生SHA与完整资源见worklog相应小节。

C03根评分删除自加的整份100字段重复stack与拆分，保留原根观察对象；只合并小网输入。固定正式246输入logits逐位相同，1/32行100次ABBA约0.303→0.277/1.00→0.834秒，17项原生/CUDA回归通过。最新27135eea完整480为472.811秒/152.88万，尚未达标。每场缓存诊断3797场/39402根，同场35605次构筑输入均未变，submit实际编码约3752行；非空队列GPU状态接入及256×768/64-64-256对照待下一阶段。


C03/P03移至原GPU根arena评分：沿原FramePipeline图执行，单独缓存root只读图，地址来自原始根而非随机化链；原返回包附logits/value，PairedTokenBatch消费，消除parent重复根模型计算。P10 fixed模式直接用现有_last_obs，删除第二次编码。P11在原tick延后状态读取，原锚点精确缓存处理时推进其他活跃GPU链，收完溢出重试才释放页；没有新增采样器或缓存。P12只增计时及纠正anchor计数。P13将现有上层队列等待设10ms，属于合批实验，不宣称旧设置原样恢复。

上述组合已完成两组256×768：满预算449.150935秒，25%预算323.768714秒，分别比最初567.187656/353.311087减少20.8%/8.4%。单独重叠时459.203712/340.456644秒；GPU去重曾为518.849101/382.289729秒，因收益不足撤回。26项原生/CUDA组件测试117.02秒通过，10ms配置CPU/CUDA PPO两项38.58秒通过；两完整采样均196608行、原根/链数量、0回退、0更新、模型/Adam不变。小预算10%及原180万/秒门槛未过，四rank两轮16Adam尚未验收；完整资源与失败尝试见worklog/2026-11-15.md。


P14扩展2026-10-29历史UpperValueRuntime标量缓存接数组批次，沿2026-10-28已有NeuralSearchBatch池8线程并行打包/只读查键，join后按原顺序更新旧LRU；_DeckConsumers仅锚点接此原生方法，保留旧扩展与纯Python缓存兼容。P15原worker每64命令循环，将批尾正常tick前移到upper的CPU准备期间，finally收状态，尾部仍原reap，不额外增加正常tick。两者是本次新增修改，不能称原样保留。

同固定256×768：满预算359.491635秒、25%预算249.194280秒，对上一阶段449.150935/323.768714减少19.96%/23.03%，阶段10%通过。root/chain/预算/截断/真实196608行与基线一致、0CPU回退、0更新、模型/Adam不变，40项组件及后续4项CPU/CUDA交错PPO/恢复与兼容通过。满预算162.05万/秒仍低于180万，四rank生产两轮未过；不用短测试替代完整资格。


P16只将原模拟器燃烧契约在选择后排队的DrawCards换成已有OpeningAction::Draw，并在原GPU编译输入中替换对应语句。触发证据为rank2正式246的slot43/48/160（EXHAUST_ONE、两项待执行回调、portable=false），不是关闭CPU回退检查绕过。未升级/升级两个CPU往返回归0.81秒通过；GPU规则产物重建及四rank生产资格尚待完成，不能把此前rank0/1完整采样结果外推为四rank全部恢复路径通过。


P18在原64命令循环合并根提交，来源镜像推进、非submit/step命令、同步回复和批末先flush；跨无关镜像推进仍保留。P19在原StepPipeline复放最多4步后读状态，实际context_dirty在原GPU compact内归约，原32字状态区27/28保护刷新/步数边界，溢出沿原扩容重试。没有新增运行时文件、消息、线程、搜索算法或CPU特殊路径；5个原运行时文件及一个既有集成测试文件是本阶段实际修改范围。

原单步API仍默认1，生产worker明确4；原upper与锚点缓存的CPU期间继续推进，所有页租约、forced跳过、共享临时空间和权重检查保持。新增测试保留逐动作CPU oracle，额外覆盖多帧、半步已提交后注入溢出、Ritual Dagger永久输入变化，以及两根提交跨无关镜像/来源镜像推进。四机满预算第二慢408.808452→346.955725秒（减少15.13%），单独P18和全局保护试版只3.99%/6.12%，没有独立晋级。完整结果、源SHA、重试修正前后边界和剩余资格见worklog/2026-11-15.md。


| P20 | 原模拟器Actions/Monster/BattleContext的待执行怪物格挡、换意图、回合末格挡/群伤/球动作登记为原OpeningAction小对象；GPU冻结编译输入对应原句同步替换 | 修复续训CODEX/HEADBUTT场景拒绝导入，动作体/入队顺序/胜利清理标志不改。无新执行路径、CPU回退或观察传输；新增四组CPU序列化与GPU逐动作oracle回归。此为正确性补漏，不计性能阶段收益；最终验收见worklog/2026-11-15.md |


| P21 | 原GpuService父CUDA交接处在已有同步后回收无引用对象并归还闲置allocator缓存 | 修复完整PPO后第二collection启动owner时stack limit CUDA错误2；不降低栈/堆，不关图或构筑/锚点缓存，活模型/Adam保持。大块闲置显存回归先复现同错，修复后CPU/CUDA两例及完整47项通过；连续真实生产验收继续从247进行 |

# Combat49：本次新增与原代码改动清单

用户原话：“你明确标出你自己加了哪些部分的适配哪些部分的改动”；并要求“最小化所有的适配，尽可能只在原本基础上改不加东西”。

核对时的已提交版本为 `dbf8e466`，分支 `codex/combat49-legacy-engine`；另包含下述 4 个尚未提交的根观测文件。本文是本阶段的审查记录，不是新的运行入口。

**当前实现并非仅替换模型权重或模型调用。除原 GPU 采样主体外，我新增了模型消费者类、输入表、CPU fallback 的观察传输分支、PPO 接线和根结果处理。特别是 C01、C02、C05、C07 改变了原数据流。不能将这些统称为“原代码原样复用”。**

## 基线与读法

- 旧 GPU 完整主体以实际 `stepfull480-01` 归档为准，SHA256 `3a72e10edb48ae26a44655acfba717f3cc4cec707e3476c2229dd4d03258ebc3`。`d5cd40be` 是原件入库基线；38 个历史源码文件的导入不等于本次新写了 38 个算法。
- 旧 CPU WaveDiag 原件以 `248aa3f8` 的导入为基线；`native/include/zones.h` 是另外核对过的历史头文件，来源及 SHA 在原有 runtime-sources.json 中。
- 原仓库已有的推理、buffer、模型、learner 等公共文件以任务开始 `e507fdde` 为基线。不能统一用 `d5cd40be` 比较，因为缓存和 buffer 的初次改动已经在这个提交之前。
- [逐文件、逐改动段清单](combat49-adaptation-audit.json) 给出每个文件的比较基线、增删行数、当前行号、内容 SHA256、是否尚未提交；行数按文本 diff 计算，不代表语义改动大小。
- 另有忽略目录内的可直接查看补丁 `runs/local/full480-reconnect/adaptation-audit.patch`。未包含被用户拒收的 8318/661b 采样代码。

## 我新增或改变的实际执行部分

| 标号 | 文件与定位 | 我具体新增/改变的内容 | 与旧路径的关系 |
| --- | --- | --- | --- |
| C01 | `rl/algorithms/full480.py`，`collect`、`consume_upper`；`rl/algorithms/accumulating_ppo.py` 的 full480 分支 | 从实际旧 collector 导入后改成固定每槽行数；把战斗根也送进 upper 请求；加入真实动作/原值/logp/teacher 写入、VecMonitor 记账、drain、GAE、回调取消。learner 新增 5 行分流到该文件 | 原 ready/异步请求/立即推进循环来自旧件，但 **战斗根先过 upper 的流程是我改的**；`full480.py` 也是我新增的仓库入口，不能只说“原入口未动” |
| C02 | `rl/search/full480/full_seeded.py`，`plan_job`、`Stream.submit`、`Stream.accept_ready` | 根包增加完整上下文投影、构筑/map 输入、AR 前缀、配对世界；完成时直接构造当前 PPO teacher、动作和行为概率；新增方法 `accept_ready` | 原来调用 `self.batch.prepare` / `self.batch.accept_ready`；当前这两个接线点被我换成直接处理。**不是原 PairedTokenBatch 的原样调用** |
| C03 | `rl/search/rollout_inference.py`，`_DeckForward`、`_DeckConsumers`；`UpperPolicyInference` 的 deck 分支 | 新增构筑编码及两份静态投影消费者、动态 head、独立锚点结果缓存；旧 `_upper_graph_inputs` 接回；head 图容量设为 256；缓存权重登记检查和一次返回上下文 | 复用原 `ExactInputCache` 与 `CombatLeafGraphs`，但两个 Deck 消费者类和路由是我写的；未关闭构筑/锚点缓存 |
| C04 | `rl/search/inference.py`，`CombatLeafGraphs._capture` | 增加 `scratch_only`、共享 scratch 时图外独立设备输出区、最后一个图淘汰时重置 scratch handle | 修改旧图执行器的内存所有权，不是仅换模型参数 |
| C05 | `rl/search/full480/cpu_boundary.py`，`CpuFallback` 的 `self.paired` 分支；`fallback_frame.py` 的 `allocate_deck` / `execute_deck` / `score_deck` | 新增 compact → canonical 观察解码 → observations/context_roots/layout 传输，再调用当前 token 输入和旧融合网络的分支；后来接回原生 `pack_raw` 的布局生成及根共享静态上下文 | **旧版是原生 `pack_raw` 的 18 个数组直接进入单份 pinned 区和旧评分图；当前新增完整观察字典的传输分支是我写的**。该项应优先收敛回原数据路径 |
| C06 | `rl/search/full480/native/bindings.cpp`、`native/wave.cpp`、`native/include/sim/search/WaveDiag.h` | 在原 WaveDiag 上加配对状态、AR 前缀、锚点/截断返回值；新增 `configure_paired`、`compact`、`choose_paired`、`paired_results` 和 `pack_raw` 重载 | 原线程池、队列和推进主体保留；这些方法与分支是本次新增。整个 native 目录不是新写的，`raw_public.h` 未改 |
| C07 | `rl/search/full480/worker_devicestepfull480141.py`；`service_devicestepfull480141.py` | worker 增加当前模型选择、upper 回复的 logits/context、fallback 观察字典分支；service 接冻结父 GPU learner、模型校验、原型输入及按消息类型计字节 | 原 FIFO、最多 64 命令后一次 GPU tick、begin/take 保留；**根观察进入 worker、上下文回复父进程后再次提交，是我增加的数据流** |
| C08 | 新文件 `rl/search/full480/deck_inputs.py`、`deck_inputs.cuh`；原 `observer_entry.cu`、`token_builder.cu`、`frame_modules.py`、`frame_pipeline.py` | 新增 `DeckInputTables` 和 CUDA 当前模型特征构造；原 observer/builder 增加 Deck 编译分支、当前物理槽位/token/查询映射；新增构建函数及升级表溢出增长 | 两个 deck_inputs 文件是本次从零新增的模型输入适配，不是旧文件搬迁。执行仍嵌在旧 observer → builder → frame 中 |
| C09 | `rl/search/full480/combo_model.py`、`combo_kernels.py`、`fallback_ieee_model.py`、`fallback_ieee_kernels.py`；`rl/policies/deck_latent.py` | 新增 `DeckWeights`、当前输出头、global/subset/action 融合 kernel；旧 2×64 主体接当前头；从已有 `encode` 提取 `token_inputs` | 旧 attention/FFN、查询子集和融合执行保留，输入与输出头分支是我改写的；并非整套模型 kernel 原封未动 |
| C10 | `rl/search/full480/rules_entry.cu`、`finite_plan.py`、`flex_runtime.py`、`flex_stream.py`、`frame_pipeline.py`、`step_pipeline.py` | 加当前 PCG 配对状态、AR 前缀、永久变化检测、锚点和 cutoff 评分；新增 `refresh_contexts` / `complete`；根及链的 context/RNG 数组加入旧所有权生命周期 | 原页租约、GPU 原地 clone/step、整步图和完成回收保留；当前叶子值处理、随机状态输入与完成时评分是新增/修改。批量随机初始化遵照用户已放宽精确序列的要求 |
| C11 | `rl/algorithms/search_auxiliary.py`，`SearchRolloutBuffer.add_rows`、`row_positions`；`room_returns.py` 的 `last_values` 参数 | 新增固定 `(time, env)` 独立完成行写入及校验，搜索 teacher/实际动作 side data；允许使用 worker 返回的末端值 | 新增训练交接接口，不是旧 PairedTokenBatch 的样本保存。当前 PPO 更新算法未重写 |
| C12 | `run_device.py`、`layout_bindings.py`、`sim/sts_lightspeed/CMakeLists.txt`，各运行模块的 import | 去掉历史固定目录/固定模型加载，改显式产物路径与包内 import；添加原生 full480 构建目标及依赖校验 | 运行绑定和构建改动，不是纯模型数学改动；部分文件只有 import 改动，不能与新增搜索逻辑混算 |

## 原样保留与并未由我修改的文件

原件导入后，`root_ledger.py`、`state_pages.py`、`step_kernels.py`、`frame_kernels.py`、`cooperative_gpu.py` 没有本次行为改动；native 的 `raw_public.h` 保持原件。`native/include/zones.h` 是历史依赖原样导入。

`graph_kernels.py`、`graph_model.py`、`pool_gpu.py`、`ragged_model.py`、`ragged_pack.py` 只改包内 import。`ragged_runtime.py`、`search_runtime.py` 还改历史产物定位；其他 runtime 文件的模型选择、尺寸或生命周期变更均已计入 C08–C12，不归为“未改”。

**纠正之前给出的入口清单：`rl/search/paired_token_rollout.py` 和 `env/search/small_playout.py` 在本任务中没有改动。旧根记录在这两个文件生成，但当前 collector 实际走 C02 的新 `Stream.accept_ready`；之前将 paired_token_rollout.py 列为当前适配位置不准确。**

## 尚未提交与已撤回的内容

- 当前尚未提交的 4 个根观测文件：`env/core/deck_context.py`、`env/core/native_observation.py`、`sim/sts_lightspeed/bindings/slaythespire.cpp`、`tests/integration/test_native_root_observation.py`。它们把当前模型的牌堆升级计数/总量放进已有 `pack_neural_roots`，替代 Python 每牌构造；不是新增采样入口。新增 `deck_context_schema` 参数和能力版本标记也属于我加的接口。明确构建 SHA `4daefadd5b63a056b7e01ca262d4935ac61699a31d0bf5b6f69fbe8589e681f4` 下 20 项测试通过，尚未做这版完整长度性能验收。
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

用户后续要求是收敛 C01/C02/C05/C07 的额外数据流，尽量回到旧包和旧调用位置。本文只标明现状，不声称这些部分已回到原路径，也不声称适配完成。

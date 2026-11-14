# Combat49：以旧版完整采样代码为主体重新接入当前训练

交接时间：2026-11-14，北京时间。任务入口：`TODO.md` 的 `COMBAT49-LEGACY-ENGINE`。本文件是下一会话的实施合同；本轮只撤除拒收实现和交接，不继续优化或恢复训练。

## 1. 用户原意与当前结论

用户原话：

> “不是旧的接到现在上，是要把现在的训练接入到之前的实现上。”
> “采样部分指的是整个训练中除了更新部分的部分，不要再漏东西了。”
> “不是做迁移，是直接在旧训练整个采样部分代码上加东西。”
> “你把要复用的范围和对接目标写在文档里面交给下一个会话做，删掉你写的所有优化。”

执行方向必须是：**旧版完整采样代码直接作为主体，只增加当前训练必需的消费者接口。** 不在近期采样器上继续抄局部优化，不照着旧逻辑重新写另一套调度器，不保留近期实现再称为“旧版主体”。

此前确实只复用了部分服务、推理图和内核，采样主循环及部分搜索推进重新写成了适配实现；“整套接入”的表述不准确。最后完整测试也未达标。当前分支已撤除这套实现，不能当成已经完成旧版接入或可部署训练的版本。

下一会话范围是修复接入方式与等价性能优化，不是改训练实验。用户没有撤销最终目标：同 5090 完整采样至少 **180 万有效非强制搜索决策/秒**，并从正式保存点恢复带搜索的连续训练。最近的交接指令要求本会话止于清理和文档。

## 2. 唯一复用基座：实际运行过的 full480 原件

仓库锚点为 `7c947c28f215ccdb75f23e72090b21d9c7d1ba25`，运行身份为 `stepfull480-01`。**单看这个 commit 的通用 collector 不够**：当时入口做了运行时拼接，实际 collector 和真实 import 快照必须一起用。

原件目录（本地）：

`D:/sts-rl/runs/local/rollout-200s-worktree/runs/local/nine-token-speed/online480/waves/devicestepfull480141`

该目录中的事实来源：`stepfull480-01-complete.json`、`stepfull480-01-exit.json`、`stepfull480-01-sources.json`、`launch-source-manifest.json`、`snapshot-manifest.json`、`archive-manifest.json`。

完整源码归档：`step-full480-tail-v1.tgz`，21,486,659 字节，SHA256：

`3a72e10edb48ae26a44655acfba717f3cc4cec707e3476c2229dd4d03258ebc3`

本轮重新核过归档存在、大小和 SHA。已解开的只读审读副本：

`D:/sts-rl/runs/local/combat49-maintenance-worktree/runs/local/pro6000-migrate-20261112/inspect480`

不能用近期名字含 `legacy` 的目录代替此原件。实际 CUDA 编译产物、规则/状态 ABI、native 扩展也须按上述 manifest 固定；Git 中的 Python 文件不能单独代表原来的运行组合。私有模拟器源码和编译产物保留在忽略的材料目录，不提交 Git。

### 原件的实测效果与口径

| 项目 | stepfull480-01 实测 |
| --- | --- |
| 设备、负载 | 5090，480 个自然完成局，928.9251 秒 |
| 有效非强制搜索决策 | 1,837,685,220，**1,978,292.091752/秒** |
| 搜索根 / simulation 链 | 233,897 / 99,603,500 |
| 完整原子动作 | 2,361,845,725；不能当成有效非强制决策 |
| 真实动作 / transition / 选择前缀 | 372,878 / 371,210 / 1,668 |
| 搜索预算 | 普通及精英 250、boss 1000，最大深度 200 |
| 自然局结果 | 360 胜/480 局，22,983 层；属于旧模型，不是当前模型成绩 |
| CPU / GPU | 进程树 CPU 约 190.46%（约 1.9 核），设备 GPU 均值约 69.19%，显存峰值 26,804 MiB |
| 真实环境推进 | 157.706 秒，54,515 个 step batch；约 0.423 ms/真实动作 |
| upper 消费 | 18,650 批、81,497 行，40.835 秒 |

它不是当前 256×768 固定长度 PPO 的同负载对照；模型、预算、搜索目标、RNG 和训练消费者也不同。旧实现曾省略 289,713 个没有学习消费者的战斗观测，当前 PPO 需要这些行，不能照抄“省略”而丢数据。并发等待和嵌套计时不能相加凑出整轮时间。

## 3. 必须整体直接复用的范围

下表文件均相对于 `inspect480`。先建立原件与实际运行文件的逐文件映射；能保持原件的代码直接保持，只审查必要的接口差异。

| 整段范围 | 原件入口 | 对接限制 |
| --- | --- | --- |
| 整个真实采样主循环、ready 环境选择、非战斗动作、终局/reset、搜索结果回收 | `full_seeded.py`、**`stepfull480-01-collector.py`** | 后者是实际拼接后 collector；不是只取通用 `independent_rollout.collect` 名称 |
| 持续接收搜索、上下层请求、返回分发、单 GPU owner | `service_devicestepfull480141.py`、`worker_devicestepfull480141.py` | 保留 FIFO、最多 64 个命令后推进一次 GPU tick、在途请求；不改回整组等待或每批搜完再推进 |
| 搜索根生命周期、真实战斗镜像、代际、分支、完成即回收/补入 | `flex_stream.py`、`root_ledger.py`、`state_pages.py`、`cpu_boundary.py` | 原 `GameMirror` 的 load/step/drop、真实动作重放、根/兄弟隔离、非移动页所有权完整保留 |
| 每轮搜索的观测、动态打包、模型前向、抽样、apply、live 压缩 | `step_pipeline.py`、`step_kernels.py`、`frame_pipeline.py`、`frame_kernels.py`、`frame_modules.py` | 保留整步图和持续接搜索结构，不能用近期 Python 循环替换后称已经复用 |
| 2×64 下层网络的冻结推理布局、融合和临时存储生命周期 | `flex_adapter.py`、`flex_runtime.py`、`flex_ragged.py`、`combo_runtime.py`、`combo_model.py`、`combo_kernels.py` | 保留调度/打包/图执行方式；当前特征、权重及评分头只能作为必要消费者替换 |
| upper 和特殊状态评分的异步请求、图传输、缓存/所有权 | `actual-import-snapshot/devicegraph141/{graph_runtime.py,graph_model.py,graph_kernels.py}`；`actual-import-snapshot/devicefallback141/{packed_fallback.py,fallback_model.py}`；`fallback_frame.py`、`fallback_ieee_model.py`、`fallback_ieee_kernels.py` | 保留 begin/take、host 输出独立所有权、scratch 复用及生命周期；不能只复用下层模型前向 |
| 原生回退、动作映射、序列化、观测和完整 continuation | `cpu_boundary.py`、`layout_bindings.py` 及 manifest 指向的 native/规则产物 | 回调队列不能丢；先核原版本实际怎么处理，不能看到 CPU 回调就默认新增 CPU 慢路径 |
| 从采样完成到现有 PPO 更新的交接 | 原采样完成边界 + 当前 buffer/返回值/日志接口 | 当前 GAE、搜索目标、样本槽位、资源计数、checkpoint drain 都要接齐；更新实现不重写 |

原件参数包括 upper 等待 0 / batch 64、单 owner、原生回退 8 线程 / 容量 128、frame 65,536、token 容量 frame×32、query 容量 frame×8。这里是源行为说明，不能把物理容量与每根搜索预算混为一谈，也不能不测显存就强填某个容量。旧临时特征/QKV 共享、页回收、缓存权重代际、shape 变化和退出释放属于整套范围。

## 4. 当前训练是消费者：必须保持的接口与语义

| 接口 | 需保持的当前契约 |
| --- | --- |
| 模型 | `deck-latent-combat-v1` / `DeckLatentPolicy`，512 构筑潜向量、Transformer64×2 战斗网络；完整参数与 Adam 从正式 246 加载，不用旧 full480 模型权重替换 |
| observation | 当前 canonical codec、198 动作 autoregressive 合同、完整候选原生槽位/PAD-ID 候选、怪物空槽、pending 来源；所有 PPO 需要的 observation 必须齐全 |
| 构筑与遗物 | 冻结构筑向量按真实永久变化刷新；战斗计数、有效标志与蜥蜴尾巴失效不能丢；三个牌堆按当前计数/汇总语义处理，重复状态牌不能膨胀成独立模型 token |
| 学习目标 | `pre-reward-anchor`，领取奖励前战斗退出锚点、当前 terminal/cutoff 定义、room returns；旧 HP/boss-win leaf 目标不能覆盖它 |
| 搜索随机性 | 当前配对世界/PCG64、rank seed、抽样覆盖率、候选顺序、每根预算完整；旧 counter RNG 不能未经语义验证直接替换 |
| rollout 动作 | 用户指定 **搜索筛选、模型排序**：已访问合法动作中 Q≥最大 Q−0.005，再取当前模型最大 logit；同分用显式 nonce 均匀选。结果必须直接推动真实环境，不能仍是 shadow 教师 |
| PPO 数据 | 每 rank 256×768=196,608 个真实行；保留 (time, env) 身份、实际动作、动作 mask、实际行为 log probability、旧值、奖励、终局、search teacher/来源。搜索分支不写成真实样本 |
| PPO 损失 | 搜索选中的行不伪装成模型 on-policy actor ratio；保留 value/辅助损失；未搜索行维持 PPO。对应 `search_auxiliary.py`、`ppo.py`、`action_selection.py` 的契约测试保留 |
| 恢复边界 | drain 后完整模型、Adam、learner、各 RNG、256 个环境及 journal；普通恢复保持旧 run/W&B 身份、阶段、评估次数与曲线 |
| 可观测性 | 本机/云端沿用正式轮次；全 rank 固定 scalar schema；raw 搜索计数先逐 rank 保存，再聚合。不能把诊断轮次上报成正式 247 |

本轮只保留了 CPU 语义参考采样和上述动作/PPO 契约，不把它当作性能解决方案。已撤除的 `cuda-resident` / `cuda-batched-native` 配置仍可读，以支持模型加载与保存点检查；尝试执行会明确失败，**不会静默改成 CPU 或 shadow**。下一会话实现原件接入后才能重新开放对应执行入口。

### 外层训练参数与更新阶段不改

四机原计划白名单实读：每机 256 环境、768 步，batch 1536、1 epoch，LR `5e-5`、gamma `1.0`、GAE lambda `1.0`、entropy `0.02`、value coefficient `0.5`、max grad norm `1.0`、clip `0.2`、seed `20261111`（rank 搜索 seed 为 `20261111 + 1000003×rank`）。每轮每卡 16 次 Adam。

保留 `configs/train/deck_latent_448_fp32.json`、现有 packed PPO 更新、FP32 和原多机训练架构。普通/精英搜索预算 64、boss 256、深 200，不减预算、环境或实际工作量来过吞吐门槛。实际参数还须以保护包和现役计划复核；不根据文件名猜分块方式。448 快速更新仅 warmup，anchor/search 的普通 packed 分流是已有约定。

## 5. 正式状态、恢复材料与接手位置

本轮只读复核四机无训练 PID、无 GPU 进程；STS 四 rank `alive=false`、step `193462272`，数据新鲜。**正式停止于完整第 246 轮**，第 247 轮显存页扩容 OOM 现场保留，未发生正式 247 保存。

四机：rank0 `connect.westc.seetacloud.com:16932`；rank1 `connect.weste.seetacloud.com:30831`；rank2 `connect.westd.seetacloud.com:27357`；rank3 `connect.westd.seetacloud.com:44669`。均为 5090；CPU 不完全相同，因此速度和资源必须逐 rank 报告。凭据由既有安全访问辅助程序管理，不写入本文。

定义远端根 `O=/root/autodl-tmp/combat49-northwest-4x5090`。正式入口仍是 `O/source-bc081439` 的既有计划，**它是旧部署现场，不是本交接指定的新代码基座**。禁止盲目重跑该计划或任何旧 launch/resume 脚本。

正式保护包：`O/references/search-memory-20261114/rankN/step-000193462272`，包含 `model.zip`、`learner.pt`、`environment.pt`、`manifest.json`。历史四 rank 成员 SHA、跨 rank 模型/Adam 一致，302 个 Adam 槽有限，step 3936，256 环境/rank 和四类 RNG 完整；这些材料不删除。

独立存储副本位于 `connect.westc.seetacloud.com:55105`：

`/root/autodl-tmp/combat49-resume-4x256-20261114/formal246-search-acting-20261114/rankN/step-000193462272`

这个目录名不表示其权重已按新 acting 行为训练。正式 246 仍是原 `deck-latent-ppo-search-v1`，stage=search/start240/good5/last240，保存中的 fraction=.375/coefficient=.025；按原 schedule 下一轮 .4/.03，最终 ramp 到 1/.1。过去后端 `qualified=true` 的记录不能充当新执行主体通过资格。诊断检查不能增加正式 good_checks。

当前工作区：`D:/sts-rl/runs/local/combat49-maintenance-worktree/runs/local/combat49-legacy-engine-worktree`，分支 `codex/combat49-legacy-engine`。从本交接提交继续；先读 AGENTS、唯一 TODO 和最新 worklog，并 fetch 共享分支检查交接。不要改主目录或其他会话脏工作区。

SSH 复用辅助程序：`D:/sts-rl/runs/local/combat49-maintenance-worktree/runs/local/nw5090-compare-20261114/all_remote.py`，不输出它依赖的凭据文件或完整 `training-plan.json`。读计划只选择必要白名单字段。

监控保留 1–246 曲线及正式停止状态；[云端日志](https://sts-training-monitor-tony.bentmuffin.chatgpt.site/worklog)已接通。WANDB_MODE 仍 offline，不能宣称 W&B 云在线。

## 6. 本轮实际撤除和保留的边界

撤除前代码固定在 `661b3f43`（最后执行候选源 `8318d305`），Git 历史和失败输出保留，不另外复制一套“备用”运行实现。

- 删除新增 `rl/algorithms/legacy_rollout.py`、`legacy_search_stream.py`；`rl/search/resident_*`、`deck_worker.py`；`rl/policies/resident_*`；`env/search` 下新 CUDA battle/bridge/vector、resident、host runtime、anchor wave、frontier、paired RNG、state pages/root ledger，以及 `env/core/cuda_observation.py`、`deck_context_native.h`。
- 撤回这些实现对原 observation、run fork、搜索 scorer、图推理、model inference、VecEnv、room-tail 和 learner 路由的优化改动。公共文件恢复到 `32ad6dd6` 的对应实现；**该 commit 只是清理用的当前模型/语义基线，不是下次要使用的 full480 采样基座。**
- 删除专用优化测试和失效文档入口，保留当前模型/训练目标/真实动作/PPO/普通恢复测试；`tools/validation/migrations.json` 标明退役，未新增兼容运行别名。
- 保留更新阶段、原训练配置、标量 schema/验证显存等已有故障修复、监控和证据。模型里的旧 service/finalizer 名称只保留在“不保存瞬态对象”的兼容排除表里，没有执行实现。
- 远端失败源码、编译产物和忽略的本地诊断文件是证据，不是 active 分支代码；没有清空恢复盘、删除 checkpoint、改写历史、部署或启动训练。

删除后已做 11 项 CPU/native 契约检查（含动作/PPO、拒绝撤除后端、fork/恢复隔离、207 张状态牌仍为三个牌堆 token）通过。另用正式 rank0/246 包逐 SHA 校验、CPU 加载模型与 302 个 Adam 槽、恢复一个真实环境、一次合法预测和一次环境推进，权重不变、Adam3936、0 更新、0 GPU 使用。CPU 加载有历史 CUDA 指标对象无法反序列化的提示，未将其称为完整训练恢复。未重跑 GPU 测速、collection 或测试训练。

证据：本地 `nw5090-compare-20261114/withdrawal-check-result.json`；远端 `O/search-memory-20261114/legacy-engine/withdrawal-check/{unit.log,checkpoint.log,result.json}`。

## 7. 未达标证据：不能沿用已有结论

最后 `8318d305` 的 `scratch-reuse/qual` 用同正式246、FP32、256×768、full1/.1、64/256、深200，四 rank 完整一轮退出0，样本/16Adam/保存通过，但 collection 分别 **283.890 / 244.170 / 281.341 / 402.538 秒**；有效速度 **49.89 / 60.31 / 51.37 / 35.38 万/秒**。这是失败候选，不是正式推进。

rank0/1 的 native fallback 为 0，rank2/3 为41/67根，不能把全部差距解释成特殊状态。rank1 CPU均值约168.29%、GPU约54.33%，也不是 CPU 满核心。其上层评分、环境推进和调度仍有明显额外开销；战后 anchor forward 只有约3.522秒，缓存实际命中，不能泛称“锚点评分拖慢全部”。计时存在并行/嵌套，见原始记录，不能相加。

本地证据目录 `nw5090-compare-20261114` 下 `legacy-scratch-verified.json`、`legacy-scratch-collection-PORT.json`、`legacy-scratch-progress-PORT.json`、`legacy-scratch-row-PORT.json`。远端原始诊断在 `O/search-memory-20261114/legacy-engine/scratch-reuse/qual`。之前 `9417014f/shared-full` 也未达标；不继续以这些重写为底打补丁。

## 8. 下一会话实施顺序与验收

1. 复核四机真实进程、正式246包、监控、工作区与最新留言，确认没有并发新部署。核 old full480 归档、实际 collector/import/产物 manifest；阅读整个非更新链路，在下一次测速前一次性列清所有必要接口差异。
2. **直接从原件建立执行主体**，可迁入所属包并改 import/路径，但保留可审查的原件→当前文件映射。只接当前模型/context/leaf、动作 codec、真实 PPO buffer、固定长度 drain 和恢复/日志；不得使用已撤除的近期 sampler/adapter 作主体。每个新增部件说明旧原件为何没有对应消费者，不能用“优化”掩盖重新设计。
3. 先跑固定根/seed/预算的状态、动作、奖励、mask、候选、RNG、teacher 和恢复检查；验证暂存状态不串根/兄弟，不在 retry 或强制动作中多耗随机数。图/缓存必须过 shape、权重变更、淘汰、临时内存释放后重放、显存归还。先过语义再测速；不要一项项小改、小测找收益。
4. 在相同 5090、同正式246来源做有界完整 256×768 collection，记录 cold/warm、总墙钟、有效非强制决策与完整原子动作分别计数。既有资格上限 collection 1800秒、总GPU窗口7200秒；超时/提前中止如实标失败，不降低负载。不能将固定帧 replay、短热态或纯搜索内核速度当完整采样速度。
5. 同246/5090/相同覆盖率和预算的原 shadow 对照为 rank0–3 **114.122360 / 106.325424 / 118.469130 / 175.998240 秒**；新 acting 完整采样各 rank 不超过对应值1.05倍，并满足至少180万有效非强制决策/秒。不同覆盖率、不同初态不得直接套这个比较；有差异先固定对照，不用旧480异构负载冒充严格A/B。
6. 正式启动前，新主体跑完整两轮采样/更新及中间普通恢复；四 rank 保存成员 SHA、模型/Adam 一致，196608行/rank、1epoch/16Adam、无缺样本/非有限/错误目标，实际 rollout 动作确来自搜索筛选。各轮记录 CPU进程树、RSS（共享页可能重复）、系统内存、Torch allocated/reserved/peak、设备VRAM/GPU利用率；不可得写null。
7. 通过后再固定 source/产物/计划，保护最新边界，恢复原run连续训练，**不设置首两轮自动暂停**。更新云端曲线和日志；每小时维护消息保持简短。没有通过则记录准确未完成项，不能再报告“已经完整复用旧实现/已到200万”。

本轮清理检查只证明删除后的保留契约和一个真实环境仍可用；不证明新采样已实现、性能达标或四机带搜索恢复已验收。

## 9. 原件复核与本次必要接口差异（13:10接续记录）

用户新增硬约束：一小时内；不能复用11月13–14日写的采样代码；必须直接在原本整个采样代码上增加当前消费者；不能因适配关闭任何旧优化，内层与锚点均须缓存。用户再次确认锚点已有缓存资格，不能把它当作整体性能瓶颈。第7节rank1约3.522秒证据仍适用。

[原件来源映射](combat49-full480-source-map.json)记录实际sources清单40项和patched collector，共41项逐SHA核验；归档SHA未变。整个原件已从归档解到本工作树忽略目录 `runs/local/full480-reconnect/original`，没有拿近期legacy实现覆盖。该清单不是完整传递依赖闭包：旧 `search_runtime`、`owned_pool`、`ragged_runtime` 等仍沿当时外部材料解析，不可擅自用名字相同的新文件代替。

另发现7c947c28通用collector与实际运行collector有一处真实差异：Git版通过`stream_type`属性取stream，实际快照直接使用注入的`SmallPlayoutStream`。按full_seeded原拼接重建后，仅该处不同；映射保存原始diff。后续必须以实际patched文件为底，不能用Git函数名代替。

本次只完成原推理消费者中的构筑/投影与锚点结果缓存接入，`rl/search/rollout_inference.py`和`inference.py`在开工e507fdde时与7c947c28全文相同，修改直接落在这两个旧文件上。未新写sampler/搜索调度器；未把失败的8318d305等采样实现带回。保留原精确缓存、CUDA图、共享scratch和device output，新增纯消费者的独立设备结果发布以维持二者同时成立。

在下一次完整测速之前还必须一次接齐下列原件接口，不得把此组件验收写成整套接入：

| 原件位置 | 必需改接的消费者合同 | 必须保留的旧执行机制 |
| --- | --- | --- |
| `worker_devicestepfull480141.py` / `combo_model.py` | 旧`policy.small_playout`/128输入/34输出硬布局与当前DeckLatent实体输入/198动作头不同；模型权重检查须接受正式246，而非替换其权重 | 原融合网络执行、帧内打包、特征/QKV存储共享、整步图 |
| `frame_pipeline.py` / 原observer与builder产物 | 当前canonical候选、pending来源、空怪物槽、三个牌堆计数及动态遗物字段；缓存构筑投影进入GPU帧 | GPU观测与动态压缩，不能改为逐lane Python重建观察 |
| `flex_runtime.py` / 原rules产物 | 旧counter随机数/HP及boss-win叶子改接当前配对PCG64、奖励前锚点/截断合同 | 根/兄弟隔离、非移动页租约、重试不多耗RNG、完成即回收 |
| `stepfull480-01-collector.py` / `full_seeded.py` | 完整自然局NOSL消费者改接固定256×768真实PPO行、搜索筛选模型排序、原值/logprob/teacher、GAE与日志 | 原ready/submit/receive/立即推进和异步upper，不另写循环 |
| `service_devicestepfull480141.py` / worker退出边界 | CPU collector/冻结GPU owner接当前更新及普通恢复；原固定模型digest、CPU-only断言与只读退出不能直接作为PPO更新协议 | 单GPU执行owner、64命令后的GPU tick、在途请求drain、完整权重/Adam/RNG恢复 |

正式246保护和停止状态未改变。256个恢复状态上的512×8模型推理与缓存检查是消费者验证，不是256×768 collection；详细回执、失败修正和未完成范围追加在worklog同ID。

## 10. 固定 PPO 接线改动清单

本节按用户要求列出本次新增改动，比较基点为 `9390b0e3`。这是审查清单，验证结果以 worklog 的 COMBAT49-LEGACY-ENGINE 为准；未通过完整长度和恢复测试前不称完成。

| 编号 | 原位置与本次改动 | 开销与验收重点 |
| --- | --- | --- |
| P01 | `rl/algorithms/full480.py:collect`：原 ready/submit/receive 循环增加每槽步数及768行停止条件；自然结束后仍由原 VecEnv 自动开局 | 不新增调度循环；检查每槽恰好768，终局、自重置、末行bootstrap与drain |
| P02 | 同一 `collect` 的原真实动作推进边界：把已有完整观察、动作、mask、old value/logprob、reward、episode_start直接写原 SearchRolloutBuffer 的(time, env)位置 | 不加中间buffer/add_rows；保留完整PPO观察会增加原自然局省略模式下未做的写入，须实测，不以数据量猜归因 |
| P03 | `full480/flex_runtime.py:submit`、`frame_pipeline.py:score_roots`：同一GPU owner用原只读帧读取未随机化的原root，结果经原reap/native_result到PairedTokenBatch.accept_ready；附小网logits/value及真实行为logprob | 不再由父进程运行根模型；原root只增加每槽8字节地址表及一个共享scratch的只读图。原整步图、缓存与CPU回退合同保留；shape/临时显存释放和完整行对照验收 |
| P04 | `collect`：未搜索战斗行、战斗退出锚点、真实强制动作进入原upper队列，获得真实old value | 真正的PPO行必须估值；内部搜索的强制动作跳过保持。新增外层评分行须单列 |
| P05 | `collect`收尾：原搜索根visits/Q/logits调用现有improved_prior，写原search_targets；随后调用原apply_room_returns | 新增teacher转换；原稿逐根Tensor计算已改512根合批，记录search_target_seconds。未修改teacher公式、GAE或PPO更新 |
| P06 | `rl/search/full480/flex_stream.py:reap`、`frame_pipeline.py:complete_values`、`finite_plan.py:aggregate`：原搜索完成边界接pre-reward-anchor/深度截断值 | 新增完成值消费函数；复用已有GPU anchor/misc观察kernel、只读小网图、构筑和锚点结果缓存；永久变化分支重算相应context。不能把此新增工作藏进“只换模型” |
| P07 | `rl/training/small_playout.py:configure_collection`、`env/search/small_playout.py:prepare`、`full_seeded.py:Stream`：读取原训练64/64/256预算、覆盖率及完整配对world合同 | 训练的完整world可能使实际链数不等于自然局精确预算；两种结果必须分别标实际链数/根数/原子动作 |
| P08 | `rl/algorithms/accumulating_ppo.py:collect_rollouts`：full480配置直接调用上述原collector；原训练更新继续消费原buffer | 不新增trainer或训练入口，不改变batch/epoch/Adam配置 |
| P09 | `collect`与`full_seeded.py:record_start/record_end`：原完整局记录允许跨固定片段；保存现有随机流状态、先发布本轮search统计再计算returns | 完整退出、取消和普通保存恢复验证；不能将上一轮chains记到本轮 |
| P10 | `full_seeded.py:submit`：固定PPO使用原_last_obs已有完整观测的批量拥有副本，删除第二次native root编码 | 数组必须拥有存储，不得随真实环境步被覆写；逐字段与原buffer对应行比较 |
| P11 | `step_pipeline.py:tick`原图重放允许延后收取完成状态；`frame_pipeline.py:complete_values`在原锚点缓存查键期间推进其他活跃链，使用已有根遗物数组消除重复设备读取 | 在原函数内复用同一整步图/溢出重试；当前完成页保持租约，消费共享帧、释放页前必须收完；完整256×768两预算已测，结果见worklog/2026-11-15.md。之前GPU排序去重整轮收益不足10%且小预算回退，已撤回 |
| P12 | 原collector/worker/frame增加setup、循环、drain、finalize与root/step/completion计时，修正anchor计数 | 只增加计时/统计；父进程与owner重叠，不跨进程加总。新字段用于解释总墙钟 |
| P13 | `accumulating_ppo.py:collect_rollouts`的full480默认配置使用原upper队列10ms收集窗口，原upper_batch=64不变，显式配置继续优先 | 这是调度合批实验，不是假称恢复旧参数（原实际full480为0ms）；不改每环境RNG、预算/目标/更新，须记录窗口及完整行数，并通过每轮10%耗时门槛 |

| P14 | `rollout_inference.py:_DeckConsumers.anchor_values`和历史`sim/sts_lightspeed/bindings/upper-value.cpp`：旧UpperValueRuntime接批量数组输入，旧NeuralSearchBatch池在大批次用8线程打包和只读查键；串行保持输入顺序更新原LRU | 替换我增加的新模型锚点Python逐行查键/输出组装；完整字节/shape/dtype、限额、淘汰、权重失效、原GPU小网保留；旧扩展仍兼容原缓存。40项及后续4项已过；完整256×768两预算通过，结果见worklog/2026-11-15.md |

| P15 | 原worker每64命令循环内，将该批尾原GPU tick前移到第一份upper请求的CPU准备前；finally收状态，批尾仍走原reap且不另加正常tick | 复用P11延后收取及原独立模型图；CPU/CUDA真实PPO与恢复已验证发生交错。属于P14同阶段消除串行等待，完整两预算均超过本阶段10%；计时upper包含重叠推进，不再把它当纯上层模型时间 |

另行审查的模拟器修改：`Actions.h/Actions.cpp/BattleContext.cpp`将实际开战回调和OnAfterCardUsed改成带类型的小对象，执行仍调用原动作函数；`ActionQueue.h`只增加队列只读访问。原state_codec/CPU桥接增加这类对象的序列化/恢复，并在原ArenaProxy/BattleArena/GameMirror边界允许CARD_SELECT。未知回调仍拒绝导入，不能宣称任意函数均已支持；固定PPO遇到未知回调明确报错，不能静默用旧terminal-HP回退产生错误标签。正式246初态186个战斗快照已全部可导入，完整片段中的覆盖仍需实跑。

PPO阶段未改原页租约、根账本、命令队列、整步GPU图、帧容量、共享QKV空间、内部强制跳过和权重失效检查。以上声明须与逐文件diff及测试结果一起审查，不替代速度证明。

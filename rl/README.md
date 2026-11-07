# 学习运行时

数据流：`env` 产生观测/合法动作 → `policies` 前推 → `search` 可选搜索 → `algorithms` 收集与更新 → `training` 编排；`evaluation` 消费 checkpoint，`apps` 负责展示。

| 包 | 职责 |
|---|---|
| algorithms | PPO/分层回报/NOSL learner 与 replay |
| policies | tokenizer、共享编码器、combat/overworld actor-critic、checkpoint 观测适配 |
| search | Python NOSL、native adapter 消费者、随机流和战斗价值搜索 |
| training | CLI、向量环境、多 rank 通信、reward wrapper |
| tasks | combat/whole-run 环境与模型组合 |
| evaluation | 整局/战斗/搜索/规则评估协议 |
| rules | 规则表示、teacher 数据、模型及其专用训练 |
| benchmarks | 固定预算训练及资源采样（旧低层探针在 tools/benchmarks） |

新代码使用 canonical 包路径。`_compat/` 仅保留历史 checkpoint 所需的 20 个库别名；旧 CLI 包装已删除。不会导入两套类，不在这里实现新功能。映射/退役说明见 ../tools/validation/migrations.json。
训练/评估参数放 ../configs；启动环境放 ../experiments/launchers；输出放 ../runs/local。不要在 rl 根目录增加实验脚本或测试。

训练异常在关闭 resident 环境前，将现场 JSON 与单槽环境 journal 写入
`training_state_dir/failure-scenes/rankN/`。有明确故障槽时只保存该槽；否则最多保存
4个在途槽，并列出全部在途索引。native 连续采样的截断在自动 reset 前留档，
每进程最多4次；正常步骤不导出完整现场。`manifest.json`记录各项导出成功或失败，
记录失败不掩盖原异常。这些材料用于直接读敌人/牌堆或短 CPU journal 回放，
不是 learner 完整恢复包，不能拿来恢复训练或将删失局计为死亡。

### 小型下层 playout policy

`policies/small_playout.py` / `algorithms/small_playout.py` 提供独立下层闭环；
`search/small_playout.py` 调用 C++ SmallPlayout，评估接入 `SmallPlayoutStrategy`
并可用 `PhaseStrategy` 与上层组合（环境必须显式使用`native-v1`完整动作）。当前支持固定战斗夹具的真实采样、搜索访问分布
交叉熵更新、逐轮报告和保存恢复，并通过 `small-playout` 战斗策略接入整局上层 PPO。
CPU/CUDA生命周期测试覆盖两轮更新、保存及恢复后两轮；四机生产验收记录在worklog。

独立完整局采集只使用上层战略目标与small-playout搜索监督，不消费旧NOSL的
回合边界/战斗终局value观测。该采集作用域省去这些辅助观测的预览、编码和复制，
保留普通Gym观测及终局target/outcome；退出或异常时恢复默认行为。普通NOSL和
outer-return路径继续提供原有value观测，此优化没有CLI或checkpoint开关。

受支持的原生向量环境包装栈会把普通上层后继观测直接编码到已有完整行缓冲。
公开Gym观测、向量返回值和终局观测仍独立持有数据；TimeLimit截断在自动重置前
保存独立终局观测。未知用户包装器保留普通路径，异常也会释放内部写入作用域。

独立采集中的CUDA统一上层在最多64行时复用已有CUDA graph传输与缓存，按8行分桶，
最多保留8个图。静态推理保留全部实体槽及原attention mask；训练仍按原方式压缩
masked PAD。图只属于一次collection，正常结束或异常均释放，不进入checkpoint；
每轮重新捕获当前权重。CPU、旧上层和更大batch沿普通前推。采样随机流、FP32精度、
mask/log-prob计算及学习目标保持，图计数和耗时进入既有upper_graph指标。

`sts1-small-playout-v1` 是纯 policy，无 value 头：32→64 ReLU→32 状态表示，
24→32 动作投影与 24→1 线性项，加每动作 key 的32维向量和bias。
StS1共785个key（结束/跳过/空选择3个、每卡基础/升级2个、药水），总30889参数，
FP32为123556字节。多选动作按所选卡的key向量/bias取均值，其他卡牌特征求和。
原生权重顺序为fc1 W/b、fc2 W/b、key embedding/bias、action projection/linear；
Torch与原生有不同浮点归约顺序，以2e-5绝对/相对容差验证，不声称跨平台逐位相同。

small-playout 的候选列表使用动态 native handle，不受旧策略96个PICK槽限制。
超过96个候选时仍按原规则仅合并效果等价项，并保留全部不同选择及执行索引；
原来可编码的局面、特征维度、权重和搜索预算不变。固定槽策略/packed observation
仍明确拒绝超过96种不同选项，不能将动态搜索支持误称为旧Gym动作ABI扩容。
对应回归为`test_dynamic_playout_preserves_choices_beyond_fixed_policy_slots`。

paired-token 独立采集在原生构建提供 `small_playout_world_batch` 时，使用
`PairedWorldPlayoutStream`：一个后台提交线程和独立原生线程池推进共享世界线，
每批至多64个根，世界内保留原动作顺序，按原世界顺序归约回报。主线程继续负责
完整根模型评分、环境推进和样本接收；活动槽不可重复提交，结束或异常时先等待
所有在途搜索，再释放线程池。每轮重新取得当前轻量头快照，不把工作线程或队列
存入 checkpoint。预算、合法动作、根部标签和符文圆顶旁路保持原契约。
当前原生接口支持至多480个根、每根1–1000次模拟、深度1–200且不带独立prior；
其他模型和不含该接口的旧构建继续使用原逐根异步路径。冻结评估的普通
`PairedTokenBatch.decisions` 接口保持兼容。回归见`test_paired_world_stream.py`。

32维状态含玩家HP/maxHP、格挡、能量、回合、药水数、四牌堆计数、力量/敏捷、
虚弱/易伤/脆弱/人工制品、五个固定敌槽的HP/格挡/当前伤害意图、选牌状态。
24维动作含五类动作、费用/升级/卡类型/成长值/免费/消耗/保留/虚无、选择区域与任务、
多选数、目标HP/maxHP/格挡/力量/易伤/虚弱/原槽位、待完成选择数量。
它是有损的小网投影，未编码全部能力/牌堆身份/遗物；合法动作、实体位置和树key仍用
完整模拟器契约，不用投影合并树状态。draw/discard/exhaust/generated/hand分别映射，
非目标动作不虚构敌人目标。Frozen Eye 保留完整可见牌序；Warcry/Headbutt 等已知牌顶
随真实 transition 跟踪。Runic Dome 特征隐藏意图，整局采用纯 policy；尚未实现其意图信念搜索。

在线随机起步时所有房间UCT；显式加载备用pair时普通UCT、精英/Boss独立先验PUCT（mix=.5，temperature=1）。叶子由小网执行policy playout。
当前搜索目标版本为`terminal-hp-boss-win-turn128-v1`。根选最高保守回报值，精确平局按显式nonce均匀选择，不按动作类别或访问数覆盖。
每个叶后最多200个完整原子动作，树深另限1024。真实胜/负/玩家逃跑与两类截断独立计数。
普通/精英只使用真实战斗终局剩余HP，搜索内除以同根固定初始maxHP作尺度；Boss仅胜1、其他真实终局0。
不附加药水、金币、回合、敌方HP等奖惩，普通逃跑亦按剩余HP。未结束模拟没有终局标签，trace的value为None；
搜索回传sum(已观察的非负终局回报)/尝试数，终局观测次数与删失计数分列。
这是保守下界统计，可能低估较长/难以模拟完的分支，不是已完成样本均值或无偏胜率，更不能把删失当作真实死亡。
不存在876前向或学习的critic补尾。根与chance重放复用NOSL未来随机性重采样；
所有动作使用同一渐进扩展ceil(sqrt(visits+1))，无零费/格挡/卡牌类别优先规则。探索系数按simulation预算
<500 / <1000 / 其余分别为.8 / 1.41 / 2.5。完整状态和动作键采用逐字段强混合，
避免已复现的Codex候选哈希碰撞；旧NOSL的历史键保持兼容，其碰撞修复另列TODO。
当前整体实验上下层均随机起步，不预训练、不加载独立先验。每轮实际经过的多动作战斗根
做在线CE：0.8×实际推荐动作one-hot + 0.2×visits分布；推荐动作按上述搜索统计选择，不能替换为argmax visits。8 epoch，AdamW 3e-3、weight decay 1e-4，按用户要求使用旧下层每rank有效batch102400（全局409600），GPU分块由`--small-playout-micro-batch-size`显式指定，FP32、无梯度裁剪。该有效batch已主动偏离参考CE的4096；epoch仍为8，不能宣称更新次数不变。
新搜索版本独立写入模型、manifest和policy pair；旧训练checkpoint/旧pair拒绝静默切换目标，须使用其归档源码/native。仅num_timesteps=0原随机初始模型可以显式开始新实验。特征/参数布局保持不变，不代表搜索标签统计等价。
在线小网的两项batch覆盖参数仅改变下层分批，保留已有Adam/权重/环境恢复；省略时沿用checkpoint。日志以`small_playout_train/batch_size_per_rank`、`batch_size_global`、`micro_batch_size`及`upper/effective_batch_size_global`、`upper/micro_batch_size`区分含义。小网路径不再上报已禁用普通PPO的`train/effective_batch_size*`占位值。
按全局有效样本量平均梯度；空rank/末尾不等长仍参加相同同步。没有旧replay，单动作根和Dome无搜索标签。
下层权重、Adam、累计更新数与整局环境/RNG一起保存恢复，下一轮重建原生权重。

独立游戏调度目前仅供 `tools.analysis.probe_upper_learning` 的有限冻结采集诊断：
在 `unified-upper-rollout-scale-v1` 协议添加 `independent_scheduler` 对象，字段为
`mode`（`barrier`/`independent`）、`threads`、`upper_batch`、`upper_wait_seconds`、`seed`。
两模式复用相同游戏独立随机流、small-playout 和上层宏观回报构造；前者等待整批，
后者完成即推进，只在上层需要决策时组批。每局政策/搜索随机流分离，版本为
`independent-games-pcg64-v1`；这是显式的新采样合同，不承诺与旧全局Torch随机序列相同。
支持FP32、resident native、joint small-playout、完整自然局、aux/replay关闭。
不更新或保存生产模型；两模式均不生成旧大网未用于学习的战斗value诊断行。
正式入口的 `--independent-rollout-config config.json` 接受同样字段，但只允许
`mode=independent`、完整自然局error预算且reasoning关闭。旧checkpoint默认不切换；
已训练源必须用 `--fork-state` 在完整自然局边界显式迁移。配置随model/manifest保存，
普通resume省略JSON文件即可保留，不能在普通resume中静默改变随机流/调度。
独立训练可在同一调度的完整自然局边界显式增加正整数 `max_combat_turns`，随后普通恢复沿用。
例如设为 50 时，真实战斗进入第 51 回合即冻结该槽，并从上下层更新中剔除该游戏的全部样本。
本轮不补局，不计胜负；按种子、槽位和 `combat-turn-limit:51>50` 记录异常。此限制独立于搜索模拟和整局动作预算。
每轮按下一批唯一game seed生成各局随机流，checkpoint只支持完整cohort边界；
最终随机流保存用于审计，不声称支持线程池/半轮恢复。
`speed/rollout_env_steps_per_second` 为全rank实际环境操作总数除以最慢rank的collection秒数，
包含选择前缀；原 `env_steps_per_second` 仍包含更新/同步/保存的整轮时间。
独立进度使用 `transition_batches`/`env_steps`/`simulator_transitions`，不冒充同步向量步。
新采样方式须另通过同硬件完整collection与两轮更新/保存/普通恢复资格后才能用于现役任务。
目标熵、CE减目标熵的KL、teacher top1、uniform CE与策略熵一同上报，避免把变化的标签难度误当学习进步。上层配置由实验args/checkpoint显式决定；当前完整游戏实验见`configs/train/small_playout_terminal_full960_distributed.args`，两遍192/rank小批、FP32和λ.97。历史全批设置不得当作当前入口默认值。
用户授权的单回合保护使用引擎cardsPlayedThisTurn（含自动/重复打牌）：达到128后等待已开始的效果/选牌结算，只允许合法结束回合。实际搜索、模拟和Dome纯策略共用同一候选限制；这是预算约束，可能截短合法超长连招，不是特定卡牌偏好。
完整游戏模式每rank启动240局，完成槽保持、不补局。新实验动作上限2424来自旧27/28共115200自然结束局观察最大1212的两倍，墙钟1800秒。`--episode-rollout-censored exclude`仅对这些预算超限剔除整局所有上下层样本并继续；不吞掉OOM、无效状态或一般异常。各槽种子/操作数/原因记录，有限现场快照单列；全删失批次零Adam、仍保存并推进下一批种子。自然结束+剔除=启动数，胜率分母仅自然结束，零分母省略，不能将删失当死亡或声称剔除无选择偏差。普通恢复必须保留该合同；历史error模式仍失败停止。
这与参考的先离线CE、再REINFORCE、最终固定小网不同，是用户要求的无预训练在线实验；当前并未声称在线启用了后续REINFORCE。终局回报只用于搜索，
没有给小网增加value监督。备用流程支持CE后独立同起点K条真实playout的leave-one-out REINFORCE，
包含截断的组不提供终局监督；备用独立先验仅做CE，不随REINFORCE更新。
StS1暂无参考项目的人类BC数据，备用先验使用模拟器搜索监督；未实现参考的证明求解和废牌清理保护，
状态/动作特征也使用StS1映射，不能把这些差异称为逐项等价移植。

备用预训练复用 `python -m rl.training.trainer --task wholerun --prepare-small-playout protocol.json`：
`collect` 协议指定可信环境journal bank、seed/rank、threads、samples、batch_roots、seconds、output；
`fit` 协议指定同bank的shards、ce_epochs、batch_size、rf_updates/rf_starts/rf_k、entropy_coef。
每阶段记录进度、真实transition计数和ResourceMonitor；80%银行起点训练，20%固定留出。
这是有限bootstrap，不是下层收敛证明；旧权重从不读取。
默认 `@configs/train/small_playout_fresh.args --steps 0 --out ...` 创建随机上下层初始checkpoint，
其中 `--small-playout-online` 接受OnlinePlayoutConfig的JSON字段。显式备用 `--small-playout-pair pair.pt`
与该选项互斥，须用不含online选项的配置，不能静默覆盖随机初始化。
兼容容器保留一个32宽单层冻结旧combat模块，整局回归禁止调用它；
实际搜索仅执行30889参数playout。上层256宽4层及card encoder均随机可训练。
新fresh配置启用`--upper-public-context`（`upper-public-context-v1`）：当前幕公开地图在
奖励、选牌、事件、商店、营火、宝箱、Boss遗物和地图界面均进入编码、稀疏传输和Map DAG。
全局HP/牌组/遗物/药水/金币/Boss信息仍共享；事件触发选牌时保留来源事件。
Falling、N'loth、We Meet Again、Note For Yourself、Designer已公开的候选卡/遗物/药水、
金币/HP成本和数量进入对应事件按钮token。未提供动态细节的按钮known位为0，
不能把隐藏结果预先填入。该选项要求global-map及event-option观测，暂不与冻结teacher组合。
旧checkpoint无标记时保持原网络和phase行为；新checkpoint保存额外事件模块并在恢复/评估中
按观测空间恢复合同。当前仅支持模拟器训练/评估，真实游戏bridge明确拒绝新合同，避免静默补零。
分布式使用 `configs/train/small_playout_distributed.args` 和已有 `experiments/launchers/launch_nosl_replay.sh`，
保持4×384×512，PPO两epoch、entropy .0125；自然完整游戏跨rank累加，包含胜败、排除超时/异常。
40万局在轮末保存停止，计数随各rank模型恢复；另有10000轮硬预算。截断/模拟量按房间层级记录，
`search_seconds`是各搜索调用耗时之和，不能当作并行墙钟；端到端以rollout_sec及iteration_wall_sec为准。

复用有超时的基准入口，例如（先设置明确的原生构建路径）：

```text
python -m rl.benchmarks.train --workload small-playout --output runs/local/small-playout/cpu.json --device cpu --threads 1 --native-search-threads 1 --n-envs 1 --n-steps 512 --batch-size 32 --updates 2 --warmup-updates 0 --simulations 250 --timeout 180
```

Boss测试显式传 `--encounter Hexaghost --simulations 1000`；`--battle-deck` 后可列出
原生卡牌ID，默认战士起始牌组。输出JSON、逐轮JSONL、最后checkpoint；第一轮保存后
实际恢复模型/Adam/未完成战斗，第二轮使用恢复对象继续。恢复仅接受相同native SHA，
旧大网checkpoint/replay不与此合同混用。C++内执行每步策略，Torch仅更新（CPU/CUDA皆可）；
分别报告实际执行transition、模拟次数、chance重放、真实动作、采样/更新/保存/恢复和资源。
测试为 `tests/integration/test_small_playout.py`，新模块缺失直接失败，不静默跳过。

### 冻结上层 teacher

搜索出口的内部推理只计算冻结上层的编码器和value头，并只缓存一个value；
普通整局动作选择继续计算完整actor/critic。两者共用同一value readout，FP32约定不变。
CUDA搜索的value-only transport可使用`UpperValueRuntime`原生流水线：阶段分组、输入
校验/裁剪/padding、逐字节LRU及miss组批/回填一次完成。每次调用仍遍历当前模块注册表
校验冻结状态；实际GPU批次的顺序、padding宽度与原Python路径一致。模块不拥有模型参数，
随collection清理，不写入checkpoint；普通CPU及旧扩展继续既有Python路径。
该差异只影响计算/传输内容，不改变搜索预算、归一化目标或上层训练状态。

COMBAT06（用户的1号实验）的 `combat-outcome-means-v1` 保留旧战斗终点HP标签和
Boss权重，输出胜率与获胜时HP效用均值两个数，不含分位数头。loss只有胜负BCE与
胜利样本的HP MSE，两者均按完整逻辑batch归一化；死亡不给条件HP分支零标签。
无合成utility MSE或transition辅助。搜索普通战斗用 `p_win * hp_mean`，Boss用
`p_win * (0.9 + 0.1 * hp_mean)`；HP效用沿旧定义为结算前 `min(HP,200)/200`。
上层876永久冻结，全下层含actor/context随机初始化共同训练；独立profile为
`configs/train/combat_two_means_876_4x5090.json`，复用已有准备/分布式入口。
新head/loss身份进入checkpoint及replay合同，不能沿用分位数或有限曲线的replay。
当前仅支持冻结上层、resident native、A0战士Act1–3实验；2号单HP均值尚未实施。

Boss消融复用上述两头与训练入口，`OuterReturnPolicy.boss_ablation` 可取
`full-or-zero` 或 `real-hp`。两者在全部Boss房间都用纯胜率搜索，搜索遇到真实胜负
也只回传1/0；普通战斗不变。前者Boss血量标签为胜利时实际终局maxHP/200或失败0，
胜败均参加HP MSE；后者保留原真实血量标签和胜利掩码。故前者Boss血量头不再表示
“获胜条件下的血量”，历史`hp_given_win`指标名仅为兼容，解释以身份合同为准。
标签不裁切；原sigmoid头仍限于[0,1]，maxHP超过200时存在饱和限制。
协议在`configs/train/combat_boss_win_ablation_4x5090.json`。本实验使用
`configs/train/combat_boss_full_or_zero_876_4x5090.json` 经已有
`--prepare-combat-outer` 构造随机内层、step0、空Adam和回放；训练不传resume/fork。
第二组也用相同seed重新初始化，仅改变Boss HP监督模式。已停止的错误续训分支
只作为审计记录保存，不作为这两组实验的初始化或对照数据。

COMBAT05 的 `combat-finite-upper-curves-v1` 是独立目标实验：下层沿 COMBAT01 的随机
scalar actor/critic/context 架构，整个下层共同训练；876 上层仍永久冻结。战斗真实结算
之后，按原始 encounter、Act、胜利/玩家逃跑出口、持有药水数和 HP/maxHP 查询有限
冻结曲线，替代实时上层 continuation V；死亡/最终通关仍使用真实终局回报，截断不标注。
搜索和实际轨迹共用 `env.core.finite_curves.FiniteCurveBank`，保持奖励前缀和单世界线。
曲线不查询实时牌组、遗物、金币、奖励内容或药水种类；这些变量对真实战斗及结算的影响
仍由模拟器执行。小偷逃走与玩家烟雾逃跑仍是不同出口；本消融不评价失去金币的机会成本。
曲线来自固定参考状态的 FP32 teacher，按参考样本取均值，线性插值，不强制单调。
参考稀少的事件组只允许 bank 中显式列出的别名，缺少映射时报错，不能隐式退回 teacher。
bank 内容嵌入 checkpoint constructor，其 SHA/teacher 身份进入 replay 和完整恢复合同；
不能混入 COMBAT01/04 replay。准备材料额外需 `curve-bank.json`，profile 固定其文件 SHA。
抽取证据与生产验收见 TODO COMBAT05，支持范围为 resident native、A0 战士、Act1–3。

`policies/outer_return.py` 提供 COMBAT01 策略组件：校验 manifest/逐文件 SHA 后加载
提取材料，876提供永久冻结上层；正式实验的整个下层随机初始化并共同训练，
使用FP32无界标量critic，不沿用旧HP分布/辅助头。`combat-public-run-context-v2`
为下层补充 act/floor/下一Boss/当前与最大HP/永久牌组ID、升级与misc/角色/进阶/keys，
并保留GRID已选牌的进度。显式876下层初始化仅为兼容对照，不能称从头训练。
旧boss标记仅保留为旧encoder输入，不能参与新value或搜索目标。

`combat-outer-return-remaining-max-v1` 是独立的 A0 Act1–3 实验目标，保持 qnorm
关闭。源状态的剩余奖励上界为 `M = .002*(51-floor) + .1*(act==1)
+ .2*(act<=2) + 1`。出口按 `(真实已结算reward + 后续raw value)/M(source)`
回传；冻结876上层继续输出raw value，若出口直接进入下一战斗，下层预测先乘
`M(next)`恢复raw单位。执行样本保存各自源状态的M，跨collection pending恢复也
保留它。自然终局只取真实reward，不除终局的零上界。预测不裁剪到[0,1]，
`outer_return/target_below_zero`、`target_above_one`记录越界；旧replay不可混用。

完整采样的外层开销由搜索侧 `outer_settlement_seconds`、`upper_prepare_seconds`、
`upper_cache_*` 和 `upper_graph_*`，及执行标签侧
`outer_return/collector_upper_prepare_seconds`、`collector_upper_value_seconds`、
`collector_upper_rows/batches`分别报告。teacher前推计时包含CPU取回结果的等待，
是调用方观测耗时，不是单独GPU kernel时间；结算可与lower GPU前推重叠，
不可将嵌套/重叠项直接相加当作整轮独占成本。
搜索专用 `outer_value_seconds/rows/batches` 统计战后teacher调用的完整耗时与
工作量（含准备、缓存查询和推理），排除真实上层动作选择；`outer_exit_rows`
另计全部搜索出口。它与 `upper_*` 的细分计时嵌套，不能重复相加。

native `outer_return=true` 要求 fixed_worldline，并拒绝旧HP/Boss效用。所有战斗
出口返回拥有独立存储的typed终局BC，由resident NativeSearchBatch在同一采样GC上
结算，按upper/combat/terminal分流；内部叶用新lower，出口用冻结teacher或自然终局
真实reward。resident普通叶子直接选择pending状态行，由共用native codec填充lower
graph输入区，保留root归属、实时HP/钱包、GRID进度与永久成长；出口仍按完整GC结算。
resident collector把本次已编码根观测交给batch，仅复用CONTEXT_KEYS；按完整env索引
取行并核对shape/dtype，在搜索开始时复制成自有数组，避免再次序列化根的全战斗状态。
其他调用方省略root_observations时仍从环境编码；不得提供上一个状态的观测。
混合batch在lower缓存关闭时先发起普通叶子的GPU传输/前推，CPU同时结算出口，
再准备upper输入并查询缓存；两模型使用各自独立的输入区和图池。复用lower计算连战
续值、返回搜索或传播异常前等待传输完成并复制输出。每个图实例最多一个未完成调用，
重复提交报错，clear或异常也先完成传输，不能在设备使用中覆盖或回收pinned输入。该重叠不跨
search advance、learner更新或checkpoint边界；计时中的output_seconds是实际等待，
可能小于设备总工作时间，outer_overlap_batches/rows记录实际参与重叠的工作量。
非resident传输明确拒绝，不降级到HP。GPU完整轮验证与learner/启动接入
状态以TODO COMBAT01为准，策略组件可加载不等于已完成训练验收。

`policies.frozen_upper.FrozenUpperTeacher` 从已核验的 entity-joint core 复制上层
tokenizer、actor、critic 和全部依赖；卡表 storage 与原下层分离。来源必须记录
checkpoint SHA256。父模块 `train()` 不改变 teacher 的 eval/无梯度状态，外层
autocast 不改变其 FP32 前推；移动设备可以，转换权重精度明确报错。调用方只给
下层建立 optimizer，在 collection、更新和保存边界调用 `assert_integrity()`，
检查参数、buffer 和训练标志。恢复时必须复制 tensor，拒绝来源/合同/权重变化。

teacher 按旧 observation space 投影输入，`decide()` 复用 checkpoint-v7-act3
动作适配和 masked argmax，只接受上层决策；真实死亡、截断和连战必须由调用方
区分。该模块本身不提供新的战斗回报、搜索结算或训练入口；正式训练入口和已验证配置
见 `../experiments/launchers/README.md`，当前训练与性能验收以 TODO COMBAT01/02 为准。
搜索内部按phase先选teacher真正消费的字段再stack，逐字段核对旧checkpoint形状，
复用同一padding裁剪和网络；公开forward/prepare_numpy仍要求完整checkpoint schema。
upper_prepare_seconds包含这一内部字段堆叠。upper前推的调用方计时可能包含同一
CUDA流上尚未完成的lower工作，不能把upper_graph_output_seconds解释成独占teacher
耗时。完整外层增量须用相同输出轨迹的端到端对照衡量。

## Entity combat 实验架构

### 完整动作等价组 equivalent-plays-v1

战斗选牌的 `overflow-equivalent-v1` 保留96个候选槽以及旧checkpoint/replay形状。
原始候选不超过96时，内容和顺序逐项保持；超过时使用原生卡实例等价关系合并副本，
每个不同效果保留第一个真实执行索引，不能按卡名合并费用、强化或特殊数据不同的牌。
这是对原先拒绝的长战斗状态（例如排斥机晕眩堆积后发掘）的兼容扩展；
超过96个不等价候选仍明确报错，后续须单独迁移动作空间，禁止静默截断。
canonical、packed、原生搜索和执行共用 `combatChoices`；已有合法replay无需迁移。

保持原198维autoregressive-v2（以及旧1392维joint-v2）权重和观测接口。
普通出牌按卡牌ID、升级、specialData、基础/当前费用、freeToPlayOnce、retain及目标分组；
Ritual Dagger/Genetic Algorithm的永久实例分开，药水、结束回合和选牌前缀不合并。
环境与C++搜索使用同一个原生分组入口，以最早合法槽位为代表执行。
PolicyStrategy（Eval/Watch）对组内概率求和后argmax或采样；底层SB3 predict仍为原始槽位接口，
不能替代带真实环境状态的决策适配器。真实CommunicationMod尚缺完整实例字段，
拒绝带此新合同的模型，避免静默采用不同动作语义；旧模型入口保持兼容。

原生搜索每个节点只保留一条组边，合并先验、访问和Q，每组一次Gumbel。
组边下相同公开后继共用子节点，忽略隐藏RNG、牌堆顺序和普通出牌时的手牌排列；
自回归选牌进行中仍保留前缀和手牌位置，永久成长卡实例仍参与标识。
不同可见随机结果仍为不同chance子节点；不跨祖先/不同动作序列共享节点，本阶段不是完整DAG。
此合同是未来分布的抽象，不保证不同alias在同一固定RNG下产生相同轨迹。

NOSL监督为 `-sum_G pi_G log(sum_{a in G} p_a)`，熵/KL/top1/合法动作数按组计算。
198维目标中的alias质量如何分配不改变该损失。replay v4保存原始根分组，
仅持久化60个出牌位置映射（每行120字节），取batch时还原；不从不完整旧观测猜分组。
旧neural replay必须重新采集，legacy未合并动作可按identity迁移；保留旧checkpoint权重可加载预测。
新训练模型记录nosl_action_group_contract，观测、HP outcome、PPO目标和模型形状未变。

TRAIN32蓝线复现实验用`configs/train/entity_combat_blue_repro_runtime.args`，
`--no-nosl-is-az-merge-equivalent-actions`恢复POL01之前的独立槽位搜索：
不合并重复出牌边，AR后继始终保留手牌顺序与前缀，恢复历史计数字段的状态判定。
保留原NOSL隐藏信息处理，不新增跨路径DAG。模型标记`individual-actions`，
采样/replay保存identity映射，组损失在此映射下等于普通槽位交叉熵，
PolicyStrategy按原槽位决策；native search配置随模型及恢复manifest持久化。
默认仍保持旧POL01模型的合并合同，旧模型载入不静默变更语义。
新实验保留线程批处理、FP32归约、候选溢出修复和STORAGE05，不回退已确认正确性修复。
TRAIN32进一步使用OBS18/native layout v3，补齐腐化/壁垒等布尔状态并包含Pen Nib。
按用户要求不迁移旧checkpoint/replay；旧实验只在保留的旧构建运行。

模拟器SIM02/03/04修复不改变198维动作宽度：Dark Shackles临时负力量及Artifact，
Panache每5次出牌循环并在回合开始重置，Forethought+（预谋+）支持任意子集含空集。
预谋+用现有AR升序槽位前缀和STOP，完整子集按升序槽位依次移到牌底；
各牌基础费用>0才赋予下一次免费标记，非升级版仍强制选一张。

`--arch entity-combat` 使用 `policies/entity_combat.py`：卡牌ID与效果程序、
实体Transformer（默认256宽/4层/8头）、带计数的牌堆query汇总、
来源条件目标pointer及自回归多选。没有整局神经网络或历史记忆；
全局规则负责战斗外决策，遇到未覆盖页面报错，不用随机上层预测回退。
卡牌语言字段在新网络内按ID/升级统计聚合，不能利用隐藏抽牌顺序。
现有观测对draw/discard提供升级统计，exhaust只有基础ID计数。

eval且关闭梯度时，卡牌ID/升级对应的语义向量按权重版本与autocast精度缓存；
训练时正常重算并反传。optimizer或load_state_dict改变权重后，下一次推理原址刷新，
CUDA graph重放前也检查刷新；图中不再重复执行卡牌程序卷积。
该缓存不进入checkpoint。自定义权重修改须保留PyTorch版本计数，勿通过`.data`绕过；
替换参数、迁移设备或改变dtype后须清除既有leaf graph（图持有参数地址）。

新训练使用`autoregressive-v2`，输出宽198：结束回合1、
卡牌×目标60、药水×目标30、单选96，以及逐张选牌10+STOP 1。
按手牌槽位升序选择，当前前缀通过select_hand_mask/pick_remaining可见，
每个集合仅一条路径。原生搜索和网络每次最多展开11个选择，不枚举完整集合；
前缀只改选择状态，STOP才执行模拟器集合操作，搜索depth只计真实transition。
新多选支持CUDA graph；旧`joint-v2`保留1392维枚举路径，仅用于旧checkpoint兼容。
新模式不构造旧181维source或输出前缀；搜索IPC/replay均为198维。
整局环境在战斗外按phase复用规则动作ID，规则行不前推战斗模型。
379维autoregressive-v1为未投产的中间协议，不支持加载；旧joint-v2模型继续支持。
两种动作协议和replay不能混用，环境快照也核对协议；新协议须从零训练。
UI适配器内部完成前缀后，按降序向真实游戏发送选牌并等待逐条确认。
当前原生多选支持GAMBLE/EXHAUST_MANY的至多N（允许空集），不宣称支持未实现的必须N任务。
固定n_envs×n_steps计策略决策，含选牌子步；日志selection_prefix_decisions与
simulator_transitions分列，比较旧吞吐时需区分两者。前缀没有实际动作后果aux监督。

支持组合：whole-run环境、autoregressive-v2或历史joint-v2动作、C++ NOSL、inner-only、rules上层，
CPU或CUDA/BF16，普通推理或既有CUDA graph叶缓存。PPO标量critic、
旧动作及旧split checkpoint仍走原实现；新结构须从头训练，旧权重不能直接迁移。
恢复同架构checkpoint可用统一trainer的resume；含replay/环境的连续恢复
仍使用distributed_sb3的checkpoint bundle。

搜索效用保持 `combat-end-hp-v1`：失败0，获胜为结算回血前
`min(HP,200)/200`。网络输出获胜概率p与32个胜利条件HP效用分位点q；
搜索叶值为 `p * mean(q)`，终局仍使用模拟器的精确效用。
树内继续备份这一标量，不增加风险惩罚或胜率优先级。
分位点是被截断后的效用，不能先求原始血量均值再截断。

训练使用原搜索策略软标签，加BCE胜负、胜利样本pinball与
0.25期望效用MSE。当前entity默认及四机runtime关闭动作后果辅助头：
不计算后果标签、前推或反传；已保存的旧头参数留在checkpoint中以便恢复，
旧replay中可能存在的后果标签被忽略。显式启用旧系数0.1的checkpoint仍可加载。
各项按完整逻辑batch平均，微批/多rank不重新
按各自胜利数归一化。明确胜负来自simulator outcome；
截断、未知或逃跑标签不进入该监督，失败没有伪造的“获胜血量”标签。
Replay版本3及 `combat-outcome-quantiles-v1` 拒绝旧标量回放混入。

固定预算示例（首次实验超参数，不是已验证的最佳配方）：

```text
python -m rl.training.trainer --task wholerun @configs/train/entity_combat.args --steps 24576 --seed 0 --device cuda --amp-mode bf16 --out runs/local/entity-combat
```

配置保持96 env、128步；按TRAIN24基准使用逻辑训练batch98304、微批12288、每次采样40次联合policy/value更新，
回放524288，旧样本采样上限0.9，LR5e-5，CUDA graph batch对齐32。`--steps 0 --vec dummy --device cpu` 可只创建初始checkpoint。
默认`constant`；可选实验`--nosl-is-az-updates 4 --nosl-is-az-update-schedule value-warmup` 在前5轮额外增加36次value更新，
第6–20轮按余弦减到0；第10/15/20轮总更新数为31/13/4，之后固定4。
额外更新先于联合更新，训练共享表示与胜负/HP分布；策略专用头的参数及Adam状态
在这些更新中保持不动，但共享表示变化仍会影响策略输出。目标来自真实自博弈终局，
并非另行采集的均匀随机策略数据。该曲线是实验设计，尚未证明优于固定U4/U40。
调度按累计global steps除以四机整轮动作量计算（生产每轮49152），保存恢复不重启warmup。
日志区分scheduled_updates、value_only_updates、joint_updates；零有效标签时实际更新可以为0。
阶跃实验用 `--nosl-is-az-updates 40 --nosl-is-az-update-schedule '{"mode":"step","hold_rollouts":100,"final_updates":16}'`：
前100轮全部为40次联合更新，第101轮起全部为16次联合更新，没有额外value-only更新。
JSON对象也可作为args文件中独立一行。`step`同样仅支持entity-combat C++ inner-only NOSL；
调度随checkpoint保存，用绝对采样步数恢复，暂停/恢复不会重新等待100轮。TRAIN33的FP32运行配置使用此规则。
旧checkpoint和未指定该选项的入口默认constant；value-warmup仅支持entity-combat的C++ inner-only NOSL。
初始化与分布式运行共用 `configs/train/entity_combat_learning.args`；
`entity_combat_runtime.args` 用于既有四机launcher，目标50M，BF16。
该实验使用Full Gumbel128，关闭Q归一化、value scale=1，并启用`--nosl-is-az-fixed-worldline`：
每棵搜索树内固定未来RNG。该开关贯通初始化、采样、分布式恢复和checkpoint；旧模型缺字段时默认关闭。
`paper_full_gumbel` 保留论文路径的sequential halving、visit-weighted backup
和mixed-value完成法；从本版本起Q归一化和scale尊重显式配置，默认仍为true/0.1。
旧checkpoint中保存的canonical true/0.1不变；曾携带非默认Q字段但被旧strict
实现忽略的checkpoint会开始采用该字段，复现旧行为时须显式设true/0.1。
Python、原生扩展、dashboard必须同步升级，防止旧扩展静默覆盖新Q设置。
新监督不保证消除HP反转；训练后仍须固定局面/未来随机流评测。

### Boss 获胜优先效用

`configs/train/entity_combat_boss_utility.args` 创建新的 entity AR 模型，`entity_config.boss_win_weight=0.9` 随模型保存；对应四机 runtime preset 为 `entity_combat_boss_utility_runtime.args`。定义 `h=min(战斗结束且未结算回血的HP,200)/200`，普通战斗效用为 `P(win) E[h|win]`；Boss 为 `P(win) (0.9+0.1 E[h|win])`，观察到的输/赢分别为0与`0.9+0.1h`。这是有血量次级偏好的有限权重目标，不是严格字典序的“任何概率提升都优先”。胜率BCE和条件HP分位数监督仍使用真实win和原始h；效用MSE及其监控目标才应用同一变换，不能把变换后的值又当HP分位数标签。

OBS19新增公开的`combat_is_boss`，由房间身份生成，涵盖A20两场Boss与心脏，不依赖当前仍存活的怪物。它通过canonical、root、packed leaf全局字段、replay与模型进入训练，模型也接收此位区分目标。native compact物理排列仍为v3，但绑定报告OBS19且Python要求匹配。全新训练不迁移旧OBS18 checkpoint/replay；暂停实验保留其源码与扩展构建用于恢复。

### Boss 开场专项环境

`distributed_sb3 --boss-opening-dataset PATH` 复用现有 entity NOSL/outcome learner；仅支持无 reasoning、环境数为4的倍数，按2:1:1分配冠军/铜人/收藏家。`env/core/boss_opening.py` 的 `boss-opening-skip-route-v1` 数据集保存原seed、规则版本、完整路线语义动作及开场SHA，重放到真实第二幕Boss第一手。原生跳战前缀不产生训练样本；保留路线产生的HP、最大HP和药水，不添加半血起点、阶段奖励或动作标签。这是跳战合成资源分布，不能当自然到达分布。

新实验使用 `--fork-state ... --fork-reset-environments` 保留模型、Adam、兼容 replay/RNG，仅重置环境/pending fights。原replay按FIFO逐步替换；old fraction .9指当前轮之前的样本，不是冻结90%的源样本。普通resume必须恢复同一数据集SHA和actor分组；环境journal可恢复任意正常进行中的战斗，不构成中盘课程。缓存只共享只读GameContext及开场，演员/搜索各自clone BattleContext，胜利不结算到共享GameContext。日志 `episode_scope=boss-opening`、`boss/*` 是完整单场胜率；W&B不发布整局act/floor通关率。

评测仍用 `python -m rl.evaluation.eval MODEL --boss-opening-dataset PATH --episodes N ...`，N必须等于文件条目数，按条目逐一测试且每条只测一次；沿用solver/Gumbel/预算参数及有界runner。训练/验证按原seed分开，胜/负依据真实战斗结局，逃跑/截断单列。输入观测、动作和outcome监督版本未改变；新增的是episode采样契约，不可普通resume全局环境到Boss环境。

当前支持entity-combat + autoregressive-v2 + C++ NOSL（native线程或既有worker传输）、纯policy预测及`NativeSearchConfig.from_model`评测。原生终局backup与模型叶估值使用相同权重；合并不跨一次根搜索，不混合Boss与普通目标。与历史boss-transition回血目标、model-evaluated victory模式不可同时启用；旧Python CombatValueSearch显式拒绝该新模型，防止用旧HP终局混入新效用。默认权重0保留旧目标，普通战斗不改变收益函数；模型共享参数与新采样造成的训练差异属于本次目标实验。

### Whole-run 上下层联合训练

`python -m rl.training.trainer --task wholerun @configs/train/entity_joint.args` 沿用 WholeRunTask、原采样器和 NOSL 更新入口；四机入口使用 `entity_joint_runtime.args` 与新初始化的 `EntityJointPolicy` checkpoint。只支持 OBS19、autoregressive-v2、C++ inner-only NOSL、整局环境；多机使用 ssh-ring。这里 inner-only 仅指战斗 NOSL 的监督范围，上层由独立 PPO 学习，不冻结。不能将旧 entity-combat 模型直接当联合模型恢复。

下层保持 entity combat 网络、初始化次序、Boss.9 效用、动作/未来合并、G1/rawQ1/strict128、U40/BF16/96×128 及 replay 学习配置。上层复用既有 split-policy 的 OverworldEncoder、共享语义 tokenizer 类型、Map DAG 和各阶段动作头，宽256、4层、8头；与下层参数/Adam独立，避免上层梯度改变战斗表示。全新实验上下层都随机初始化。

上层默认 Adam LR1e-4，每轮收集到的已完成宏动作做最多4次完整批次 PPO 更新，micro256，gamma1、GAE.95、clip.2、entropy.02、value.5、梯度裁剪.5，近似KL超过.03停止本轮剩余更新。按全局实际样本数归一化优势与梯度，空样本rank仍参加同序集合通信。奖励为每推进一层+.002、通过一/二幕额外+.1/+.2、最终胜利+1；不使用HP、卡牌优劣或Boss战术标签。这是独立的战略目标，原整局episode_reward日志仍沿用环境原奖励。

分布式入口的 `--upper-micro-batch-size N` 可覆盖上层前后向分块大小，须为正整数且只支持 `entity-joint`。每个 epoch 仍累积本轮全部已完成的上层样本，按全局行数加权后执行一次梯度同步、裁剪和 Adam；有效 batch、epoch 数和学习率不随分块变化。覆盖值写入模型构造配置及恢复 manifest，后续普通 resume 即使省略该参数也保留该值。BF16 分块变化不保证逐位相同；先在同一 checkpoint/样本上检查数值、显存和完整更新耗时，再用于生产续训。

`--upper-gae-lambda L` 单独覆盖 `entity-joint` 上层 GAE，范围为有限的 `[0,1]`；省略时沿用 checkpoint 的 upper_config，保存和普通恢复保持该值。普通 `--gae-lambda` 不控制这一路上层目标。λ=1 仍在 collection 截断处使用 bootstrap；只有连续记录的下一状态估值相互一致时，内部 value 项才严格抵消，不能等同于完整无截断 Monte Carlo。此参数改变统计目标，应作为独立实验使用。

`--upper-epochs N` 单独覆盖上层每轮最大完整批次更新次数，正整数，省略沿用 checkpoint；保存后普通恢复保留覆盖值。每次更新仍使用全部有效上层样本，保留原 `1.5 * target_kl` 提前停止。它不改变下层 epoch 或采样长度。`upper/requested_updates` 与 `upper/updates` 分别报告上限及实际次数；逐 epoch 的 policy/value loss、KL、clip fraction 在该次 Adam **之前**计算，不能视为更新后的独立评估。修改更新次数是统计配置实验，不是等价性能优化。

`--upper-auxiliary-search public-v1` 在 joint NOSL 上层增加有限公开分支监督，`off` 关闭，省略沿用 checkpoint。行为动作及 old log_prob 仍由原策略采样；搜索只对已审查的确定性动作建树，未知随机转移留作 value 叶子。模型无关分支在 `env/search/upper_public.py`，冻结 FP32 teacher、条件 CE 和预算配置在 `algorithms/upper_auxiliary.py`；辅助反向加入原上层 Adam，不增加更新次数。当前支持奖励、商店、六类事件、休息及来源明确的删牌/升级选择，不等同于所有合法动作均可搜索。默认每 phase 四根、每树64 transition/深度3/两秒、每轮分支六秒；温度.02、Q跨度门槛.01、系数.02，均以原 reward 单位计算。`upper/aux_*` 报告实际覆盖、标签、预算与额外时间；空样本不制造标签。此为独立统计方法实验。`configs/train/small_playout_upper_aux.args` 是在现有分布式配置后应用的10更新实验覆盖文件，不能当作完整启动配置。

`entity-joint` 可通过 `--card-vector-checkpoint` 使用冻结的匿名规则卡向量。该模式关闭 card-language 观测，把 `[card id, base/upgraded]` 的完整固定表同时用于上下层；表注册为持久 buffer，不属于任何优化器参数，上下层仍各自训练独立的 `129 -> 256` 投影。输入可为原 standalone rule-encoder checkpoint，也可为 `frozen-card-vector-table-v1` 紧凑表；两者都严格校验371张全局卡表及基础/升级两版，不能用只有75张未升级战士牌的PCA展示数据替代。普通不传该参数的 entity-joint 初始化与既有 card-language 路径保持不变。

上层将一次进入战斗的决策延迟到战斗真实结束才结算，跨采样轮持久保存 pending；GAE以战略动作而非战斗点击为时间单位。尚未完成的末尾动作等待后续轮，截断/异常状态丢弃而不当作死亡。跨轮完成的动作保留原行为log_prob，属于延迟宏动作的PPO近似，不能宣称所有样本均由本轮同一策略产生；通过clip及KL门槛约束更新，不复用上层历史replay。下层replay完全沿用原契约。

模型zip包含两个Adam；完整bundle还保存上层pending/episode序号以及原model/replay/RNG/environment。普通resume从同实验完整bundle继续，不只加载权重。`upper/*`记录战略样本/更新数/KL/损失，原`nosl_is_az/*`仍是战斗学习指标。回归见 `tests/unit/test_upper_ppo.py`、`tests/integration/test_entity_joint.py` 和 `test_distributed_lifecycle.py` 的 joint_wholerun CPU/CUDA两轮更新再恢复路径；不得将本机测试称作四机生产验收。

### 固定规则下层的上层学习实验

`entity_joint_rulebase_lower.args` 和 `entity_joint_rulebase_lower_runtime.args`
将战斗决策固定为已完整评测的 `ironclad-turn-v8-awakened-phase`：2个独立未来、
每未来8192节点、beam 8、depth 16、一回合、nonce 20260918；符文圆顶使用同一套
只读公开信息的v3规则回退。完整native动作会严格映射到联合模型的
`autoregressive-v2`动作，包括目标和多选前缀，不能退化为按手牌位置重复调用规则。

该模式关闭战斗replay，不收集NOSL监督，不执行下层Adam、下层梯度同步或普通逐步
PPO，并冻结战斗网络；每轮只有上层宏动作PPO更新。runtime preset保持TRAIN41的
每rank `384×128`（四rank全局196608 transition）、四层MapDAG、最多10 epoch、
LR1e-4。为了只比较下层替换的影响，应从TRAIN41相同的fresh step0 checkpoint启动，
并核对policy tensor SHA、冻结卡向量SHA与上层空Adam一致，不能重新随机初始化后直接
声称配对。`nosl_is_az/rulebase_lower_*`记录规则决策、节点和耗时；验收要求
`train/optimizer_steps=0`、下层权重逐字节不变且`upper/updates>0`。

`--vec native` 使用同一常驻 C++ 线程池批处理规则搜索；树维护和评分直接读取原生
状态，不逐节点构建 Python 字典。需要包含 `rule_turn_plan` 的扩展，缺少时明确报错；
Python `NativeTurnPlanner.choose` 保留为串行参考。符文圆顶与未完成多选仍走共用
动作适配，不能切换规则版本、预算、nonce 或平局规则来冒充等价加速。
`rulebase_lower_batch_seconds` 是实际批处理墙钟时间，`rulebase_lower_search_seconds`
是各根耗时之和，多线程时二者不能互换。跨设备性能用实际整轮采样和更新耗时衡量。

这个规则搜索每个战斗根有 `2×8192` 节点预算（动作过多时按既有每根最低配额扩容）。
判断上层推进效率时以相同上层样本数/环境transition下的楼层、过幕和胜率曲线为主，
同时报告规则搜索耗时；不能仅按训练小时比较。

`entity_upper_rulebase_batch4x_runtime.args` 将上述规则下层实验的每rank采样
扩大为 `1536×128`，四rank每轮786432 transition；上层仍按当轮实际完成的
宏动作做完整批次更新，micro4096、最多10epoch、LR1e-4，总预算2亿。
实际宏动作样本数取决于战斗长度，不能将环境transition等同于上层样本。

### 宝箱选择指标

`game/chest_open_rate` = 打开次数 / 宝箱选择次数；`game/chest_relic_take_rate`
= 至少领取一件遗物的宝箱数 / 打开且提供遗物、已完成领取或放弃选择的宝箱数。
两者各自使用每rank最近 `metrics-window` 次机会，跨rank按实际分母合并。
金币/药水点击、第二件套娃遗物不重复计数；空箱、精英奖励和Boss遗物不计入
条件领取率。尚未选择或中途截断的遗物页面不视为放弃；只要发生宝箱决策，
无需等整局结束即可上报。对应 `*_window` 分子/分母也写入日志；没有分母时
不报告0概率。普通resume与其他滚动指标相同，显示窗口重新累计，已打开待领
宝箱的归属由环境journal重放恢复；这些计数不改变奖励、观测或训练目标。

`distributed_sb3 --continuous` supports fixed-learning-rate, frozen-upper inner-only NOSL and joint-upper small-playout training. The latter retains the upper and compact policy's own fixed learning rates and update budgets. Full recovery saves and a coordinated `--stop-file` are required. It has no update or wall-time target; do not combine it with `--total-steps` or a finite `--updates` target. Omit `--completed-game-budget` when there should also be no completed-game cap. Resident deployment uses `continuous: true` with null round/time limits. Finite experiments retain their explicit budgets.

`distributed_sb3 --training-scope lower --fork-state <bundle>` permits bounded
lower-only experiments from a mature joint checkpoint. It keeps the upper
actor/critic and other learners' parameters and optimizer states unchanged,
while retaining the existing collection, search, lower update, reporting and
recovery paths. Supported configuration: native vector backend, online
small-playout, inner-only NOSL, no replay, reasoning or upper auxiliary search.
Set a finite `--updates` and iteration budget for experiments. Omitted scope
inherits the saved checkpoint (`all` for historical checkpoints); changing it
on ordinary `--resume-state` is rejected. Frozen upper rows report actual
decision counts and `upper/frozen=1`, `upper/updates=0`, without fabricated loss
measurements. Completed upper samples are released; unfinished macro actions
remain available for collection and recovery.

`configs/train/small_playout_upper_aux_768.args` forks the upper-auxiliary experiment from the preserved round-215 bundle, retaining 384 environments per rank and 10 upper updates while collecting 768 vector steps. Use this standalone config instead of layering it over the finite 512-step config. Preserve the source environments, optimizers and learner RNG on fork. Log rounds from the explicit step anchor: `215 + (global_steps - 169082880) / (4 * 384 * 768)`; dashboard comparisons use `branch_round: 215`, `branch_step: 169082880` and each run's own `iteration_steps`. Equal round counts across the two lengths represent different sample budgets.

`combat-outer-return-raw-trajectory-mix-v1` 保持876冻结、下层从零训练，
每场真实完成战斗共享一个原始尺度标签：`0.5 * (战斗期间已获战略奖励
+ 出口后续raw V) + 0.5 * mean(整场实际状态搜索前的lower raw V)`。
自然整局终止不补尾，截断丢弃；连战出口使用下层原始预测。轨迹均值包含
只有结束回合动作的状态，使用采样时detach值，跨collection与恢复保留；
不使用MCTS回传值，不按每个状态扣除奖励前缀，不进行监督归一化或裁剪。
搜索根和普通lower叶除以搜索源状态剩余奖励上界M；结算出口将原始reward
与raw continuation相加后除同一M，连战lower不乘M(next)。qnorm保持关闭。
该目标改变统计方法；不能混用旧normalized checkpoint/replay，也不等于
等待整局真实回报的MC实验。profile为combat_mixed_raw_outer_return_876_4x5090.json。

`combat-actual-run-return-search-normalized-v1` 的监督仅来自实际轨迹：每个
下层状态至整局自然结束的未折扣战略奖励和，包含之后全部战斗和上层决策。
未结束整局跨collection保留，截断整局丢弃，不以critic补尾、不归一化标签。
876仅提供冻结上层actor；搜索普通叶与假想战斗出口均由下层raw critic
预测后除源状态M。额外输入combat_mc_exit区分结算前战斗出口，真实经历
的出口作为policy_weight=0样本使用同样MC标签。上层critic不参与搜索和
监督（上层选动作的共享actor/critic前向可能仍计算未使用的value）。
新objective/观测标志进入checkpoint合同，不能混用旧目标replay或模型。

MC搜索尺度由两部分共同决定：根/叶保持raw V/M，Full-Gumbel再按
`(50 + max_visit) * gumbel_value_scale * Q`评分。当前MC profile使用
`gumbel_value_scale=8`，等价于把有效分母改为M/8（约0.125–0.175）；
不把值裁剪到[0,1]，不做每棵树的min-max归一化，真实奖励监督仍为raw值。
尺度系数属于已有checkpoint/replay的search身份，旧gain=1模型仍可按原
身份读取；新实验fresh lower/Adam/replay，不覆盖旧模型的搜索身份续训。

上层奖励监控沿用现有奖励动作段：`upper/reward_gold_pickup_rate` 与
`upper/reward_potion_pickup_rate` 为领取份数除以已经处理的奖励份数。
领取时计 taken/resolved，奖励页 PROCEED 时把剩余条目计入 resolved；
同一奖励多次出现在决策页不会重复计数，未处理条目不提前作为失败。
金币按条目而非金额，药水按瓶；满栏/Sozu 等离开时不可领取的条目也计未领取，
并单列 `unavailable_left`。`upper/reward_{gold,potion}_policy_probability`
是存在合法对应领取动作时，已有 masked 策略分配给这一类动作的概率质量均值，
并非整张奖励页最终领取的概率。日志同时保留 resolved/taken/policy_decisions/
policy_probability_sum；跨 rank 先聚合计数再除，无分母时省略比例，旧日志不补零。
当前支持 joint-upper 的 NOSL/小网整局采样入口；计数不参与损失、动作或 RNG，
无需新增 pending 状态，尚未处理的奖励通过原环境保存恢复。

同一 joint-upper 入口记录火堆七种主选择（rest/smith/recall/lift/toke/dig/skip）。
`upper/campfire_{option}_choice_rate` 是实际选择次数除以该项合法的主选择次数；
`policy_probability` 是这些相同状态上已有 masked 策略的平均概率，先在合法火堆
选项内归一化，排除药水等其他操作。仅实际提交合法主选择时计数，升级/删牌后的
选牌、挖掘/捕梦网奖励和后续离开不重复计数。`PROCEED` 只有在 rest phase 才是 skip。
原始 `available/selected/probability_sum` 与 `campfire_decisions` 先跨 rank 聚合；
无可选样本时省略条件比例，不把未解锁项目伪造为零偏好。另存 `share`（所选次数/
全部主选择次数）供总体分布分析；条件选择率不同分母，不必相加为100%。计数每轮
重置，普通 checkpoint 恢复不重放已统计选择，也不回填旧轮次；不改变训练/RNG。

Strategic targets use the `fp32-tail-fp64-gae-v1` numerical contract. Within an
observed trajectory, GAE reuses the following record's behavior value so adjacent
terms cancel at lambda=1. Only an unfinished trajectory tail is evaluated again,
in FP32 with autocast and TF32 disabled. GAE/return accumulation and global
advantage moments use FP64; loss inputs are cast to FP32. This does not change
reward, gamma/lambda, sampling length, optimizer or update count. The configured
AMP mode still applies to collection and training forward/backward. Runtime
`upper/bootstrap_samples`, `bootstrap_seconds`, `bootstrap_precision_bits`,
`gae_precision_bits` and `target_precision_version` identify the executed path.
Existing checkpoints remain loadable; completed upper rows are not replayed.

完整游戏 cohort 使用 `--episode-rollout` 和 native vector：每个环境每轮只运行
一个自然完整游戏，结束后保持终局；全部结束才更新，下一轮用新种子重新初始化。
种子为 `seed + seed_offset + completed_rollouts * world_size * n_envs + rank * n_envs + slot`，
旧包 `seed_offset=0`。
`--episode-rollout-max-steps/seconds` 只作失败看门狗，超限报错，不能把未结束局
算死亡、替换或训练。`--n-steps` 此时仅决定未使用的 SB3 scratch buffer 大小。
`global_timesteps` 累加实际执行的动作（含原子选择前缀，不含终局等待），
`rollout_round` 为显式完成批次数；不能用固定 stride 换算。恢复保留模型、双
Adam、RNG、完成批次和已结束环境的严格日志回放（terminal journal schema 2）。
旧进行中环境 journal schema 1 继续支持。此模式目前仅支持 FP32 joint-upper、
small-playout、无辅助/旧 replay；不改变动作、公开观测、奖励或战斗预算。

仅统一上层、零删失且完整自然终局的保存边界允许显式扩容：
`--fork-state <完整包> --fork-grow-environments --episode-rollout --n-envs <更多局数>`。
必须新 run/W&B 身份，不可与任何 reset-environments 参数合用；普通 resume 仍拒绝改变
cohort 几何，world size、seed 和采集预算也不得同时改。源环境逐槽恢复并核观测摘要，
仅追加的槽按下一cohort中对应seed初始化；源模型、Adam、计数、历史窗口和learner RNG保留。
下一轮仍走正常的完整cohort begin：旧终局槽和新槽一起采集新游戏，禁止丢弃未完成源游戏。
迁移将 `seed_offset` 减去 `已完成轮数 × world_size × 新增局数`，下一全局seed连续、
后续stride采用新局数。offset和扩容来源随模型及manifest保存，普通恢复不再施加迁移。
`ENVIRONMENT_EXPANDED_ON_FORK`区分恢复数和追加数，并记录seed范围；不能将新增槽称作源恢复。
扩大cohort改变统计样本和每局更新频率，须独立有界资格、完整墙钟及学习效果评测。

`UpperConfig.minibatch_size > 0` 启用每 rank 小批 PPO；0 保留旧整批实现。
完整轨迹先计算 FP64 GAE，再全局标准化优势，每遍独立打乱并使用每条记录一次。
四 rank 每步各贡献最多指定数量，末尾不足时按真实全局条数加权；空 rank 只
参加零梯度归约，绝不重复/丢弃样本。每次归约后统一 clip/Adam，KL 保护保留。
`epochs` 是数据遍历次数，不再等于 Adam 次数；`requested_updates`、`updates`、
`samples_used` 和 `epochs_completed` 报告实际工作量。AdamW 可独立设置 value
MLP/head 的学习率，其他上层参数用主学习率；默认 `optimizer=inherit` 保持旧
checkpoint 行为。完整960局实验使用 `small_playout_full960_fresh.args` 生成随机
初始模型，四机启动使用 `small_playout_full960_distributed.args`。


统一上层的 `--upper-critic-target mc|frozen-lambda` 只选择 critic 的监督标签；
仅支持 `upper-latent-actions-v1`，省略时沿用 checkpoint，旧包默认 `mc`。
两种模式都要求完整自然终局，先检查完整 MC 回报；`frozen-lambda` 使用采样时保存的
旧 value 和既有 GAE λ 计算固定 λ-return，整次 critic 拟合中不重算目标。
actor 的冻结 GAE、奖励、终局定义、搜索、样本遍数和上下层 Adam 均保持原合同。
这是统计目标实验，不能把 value loss 下降当作实际策略改进。
`upper/critic_target_version` 为1（MC）或2（冻结 λ），配置随模型和 manifest 保存；
普通恢复保留该目标，目标覆盖保留已有优化器状态。

统一上层有界优化实验可在现有分布式入口使用 `--upper-learning-rate`、
`--upper-value-learning-rate`、`--upper-effective-batch-size` 和 `--upper-critic-epochs`。
`--upper-critic-effective-batch-size` 可独立设置 critic 的正整数全局目标 batch；
旧包配置默认为0，表示随 `effective_batch_size`，保持原共享 batch 行为。
省略 CLI 时保留已保存的独立设置；actor 的有效 batch/遍数不会因此改变。
省略时保留 checkpoint；显式覆盖在 Adam 加载之后更新对应配置/参数组，保留所有动量和步数。
全局目标 batch 不再乘 rank 数，实际每遍更新 `ceil(全局上层样本数 / 目标batch)` 次，
所有完整游戏样本均使用；micro batch 只控制执行分块。日志记录实际 actor/critic LR、
目标及实际平均有效 batch、更新次数和样本使用，资格仍须完成多机生产采集与保存恢复。
其中既有 `requested_effective_batch_size` / `effective_batch_size_global` / `requested_updates`
保留 actor 口径；critic 对应 `requested_critic_effective_batch_size` /
`critic_effective_batch_size_global` / `requested_critic_updates`，实际计数仍为 `critic_updates`。

统一上层的 `--upper-measure-value-loss` 启用同一完整训练 cohort 的冻结前后测量；
默认关闭，省略参数时沿用 checkpoint，`--no-upper-measure-value-loss` 显式关闭。
`upper/value_mc_mse_pre_update` / `post_update` 分别在任何上层更新前、全部 critic
更新后，以固定权重和 eval/no-grad 对全部样本重算 MC MSE；对应
`value_target_mse_pre_update` / `post_update` 使用实际 critic 标签，冻结 λ 模式下
与 MC 区分。标签使用训练实际 float32 值、平方误差以 float64 累计，按全局样本数加权；
`value_measurement_samples` 给出分母，`critic_target_version` 给出目标版本。
当前 SSH 标量归约仍使用 FP32；FP64 是本地平方误差累计精度，同一 MC/目标指标
也可能因 ring 分片的求和次序产生微小舍入差，比较须使用有依据的浮点容差。
测量保留模型训练模式和 Python/NumPy/Torch RNG，不执行 optimizer step。
`value_measurement_pre_seconds_rank` / `post_seconds_rank` 和总
`value_measurement_seconds_rank` 包含 GPU 同步与指标归约；开销计入完整轮墙钟。
原 `upper/value_loss` 仍是末遍各 batch 更新前误差的加权均值，不是冻结前后重算值。
以上均为**参与拟合的训练数据**，不能凭降幅证明过拟合；泛化诊断须另用未参与更新、
按完整游戏划分的留出集。空 rank 参与同一归约，旧轮次没有这些字段时不填零。

已有 `tools.analysis.probe_upper_learning --unified-candidate-protocol ...` 的协议
`diagnostic.full_critic_mse: true` 会另外对按完整游戏拆分的**全部** train/holdout
决策记录冻结 MC MSE、全局样本数和每 rank 测量耗时；`full_critic_pre_update` 为
共同源权重，逐臂 `full_critic_post_update` 为更新后的权重。它复用同一测量原语。
原 `probes` 仍受 `max_probe_decisions_per_split_per_rank` 限制，是另一个采样分母；
不要将 probe MSE 与全 split MSE 混用，或将有界诊断的总成本混入生产单轮吞吐。

`tools.analysis.probe_upper_learning --unified-candidate-protocol <协议> --probe-output <目录>`
也接受 `unified-upper-rollout-scale-v1`：沿同一分布式恢复与完整采集入口，在上下层更新前
结束，逐rank输出 `rollout.json`、自然局/动作/决策分母、CUDA同步采集时间、资源和源参数/Adam摘要。
允许显式完整边界扩容fork；禁止continuous、reset、W&B训练记录或模型保存。
测量包括首次使用开销，不丢弃warmup局；启动/恢复与总进程时间另记，采集时间不含更新/同步/保存。
`unified-upper-candidate-screen-v1` 的 `diagnostic.profile_updates: true` 为每个临时更新臂
记录1秒资源采样及精确Torch峰值；固定有效batch/样本顺序时可比较microbatch执行分块。
CPU资源缺GPU项填null；单次诊断吞吐不是完整训练资格，也不能证明同墙钟学习收益。

独立完整游戏采集的原生 small-playout 路径可省去未被消费的战斗观测：
战斗中内部传输只保证 `phase/act/floor` 当前有效，非战斗传输完整上层字段；
下层搜索仍读取完整模拟器状态，终局 `info` 保留完整观测。该路径仅在具体回调类
直接声明 `_uses_rollout_observations = False` 时启用；未知回调及未重新声明的子类
默认使用完整观测。声明表示 `_on_step` 不读取观测或保存模型，不能只因父类支持而继承。
开始/结束回调、正常返回、提前取消及异常退出后恢复完整 `model._last_obs` 和向量缓存，
公开环境接口和 checkpoint schema 不变；未知环境包装器也保留完整路径。
`nosl_is_az/combat_observation_rows_omitted` 记录每 rank 省略行数的全局平均。


统一上层每轮还记录 `upper/policy_loss_pre_update`：在任何actor/critic更新前，
固定actor对本轮全部上层决策重新前向，使用采集时old_logp和本轮全局标准化优势，
计算与训练一致的clipped PPO surrogate（不含entropy正则）。不是首遍移动参数的均值，
也不是交叉熵预测误差；ratio约1且mean(A)约0，因此接近0是预期行为。
同步记录 `policy_kl_pre_update`、`policy_entropy_pre_update`、`policy_nll_pre_update`
（采集动作负对数概率）、`policy_ratio_abs_error_pre_update`（mean(abs(ratio-1)))，
帮助区分策略分布变化和相对目标归零。NLL/熵也来自当前训练分布，不是独立评测。
`policy_measurement_samples/version` 标记全局分母与版本；本地FP64求和后按现有
SSH归约协议汇总，`policy_measurement_pre_seconds_rank` 包含GPU同步和归约，
计入整轮时间。保留模式、Python/NumPy/Torch RNG与梯度/Adam，不改变更新配置；
空rank也参加归约。此项仅统一上层模型支持，旧模型/旧日志不补零。

战斗搜索的公开实体输入 `sts1-playout-public-entities-v1` 由
`small_playout_public_features(game, battle)` 导出，布局见
`small_playout_public_layout()`。inventory 是当前公开的永久牌组、遗物及其属性；
entities 是本战斗的牌堆、角色、怪物、状态、可见招式、药水及待选来源。
普通抽牌堆未知部分只提供多重集，已知顶部与冰冻之眼例外；符文圆顶隐藏当前意图。
source_rows 只引用动态实体行，target_slots 对应原生怪物槽位，-1 表示没有对应项。
它们仅用于索引，执行句柄、物理UID、RNG不作为数值特征。候选数无96上限。
此接口与旧的 `sts1-small-playout-v1` 独立，不能将新权重按旧格式导出。

下层新架构的隔离试验从原生在线 AdamW 的 v1 checkpoint 使用
`--fork-state <bundle> --training-scope lower --fork-public-lower root`。
`root` 每个真实决策提取公开信息，在本次 MCTS 内复用上下文，模拟每一步仍读实时的
32 个状态和 24 个动作属性；`state` 在每个需要打分的模拟状态重算全部公开实体。
两者搜索预算、目标和合法动作相同，策略并不等价。初始化时 1024×64 适配矩阵为零；
`--fork-public-weights <selected.pt>` 可装载与旧下层摘要匹配的离线权重和 Adam，
不传则仅下层 Adam 重新开始。普通恢复继承模型、优化器、模式和训练范围。
`python -m tools.analysis.probe_lower_public --help` 给出成熟固定数据选轮的参数。

`--fork-public-lower turn` 在每条模拟的首个打分点、回合变化及选择界面刷新上下文；
`state` 为每个打分点刷新。两者使用同一组实体与参数，刷新频率是策略的一部分。
固定数据实验可以先 `python -m tools.analysis.probe_lower_public --collect --source <bundle> --output <bank>`，
再用 `--bank <bank> --source <bundle> --output <fit>` 选轮。严格区分离线动作标签与搜索后实战效果。

逐实体动作打分目前用于**离线诊断**。同一状态的序列含手牌、真实敌人槽、药水槽及
牌堆/遗物等公开摘要；当前可选的非手牌实例另有 token。每个动作在成熟下层的 logit 上
叠加来源分、目标分及可学习的来源–目标配对分。一次前向给全部合法动作打分。
这套权重不能导出到原生 v1 接口，也没有继承线性公开模型的 rollout 用时。

- `python -m tools.analysis.probe_action_tokens --help`：单批完整对局划分训练/验证/测试，
  对照仅相加与可配对打分，以及三个模型容量。只用验证局选择遍数。
- `python -m tools.analysis.probe_lower_public --collect --collections 2 --cohort-offset N ...`：
  固定同一教师、使用该 rank 的真实环境生成连续新局；offset 以整个 world 的一批局为单位。
- `python -m tools.analysis.probe_fresh_action_tokens --parent <bundle> --first <首批> --others <新批...> --holdout <独立测试批> --output <新目录>`：
  对照首批重放与每批取不重复决策，首组更新后复制模型及 Adam，更新与样本呈现数一致；
  每批须有 480 个自然结束的游戏，局 seed 不可重叠。加 `--compact-controls` 用相同协议训练原模型与线性公开输入对照。
  结果给出按决策加权、按整局重采样的成对区间。

同一多批实验可用 `--continue-arm 32|64|128|sketch --resume-lower <105步权重和Adam> --validation-bank <独立验证批>` 续训。每遍会遍历允许的对局决策，保留末尾不足整批的样本。
`--check-every`、`--patience`、`--min-gain` 定义验证KL平台，触发后从验证最优的权重和Adam降低学习率，第二次降率后再次达到平台时停止。另记录预算用尽状态。最终在独立测试批对比成熟模型、第105步和验证选中权重的KL及与搜索推荐动作相同的比例。

冻结的原版 COMBAT37 可通过 `python -m rl.evaluation.eval <model.zip> --task wholerun --solver paired-worldlines` 测共享世界线。每条世界线先确定可见信息所允许的抽牌堆和游戏随机状态，给各个合法动作复制相同起点和策略随机流，再用原紧凑策略模拟到终局或截断。普通/精英总250次、首领1000次，每动作分到的次数至多差1，余数随机分配。按各动作的归一化终局均值评分，距最高分0.005以内用原下层logit排序，剩余平分仍均匀抽选。普通战以根最大生命归一化，首领以胜负计，截断保留原保守计分。此评测契约 `paired-worldlines-eps005-terminal-hp-boss-win-turn128-v1` 适用于单个v1策略，符文圆顶沿用公开策略。传统 `small-playout` 保留UCT。

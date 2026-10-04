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

当前支持entity-combat + autoregressive-v2 + C++ NOSL（native线程或既有worker传输）、纯policy预测及`NativeSearchConfig.from_model`评测。原生终局backup与模型叶估值使用相同权重；合并不跨一次根搜索，不混合Boss与普通目标。与历史boss-transition回血目标、model-evaluated victory模式不可同时启用；旧Python CombatValueSearch显式拒绝该新模型，防止用旧HP终局混入新效用。默认权重0保留旧目标，普通战斗不改变收益函数；模型共享参数与新采样造成的训练差异属于本次目标实验。

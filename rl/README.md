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

训练使用原搜索策略软标签，加BCE胜负、胜利样本pinball、
0.25期望效用MSE、0.1实际动作后果SmoothL1。
后果为下一决策边界的HP/格挡/能量/敌方总HP/力量/手牌数变化；
只监督实际执行路径。所有项按完整逻辑batch平均，微批/多rank不重新
按各自胜利数归一化。明确胜负来自simulator outcome；
截断、未知或逃跑标签不进入该监督，失败没有伪造的“获胜血量”标签。
Replay版本3及 `combat-outcome-quantiles-v1` 拒绝旧标量回放混入。

固定预算示例（首次实验超参数，不是已验证的最佳配方）：

```text
python -m rl.training.trainer --task wholerun @configs/train/entity_combat.args --steps 24576 --seed 0 --device cuda --amp-mode bf16 --out runs/local/entity-combat
```

配置保持96 env、128步；逻辑训练batch16384、微批256、每次采样4次联合policy/value更新，
回放131072，旧样本采样上限0.9，LR5e-5。`--steps 0 --vec dummy --device cpu` 可只创建初始checkpoint。
`--nosl-is-az-update-schedule value-warmup` 在前5轮额外增加36次value/辅助头更新，
第6–20轮按余弦减到0；第10/15/20轮总更新数为31/13/4，之后固定4。
额外更新先于联合更新，训练共享表示、胜负/HP分布和动作后果；策略专用头的参数及Adam状态
在这些更新中保持不动，但共享表示变化仍会影响策略输出。目标来自真实自博弈终局，
并非另行采集的均匀随机策略数据。该曲线是实验设计，尚未证明优于固定U4/U40。
调度按累计global steps除以四机整轮动作量计算（生产每轮49152），保存恢复不重启warmup。
日志区分scheduled_updates、value_only_updates、joint_updates；零有效标签时实际更新可以为0。
旧checkpoint和未指定该选项的入口默认constant；value-warmup仅支持entity-combat的C++ inner-only NOSL。
初始化与分布式运行共用 `configs/train/entity_combat_learning.args`；
`entity_combat_runtime.args` 用于既有四机launcher，目标50M，BF16。
该实验使用Full Gumbel128，关闭Q归一化、value scale=1。
`paper_full_gumbel` 保留论文路径的sequential halving、visit-weighted backup
和mixed-value完成法；从本版本起Q归一化和scale尊重显式配置，默认仍为true/0.1。
旧checkpoint中保存的canonical true/0.1不变；曾携带非默认Q字段但被旧strict
实现忽略的checkpoint会开始采用该字段，复现旧行为时须显式设true/0.1。
Python、原生扩展、dashboard必须同步升级，防止旧扩展静默覆盖新Q设置。
新监督不保证消除HP反转；训练后仍须固定局面/未来随机流评测。

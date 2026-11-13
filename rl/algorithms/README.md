# 更新与存储

`accumulating_ppo.py` 是 SB3 扩展：含 PPO 梯度累积、分层 rollout 和 NOSL combat learner。`hierarchical_returns.py` 定义宏/微回报；`nosl_replay.py` 负责紧凑 FIFO、fresh/old 配额与采样。

输入是 env rollout 和 search root targets，输出是 policy/optimizer 更新。只在完成战斗后提交最终 HP 标签；当前 F05/F11/F12 等语义问题见根 TODO。
`ppo.py` 定义纯 PPO 目标；`updates.py` 共享 PPO 前推、reasoning batch/反传和累积平均→裁剪→optimizer 更新边界。本地与分布式编排都调用这些原语；NOSL 模式下 combat 由 AZ 监督，不再同时进入 PPO 目标。统计加权与 replay 配额属于算法选择，不能由传输层悄悄改变。

新增学习方法可以组合这些原语，也可以独立实现目标；不要求继承 PPO 或伪装成 NOSL。继续拆 collector/learner 时用 tools.validation.bitwise 对照 rollout、有效 replay、Adam 和权重，不能只测最终 action。

NOSL CUDA learner 对至少32768条的有效 batch 用最多4个线程提取独立 replay
字段。采样 RNG、索引顺序和 fresh/old 配额不变；`get_batch` 返回前线程全部结束，
调用期间不可写 replay。CPU 和小 batch 默认串行。
紧凑字段先放入 pinned host memory，再异步传输并在 GPU 转回 observation space
规定的 dtype。每次更新末尾的同步诊断回传完成后才释放 host buffer；不缓存旧权重，
不改变 microbatch、梯度归一化或 Adam 更新顺序。
浮点逐位对照需固定设备、依赖、确定性算法及 CUDA BLAS 配置；默认 CUDA 的末位
差异不能当作跨设备 bitwise 保证。吞吐还须覆盖完整 collection/update/save 周期。


`budget_calibration.py` 提供成熟策略的独立预算头附加、审计特征保存及有界拟合；复用 `updates.optimizer_step`，不执行 PPO 或下层动作监督。特征由 `AdaptiveTokenRegretPolicy.budget_features` 统一生成并 detach。预算头以单独文件绑定原模型 SHA 和 tensor digest，不能替代完整训练恢复文件。采集和冻结评测复用现有 episode runner/策略与 eval CLI，支持范围见 `rl/evaluation/README.md`。

Deck-latent 的 `SearchTrainingConfig.action_selection` 默认 `shadow`，用于兼容
历史旁路教学 checkpoint。显式 `search-filter-policy-rank-v1` 则在真实环境推进前
完成当前根搜索：保留已访问动作中 Q 距最大值不超过 .005 的集合，选择模型
logit 最大者，完全同分用独立 nonce 均匀选择。此规则与 `PairedTokenBatch`
共用 `env/search/action_selection.py`，不增加搜索预算或丢弃候选槽。
调度配置对应 `search_action_selection`，训练目标标记为
`deck-latent-search-acting-v1`；改变此字段是采样行为迁移，不能称等价加速。

真实动作和同分选择概率写入 rollout；`_search_executed` 区分确定性搜索决策。
这些行参加真实回报 value 与原搜索辅助损失，不参加模型随机采样的 PPO
actor ratio、entropy 或 KL。其余行保持 PPO，逻辑 batch、全行分母和 Adam
边界不变。当前固定步长 vector 入口在动作前等待当步搜索，不能用旁路搜索
跨时间步并行的历史采样速度作为此模式的性能验收。

# 训练编排

`trainer.py` 是单机 task CLI；`distributed_sb3.py` 是 SSH ring 多 rank 训练编排。RLlib 与 overfit 专用训练入口已退役，需要历史实现时从 Git 恢复。

`cached_mask_vec_env.py` 处理 worker 与 mask/packed leaf IPC；`ring_allreduce.py` / `ssh_transport.py` 处理同步；`reward.py` 提供训练 reward wrapper。小预算训练使用 rl.benchmarks.train，与生产共享算法。

当前实验入口为 `python -m rl.training.distributed_sb3`。算法参数、机器地址、结果目录分别见 configs/train、experiments/launchers 和忽略的 runs/local。

## 在现有纯 PPO 训练上启用战斗结束锚点

先在共同更新边界停止并保护每个 rank 的完整恢复包，再用现有 checkpoint 工具迁移：

```text
python -m tools.checkpoints.replace_combat --mode anchor-bundle --source SOURCE_BUNDLE --out NEW_BUNDLE
```

仅支持 shared whole-run、sequential-v1、gamma/lambda=1、无搜索的 CPU checkpoint 迁移。
每个 rank 独立迁移后，以同一个 run ID 和 `--resume-state NEW_BUNDLE` 继续分布式训练。
已有权重、Adam、RNG、步数、自然局计数和未结束对局均保留；只在战斗 scalar projection 的
六个派生特征之前插入两个零权重输入，并对 Adam 一二阶矩做相同插入。迁移逐局检查旧观察和
合法动作完全相等，比较 CPU FP32 策略概率/value（绝对及相对容差 2e-5），记录实际最大误差。

环境恢复状态的可选 `combat_exit_anchor_start_action` 标记旧动作日志中启用锚点的位置。
历史前缀仍按无锚点规则重放，后缀按锚点规则重放；新对局从位置 0 启用。旧恢复包缺少该字段时
默认 0，因此兼容既有无锚点及从头有锚点训练。迁移后的包必须用支持该字段的运行时代码恢复。
奖励仍由同一结算流程提供：战斗动作目标截到奖励揭示前的旧 V，锚点自身与上层仍用 lambda=1 整局回报。

## COMBAT04：保留context的TRAIN47目标对照

`configs/train/combat_train47_context_876_4x5090.json` 沿用上述入口的
`--prepare-combat-outer` 材料准备模式。876上层继续冻结、FP32 masked argmax；
下层及context分支随机初始化，采用TRAIN47的胜负BCE、胜利条件HP分位数及utility MSE，
Boss权重0.9，转移辅助损失系数0。无效的transition头保持源配置的停用状态；其他下层
参数从第一次更新共同训练。优化器、replay、计数全部从零开始。

`OuterReturnPolicy` 的显式 `objective='combat-outcome-quantiles-v1'` 选择该受支持变体；
默认值仍是COMBAT01。`outer_return_context` 只决定额外输入，环境的
`combat_value_objective` 独立决定真实终局标签；旧环境未指定目标时保持原行为。
原生 `public_run_context` 仅传输叶的永久卡牌成长，不改变TRAIN47搜索终值。
必须使用包含该传输扩展的新native，恢复身份严格记录目标、损失配置、teacher和native。
沿用历史持久字段名 `outer_return_contract` 保存这两种冻结上层合同，其存在不代表选择外层目标。

本对照保留额外context，因此不是原始TRAIN47架构的完全重跑；上层value不参与下层监督。
部署仍需两轮真实生命周期和GPU graph形状/淘汰/权重变化/资源释放验收。

## COMBAT01：876冻结上层、完整随机下层

`configs/train/combat_outer_return_876_4x5090.json` 定义当前合同。876只提供独立的
上层actor/critic及依赖；下层actor、encoder、公开context和FP32标量critic随机初始化，
所有下层Parameter从第一轮共同更新。固定卡向量是输入buffer。没有预校准、局部冻结或
上层optimizer；禁止旧HP/Boss目标、value-only更新和旧Adam/replay恢复。

材料导出见 `tools.checkpoints.replace_combat --mode entity-material`；现有入口创建新step0：

```text
python -m rl.training.distributed_sb3 --rank 0 --world-size 1 \
  --prepare-combat-outer configs/train/combat_outer_return_876_4x5090.json \
  --material-directory MATERIAL --material-sha256 MANIFEST_SHA256 \
  --output runs/local/combat01/initial.zip --device cpu --native-search-threads 2
```

该操作只构造并核验，不训练；目标文件必须不存在。输出JSON记录profile/native/teacher
身份、空Adam/replay、step0和可训练参数数量。必须使用当前源码重建的显式
`STS_LIGHTSPEED_BUILD_DIR`；原交接二进制没有新目标接口，不能直接使用。

训练支持resident `--vec native`、每树一个在途叶、完整GameContext单世界线。
实际退出按自然终局、上层决策、直接连战分别结算；跨collection未结束的战斗保留
reward前缀和pending，截断不冒充死亡。上层始终FP32 masked argmax，奖励仍是
TRAIN47整局定义、gamma1；新目标版本为 `combat-outer-return-frozen-upper-v1`。
通过[四机manager](../../experiments/launchers/README.md)启动新run，参数沿用876，预算显式有限。

新checkpoint和replay带目标、teacher、native、公开上下文与搜索身份，改变身份即拒绝
resume。移动完整材料时可设置 `STS_OUTER_MATERIAL_DIR`，文件及manifest SHA仍必须一致。
恢复用每rank相同global step的完整bundle；不能混用旧TRAIN47恢复状态或仅复制rank0。
CUDA恢复必须保留CUDA RNG。恢复本身不承诺异步搜索轨迹逐位重演。

CPU eager和CUDA graph训练、whole-run policy/native评测已接入该合同；Watch、dashboard
和真实bridge尚未适配完整上下文，不能直接部署新checkpoint。缺字段会明确拒绝。
Windows开发GPU可显式 `--no-leaf-compile`；四机Linux保持compile开启，其现场图缓存和
完整训练验收不能由Windows结果替代。

PPO、NOSL 与 reasoning 使用同一个梯度同步边界（ring 或 file transport）和 algorithms/updates.py 的 optimizer 更新；编排负责 rank、barrier、批次调度和日志。新的策略/学习方法应显式选择其目标和采样方式，不能通过复制整套 trainer 来添加算法。大 main 的生命周期拆分仍待做。

环境构造与 checkpoint schema 适配在 `environments.py`，采样基准复用同一工厂。
`--vec native --native-search-threads 4` 将环境留在当前进程，原生搜索线程池并行处理
不同树的 select/backup；每棵树仍只有一个待求值叶，模型每波只接收一个数字数组批。
支持 compact legacy/joint-v2/autoregressive-v2（含 card-language），不支持 combat-language
或同树多个在途叶。原生扩展须重建；缺失批处理接口会明确报错。
此模式的环境真实 step 仍由 Python 编排，只有搜索/叶编码走 C++；默认后端保持 subproc。
无叶子 IPC 不等于端到端零开销，`native_tree_seconds` 与 `native_pack_seconds` 是
begin/advance 的内部拆分，不能再次加到 collection 总时间。环境 checkpoint 状态沿用相同合同。
分布式 JSON 与 W&B 的 `nosl_is_az/native_{begin,advance,finish,tree,pack}_seconds`
记录每轮 collection 的跨 rank 均值；不随该轮 learner 更新次数重复累计。
非 native 路径或没有搜索的轮次为零。`cpp_leaf_model_seconds` 扣除了叶解码，
仍含推理包装和数据传输，不能解释为纯 GPU kernel 时间。
`--leaf-inference cuda-graph` 仅加速 CUDA compact combat 叶节点；其它路径默认 eager。
`--leaf-graph-batch-multiple`、缓存生命周期和数值边界见 [搜索说明](../search/README.md)。
Grouped worker 对同一组 native 数字观测按字段合并 IPC，调用方仍收到原来的
逐环境 RPC 结果；混合终局、不同 batch 长度与请求顺序有真实子进程回归。
`--rollout-resource-interval 2` 让同一个采样器覆盖完整训练轮：每个 rank 的 JSON 中，
`rollout_resources` 保留 collection 快照，`iteration_resources` 还包含 learner、同步、
本轮周期保存与 checkpoint 发布等待。两者包含 CPU/RSS/cgroup/GPU 采样，0 关闭；
定义与复现实验入口见 [资源基准](../benchmarks/README.md)。

生产恢复默认使用 `--training-state-dir DIR --save-every-updates 1 --max-checkpoints 3`。
在需要测量 checkpoint 固定成本时，launcher 可通过
`NOSL_SAVE_EVERY_UPDATES=N` 将完整 recovery bundle 改为每 N 个 update 保存一次；
stop-file 和正常结束仍强制发布最后一个 bundle。该选项只改变恢复窗口，不改变采样、搜索、loss、optimizer 或 replay。
每个 rank 各存 `DIR/rankN/step-XXXXXXXXXXXX/{model.zip,learner.pt,manifest.json}`，
支持环境恢复的 schema 2 另存 `environment.pt`。
manifest 最后原子发布，包含配置、run/W&B ID 和文件 SHA256。恢复时四 rank 必须
选同一个已完整保存的 global step，传入各自的 `--resume-state`；不得用 rank 0 的
replay 代替其它 rank。模型/Adam、有效 replay slots、父进程 Python/NumPy/Torch/CUDA
RNG 恢复，global step 连续。完整性或 rank/world 不匹配立即拒绝恢复。

schema 2 恢复原生环境、wrapper 状态、当前观测、episode 窗口和未结束战斗的
pending labels，并校验恢复观测摘要。schema 1 的旧 bundle 只恢复 learner，需重启
环境。显式 `--fork-reset-environments` 也会重启环境并丢弃环境局部进度；改变环境
数量的 fork 必须使用这一选项。恢复与 fork 不重新初始化已训练的 shared embeddings。
新环境不安装已退役的 occupancy 预热 wrapper；旧 schema 2 快照若记录已 release
的 wrapper，恢复时重建其 episode bookkeeping。尚未 release 的旧快照明确拒绝，
需用 `--fork-reset-environments` 重启环境，避免恢复后永久等待。
learner/environment 使用 pickle protocol 5 保存 NumPy buffer，保持 bundle schema
不变；历史 protocol 2 仍可读取。实际恢复须使用对应 rank 的完整 bundle。
同一下一训练 batch 的 learner/Adam/RNG 有逐位回归；异步搜索调度及 CUDA 非确定性
仍可能改变后续轨迹，不能将状态恢复等同于整场训练的逐位重演。
每轮完整快照包含每 rank 的 replay，写盘成本需计入实际生产吞吐。
`dist/iteration_wall_sec` 与吞吐分母包含指标聚合、恢复点写盘和跨 rank 发布等待，
`dist/checkpoint_sec` 单独记录保存/发布等待；不包含随后 W&B 客户端排队的时间。
周期性的 `--output` 和旧 `--checkpoint-dir` 保存也计入本轮保存时间；
整个 run 结束后的最终保存属于收尾，不计入单轮时间或资源范围。
原 `dist/rank0_total_sec` 保留旧计时口径，避免误当作新端到端口径。

`--stop-file PATH` 在下一次更新/跨 rank 同步后触发保存并协同结束；多机要求 ssh-ring。
正常维护创建所有 rank 的 stop 文件并等待全部退出、核对同一恢复代数，再删除这些
请求文件并恢复。不要在训练中直接 kill 后声称最后一轮已保存。

战略上层的 `--upper-target-kl X` 可在普通恢复时覆盖 KL 提前截停目标：正数在
当前更新批的近似 KL 超过 `1.5 * X` 时拒绝该批并停止本轮剩余更新；`0` 显式关闭
KL 截停，保存为 `upper_config.target_kl=null`。不传参数则保留 checkpoint 设置。
关闭只影响截停判断，仍记录原近似 KL、epoch 样本加权统计和实际更新覆盖；
PPO 裁剪、梯度裁剪、非有限 loss/梯度的故障检查保持。`upper/kl_early_stop_enabled`
标记当前机制，旧配置默认目标 `.02` 不变。改变此项是训练行为变更，须记录恢复边界。

完整自然局训练可设置 `--checkpoint-keep-round-multiple 5`：每轮仍保存，四rank共同
发布后只保留最近一个 `rollout_round % 5 == 0` 的完整包和最新完整包；重合时只留一份。
该模式要求 `--save-every-updates 1`，覆盖按 `--max-checkpoints` 计数的保留方式。
写入期间不预删旧锚点/当前包，临时最多三份；空间不足或写入失败保留旧恢复点并报错。
首次到第五轮前只有最新包，没有可用锚点。原永久 `references` 目录不参与轮转。
新manifest记录显式 `rollout_round`；旧cohort包从model.zip数据读取该计数，不能对
真实动作数global step取模代替轮次。


`NativeSearchVecEnv` 在原生扩展提供 `pack_neural_roots` 时，可批量构造 AR 战斗的完整根观测；搜索叶与真实根复用同一原生 pack/decoder。仅已知训练包装栈（TrainingState、CheckpointObservationCompat、ShapedReward、TimeLimit）允许在单次同步 `step_wait` 内延后非终局观测，返回前全部补齐。非战斗、真实终局及外层 TimeLimit 截断保留完整观测与原 autoreset；未知 wrapper、旧扩展或其他观测合同沿用串行路径。此优化没有改变观测版本、搜索目标或保存状态格式。回归入口：`tests/integration/test_native_root_observation.py`，必须指定候选扩展构建。

统一上层合同 `upper-latent-actions-v1` 使用相同公开观察/动作ABI，模型权重与旧上层不兼容，按 `UpperConfig.architecture` 显式分流；旧 checkpoint 默认 `legacy`。`configs/train/unified_upper.json` 固定新模型/更新配置。状态实体含公开地图有向边，少量latent交叉注意力汇聚，动作按操作类型和引用对象统一评分，实体候选不使用无意义的槽位偏好；保留药水槽和地图坐标。

两套网络（含embedding）完全独立。actor使用采集旧V得到的GAE和旧logp，critic使用完整自然局实际奖励和；标签在任何更新前固定。仅支持完整自然局及整局排除语义，未完成轨迹拒绝更新。每遍按全局有效样本数均衡分块、逐micro按样本数累积，再同步/裁剪/Adam；不按界面等权、不复制/丢弃尾批。`upper/updates`为actor Adam，`upper/critic_updates`单列critic Adam；两个参数组共用保存容器但没有共享训练参数/动量。

`configs/train/unified_upper_420.args` 按用户要求持续训练，没有总轮数上限。`--iteration-timeout-seconds 415`覆盖采样、学习、同步、保存和W&B上报；到期进程退出124，写round-timeout记录，不能把本轮当完成或自动循环重启。`ROUND_COMPLETE`记录上报之后的完整耗时；仅从四rank都有该标记且bundle校验完整的共同代恢复。硬限是异常保护，生产资格仍需测得正常完整轮≤420s并留余量；不是硬实时平台保证。每次启动/构建初始化不属于训练轮。

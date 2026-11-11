# COMBAT49：大外层、256环境与采样加速恢复

状态：采样资格已通过，四机已使用 `427ddedb` 从零启动。关联任务：`TODO.md` 的 `DECK-LATENT-FAST-RESTART`。实际进度与验证结果追加到 `worklog/2026-11-11.md`。

## 用户意图与顺序

用户要求恢复每机256环境、外层恢复之前的大模型，并强调“加速项需要打开后再启动训练”；随后要求先将即将启动的训练及所需优化记录到Git。

顺序为：保存停止COMBAT48 → 适配缺失加速路径 → 语义与生产长度性能验证 → 从零启动COMBAT49 → 四机完整两轮验收。不能仅把不兼容开关置为true，也不能在等待优化期间先启动慢版本训练。正式连续训练沿用无总步数上限授权，所有测试有独立预算。

## 计划训练配置

| 项目 | 计划值 |
| --- | --- |
| 初始化 | 全部随机权重、空Adam、0步、新环境，不加载旧48权重 |
| 构筑编码器 | 恢复之前大外层规格：512宽 × 8层，8头 |
| 外层决策网络 | 512宽 × 2层，8头；保留地图编码 |
| 内层战斗网络 | 2层 × 64宽、4头、FFN128；普通32token，最多128，不裁合法候选 |
| 战后锚点网络 | 保留256宽、真实战后资源和构筑latent输入 |
| 环境与采样 | 4台 × 256环境 × 768步；每机196608、全局786432真实环境transition/轮 |
| 学习率与PPO | 5e-5，1 epoch；batch1536 × accumulation8 × 4台 = 有效global49152；16次Adam/轮 |
| 优势归一化 | 延续当前rollout级全局统计，不改回历史group256 |
| 精度 | FP32 highest，AMP/TF32关闭 |
| 契约 | 保持当前奖励、autoregressive-v2动作、sequential-v1奖励流程 |
| 初始目标 | gamma=GAE lambda=room lambda=1；whole-run预热，不开战斗锚点目标 |
| 阶段调度 | `configs/train/deck_latent_schedule.json`：100轮首检，之后每20轮检查，连续3次合格后切锚点；不指定固定启用轮次 |
| 搜索 | 初始关闭；search_qualified=false，独立资格通过前不得自动启用 |
| 环境执行 | native-subproc、forkserver、每worker3环境（每机86worker）、单线程Torch |
| 更新物理分块 | 当前768行/131072 token仅作候选起点，大模型重新验证显存；不得改变有效batch/Adam次数 |
| 更新复用 | context_grouping=true；context_gradient_replay=false，后者此前同数据更慢 |
| 正式运行 | continuous，无总步数上限；保留停止文件及完整checkpoint恢复协议 |
| 计划root | `/root/autodl-tmp/combat49-deck-latent512-t64x2-20261111` |
| 计划run | `combat49-deck-latent512-t64x2-ppo-search-r1` |

部署注册使用 `experiments/launchers/unified_upper_comparison.json`，实际操作与原始结果放忽略的 `runs/local/combat49-deploy-20261111/`。现役监控仅在新训练启动后切换；此处不存凭据。若升级原生构建，须另记SHA及兼容验证。

## 速度差异的已知证据

旧COMBAT40源 `ce4c10fd`：509–526连续18轮四rank最慢采样/完整更新均低于20秒；517–526轮最慢采样均值18.400秒，每轮786432真实transition。定位：`worklog/2026-11-09.md` 的 `PURE-RL-RANK3-REPLACE` 小节及其资格归档。

当前COMBAT48源 `33d45a8a`：25–34轮四rank最慢采样均值73.887651秒、范围72.491–76.093秒，每轮393216真实transition。每transition吞吐约低8.03倍，仍warmup、搜索模拟数0、无OOM退避。模型、阶段和观察不同，此比较不是隔离因素的因果A/B。

已确认快路径回退，但尚未测出当前模型中各项占用秒数；不能把全部差异归因于一个开关，也不能承诺更大新模型必然恢复20秒。

## 逐项加速审计

| 项目 | COMBAT48源33d45a8a实际状态 | 新训练要求 |
| --- | --- | --- |
| GPU执行图 | DeckLatentCore显式collection_graphs_supported=false，collector未安装图运行时 | 适配新构筑/内层/锚点几何后启用，保留有界缓存、共享临时池和输出所有权 |
| host元数据派发 | 新路径有GPU unique/tolist/nonzero同步，旧图路径用CPU phase/宽度 | 恢复host派发，未知调用方仍有正确fallback |
| 按dtype打包、少量gather/copy | 旧图路径包含，新core逐phase gather整个观察字典 | 新图输入包含新增字段，打包后恢复视图，不删除有效信息 |
| 上层界面合批 | 新core覆盖actor_critic，绕开merged-upper路径 | 恢复，保留真实phase、动作头及无样本参数的梯度语义 |
| 地图神经编码缓存/合批 | 旧采样图的map cache未被新core使用，外层按phase重复调用 | 恢复内容/位置/权重失效检查和跨phase复用 |
| 原生批量战斗观测 | _batchable明确排除deck_latent_context，逐环境重建观察 | 适配新增牌堆升级、构筑misc、地图字段，逐字段对照reference codec |
| 原生目的地直写/切片view缓存 | 被上一项连带绕开 | 恢复连续行直写，非连续/旧schema保留scatter；终局/重置观察独立拥有数据 |
| 子进程共享内存 | 仍在，模拟器也仍是C++，不是纯Python模拟器 | 保留并验证新schema和恢复后的所有权 |
| forkserver、每worker3环境 | 仍在；此前3→8→16→3完整对照显示8/16更慢 | 保留3作为起点，启动和稳态时间分开 |
| 样本写入与环境推进重叠 | prefill及生产callback契约仍在 | 保留，未知callback仍走安全顺序 |
| buffer复用/并行清零 | SharedRolloutBuffer仍在 | 保留样本顺序/RNG；GPU驻留仍取决于实际数据字节和显存，不保证无条件开启 |
| inference_mode/异步有限值检查 | 分布式pure PPO及共享分布类仍保留 | 保留，不误报为遗漏 |
| 卡牌查表缓存 | 新内层仍调用inference_card_table | 保留权重版本失效 |
| 地图数值拓扑缓存 | 环境spiremap缓存仍在，区别于未接上的神经编码缓存 | 保留公开地图及合法候选语义 |
| 构筑latent缓存 | 已存在，但每步仍比较完整key | 新采样运行时复用精确身份，权重/shape/环境重置正确失效 |
| 动态牌堆/遗物稀疏更新 | 有数学接口，普通PPO未接原生事件流，仍重建摘要 | 后续增益，不能声称已恢复；先适配已上线旧快路径 |
| 更新端context分组/packed FP32 | 已启用且先前完整A/B有收益 | 保留；大模型核验物理块，不改统计方法 |
| context梯度重算、旧专用Triton/FP16 | 前者实测较慢，后两者未对当前几何取得等价端到端资格 | 不强开，不把被拒候选称为遗漏的有效优化 |

源路径：`rl/policies/deck_latent.py`、`collection_graphs.py`、`shared_wholerun.py`；`rl/algorithms/accumulating_ppo.py`、`shared_buffer.py`、`packed_ppo.py`；`rl/training/native_vec_env.py`、`native_subproc_vec_env.py`；`env/core/native_observation.py`、`deck_context.py`、`spiremap.py`。

## 实施与启动验收

1. 先提交计划/状态；再分别提交带测试的原生观测适配、图执行与合批适配；最后记录资格及部署。不得把未验证草稿写成已启用优化。
2. 固定seed20261111、同模型/环境初值/随机流、完整256×768工作量做基线/候选/复测；不更新生产权重。单臂最多900秒、组合最多3600秒，超时停止诊断。不得减样本、少做Adam、截断游戏或降精度来冒充等价提速。
3. 原生差分覆盖完整公开观察/mask/动作/奖励、终局/锚点、自动重置、牌堆升级及选择候选。显式指定原生构建；缺扩展不是通过。旧checkpoint仍须加载、一步预测和环境接入。
4. 图执行覆盖混phase、不同batch/候选长度、权重变化、缓存淘汰、临时内存释放和任意重放次序；必须完整生产长度collection，不能仅测热态前推。采用预先确定的合理浮点容差，并对照动作及环境轨迹。
5. 分开记录初始化/捕获、采样、编码/IPC、前推、buffer/回报处理、更新、同步、保存和端到端时间。记录CPU树、RSS、系统可用内存、Torch allocated/reserved/peak、VRAM和设备GPU利用率；未测项填null。
6. 同工作量完整采样须明确优于未适配基线，且通过语义/资源生命周期验证，才启动COMBAT49。报告具体收益及绝对时间，不提前声称20秒达标。
7. 正式两轮均须每rank196608行/16Adam、四机模型与Adam一致、全部checkpoint manifest通过、无漏行/未处理OOM、监控新run可读；核实warmup whole-run/λ1/无搜索后继续连续训练。

## 已保护状态

COMBAT48已在第37轮/global14548992完整停止，四机exit0、全部checkpoint成员SHA通过。各机保护目录：COMBAT48 root下 `references/before-large-fast-restart/rankN/step-000014548992`。

- policy SHA256：`88217b752934ed7a5f53a59f733c838ae4123e57429c1b168ce71b16411c0ea5`
- Adam SHA256：`e62fa00dab36a013b6b9f456c2eb7a97934cd5d088fe0212e3d97ee3a82cfd2d`

COMBAT49 初始模型为 38,641,205 参数，SHA256 `ae6b2d41fdf4df0242030900220ae788e68655e585e6f1e336380f458d3362a0`，四机一致。原生适配在 `bd25d0dd`，图执行及构筑缓存适配在 `427ddedb`。最终运行源码在 rank 0 的固定种子、完整 196608 行冻结采样中用时 37.209 秒，两次未加速基线为 90.831 和 90.018 秒。完整四机验收见 [当日工作记录](../../worklog/2026-11-11.md) 的 DECK-LATENT-FAST-RESTART 小节。

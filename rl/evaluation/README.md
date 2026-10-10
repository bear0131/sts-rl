# 评估后端

`entity-combat` checkpoint自动组合规则上层与所选战斗策略，不存在神经上层
回退。可显式传 `--upper-strategy rules` 记录实验意图。真实bridge遇到非战斗
决策时等待手动操作；尚不支持通过该战斗模型自动完成整局。

统一 episode 入口：`python -m rl.evaluation.eval CHECKPOINT --task wholerun --solver native --search-mode strict --num-envs 4 --episodes 10 --timeout 1800 --output runs/local/eval.json`。

要保留可对照的局外历史（不记录战斗逐动作），加上
`--trajectory-output runs/local/<run-dir>`。入口会写出 `report.json`、
`episodes.jsonl`、`episodes.md`、`summary.md`；历史包含地图路线、问号事件的
每一步选项和资源差分、商店购买/删牌、战斗前后 HP 与牌/遗物/药水差分、奖励
候选及领取、宝箱和 Boss 遗物候选/选择。事件的 typed 候选缺失时保留事件名与
选项编号，不从本地化文本推测实体。

最新规则上层 + 回合规划搜索的固定示例：

```text
python -m rl.evaluation.eval CHECKPOINT --task wholerun --upper-strategy rules \
  --solver turn-planner --planner-worlds 2 --planner-nodes 8192 \
  --planner-beam 8 --planner-depth 16 --planner-turns 1 \
  --episodes 100 --seed 200000 --num-envs 8 \
  --trajectory-output runs/local/eval-trajectories
```

训练模型直推或搜索仍使用同一入口：省略 `--upper-strategy rules`（默认
`model`），并按需选择 `--solver policy`、`native`、`value` 或
`turn-planner`。因此评测算法和轨迹格式不会随 checkpoint 复制一份。
`--solver policy` 为直接策略；`value` 为原有 value-guided lookahead；`leftmost` 为最小合法动作。`--task combat` 保留 gauntlet 的 combats_cleared 指标。native 当前只接 whole-run 场景；旧 Python NOSL fullrun/parallel、独立 eval_combat 入口已删除。

`--solver rules` 只替换 combat 为规则策略，其余阶段使用位置参数的上层 checkpoint。
`--upper-strategy rules` 另外替换抓牌、路线、商店、营火等上层决策；未覆盖的事件显式回退到
位置参数的模型，并记录回退次数。两个开关独立，可比较固定上层和联合规则；
`--route min-elite-max-rest` 最后覆盖地图决策。默认 `--upper-strategy model` 保持原有行为。
`--combat-checkpoint LOWER.zip` 可为 policy/native/value 单独指定下层；`--compare-combats N`
沿参考轨迹取前 N 场开局，在相同完整初态上配对规则/policy/native；`--trace FILE.jsonl`
保存可见状态、选择与规则分项。信息边界、支持范围与开发/留出协议见
[战斗规则基线](baselines/README.md)。

- `runner.py`：环境常驻/重置、预算、取消、合法性、结果与统计，CLI/dashboard 共用。max_steps 截断记为 step_limit，不能当死亡。timeout 在 step/leaf batch 之间检查；native 卡死须由外层进程硬超时处理。
- `strategies.py`：DecisionRequest → 批量 Decision。新增策略实现 DecisionStrategy，传给 runner 即可；只在需要 CLI 时注册到 STRATEGIES。PhaseStrategy 按阶段及显式谓词组合策略，runner 不识别具体模型或搜索算法。
- `Scenario`：工厂、seeded reset、结果解释。固定初态/新任务实现自己的 Scenario，无需复制 episode loop。
- `rl/search/native.py`：共享 native 协议、compact leaf 编码/批量推理/回传，rollout、交互搜索、固定场景和 episode eval 共用；训练侧保留探索噪声、样本与监督目标所有权。
- `nosl/` 保留 corpus/固定场景等实验协议，`rules/` 保留现役 v7 评测；不同统计任务不能伪装成同一个胜率。

固定战斗工具在 `nosl/scenarios.py`，卡对/填充牌参数在 configs/eval/nosl_scenarios.py；strict Bash/Clash 消融统一用 `nosl.eval_full_gumbel_bash_clash` 的 mode/预算参数，旧 matrix/repeated/trials/sequence 已删除。`nosl.eval_nosl_order_sensitivity` 保留多卡对与 depth 敏感性诊断，其场景实现同样调用共享库。地图路线目标在 `baselines/routes.py`，用于 baseline 和组合策略。

一次评估复用常驻模型，指定独立 combat checkpoint 时加载上下层各一份；多环境的直接策略和 native 叶子批量前推。num_envs 表示常驻环境数；改变 batch size 可能改变浮点舍入，随机策略的 RNG 消费顺序也可能变化。参数预设在 configs/eval，结果在 runs/local。生成器调用方提前停止消费时应显式 close，以释放环境。

评测加载使用共享 `load_checkpoint(..., inference_only=True)`，将 SB3 为推理重建的未使用训练 buffer 限为2步×1环境，避免按 checkpoint 的大规模训练几何分配内存。权重、双Adam和搜索/上层配置仍从原ZIP加载；真实评测环境数由 `--num-envs` 控制。此加载选项不用于训练resume，普通共享加载默认保持保存的训练几何。

Native whole-run 评估可显式加 `--leaf-inference cuda-graph`，复用已有
`CombatLeafGraphs`；默认 `eager` 保持原路径。图后端只支持 CUDA、eval 模式、FP32
的 compact combat 模型，其他组合明确报错。缓存归各个搜索策略独立所有；同一模型上的
eager/graph 策略和 paired-combat 候选不会重配置彼此。独立 `--combat-checkpoint` 时
缓存只引用下层，上层仍走原策略。缓存容量为32，使用实际 batch/遗物宽度，不补齐 batch、不启用
AMP，不改变搜索 simulations/depth/leaf concurrency 或模型参数。最终
`effective_strategy` 记录后端配置、capture/hit/eviction 与时间统计。固定叶批的热态
收益不能替代整局计时或生产长度 collection 的 CUDA 资源生命周期验收。

Native 搜索可用 `--fixed-worldline` / `--no-fixed-worldline` 显式控制整棵搜索树是否
共享一次根部重确定的随机未来；未指定时继承 checkpoint 的配置。做消融时两边都应显式传参，
并固定 checkpoint、episode seed、搜索预算和并行度。
`--gumbel-scale 0` 显式关闭根节点Gumbel噪声；未指定时继承checkpoint。确定性评测还应
关闭`--stochastic`并固定seed、master nonce、worldline设置和并行度。
`--native-search-threads N` 让整局评测复用训练rollout的同进程`NativeSearchBatch`和
C++线程池；0保留同步兼容路径。该模式要求compact观测与leaf batch 1，并在
`effective_strategy`记录实际线程数。
生产吞吐口径还可显式使用`--leaf-amp bf16 --leaf-graph-batch-multiple 32
--float32-matmul-precision high`，分别对齐训练的autocast、图输入补齐和矩阵精度；
这些参数可能改变浮点舍入，评测结果必须记录，不能冒充FP32语义等价优化。
若本机PyTorch Inductor不能编译该模型，可加`--no-leaf-compile`；它仍调用相同的
`graph_actor_critic`并捕获CUDA graph，只跳过捕获前的`torch.compile`，manifest会记录该差异。

CLI 正常结束写 `state=complete`；episode 循环超时、异常或 Ctrl-C 时仍按 `--output` / `--output-csv` 保存已完成的局，JSON 写 `state=error` 与错误类型，并继续以失败退出。部分结果不能当作原定完整评测集；硬杀进程、加载前失败或磁盘写入失败无法靠这个 finally 边界保留结果。dashboard 的策略初始化失败也会显示 error，避免一直停在 running。

## 固定血量敏感性协议

复用原有 CLI：`python -m rl.evaluation.nosl.eval_value_hp_sensitivity CHECKPOINT --suite --write-corpus runs/local/hp-corpus.npz --output runs/local/hp-first.json`。
之后的 checkpoint 用 `--corpus runs/local/hp-corpus.npz` 替代 `--write-corpus`，确保输入逐字节相同。
默认 CPU 单线程、batch 64；`--encounters`、`--seeds` 和 `--axes player enemy` 可缩小预算。
默认五类 encounter × 三 seed，覆盖自然生成的 Acid/Spike 大史莱姆、Cultist、JawWorm、GremlinNob、Shelled Parasite，分别检查 root 和真实 End-Turn preview boundary。
`hp_sensitivity.py` 是无 CLI 的共享库，corpus 使用不含 pickle 的 NPZ，保存命令拒绝覆盖。

JSON suite schema=2 包含 checkpoint/corpus SHA-256、每条完整曲线、局部反向步数/幅度、低于零及超当前我方 HP 的偏差；异常非有限预测以失败退出。
每条曲线逐点只改一个当前 HP 坐标及其所有 scalar 别名，不推进模拟器；enemy slot 保持不变。
**monster_scalars[0] 是 presence，不能当 max HP**；未饱和的 current HP 和 HP ratio 才能还原 enemy max HP。
当前支持 scalar HP 编码、我方 max HP < 200、敌方当前 HP < 200；遇到 tokenized combat-status HP 或被截断的 HP 编码明确报错，不能静默制造不一致 observation。
语料基态无遗物、固定牌组，不是训练分布代表性样本。改 HP 可能让 intent/分裂阶段与可达状态不一致；统计是敏感性报警，**不等于真实 transition 的单调性定理，也不等于无回血条件下的总体违规率**。
解读时报告幅度、样本位置和异常片段；要验证游戏行为，另用实际 native action 路径复测。

同一入口的 `--terminal`（与 `--suite` 互斥）检查有精确答案的终局输入：
无遗物、五张 Strike，将敌人设为HP1/block0，在每个单敌人场景以我方HP1/5/20/80执行真正致死攻击，
比较模型输出与 C++ 终局 HP。用独立的 `--write-corpus runs/local/terminal-corpus.npz`
保存此语料，之后 `--terminal --corpus ...` 复用。两类 corpus 不可混用。
该协议没有结算回血，pre/post HP 相同；输出每状态预测/误差及 MAE（HP单位）。
当前仅支持单敌人、能被该固定攻击直接击杀的场景；条件不成立时明确报错。
native search 对终局本来就直接返回精确 HP，所以网络的 terminal 误差反映监督/校准情况，不能单独当作搜索返回值错误。
# 评测轨迹显示约定

轨迹 Markdown 使用中文时间线排版：地图、问号事件、战斗摘要、战斗奖励、商店、宝箱和首领遗物按发生顺序混排；战斗不展开逐动作，只记录战斗前后生命与资源变化。生命和金币使用颜色标记，牌名、遗物和药水使用不同颜色。显示名称在报告层维护，特别保证 `Combust` 显示为“自燃”，`Inflame` 显示为“燃烧”。

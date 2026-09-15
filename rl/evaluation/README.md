# 评估后端

统一 episode 入口：`python -m rl.evaluation.eval CHECKPOINT --task wholerun --solver native --search-mode strict --num-envs 4 --episodes 10 --timeout 1800 --output runs/local/eval.json`。
`--solver policy` 为直接策略；`value` 为原有 value-guided lookahead；`leftmost` 为最小合法动作。`--task combat` 保留 gauntlet 的 combats_cleared 指标。native 当前只接 whole-run 场景；旧 Python NOSL fullrun/parallel、独立 eval_combat 入口已删除。

- `runner.py`：环境常驻/重置、预算、取消、合法性、结果与统计，CLI/dashboard 共用。max_steps 截断记为 step_limit，不能当死亡。timeout 在 step/leaf batch 之间检查；native 卡死须由外层进程硬超时处理。
- `strategies.py`：DecisionRequest → 批量 Decision。新增策略实现 DecisionStrategy，传给 runner 即可；只在需要 CLI 时注册到 STRATEGIES。PhaseStrategy 按阶段及显式谓词组合策略，runner 不识别具体模型或搜索算法。
- `Scenario`：工厂、seeded reset、结果解释。固定初态/新任务实现自己的 Scenario，无需复制 episode loop。
- `rl/search/native.py`：共享 native 协议、compact leaf 编码/批量推理/回传，rollout、交互搜索、固定场景和 episode eval 共用；训练侧保留探索噪声、样本与监督目标所有权。
- `nosl/` 保留 corpus/固定场景等实验协议，`rules/` 保留现役 v7 评测；不同统计任务不能伪装成同一个胜率。

固定战斗工具在 `nosl/scenarios.py`，卡对/填充牌参数在 configs/eval/nosl_scenarios.py；strict Bash/Clash 消融统一用 `nosl.eval_full_gumbel_bash_clash` 的 mode/预算参数，旧 matrix/repeated/trials/sequence 已删除。`nosl.eval_nosl_order_sensitivity` 保留多卡对与 depth 敏感性诊断，其场景实现同样调用共享库。地图路线目标在 `baselines/routes.py`，用于 baseline 和组合策略。

一次评估只加载一份模型，多环境的直接策略和 native 叶子批量前推。num_envs 表示常驻环境数；改变 batch size 可能改变浮点舍入，随机策略的 RNG 消费顺序也可能变化。参数预设在 configs/eval，结果在 runs/local。生成器调用方提前停止消费时应显式 close，以释放环境。

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

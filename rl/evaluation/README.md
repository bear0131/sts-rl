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

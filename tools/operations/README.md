# 运维辅助

`pull_latest_ckpt.sh` 是原有远端 checkpoint 下载命令；使用前读参数/目标。脚本移动不代表本次已连接远端。

四机 NOSL 启动/恢复统一使用 `experiments/launchers/launch_nosl_replay.sh RANK`。
必须提供 `NOSL_RUN_ID`、`NOSL_WANDB_ID`；`NOSL_ARGS_FILE` 选择完整参数，
`NOSL_OVERRIDES_FILE` 是运行中的单坐标实验覆盖，`NOSL_RESUME_STEP` 选择四个
rank 都完整保存的 global step。输出为 `runs/$NOSL_RUN_ID/`，每轮独立保存
各 rank 的 learner 状态，保留三代。直接恢复前先核对 manifest 身份、step 和校验和。

`python -m tools.operations.monitor_nosl --log RANK0.log --state WATCH.json`
读取日志并输出监测建议，不直接停止训练或改配置。每半小时运行；状态放所属
run 目录。layer 5 连续三点均值相对历史最佳三点均值下降至少 50%、绝对下降
至少 0.08，且仍下行时判为明显崩溃；不会因单个低点、起步低胜率或暂时没新点调参。
每次改动后至少观察 6 次 collection 且 1 小时；之后持续低位不恢复也可提出下一步。
NaN/非有限训练指标归为运行故障，不能据此推断搜索参数有问题。

受用户授权的实验回退顺序在 `rl/training/monitoring.py`：Gumbel scale→1，
Q scale→0.1，interior value mode→0，coverage→0，paper 开关→true，
End-Turn leaf→false，simulations→128。每次只变一个参数；最终基线对应
`configs/train/nosl_paper_replay_fresh10_20260913.args`。这不是自动宣称所有配置
同义，也不把一次在线回升当成因果证明。记录前后 step、趋势与完整配置；需要归因
时保留干预前 checkpoint，并在固定 checkpoint/seed 上复测候选参数。

执行建议前：核实当前 run 身份及四 rank 健康，创建 STOP，等待所有 rank 在同一
global step 保存退出，保留该干预前 generation 的副本，写覆盖参数，清除 STOP，
同 ID 恢复。确认继续上报后用 `--record-change PARAMETER --at-step STEP` 记录。
不得仅记录建议就当作已改动；没有明显变化时保持安静，故障或完成才通知用户。

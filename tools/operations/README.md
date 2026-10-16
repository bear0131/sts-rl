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

## 四机 NOSL 本地启动入口

Windows 本地统一使用
`experiments/launchers/manage_nosl_replay_4way.ps1` 编排四个远端 rank；远端实际命令仍只有
`launch_nosl_replay.sh` 一个入口。脚本只依赖 OpenSSH 和
`%USERPROFILE%\.ssh\sts_dist_shared_ed25519`，不会保存密码。新机器首次启用时应先把该公钥加入
远端授权；密码只用于这次引导，不写入命令、文档或 Git。

先检查当前端口、构建目录和准备执行的身份：

```powershell
$ports = 22324,50418,25377,17011
.\experiments\launchers\manage_nosl_replay_4way.ps1 -Action plan -Ports $ports
.\experiments\launchers\manage_nosl_replay_4way.ps1 -Action status -Ports $ports
```

从保留 replay 的四 rank recovery 恢复时，必须使用相同的 run/W&B ID、相同 global step，且四份
`manifest.json`、`model.zip`、`learner.pt` 都完整。`preflight` 会校验 checksum、rank/world、run、
W&B 和 step，并拒绝已有训练进程或遗留 `STOP`：

```powershell
.\experiments\launchers\manage_nosl_replay_4way.ps1 `
  -Action preflight -Ports $ports -ResumeStep 2113536
```

启动并守到下一轮完整 recovery 发布：

```powershell
.\experiments\launchers\manage_nosl_replay_4way.ps1 `
  -Action start -Ports $ports -ResumeStep 2113536 `
  -TargetStep 2162688 -WaitForTarget
```

脚本默认值对应 `u40x4-20260915`。新实验必须显式换 `RunId`、`WandbId`、`WandbName` 和需要的
args/checkpoint；`ResumeStep=0` 表示从头启动，并会拒绝复用已有 recovery 的 run 目录。端口或某台
机器的 native build 变化时分别传 `-Ports`、`-BuildDirs`。只等待已有训练可运行：

```powershell
.\experiments\launchers\manage_nosl_replay_4way.ps1 `
  -Action wait-update -Ports $ports -TargetStep 2162688
```

启动器不自动同步源码或重编译 native 扩展。部署属于独立操作：同步后先在每台机器显式构建并
运行原生回归，再执行 `preflight`。这样不会因为一次启动静默覆盖远端未提交文件，也不会把
checkpoint 的恢复成功误当成 native 代码已经对齐。

HP ordering 坏事件统计的权威全局比例为
`rl.training.monitoring.hp_order_fraction(record)`，其输入是等权 all-reduce 后的
`rollout_value_gt_current_hp_ge_future_hp_matches/count`（两者均为 rank mean，
相除等于全局 sum/sum）。历史版本直接平均 rank fraction，可从保留的两项 count
修正；多 collection 合并也先分别加总分子/分母。count=0 表示无数据；运行日志为
兼容保留 fraction=0，但分析函数返回 None，不能当作完全校准。
该事件只看已完成战斗 root、update 前的预测，future HP 是结算后 HP；它不是
无回血条件概率。与胜率和固定 HP corpus 共同观察，固定探针入口见
`rl/evaluation/README.md`。旧搜索回退序列仅适用于原实验配置，换 run/基线必须
重新定义干预协议，不能对已是 strict 的新 run 机械复用。

## 长 collection 的 W&B 存活记录（OPS02）

`wandb_heartbeat.py` 用于 **Linux + W&B 0.28.1**，训练已在线运行、只在轮末上报且
W&B 在轮间误报 crashed 的场景。它通过现有 core 的 Unix socket 调用 `wandb.attach`，
由同一个 core 排序每分钟一条隐藏的 `ops/heartbeat_unix` 记录。不会新建 run、调用
`finish`、修改模型/配置/checkpoint 或强制覆盖服务端状态。启动文件已被 SDK 删除时，
从指定 core 的 fd inode 精确找到监听 socket；不猜路径或连接其他 run。

**日志语义**：W&B `_step` 和 history 行数包括健康记录，因此会增加；它们不是训练步数。
原有 `global_step`、`legacy_step`、训练指标值和以这些字段为横轴的图表不变。健康记录
不复制这些字段，也不写训练 summary；健康指标不自动生成图表。只支持训练调用
`wandb.log(payload)`、不传显式 `step` 的配置。使用显式 step 的训练不能直接使用本工具。

核对 rank0 的 PID、其 W&B core PID、run 身份和最近完整轮次后启动，例如：

```bash
python -m tools.operations.wandb_heartbeat \
  --pid TRAINER_PID --core-pid WANDB_CORE_PID \
  --run ENTITY/PROJECT/RUN_ID --log /absolute/path/rank0.log \
  --max-seconds 604800
```

工具复用当前用户的 `api.wandb.ai` netrc 登录，不接受命令行密钥。一次只启动一个实例；
启动/重启后的 PID 变化需要重新核验并显式启动，不自动跟随新的训练。使用当前训练的
Python 环境，可把这个单文件放在独立运维目录运行，无需覆盖正在训练的代码。

默认每 60 秒验证进程的启动标识、存活/暂停状态、CPU 进展、日志中的完整训练步数，
并查询服务端状态。进程退出/暂停/换代、步数回退、非有限指标、5 分钟 CPU 无进展、
25 分钟无新完整轮次、服务端终止、查询失败或总时限到达时停止。没有停止/重启训练
的权限逻辑，也不自动恢复辅助进程；网络错误后需核实状态再重启辅助进程。
25 分钟用于约 11 分钟一轮的当前训练；它是卡住检测的上界，不是立即检测所有 rank
故障的保证。首次启动依据日志 mtime 估计已有进展的年龄，后续只凭真实步数增长续期。

该工具专门补充存活记录；既有 `monitor_nosl` 的参数干预建议不是此功能的合适入口。
本地验证：明确原生构建运行 `tests/unit/test_wandb_heartbeat.py`；使用安装了 W&B
0.28.1 的 Python 直接运行 `tests/integration/test_wandb_heartbeat_lifecycle.py`，
后者是离线 unittest：父进程的两次训练记录、中间三次附加健康记录、辅助进程退出后
父进程继续写入及最终 finish 均通过真实 SDK/core 验证，不训练模型、不访问线上 run。

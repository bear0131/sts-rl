# Four-Rank NOSL Startup: Authoritative Procedure

## COMBAT01：新run开机入口

当前新目标使用同一manager的 `-CombatDeployment` 模式；本节参数优先于下文历史
TRAIN47操作。它不会读取历史主机/HP模型/旧run默认值。训练profile保持876的
384env/rank×512、128搜索、batch102400、micro8192、每轮40次同步更新；876上层
永久冻结，完整下层随机初始化、从第一次更新同时训练actor/encoder/critic。

1. 将 `experiments/launchers/combat01_876_4x5090.json` 复制到忽略目录
   `runs/local/combat01-deployment.json`。填写当前四个SSH `host/port`。默认新的
   `remote_repo=/root/autodl-tmp/combat01-876-20261021`、转发 `ring_port=29880`已提供；
   重试时须改为尚不存在的独立目录。核对 `material_directory` 与已授权的 `ssh_key`；不用旧机器地址。
   四台均须有兼容驱动的5090/32GB、约90GB容器内存、SSH和C++构建工具。
2. 用下列现有入口依次执行 `plan`、`preflight`、`start`、`wait-update`：

```powershell
& .\experiments\launchers\manage_nosl_replay_4way.ps1 `
  -CombatDeployment .\runs\local\combat01-deployment.json `
  -CombatPython D:\sts-rl\.venv-eval-gpu\Scripts\python.exe -Action plan
```

将最后的action改为相应阶段即可。`plan`只在本地生成配置审查结果，不连接机器。
`preflight`要求受版本控制源码干净，打包当前HEAD；四台新目录必须不存在，避免覆盖。
它安装锁定Python3.11环境、同源构建native，验证真实CPU生命周期/分支合同、876材料
和资源响应、Linux编译图缓存生命周期，再比较四台源码/依赖/native/teacher/随机下层
身份。预检每台至多40分钟，失败留日志和文件；修复后用新的独立目录重试。
模型、native或配置在预检后变化会阻止start，不能跳过一致性检查。

预检最低空闲量为28GiB显存、65GiB容器可用内存和30GiB磁盘；RAM同时检查cgroup
上限，不能把宿主机内存误报成容器配额。四台生成各自run专用SSH密钥，交换公钥并
检查全部12条peer连接；私钥不复制到其他机器。

`start`并行启动四个有预算supervisor。默认首轮验收最多2轮、1572864个全局槽位step、
80次同步lower更新、3600秒，seed9001000；这已真实训练整个下层，无预校准阶段。
需要长训时在预检前显式设置 `budget.rounds/max_seconds/seed`。每台提前请求协同保存，
到截止时终止进程组；硬截止只能恢复已经发布的共同代。不会无限自动重启。

`wait-update`在有界时间内等待四台实际lower更新指标及共同checkpoint，校验四份bundle
文件哈希后才宣布训练启动成功；PID存在不算通过。`status`显示现场进程出生身份、
退出原因、最后指标和共同代。W&B可选，设置新的 `wandb_id` 即启用；不设置仍有完整
JSON指标/资源记录和checkpoint。结果位于远端 `runs/<run_id>`，管理日志位于本地
`runs/local/deployment-<run_id>`。保留2代完整recovery，临时大输出不入Git。

该自动入口只创建新训练，不接收旧resume_step。之后恢复须按
[训练恢复合同](../../rl/training/README.md)选择四台共同代、各自bundle与相同目标/native，
明确有限预算后使用原distributed入口；不能把旧TRAIN47/HP状态作为COMBAT01恢复源。
未来实际5090、Linux依赖和四rank训练结果仍由现场预检/验收生成，本地CPU/4060结果
不替代它们。模板的 `launch_ready=false` 表示尚无现场四机验收；start实际依据四份预检
报告及其一致性，不能手改该标记绕过检查。

## Historical TRAIN47 procedure

This is the visible, authoritative startup record for the four-machine NOSL
trainer. The manager is the only supported production entry point:
`manage_nosl_replay_4way.ps1`.

For machines reached through different SSH gateways, pass four `-Hosts`
and four `-Ports` entries in rank order (0, 1, 2, 3). The manager uses each
host for SSH and passes the same ordered hosts to the rank launchers for
peer connections. `-Action plan` shows the mapping without connecting;
the default hosts remain `connect.bjb2.seetacloud.com` for all four ranks.

## Required Sequence

1. Stop and verify the previous run. Every rank must have no live
   `rl.training.distributed_sb3` process.
2. Run manager preflight for all four ranks. It must validate the exact run,
   checkpoint, recovery identity, and accepted native extension.
3. Let the manager clean the exact run-specific transport root once on every
   host. The manager then exports `NOSL_SKIP_TRANSPORT_CLEANUP=1` before any
   rank launcher starts.
4. Launch ranks through the manager in its normal 0, 1, 2, 3 order. The rank
   launchers may only create directories; they must not delete the shared
   transport root.
5. Confirm all four ring messages, then confirm rank 0 prints a W&B `View run
   at` URL, and only then report the training as started. Finally confirm a
   common recovery generation and a metric line on all ranks.

The cleanup and skip behavior is implemented in
`launch_nosl_replay.sh` and `manage_nosl_replay_4way.ps1`. A run is not
considered started merely because a PID file or parent process exists.

The launcher enables Python fatal-signal tracebacks and retains the actual
trainer PID in `rankN.pid`. A waiting shell records its termination in
`rankN.exit.json` and a `TRAINER_EXIT` log line, including nonzero exits and
signal termination. The exit record is cleared at each new launch; archive
the previous logs and record before restarting an incident. An absent exit
record alone does not prove the trainer is alive (the whole container or its
supervisor may also have been terminated).

For a full FP32 run, use `configs/train/entity_combat_fp32_runtime.args`
and pass `-Float32MatmulPrecision highest` to the manager. This disables
BF16 autocast and TF32 matmuls; the launcher otherwise retains its historical
`high` default. The microbatch is 5120 to fit FP32 activations on a 32 GB GPU;
the effective batch remains 102400. Other sampling, search and learning
settings are inherited unchanged from the blue reproduction configuration.

## Forbidden Startup Sequences

- Do not let every rank execute `rm -rf` on the shared transport root.
- Do not start ranks 2/3, wait for partial ready markers, and then start
  ranks 0/1. The barrier requires all ranks and this order can erase markers.
- Do not report success before W&B initialization and a first metric/checkpoint
  are observed.

## Incident Record: 2026-09-24

The first two fresh attempts were **failed startup attempts**, not training:
their shared-root cleanup race destroyed ring ready markers, so all ranks
exited before `wandb.init`. They produced no W&B runs and no recovery state.
Those attempts must not be described as successful in later logs.

The corrected run is:

- Run: `u40x4-ruleupper-v1-original-256-align-96x128-noclip-mb12288-shared-unfrozen-retry3-20260924`
- W&B: `u40x4orig256shared24c`
- URL: `https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/u40x4orig256shared24c`
- Source: original step-0 checkpoint, not an old run's final parameters
- First common checkpoint verified: `step-000000049152`

## Manual Health Check

Every 30 minutes, perform a direct human check using manager `status` and a
direct `df -h /root/autodl-tmp` query on all four hosts. Record process state,
latest common checkpoint, latest metric, free space, and any traceback in the
dated worklog. Do not replace this with a resident monitoring script.

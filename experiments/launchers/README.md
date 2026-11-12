# Four-Rank NOSL Startup: Authoritative Procedure

<a id="autodl-pro"></a>
## AutoDL 北京 Pro 实例获取

`autodl_beijing.py` 使用官方 Pro API 申请资源，与下面管理已租机器的训练入口分开。
只需 Python 3.11+ 标准库，不依赖 Torch，不启动训练，也不操作现有训练实例。
本工具不支持普通容器的网页/私有接口。官方核对日期：2026-11-12。

- [产品区别](https://www.autodl.com/docs/instance_pro/)：普通容器不支持创建管理 API，Pro 明确支持自动创建。
- [服务协议](https://www.autodl.com/docs/agreements/)：禁止未经授权访问、干扰正常服务等行为。
  本工具走已公开支持的接口；这不是对所有自动抢购方式的法律保证。
- [API 合同](https://www.autodl.com/docs/instance_pro_api/)：需实名认证及开发者 Token，
  固定 `https://api.autodl.com`，仅传 `data_center_list=["beijingDC2"]`，仅按量计费。
  不保证可挑选北京子区、独立物理机、CPU 型号或内存容量，也不保证四台同时获取成功。

**价格限制：**公开 create 接口没有限价字段或创建前报价接口。因此不提供虚假的价格上限；
执行必须明确接受平台当前按量价。系统盘扩容、关机后的存储等费用以平台为准。
等待超时、脚本退出或获取成功都不会自动关机/释放已拿到的实例。
Pro 使用网络系统盘，其存储容量、性能和计费与普通容器不同，不能直接套旧部署路径。

从仓库根目录执行（也可将脚本和配置单独复制到其他目录）：

```powershell
New-Item -ItemType Directory -Force runs/local/autodl-acquire
Copy-Item experiments/launchers/autodl_beijing.example.json runs/local/autodl-acquire/config.json
python experiments/launchers/autodl_beijing.py --config runs/local/autodl-acquire/config.json
```

默认只离线预演，不连接 AutoDL、不读取 Token。示例为 **4 台单卡 5090-32G**、
驱动 CUDA 下限 13.2、系统盘不扩容；仅是可编辑模板，不代表用户已确定规格。
填写真实 `image_uuid` 后才能执行。通过以下只读命令列出私有镜像；公共新镜像 UUID
需从控制台/官方支持取得，不把文档中的老 Torch 镜像当作 5090 训练环境。

```powershell
python experiments/launchers/autodl_beijing.py --list-images
python experiments/launchers/autodl_beijing.py --config runs/local/autodl-acquire/config.json --execute --accept-current-price
```

程序隐藏询问开发者 Token，也支持从进程环境 `AUTODL_TOKEN` 读取；不要把 Token 填到
配置、命令参数或 Git。Token 来源：控制台 → 账号 → 设置 → 开发者 Token。
脚本不保存 Token、SSH 密码或完整实例详情，HTTP 重定向不跟随。

配置字段：`job` 是稳定任务名；`instances` 是实例数，`gpus_per_instance` 是每台卡数；
`gpu_spec_uuid=5090-p` 对应 5090-32G 性能型。`cuda_v_from` 是驱动支持下限，不会安装软件。
`interval_seconds` 默认60、最低60；`max_wait_seconds` 默认3600，限制本次等待；
`max_create_attempts` 默认60，限制同一任务累计创建请求数，包括失败请求。
一台成功后也会等待下一个间隔才申请下一台。60秒是本工具的保守选择，不是官方公布的限流阈值。

创建意图先写入 `runs/local/autodl-acquire/<job>.json`，再发送请求；成功后保存 UUID。
原命令重跑会核对完整实例列表并继续缺少的槽位，不替换已删除/停止的实例。
状态目录有操作系统互斥锁，崩溃后锁自动释放。**同一任务必须使用同一账号、配置和状态目录；
不要多开不同副本或删除台账来“重试”，平台 API 没有公开幂等键，跨机器副本不能保证去重。**

只有 `code != Success`、`data=null` 且消息恰为“暂无库存”“库存不足”“暂无可用库存”的
明确缺货拒绝会自动等待。官方未公布缺货错误码，以上是保守消息识别，尚未用真实缺货响应验证；
其他文字、鉴权/余额错误、HTTP 429/5xx、超时、格式改变一律停止，不进行模糊重试。
创建响应丢失后，原命令重跑只在唯一同名实例出现且地区/型号/卡数一致时认领；
即使列表暂时为空也不再次创建。未知拒绝会保留 rejected 状态，请核对控制台或向平台确认，
不要直接清空台账。列表分页读失败也不会继续下单。

输出“已创建”只表示平台返回了实例 UUID 并完成列表核对，不保证镜像已加载、SSH可达或训练可恢复。
状态从列表如实输出；已创建但仍在启动的实例在控制台确认。退出码0表示数量已创建，
2表示时间/累计尝试用尽，1表示需核查的失败，130表示用户中断；任何退出码下都可能已租到部分机器。

测试（不联网、不使用真实 Token、不产生费用）：

```powershell
python -m unittest discover -s tests/unit -p test_autodl_beijing.py -v
```

单测覆盖地区/数量、缺货等待、部分成功后恢复、超时后核对、进程中断、重复实例、
预算上限、锁和凭据输出限制；模拟测试不替代真实账号/API及付费创建验收。

## 统一上层的持续训练

`unified_upper_420.json` 记录 `combat31-unified-upper-420-r1` 的部署身份、初始化来源、
主机与持续训练合同，使用原有 `launch_nosl_replay.sh` 的 `NOSL_*` 环境变量入口；该JSON
是部署记录，不是 `-CombatDeployment` 的COMBAT01 schema，也没有新增启动脚本。
模型与训练参数分别在 `configs/train/unified_upper.json` 和
`configs/train/unified_upper_420.args`。原生扩展、初始模型与部署源码须核对记录中的SHA。

此训练两轮资格通过后曾设置有限100轮，用户随后明确取消总轮数上限。
在第24轮/global5717291完整保存后普通恢复，`NOSL_OVERRIDES_FILE` 使用远端
`material/continuous.args`（`--updates 1 --continuous`）；`updates=1`是已有持续模式
的解析哨兵，不是一轮上限。各rank加载自己的完整bundle，模型/Adam/环境接续。
每轮415s失败保护继续保留，超时轮不算完成，不自动重启。
运行源码仍固定为部署记录commit，原args文件由已记录SHA的override覆盖；当前仓库
的训练args也已改为continuous。监控使用 `--max-hours 0` 持续采集；新run及旧历史
分别保存。文档和本地监控提交不替换远端运行代码。
指标暂用W&B offline并保留远端本地日志；完整耗时与恢复约束见
[训练合同](../../rl/training/README.md)，一次性验收证据位置见
[UPPER-UNIFIED-420工作记录](../../worklog/2026-11-03.md#upper-unified-420实现与训练安排)。

## COMBAT01系列：876冻结上层的开机入口

当前新目标使用同一manager的 `-CombatDeployment` 模式；本节参数优先于下文历史
TRAIN47操作。它不会读取历史主机/HP模型/旧run默认值。训练profile保持876的
384env/rank×512、128搜索、batch102400、micro8192、每轮40次同步更新；876上层
永久冻结，完整下层随机初始化、从第一次更新同时训练actor/encoder/critic。

归一化外层value实验使用 `configs/train/combat_normalized_outer_return_876_4x5090.json`，
保持512步/128搜索与qnorm关闭。它是独立目标、随机下层和新replay，不能resume旧COMBAT16；
剩余reward单位及外层耗时指标见 [rl模块合同](../../rl/README.md)。部署源码身份以
`deployment-source.json`为准；现场GPU和完整采样验收以新run记录为准。

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

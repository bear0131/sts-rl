# 上层连续训练交接

记录时间：2026-11-03 11:45（Asia/Shanghai）。用户要求“把当前的任务记录下来推上去，我交给其他会话”。本文件是交接快照；实时身份和下一复核点以本地 ledger 为准，TODO.md 仍是唯一任务清单。

> **11月4日04:47续接：** COMBAT37同PID/source b12c01d4/native f35674a6 continuous，最新只读42/global23206345；完整恢复验收仍30，另已复制核验完整诊断源40/global21983219/Adam1978/816/320（critic-stability-source40-protected.json），未暂停。固定40三10窗22.4651/24.2332/26.4622层、30/56/145胜、0剔除，未停滞；共同训练wall近期9600局为79vs历史B70胜、26.8160vs26.8366层，仅接近。下一50固定21–30/31–40/41–50及新增近期参考。32/39同批λ MSE更新后+18.697%/+48.213%，已登记critic_stability_diagnostic：先修离线probe候选criticLR未写Adam的缺陷、CPU验证/隔离源码；50复核后协调完整51停界保护50/51，再CPU/CUDA门禁与源40/51各5臂、900/960s。尚无STOP/新诊断启动；预算、门槛和恢复计划读协议，不热改生产。既有probe均未覆盖criticLR，因此不因该缺陷作废。现有资格/恢复/10–40报告勿重跑，旧训练和Windows调度保持停。下段04:00记录是此前阶段。

> **11月4日04:00续接：** native修复28/29生产资格及普通恢复30全状态均已验收；30/global15855593/累计Adam1422/580/240、各480环境/RNG通过，230.459秒、1920自然局/0剔除。COMBAT37仍continuous无总限，新PID以ledger/native-capacity-continuous-launch.json为准，Python b12c01d4/new native f35674a6。固定30三10窗楼层19.32354/22.46510/24.23323、胜局8/30/56，未触发停滞；新旧重叠字段已核。下一40固定11–20/21–30/31–40、六5/共同完整max4墙钟，按ledger推进。18臂整局留出无epoch候选过门槛，保持2/4；同57600局完整墙钟比B60少37.796%，但累计楼层/胜率仍落后，不能声称更强。各资格/启动/恢复/30报告均已有receipt，勿重复。原失败28与26/27/3/B192/trial30保留；heartbeat b已回读核验，旧Windows调度不恢复。

> **11月4日03:12故障更新（历史）：** COMBAT37第28轮采集遇HEADBUTT100种/172张超过固定96槽，四机exit1，未重启。完整26/27（最后global14073832）模型/上下层Adam/RNG/480环境已保护，3/B192/trial30仍保留。先处理TODO SIM-SMALL-NATIVE-CHOICE-CAPACITY，不执行旧31停界/launch；更新次数诊断源已在运行前修订为3/27。学习已复核20、固定30待完成。以下continuous状态为故障前历史。

> **11月4日续接说明：下文保留原Windows历史，不是当前启动指令。** 当前Mac同分支；原B完整停192/global71059696且191/192全状态保护，B190复核完成。B192冻结1/1.5/2/2.5/3/1倍重复及micro256/512/1024/256全部验收，已选1920自然局/micro1024。用户最新USER-b1cb…明确测试后从头新训，四机已从原统一随机上层/COMBAT30-203下层初值、空Adam/计数0启动 `combat37-upper-fresh-lambda097-r1`，root `/root/upper-throughput-20261104`/source b12c01d4；critic整局frozen-lambda .97、actor6144×2/LR.00012、critic32768×4/LR.0001。完整1–2轮资格与普通resume3全状态均已通过，现continuous无总限；不得重复资格/launch，下一10/20描述及30后固定复核；实际阶段以ledger/receipt为准，已有launch禁止重复。新训练用censored_games=error、415失败/420门禁、不削搜索或截局。新lineage用candidates/review_fresh_lambda.py，10/20描述、30及每10固定三10/六5/共同完整max4墙钟；不能混B960窗口。原critic1/配对终点评测暂缓，A/C/C35/COMBAT30保持停。用户授权共享worklog“用户发言 · USER-…”读取并完整回复；顶层monitor已指新run真实origin0/0，Windows8773由原会话负责。唯一heartbeat b，旧Windows巡检保持暂停，健康不重启或重复资格。

## 接手先做什么

1. 在 `D:/sts-rl/runs/local/upper-unified-worktree` 工作，分支 `codex/upper-unified-420s`；原 `D:/sts-rl` 的用户改动不得动。先读根 `AGENTS.md`、`TODO.md` 的 `UPPER-UNIFIED-LEARNING`、`worklog/2026-11-03.md` 最近相关节和本文件。
2. 读 `runs/local/upper-unified/active-training.json`（忽略文件，下文简称 ledger）、`experiments/launchers/unified_upper_comparison.json` 和 `http://127.0.0.1:8773/api/state`。确认 run、PID、更新时间、最新完整轮与错误。健康时不重启、不重复资格测试。
3. **当前 B 正在无总轮数上限持续训练，下一学习复核是 B130，不是停止点。** 已完成 B60/B70/B80/B90/B100/B110/B120，不重复。若接手时已超过130，先补固定101–110/111–120/121–130窗口，不能事后挑表现好的窗口。
4. 用户交接后，本会话15分钟自动跟进 `automation` 已设为 **PAUSED**，防两个会话并发操作。远端训练和本地监控均未停。接手会话如继续自动跟进，应接管唯一巡检，勿恢复指向旧会话的重复自动化。旧 `combat18-playout` / COMBAT30巡检也仍为PAUSED，不恢复。

可用的本地只读起点（PowerShell）：

```powershell
Set-Location D:/sts-rl/runs/local/upper-unified-worktree
git status --short --branch
Get-Content runs/local/upper-unified/active-training.json
$trainingState = Invoke-RestMethod http://127.0.0.1:8773/api/state
$trainingState | Select-Object run_id, updated_at
$trainingState.ranks | Select-Object rank,pid,alive,errors,connection_error
$trainingState.rows[-1] | Select-Object rollout_round,global_timesteps,iteration_wall_sec
```

## 用户意图与不能改变的条件

- 优化单位训练墙钟内的胜率和推进效果；420秒只是完整单轮上限，快轮不是优胜标准。正式训练无100轮或其他总轮数上限；有限实验须有固定seed、游戏/轮数及超时预算。
- 排除上层分支续玩。不得为了过限截短游戏、减少搜索预算或把失败轮记为成功。415秒watchdog是失败退出保护，不自动重启失败run。
- 自主诊断、设计和执行有界实验已获授权，无需再让用户选参数；普通波动不频繁改参。任何切换先协调完整边界保存、确认四机停止，再核模型、上下层Adam、RNG与环境连续性。已有run不得覆盖、重置或从initial重来。
- 仅显著发现、配置切换、实验完成、故障或需要用户行动时通知；健康例行结果保持安静。每个可验收阶段更新日志/必要的TODO，审查diff/check后commit并push origin对应分支；记录提交不等于生产source。

## 现役身份与交接健康快照

| 项目 | 值 |
|---|---|
| run | `combat32-unified-upper-lr12-r1`，B |
| 远端目录 | `/root/upper-unified-20261103` |
| SSH主机 | `connect.bjb2.seetacloud.com`，root |
| rank0/1/2/3端口 | 42228 / 41309 / 49038 / 33203 |
| 原trainer PID | 148933 / 120679 / 106655 / 161772；以后以ledger和创建时间复核 |
| 生产source | `61f4f94d2709fe176e624c013546ae254997d178`，远端`deployment-source.json` |
| native SHA256 | `b3a058e42f4908fbb9a8027b31785395e0ab5ba1bbcf5d95420e5d7bdaa58e7e` |
| native目录 | `/root/autodl-tmp/combat-small-playout-20261030/sim/sts_lightspeed/build-small` |
| W&B / ring | offline `combat32lr12a` / 29932 |
| 本地SSH key位置 | `C:/Users/Tony2/.ssh/sts_dist_shared_ed25519`；不得复制密钥进Git |

交接API快照为完整126轮/global41536618、120960自然终局、0剔除，累计日志actor3836/critic7672/lower1008次；随后只读四机核实已共同完整127轮。四个原PID及创建时间保持、CPU继续增加，无exit/STOP/连接或日志错误。所有完整轮最大228.0042秒，系统盘最少可用17,076,867,072字节。这里是健康检查，不是重复完整checkpoint验收。

本地原始证据（均在忽略目录、不上传大输出）：

- `runs/local/upper-unified/comparison/handoff-20261103-state.json`
- `runs/local/upper-unified/comparison/handoff-20261103-health.json`
- ledger的`last_health`指向上述健康快照；`last_verified`仍为已验收的恢复31轮，二者不要混淆。

远端每rank日志：`runs/<run>/rankN.log`；完整耗时在每条 `ROUND_COMPLETE {json}`。正常supervisor的bash命令也带训练参数，不能误判为第二个trainer。只读检查原PID/创建时间、CPU增量、全部新增完整轮、错误、`rankN.exit.json`、`STOP`、`runs/ACTIVE`及系统盘空间；不要读到“进程存在”就等同健康。

本机已有只读SSH辅助库 `remote.ssh(rank, bash_script, timeout)` 位于 `C:/Users/Tony2/.codex/worktrees/combat16-rollout-cpu/sts-rl/runs/local/outer-value-normalized-20261028/remote.py`。使用前读代码；它是本机依赖，不在本分支Git中。可用标准SSH替代。

## 配置、保存点与监控

- 上层192宽/3层/6头/16 latent；actor与critic及embedding独立。actor LR0.00012、critic LR0.0001、clip0.2；actor冻结旧V的GAE（gamma1/lambda.97）2遍，critic完整自然回报MC 4遍。
- seed20261103；每轮960新完整游戏、每rank240环境；目标全局优化batch6144，micro256；均衡分批后的实际batch可略小于6144，必须核真实样本与Adam次数。FP32、AMP/TF32关闭。
- 下层从原COMBAT30第203轮小网权重开始；搜索/奖励/剔除规则与初筛一致。每轮上层全样本2+4遍；不根据MC训练loss选型。
- B从30轮/global7762009的完整保存点恢复continuous。不可变参考在各机 `runs/combat32-unified-upper-lr12-r1/references/trial30/rankN/step-000007762009`，保留模型、上下层Adam/RNG/240环境及计数。rolling recovery仅保留少量最新包，不要删除参考包。
- 已实际验收恢复31/global8051411：240环境/rank恢复、模型和上下层Adam一致、累计734/1468/248次、30→31小网刷新、样本2+4遍、最慢154.826秒。证据 `comparison/B-continuous-accepted.json`、`B-continuous-audit-rank*.json`。不重复stage/launch/audit。
- 实际运行入口是 `experiments/launchers/launch_nosl_replay.sh`；恢复时通过已有完整包与 `material/continuous.args`（`--updates 1 --continuous`）覆盖，1是continuous解析哨兵，非一轮停止。当前无总游戏/步数预算。
- 监控 `http://127.0.0.1:8773/` / `/api/state`，本地PID37092（以ledger/CIM实际身份为准），`D:/anaconda/python.exe -m apps.training_monitor`，30秒只读轮询、max-hours0。配置 `runs/local/upper-unified/local-monitor/config-B-continuous.json`，输出 `state-B-continuous.json`；现役B连续曲线，11份对照含A/C各30轮，原9份历史SHA保持。健康不重启。

**容易误用的旧入口：**

- `experiments/launchers/unified_upper_420.json`保留最初A/COMBAT31部署身份，不能拿它直接启动现役B。B身份以ledger、comparison协议arms和已执行续训记录交叉核对。
- `runs/local/upper-unified/comparison.py`的launch用于全新30轮初筛；audit按30轮停止验收。不要用于现役continuous。
- `comparison/continue_selected.py`是B30→continuous的一次性stage/launch/audit，已经执行完；重跑会使用旧选择点，不能用于下一次恢复。
- `comparison/compare_completed.py`为已结束30轮A/B/C对照；`learning_review.py`为旧25轮分析，均不能直接冒充B130复核。
- `local-monitor/restart.ps1 -Arm B-continuous`会停止并重启监控；仅确有故障/身份切换且读过代码才用。本次未运行。
- 其他忽略目录脚本可能硬编码A或历史阶段，先读再用；不要盲目prepare、launch、reset、finalize。

## 已完成的实验及最新学习结论

A/B/C均已完成30轮×960起始局和四机完整结束审计；精确初始控制、空优化器、seed/source/native/下层规则一致。A：192×3，actor LR6e-5；B：与A精确初始相同，仅LR1.2e-4；C：256×4×8/32 latent、与B相同LR，容量包而非只改width。

- `comparison/AB-review.json`：B末两个5轮窗口楼层比A高2.089/2.007，共同时间不退，早期初筛通过。
- `comparison/BC-review.json`：C末两窗比B低.266/.315层，共同时间未改善，30轮多耗5.50%；没有采用扩大模型的证据。保留A/C，不宣称大网长期更差。
- 单训练副本、稀少早期胜局、相同seed不等于相同行动/RNG轨迹；旧COMBAT30初态不同，只作背景。原COMBAT30已停203轮，不能恢复。

最新完成的 `comparison/B120-review.json`：

| 完整窗口 | 游戏 | 胜局 / 胜率 | 平均楼层 | 四机完整轮墙钟累计 |
|---|---:|---:|---:|---:|
| 91–100 | 9600 | 551 / 5.73958% | 31.16854 | 2009.6489s |
| 101–110 | 9600 | 758 / 7.89583% | 32.44688 | 2059.4536s |
| 111–120 | 9600 | 905 / 9.42708% | 33.21135 | 2136.1497s |

均0剔除；相邻楼层+1.27833/+.76448、胜率+2.15625/+1.53125个百分点。末两5轮433/472胜、33.10417/33.31854层。共同2009.6489秒内仅纳完整轮，10/9/9轮对应551/666/815胜、31.16854/32.34595/33.15660层，单位时间效果仍改善。继续原B，尚未触发任何新候选。

三窗口真实上层样本999457/1057604/1092932，actor Adam336/354/364、critic672/708/728、lower均80；样本使用严格2+4遍。训练中第二遍KL均值.00124518/.00128178/.00139887。1–120累计上层样本10703309、actor3608/critic7216/lower960。以上是同一持续训练轨迹趋势，非独立配置因果或留出集评测。

## 下一次复核如何完成

下一固定B130窗口：101–110、111–120、121–130；之后每10完整轮移动一次。协议post_screen的`next_review_round=60`是原始首次观察点，**实时下一点以ledger.comparison.next_learning_review_round为准**。

1. 保存新鲜API和只读四rank日志证据至comparison下唯一文件名，保留原始结果，不覆盖旧报告。
2. 对每轮用round/global_timesteps与四rank `ROUND_COMPLETE`匹配；取该轮四rank `wall_seconds`最大值，再累计。必须含保存/同步日志入队，排除启动/暂停/空闲；UI的rank0 `iteration_wall_sec`仅辅助，不能代替四rank最大值。
3. 同游戏数三个10轮窗口、六个5轮窗口；共同墙钟取三个10轮窗口累计耗时的最小值，分别从各窗口起点只纳预算内完整轮。游戏/剔除/样本/实际Adam求和，wins逐轮round(win_rate×games)，楼层按games加权；KL等描述指标沿旧报告`statistics.mean`口径。重叠窗口应与旧报告逐字段一致，只有浮点派生比值容许1e-14相对误差。
4. 核每轮960自然局，异常剔除如实报告并调查；上层sample uses=2×/4×样本、epochs=2/4、无KL早停。报告wins、floor、excluded、真实samples、Adam次数、KL及完整墙钟。不用MC loss或一个罕见胜局定论。
5. 若两个相邻10轮增量都满足楼层<.5且胜率<.5个百分点，或持续退化，先查数据/优化；普通单窗回落不换参。无触发则保持原配置，保存`B130-review.json`，ledger/TODO/log同步下一140。候选的最终采用标准另见协议，不把观察门槛当采用门槛。

## 论文方向及未完成事项

参考 https://arxiv.org/html/2603.06009v2 ，已登记协议 `post_screen.paper_informed_review`。最近解释与代码定位在 `comparison/paper-application-20261103-1134.json`、worklog“再次讨论PPO论文如何用于当前训练”。

- 960新局、6144优化目标batch、micro256是不同量；更多同局决策不是更多独立游戏。当前`kl_epoch_2`是各minibatch更新前、变化中的策略的采样动作估计均值，非整次actor更新后的固定探针KL，不能仅凭低值加LR。
- 尚未实现/测量：固定探针的旧→最终完整categorical KL、按phase和多合法动作分组/尾部统计、冻结上下层及critic的独立cohort梯度方向诊断。当前只保存采样动作old_logp，要测完整KL须预存探针全分布或保留旧actor；单合法动作KL=0应单列。任何诊断先登记seed/步数/超时，不能凭空声称有证据。
- 已登记但未排队的单变量候选：支持更新不足时actor LR.00024对照.00012；支持噪声/过大策略移动时clip.1对照.2、LR仍.00012。一次只选一个，不由低训练KL自动触发。
- 每个候选两臂从同一核验完整checkpoint（上下层Adam、RNG、每rank240环境及cohort计数）续训，seed20261103，20轮×960局/臂，完整轮墙钟最多8400s/臂，420上限/415失败保护，四机顺序互斥。共同时间与同19200局比较；末两5轮持续达到+.5pp胜率或+2楼层，且无明显胜率/共同时间退化才采用。单副本仍属探索。
- 960→1920更大新经验仍为设计方向，未实现/资格。1440只是最近讨论的较保守资格起点示例，并未另行登记或排队。普通resume拒绝改变n_envs/cohort，绝不能reset-env绕过；多波采集需冻结行为权重、不重复seed、完整回报/缓冲区恢复及下层更新频率口径。先另行有界协议、语义测试及四机两轮真实完整GPU资格；不能把两轮普通训练拼成一轮或削游戏预算。当前约214秒均值不能保证翻倍仍小于420秒。
- PPO-EWMA也是未实现的后续候选，须正确区分行为策略重要性修正与平滑参考策略正则，并保存参考权重；不是简单平均现役模型。扩大模型/缩小优化batch均是独立假设，不同时改多项。

## 文件和同步边界

本交接文档、TODO、worklog和纯数据协议随Git提交推送。`runs/local/upper-unified`为忽略目录，包含ledger、运行快照、报告、一次性助手和监控配置，**不在Git中**；同一电脑的接手会话可直接读取。若换机器，先核实这些证据的可用性，不能以缺本地ledger为由重新初始化远端run。远端checkpoint和原日志保留，未复制密钥、大checkpoint或大输出进Git。

原自动化ID、暂停状态和接手标记也已写入本地ledger；自动化配置位于 `C:/Users/Tony2/.codex/automations/automation/automation.toml`，通过应用工具修改，不手改其调度。生产source仍为61f4f94d，Git记录提交只负责交接文档。接手会话自行核实当前进度后继续，不要把本快照127轮当成需要回滚到的保存点。


## Mac 接手补充（2026-11-03 11:58，Asia/Singapore）

本文件上文保留原Windows交接快照；当前管理工作区为 `/Users/xb.zhong/my/sts-rl`，同一 `codex/upper-unified-420s` 分支。用户提供训练机认证后已核实四机现役B到完整130/global43180750，原PID与本次两次读取的创建时间一致、CPU推进、source/native及核心入口文件保持。B130固定窗口复核已完成，下一140用111–120/121–130/131–140，不是停止点；结果见 [Mac接手日志](../../worklog/2026-11-03.md#upper-training-mac-takeover-b130)。

本机 `runs/local/upper-unified/active-training.json` 从新鲜远端证据重建，并标明旧31轮完整包验收来自Git交接而非本次重验。只读采集使用同目录 `collect.py`/`remote_snapshot.py`，分析使用 `review.py <health目录> <目标轮>`，原始四rank结果与B130报告放 `comparison/`；这些仍是忽略文件，不随Git跨机同步。旧Windows ledger、B120原JSON与11份监控历史不在本机，不能声称原报告逐字段一致或本机8773已恢复。未来重叠窗口可与本机B130及之后报告逐字段核对。

本机免交互SSH使用 `/Users/xb.zhong/.ssh/sts_upper_monitor_ed25519`，仅在标准SSH目录保存私钥；对应公钥追加到四机authorized_keys，带`restrict`，保留既有条目。端口与远端路径不变，普通连接应显式指定该密钥、BatchMode和StrictHostKeyChecking；无需依赖临时控制socket或保存密码。

当前聊天15分钟heartbeat `b` 已创建并核实ACTIVE。旧Windows `automation` 按交接保持PAUSED，未恢复；不要再创建并行巡检。原Windows监控没有被停止或修改，但本机未直接核实其当前状态；现役健康以新鲜SSH证据为准。运行配置、checkpoint、搜索与奖励均未修改。

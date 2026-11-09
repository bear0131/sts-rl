# 用户交互入口

`python -m apps.dashboard`：浏览器 checkpoint 检查；静态文件在 dashboard_static/。
`python -m apps.watch CHECKPOINT`：终端逐步观看；`apps.play_gauntlet`：gauntlet 交互。
旧 apps.dashboard / apps.watch / apps.play_gauntlet 入口继续可用。训练和算法不能反向依赖 apps；UI 只消费 env/policy/evaluation。
日志、截图和 dashboard session JSON 写入 runs/local，不放仓库根目录。启动前查看 --help，自行指定端口/结果路径。

### 远端训练实时曲线

`python -m apps.training_monitor --config runs/local/monitor/config.json --output runs/local/monitor/state.json`
提供只读的 `http://127.0.0.1:8773` 窗口，默认每30秒通过SSH同步各rank状态及rank0的完整轮次指标，
默认48小时后停止采集并保留展示。`--poll-seconds` / `--max-hours` 可显式调整；
`--max-hours 0` 持续采集，直到本地服务关闭。
它与需要模拟器、Torch及checkpoint的交互dashboard分开，纯标准库运行，不加载模型或触发评测。
配置示例（实际地址、run和密钥路径放本地配置，不写入页面）：

```json
{"host":"root@example.com","ports":[2200,2201,2202,2203],"ssh_key":"/path/to/key",
 "root":"/remote/repository","run_id":"my-run","iteration_steps":786432}
```

远端使用已有`.venv/bin/python`读取日志、PID及保存目录。SSH需预先配置可信host key，
所有远端操作只读。指标按global timesteps去重，半行留到下次读取；日志轮转从头读并保留已有历史。
断连不会清空曲线，会显示连接异常；本地服务停止后页面会标记断开，不把缓存当实时状态。
完整游戏模式按 `rollout_round` 显示轮次，胜率/楼层使用本批自然结束局，剔除数单列；
历史固定步数模式保留原400局窗口，不能和完整批次混为同一统计口径。
上层loss/KL取各自最后一遍按样本加权的汇总，支持actor两遍、独立critic四遍；
actor与独立critic的Adam次数分别展示。整轮耗时取rank0的`iteration_wall_sec`
（含保存、不含随后W&B客户端排队），不当作四rank最慢墙钟。以上都是训练统计，非独立评测。

### 战士选牌排名

打开 `http://127.0.0.1:8765/?view=ranking`，75 种战士牌（含基础牌）与
「跳过」按从优先到放弃排列。拖动左侧手柄、点击箭头或输入全局名次；可搜索卡名/效果
和筛选类型，筛选不会改变未显示的卡牌顺序。初始顺序来自卡牌目录，不是推荐排名。
中文效果中的 `普通/升级` 数值复用现有卡表，英文 ID 用于区分同名译文。

点击「保存排名」写入本机 JSON；支持撤销、导入草稿、导出 JSON、读取已保存文件，
未保存离开有提示。独立使用不需要 checkpoint：

```text
python -m apps.dashboard --ranking-only --card-ranking-file runs/local/card-ranking/ironclad.json
```

默认路径是当前 checkout 的 `runs/local/card-ranking/ironclad.json`；有多个 checkout 时
可指定同一个绝对路径。原子保存、版本/完整唯一 ID 校验及 revision 冲突检测防止半文件和
旧页面覆盖新排名。一次只由一个 dashboard 进程管理该文件。

文件契约为 `schema_version=1 / character=IRONCLAD / upgrade_policy=shared / order`；
`order` 包含全部 canonical 卡 ID 与 `__skip__` 各一次。`env.search.card_ranking.CardRanking`
的 `from_dict(json.loads(path.read_text()))` 加载它，`choose(legal_card_ids, can_skip=True)`
返回输入候选中的索引或 `None`（跳过）。普通/升级共用卡种排名，重复卡选首个候选；未知卡
明确报错，由调用方决定回退策略。调用方只传合法候选并自行映射回动作 ID。
页面试选使用这个相同函数；这里只提供奖励选牌规则原语，未改变训练或 Watch 的策略，
也不把奖励排名套用于删牌、升级、商店或战斗内选牌。

Dashboard 的 Watch 页可在 `ironclad-overworld-v3` 规则上层（默认）与 checkpoint 模型
上层之间切换。规则复用训练 worker 的 `overworld_rule_decision()`；未覆盖的未知画面显式
回退 checkpoint masked argmax，因此不是完全无模型的整局策略。战斗下层可独立选择
checkpoint 模型/所选 NOSL 搜索，或 `ironclad-turn-v8-awakened-phase` 回合规划器。规划器
使用 native-v1 完整动作覆盖出牌、目标、选牌和药水；前端固定使用历史 R10 配置
`worlds=2,node_budget=8192,beam_width=8,max_depth=16,lookahead_turns=1,master_nonce=20260918,leaf_profile=legacy-v8`。
它不调用 checkpoint 决定战斗动作；Runic Dome 等不支持的隐藏意图场景按版本契约回退
`rules-ironclad-v3`。页面显示的 checkpoint value/prior 仅用于对照。
联合模型的 `joint-v2` / `autoregressive-v2` Watch 也支持此规则下层，复用训练环境的
`rulebase_turn_decision` 映射；多选逐个 token 后确认，不绕过原子动作契约。
撤销/seek 保留当时规划的完整子集；已选了一部分牌时须完成多选再切换规则、模型或手动控制。
查看 TRAIN42 时选择「模型上层」和「回合规划下层」；其下层网络未训练，不应用于打牌。

Watch 的 native 搜索树在每个已展开节点标题中单独显示“模型直出 V”：它是该节点观测
第一次送入 checkpoint 模型时直接得到的 value，不含搜索回传、样本平均、Q 变换或
Gumbel 分数。未实际经过模型前向的终局节点显示“未评估”，不以 0 代替。

Watch 的玩家 `power_flags` 是通过原生 `BattleContext.player_power_flags()` 读取的
布尔状态显示字段，覆盖腐化、壁垒、混乱等；它不写入旧checkpoint的数值 `powers`
或policy观测。使用旧扩展时该额外字段不可用，需要重新构建本地扩展才能显示。

Watch 可选择 Act 1/2/3 并“快进 Boss”。目标 Boss 前的战斗由模拟器直接结算；推进期间把
HP 和最大 HP 固定为开局值并清空药水，但保留牌组、遗物、金币、奖励及地图进度。目标
Boss 是真实战斗，停在所选下层策略的第一手决策。此功能是查看指定 Boss 打牌的路线
沙盒，不是可用于胜率统计的自然整局评测。

当前专项训练的开场由 `--boss-opening-dataset <dataset.json>` 显式接入 Watch。
“训练同款二层”选择 Boss 和数据集中的 source seed 后，复用
`env.core.boss_opening.BossOpeningPool` 的固定开场工厂；原生路线重放必须通过数据集
opening SHA 校验。它保留训练开场的牌组、遗物、HP、药水、随机数状态；此前战斗被
跳过，是专项训练的合成入场分布，不代表自然爬塔到达 Boss 时的资源。
点击“载入训练开场”默认人工控制，可点手牌或右侧合法动作（含目标、药水、单选及
多选确认），支持撤销与“重开同一局”。取消“手动操作”可切回当前 checkpoint/搜索。
手动操作清除缓存模型动作并校验画面版本，防止重复点击或执行过期目标；重开/撤销
复用同一开场工厂，不写回共享开场。训练开场模式禁止自定义状态及旧路线快进混用。
本功能依赖 `boss_opening_route_version() == 1` 的原生构建。
只需交互时指定 `--eval-episodes 0`，加载参数后不会另外启动整局评测。
查看留出集或失败子集时，用 `--boss-opening-label` 标明数据集来源；页面只列出数据集
实际包含的 Boss。子集应原样保留开场 journal/source_seed/opening SHA，不能把评测调度
seed 当作原始开场种子，也不能将留出失败筛选集的结果作为新的无偏胜率统计。

“完整爬塔评测”独立选择上层与下层。上层支持 checkpoint masked argmax，或
`ironclad-overworld-v3`（未知画面回退 checkpoint）；下层支持 checkpoint direct/NOSL
搜索，或上述固定配置的 `ironclad-turn-v8-awakened-phase`。评测状态记录两项选择、版本和
规划器配置；切换回合规划下层时 NOSL 参数不参与执行。


### 私有云端训练监控

远端运行 `python -m apps.training_site_sync --collect-only --config CONFIG --state STATE`，
复用Monitor的SSH/PID/日志读取，不加载模型或修改训练。`--poll-seconds 30`，
`--max-polls 0`持续运行；有预算验收可指定`--max-polls 2`。
CONFIG复用host/ports/ssh_key/root/run_id/iteration_steps，另设
`discovery_root`（含training-plan*.json的训练根父目录）。
按plan与真实PID/cmdline/cwd/启动时间识别continuous distributed_sb3；
仅run改变会自动归档旧曲线，SSH机器地址改变需更新配置。

Site每约30秒在页面或GPT读取时拉取远端快照，缓存至私有R2；
远端始终持续采集，电脑关机不影响采集或云端读取。
站点专用SSH公钥在sshd中用`restrict,command="… --serve-state STATE"`限制，
只能请求`current`或`history:<sha256>`，不能自行指定文件、打开shell或转发端口。
Site以部署secret保存专用私钥，并固定校验训练机host key；
不要复用训练运维密钥、把私钥提交Git，或把凭据送进网页。
状态文件含私有游标，受限读取入口只输出指标白名单与存活状态。

历史曲线按SHA独立缓存、勾选后加载；未记录的值为null，不补零。
数据超过90秒未更新会标记过期；断连保留上次曲线。
`/mcp`的training_status/list_training_runs/training_metrics只读，
每次最多读取100轮、12项指定指标，不提供训练控制或发送prompt。
站点源码独立保存在Sites项目源码仓库，部署身份与远端启动位置见
`experiments/launchers/training_site_monitor.json`，验证见worklog/2026-11-10.md#training-sites私有移动端与gpt训练监控。

可选直接上传仍使用`--credentials CREDENTIALS`和CONFIG的`site_url`，
CREDENTIALS含0600权限的service_token；只向精确HTTPS站点origin发送授权，
不跟随重定向。当前训练机直连Sites被边缘403拒绝，生产使用上述只读拉取。

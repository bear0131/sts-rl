# 用户交互入口

`python -m apps.dashboard`：浏览器 checkpoint 检查；静态文件在 dashboard_static/。
`python -m apps.watch CHECKPOINT`：终端逐步观看；`apps.play_gauntlet`：gauntlet 交互。
旧 apps.dashboard / apps.watch / apps.play_gauntlet 入口继续可用。训练和算法不能反向依赖 apps；UI 只消费 env/policy/evaluation。
日志、截图和 dashboard session JSON 写入 runs/local，不放仓库根目录。启动前查看 --help，自行指定端口/结果路径。

Dashboard 的 Watch 页可在 `ironclad-overworld-v3` 规则上层（默认）与 checkpoint 模型
上层之间切换。规则复用训练 worker 的 `overworld_rule_decision()`；未覆盖的未知画面显式
回退 checkpoint masked argmax，因此不是完全无模型的整局策略。战斗下层可独立选择
checkpoint 模型/所选 NOSL 搜索，或 `ironclad-turn-v8-awakened-phase` 回合规划器。规划器
使用 native-v1 完整动作覆盖出牌、目标、选牌和药水；前端固定使用历史 R10 配置
`worlds=2,node_budget=8192,beam_width=8,max_depth=16,lookahead_turns=1,master_nonce=20260918`。
它不调用 checkpoint 决定战斗动作；Runic Dome 等不支持的隐藏意图场景按版本契约回退
`rules-ironclad-v3`。页面显示的 checkpoint value/prior 仅用于对照。

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

“完整爬塔评测”独立选择上层与下层。上层支持 checkpoint masked argmax，或
`ironclad-overworld-v3`（未知画面回退 checkpoint）；下层支持 checkpoint direct/NOSL
搜索，或上述固定配置的 `ironclad-turn-v8-awakened-phase`。评测状态记录两项选择、版本和
规划器配置；切换回合规划下层时 NOSL 参数不参与执行。

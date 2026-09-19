# 用户交互入口

`python -m apps.dashboard`：浏览器 checkpoint 检查；静态文件在 dashboard_static/。
`python -m apps.watch CHECKPOINT`：终端逐步观看；`apps.play_gauntlet`：gauntlet 交互。
旧 apps.dashboard / apps.watch / apps.play_gauntlet 入口继续可用。训练和算法不能反向依赖 apps；UI 只消费 env/policy/evaluation。
日志、截图和 dashboard session JSON 写入 runs/local，不放仓库根目录。启动前查看 --help，自行指定端口/结果路径。

Dashboard 的 Watch 页可在 `ironclad-overworld-v1` 规则上层（默认）与 checkpoint 模型
上层之间切换。规则复用训练 worker 的 `overworld_rule_decision()`；未覆盖的未知画面显式
回退 checkpoint masked argmax，因此不是完全无模型的整局策略。战斗下层可独立选择
checkpoint 模型/所选 NOSL 搜索，或共享的 `rules-ironclad-v3` 纯规则策略。规则下层负责
出牌、目标、选牌和药水等完整战斗动作，不回退模型；页面仍显示 checkpoint value/prior
作为对照，但它们不参与规则动作选择。

Watch 可选择 Act 1/2/3 并“快进 Boss”。目标 Boss 前的战斗由模拟器直接结算；推进期间把
HP 和最大 HP 固定为开局值并清空药水，但保留牌组、遗物、金币、奖励及地图进度。目标
Boss 是真实战斗，停在所选下层策略的第一手决策。此功能是查看指定 Boss 打牌的路线
沙盒，不是可用于胜率统计的自然整局评测。

“完整爬塔评测”独立选择上层与下层。上层支持 checkpoint masked argmax，或
`ironclad-overworld-v1`（未知画面回退 checkpoint）；下层支持 checkpoint direct/NOSL
搜索，或无模型 fallback 的 `rules-ironclad-v3`。评测状态会记录两项选择及版本，切换
规则下层时搜索参数不参与执行。

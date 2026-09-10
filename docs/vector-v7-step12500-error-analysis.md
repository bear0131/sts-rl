# v7 step 12500 错误归因

分析对象：`docs/vector-v7-error-positions-step12500.json`。该 checkpoint 使用的是
executor contract v1：local executor 不接收 outer 已确定的 NodeSpec header，仍需重新
预测节点种类、目标和数值；计算过程还要求模型把连续状态还原成十进制 token，并直接
回归 dense delta。

因此这份报告不能直接代表 contract v4 的预期上限，但很适合定位旧接口把哪些不必要的
工作交给了 Transformer。

> 当前设计修正：contract v4 只移除“连续归一化槽 -> 精确离散数字 token”这条有损
> 反解任务。卡牌规则中的固定数字由 outer 直接从规则 token 写入 NodeSpec；运行时数字
> 由环境按字段返回。节点结尾仍保留 `DELTA_SLOT` 连续向量监督，因为“精确离散数字 ->
> 归一化连续变化”具有固定的位权和槽位缩放，是信息完整的正向映射。

## 结论概览

错误主要来自四种原因：

1. local 缺少 outer header，却被要求重新预测 header。
2. 连续归一化状态被要求精确还原成离散整数或长小数 token。
3. 父节点/卡牌参数没有显式绑定，local 只能猜 base damage、费用和重复次数。
4. dense delta 在缺少精确离散操作数时表现为稀疏、多槽、类别不平衡的回归。

contract v4 用“已知 NodeSpec 前缀”“交互式精确读取”和“卡面常量直接写入 header”
处理前三类问题；第四类继续保留为 delta head，并在精确数字上下文已经可见的条件下重新
测量，而不是预先删除。

## 1. Header 重复预测

最有力的证据是：

```text
OUTER_HEADERS error rate   0.49%
LOCAL_REPEAT              19.66%
LOCAL_STATE_CHANGE        15.09%
LOCAL_ATTACK              14.25%
LOCAL_COMPLETE            11.53%
```

outer 有卡牌向量，几乎能正确生成结构；旧 local 把 action program 固定为 `(0,)`，却把
header 也放入 label。高频混淆正好都是 local 猜成最常见节点：

```text
REGION_CARD_PAYMENT     -> REGION_CARD_PLAY   831
REGION_CARD_DESTINATION -> REGION_CARD_PLAY   414
REGION_COMPLETE         -> REGION_CARD_PLAY   363
REGION_DAMAGE           -> REGION_CARD_PLAY   229
REGION_ATTACK           -> REGION_CARD_PLAY   163
```

这不是模型容量不足，而是输入条件不完备。contract v2 已把 NodeSpec 作为 local 的
environment-owned prefix；contract v4 延续该边界。header 可参与 attention，但不参与
local loss，推理时直接用 outer 的结果启动 executor。

同一原因还解释了下列 token 错误：

```text
ENTITY_ENEMY_SLOT_1 error 67.43%
ENTITY_ENEMY_SLOT_2 error 64.09%
REGION_PILE_SHUFFLE      73.00%
REGION_ENERGY            51.23%
REGION_RELIC_COUNTER     50.00%
```

旧 local 不知道自己的 target/entity/kind。contract v4 不再让它猜这些字段。

## 2. 精确状态读取被错误地当成语言生成

主要位置：

| 位置 | 错误率 | 原因 |
| --- | ---: | --- |
| `SUBTRACT_PLAYER_HP/left` | 43.23% | 从 `hp/200` 恢复原始 HP |
| `READ_TARGET_HP/value` | 35.47% | 从 enemy HP 连续槽恢复整数 |
| `READ_TARGET_BLOCK/value` | 22.86% | 从 `block/100` 恢复整数 |
| `READ_VULNERABLE/value` | 10.93% | 从归一化 buff 层数恢复整数 |
| `READ_STRENGTH/value` | 4.17% | 从归一化 buff 层数恢复整数 |

这些位置的 marker 基本正确，错误集中在数字位。例如 player HP 的
`NUM_LSD_BEGIN/NUM_END` 均为 0% 错误，但两个数字位分别为 87.5% 和 85.4%。模型知道
“这里需要一个整数”，却无法精确执行 `normalized_value * scale` 再逐位输出。

独立实验进一步验证：即使 0..200 全部在训练中出现，完整随机 player vector 的 greedy
整串准确率也只有 2.36%；把其他槽全部固定后也只有 10.95%。这属于接口错误，不应靠
增加普通训练样本解决。

contract v4 的替代流程：

```text
model: READ_SOURCE enemy[0].hp -> reg.target_hp
environment: READ_RESPONSE reg.target_hp = 50
model: APPLY_HP_DAMAGE reg.target_hp reg.damage -> reg.target_hp_after
```

响应数字对后续 token 可见，但响应位置不计算 loss。

## 3. Buff 与格挡的“存在判断”和“精确数值”混在一起

旧向量把 Strength/Vulnerable 等层数写成 `amount / 30`。同一个连续槽同时承担两个任务：

- 判断 buff 是否存在，从而决定是否打开计算分支；
- 恢复精确层数，参与算术。

这会把一个容易的布尔判断和一个困难的反归一化任务耦合起来。contract v4 将每个已知
buff 槽改为严格 `0/1 presence`，精确层数只保存在 ValueStore。模型看到 Strength 位为
1 时输出 `READ player.buff.Strength` 和 `ADD_STRENGTH`；为 0 时省略该区域。

格挡同理新增 `block_present` 派生位，避免一点格挡在 `block/100=0.01` 时成为很弱的分支
信号。分支成立后才请求精确 block。

`VALUE_player_buff_Strength` 在旧报告中 100% 错误并不表示模型完全不理解力量。这个 token
主要出现在通用 `APPLY_DELTA field=...` 中，而旧 STATE_CHANGE local 没有字段 header，
只能在大量 buff 字段中猜。contract v4 的 `FIELD_OR_STATUS` 是已知 NodeSpec 参数。

## 4. 卡牌/父节点参数没有绑定

| 位置 | 错误率 | 旧问题 | v4 处理 |
| --- | ---: | --- | --- |
| `BASE_DAMAGE` | 18.27% | local 没卡牌向量却猜卡面伤害 | outer 从规则 token 输出精确 header 数字 |
| `PAYMENT_COST` | 6.87% | 从卡牌/当前能量猜费用 | 固定费用写 header；X 费用声明运行时来源 |
| `REPEAT_COUNT` | 12.82% | X-cost/Whirlwind 次数未提供 | 请求 `action.paid_energy` |
| `ARG_REMAINING` | 12.82% | 递归 frame 参数未提供 | runtime `CALL_FRAME` |
| `COUNT_ATTACK_HITS/right` | 14.49% | hit count 来源不明确 | parent/call-frame register |
| `READ_TRIGGER_DAMAGE/value` | 20.14% | 触发伤害没有父摘要绑定 | `parent.trigger_damage` read |

旧 checkpoint 的 `CALCULATION_READ_TRIGGER_DAMAGE` token 本身错误率为 100%。除了值难预测，
旧 compiler 还把对玩家的伤害落成通用 STATE_CHANGE，节点种类没有表达“这是父效果导致的
self damage”。contract v4 使用 `SELF_DAMAGE` 节点，并显式读取
`parent.trigger_damage` 和 `player.hp`。

多段攻击的 `CALCULATION_REUSE_ATTACK_DAMAGE` token 错误率 34.62%。旧 header 没说明该
DAMAGE 是首次计算还是复用父攻击结果。v4 header 使用：

```text
DAMAGE_SOURCE
VALUE_PARENT_ATTACK_DAMAGE
```

local 随后请求 `parent.attack_damage`，不读取卡牌，也不重新计算 Strength/Vulnerable。

## 5. 通用 APPLY_DELTA 缺少语义

主要错误：

```text
APPLY_DELTA/field                  17.87%
VALUE_piles_exhaust_count         95.45%
  -> VALUE_piles_discard_count    294 次
attacks_played -> skills_played   250 次
```

这些不是算术错误，而是节点不知道自己执行的是什么操作：旧 CARD_DESTINATION 没有明确
destination，旧 CARD_PAYMENT 没有明确 play kind，generic STATE_CHANGE 没有明确字段。

contract v4 在 NodeSpec 中保留语义枚举/字段：

```text
DESTINATION       VALUE_EXHAUST
PLAY_KIND         VALUE_ATTACK
FIELD_OR_STATUS   VALUE_enemy_0_buff_Vulnerable
ENERGY_OP         VALUE_GAIN
PILE_OPERATION    VALUE_SHUFFLE_DISCARD_INTO_DRAW
```

卡牌规则里的固定 amount 由 outer 直接输出数字；运行时 amount 使用语义来源和环境响应。
模型仍需选择“改哪个字段、做加法还是减法”，但无需从连续向量猜精确数值。

特别地，v3 禁止 `READ effect_delta.*`。否则模型可以询问“正确 delta 是多少”而绕过规则。
洗牌必须读取 draw/discard count 并输出 `TRANSFER_PILE`；遗物计数必须更新具体 counter，
summary 由 runtime 重算。

## 6. 长小数造成重复且无意义的 token 错误

`APPLY_DELTA/left` 的 digit offset 3..9 错误率达到 65%..96%。来源主要是：

```text
player.hp_frac before = 0.9753086419753086
delta                 = -0.024691358024691357
result                = 0.9506172839506173
```

这些值既长又是 HP/max HP 的派生量。要求模型逐位复写不会增加规则理解，只会增加序列和
loss。contract v4 不把 `hp_frac` 当作独立语义 effect；应用 HP delta 后再确定性派生。

同样，`block_present`、敌人 `present/gone`、relic summary 都是运行时派生值，不要求模型
生成其数值变化。

## 7. Dense Delta 回归错误

旧 delta head 的 sign 通常正确，但精确幅度不可靠：

| 槽位 | active count | 归一化 MAE | 原始单位约 MAE | 说明 |
| --- | ---: | ---: | ---: | --- |
| enemy[0].block | 338 | 0.0463 | 4.63 block | 高频总误差最大 |
| enemy[1].block | 171 | 0.0762 | 7.62 block | 槽位更稀疏 |
| enemy[2].block | 83 | 0.0945 | 9.45 block | 槽位越后样本越少 |
| player.energy | 877 | 0.0151 | 0.15 energy | sign 100%，仍非精确离散值 |
| enemy[0].hp | 82 | 0.0323 | 6.45 HP | 伤害/格挡组合回归 |
| attacks_played | 451 | 0.0127 | 0.13 count | 本应精确 +1 |

`enemy[2].present/gone` MAE 约 0.95，但各只有 2 个 active 样本，是典型的极端类别不平衡。
若继续回归，模型必须同时学习“哪个槽变化”和“变化多少”，而绝大多数 407 个槽都是零。

这些变化本质上可由符号操作确定执行：

```text
APPLY_BLOCK
APPLY_HP_DAMAGE
SUBTRACT_FIELD player.energy
ADD_FIELD player.counter.attacks_played constant.one
```

这些误差不能单独证明 dense delta target 应被移除。旧模型在预测 delta 时既没有可靠的
NodeSpec，又要从连续输入反解精确 HP、格挡和层数，因此其幅值误差混合了上游读数错误。

contract v4 保留 `DELTA_SLOT`。环境先把所需运行时值以精确离散 token 插入上下文，模型
完成符号计算后，由 delta head 输出连续变化。以 HP 为例，离散结果 `-7` 到目标槽
`-7 / hp_scale` 是固定线性映射；十位和个位也各有固定线性贡献。后续应单独测量在
teacher-forced 精确读数下的 active-slot MAE、零槽误报率和完整向量准确率。

## 8. 哪些问题仍需模型学习

新协议没有把规则学习全部交给环境。模型仍需正确预测：

- outer tree 的节点顺序、条件、目标模式和参数绑定种类；
- presence bit 为 1 时应打开哪种 modifier 分支；
- 需要读取哪些字段，以及绑定到哪个寄存器；
- Strength 用加法、Vulnerable 用 `floor(x * 3 / 2)`、block 先吸收伤害；
- 多段攻击何时复用 parent damage；
- 洗牌、抽牌、遗物触发的树形因果关系；
- 运算顺序和最终 `DELTA_SLOT` 边界。

被移除的是从连续向量反归一化并复写精确十进制 token 的任务。精确值寻址由环境完成，
响应数字不计 token loss；从离散计算上下文生成连续 delta 的正向映射仍由模型学习。

## 验证结果

以下是已废弃 contract v3 在 3,000 条随机记录上的历史统计，不代表 contract v4：

```text
最长 local 输入                 142
最长 local model-owned body      99
model-owned numeric token          0
dense delta target                 0
effect_delta read                  0
```

使用独立 seed 构造 2,000 条 tokenizer 数据、再检查 3,000 条记录，unknown token 为 0。

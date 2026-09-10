# v7 简单卡牌统一评测

checkpoint: `.tmp-vector-rule-v7-contract-v10-register-values-20260901/last.pt`; step=3500; contract=10

评测统一使用 v7 contract 10：outer 自回归，local 使用真实 NodeSpec 前缀并自回归生成模型拥有的 token，delta 按目标状态槽比较。

- cases=11; outer token accuracy=0.9986; local token accuracy=0.9090
- target effect MAE=11.9939; unknown tokens=0

| case | card | game expected effect | protocol expected | predicted effect | target abs error | outer acc | local acc |
|---|---|---:|---:|---:|---:|---:|---:|
| strike_base | 打击 | 6.0 | 6.000000238418579 | 0.9958036243915558 | 5.004196614027023 | 1.000 | 0.869 |
| strike_vulnerable | 打击 | 9.0 | 8.99999737739563 | 2.1714355796575546 | 6.828561797738075 | 1.000 | 0.895 |
| strike_block | 打击 | 2.0 | 1.9999980926513672 | 1.3674765825271606 | 0.6325215101242065 | 1.000 | 0.863 |
| strike_vulnerable_block | 打击 | 5.0 | 5.0000011920928955 | 1.954314112663269 | 3.0456870794296265 | 1.000 | 0.884 |
| strike_tampered_6_to_12 | 打击 | 12.0 | 12.000000476837158 | 2.3587774485349655 | 9.641223028302193 | 1.000 | 0.987 |
| perfected_strike_zero_strikes | 完美打击 | 6.0 | 6.000000238418579 | 0.9956486523151398 | 5.004351586103439 | 1.000 | 0.869 |
| perfected_strike_three_strikes | 完美打击 | 12.0 | 6.000000238418579 | 0.9947884827852249 | 5.005211755633354 | 1.000 | 0.869 |
| perfected_strike_vulnerable | 完美打击 | 9.0 | 8.99999737739563 | 2.172207459807396 | 6.827789917588234 | 1.000 | 0.895 |
| impervious_base | 岖然不动 | 30.0 | 30.000001192092896 | 10.201548784971237 | 19.79845240712166 | 1.000 | 1.000 |
| impervious_frail | 岖然不动 | 22.0 | 30.000001192092896 | 10.198230296373367 | 19.801770895719528 | 1.000 | 1.000 |
| impervious_tampered_30_to_60 | 岖然不动 | 60.0 | 60.00000238418579 | 9.656708687543869 | 50.34329369664192 | 0.984 | 1.000 |

## Perfected Strike 计数诊断

当前协议隐藏牌堆组成，只提供 draw/discard/exhaust 数量；因此没有 STRIKE_COUNT 环境读取时，3 张 Strike 的语义值不会进入 teacher target。

| case | hidden Strike count | game damage | protocol base damage | teacher applied count | count read token |
|---|---:|---:|---:|---|---|
| perfected_strike_zero_strikes | 0 | 6.0 | 6 | True | False |
| perfected_strike_three_strikes | 3 | 12.0 | 6 | False | False |
| perfected_strike_vulnerable | 0 | 9.0 | 6 | False | False |

## Case notes

### strike_base

打击：无易伤、无格挡。

### strike_vulnerable

打击：敌人2层易伤，6点伤害按易伤结算为9。

### strike_block

打击：敌人4点格挡，只有剩余伤害扣血。

### strike_vulnerable_block

打击：易伤与格挡同时存在。

### strike_tampered_6_to_12

打击：将卡面基础伤害6篡改为12。

### perfected_strike_zero_strikes

完美打击：隐藏三堆中0张Strike，语义上期待4伤害。

### perfected_strike_three_strikes

完美打击：隐藏三堆中3张Strike，语义上期待12伤害。

### perfected_strike_vulnerable

完美打击：不含Strike时受易伤影响，当前teacher基础项为6。

### impervious_base

岖然不动：获得30点格挡。

### impervious_frail

岖然不动+脆弱：游戏语义为floor(30*0.75)=22，当前v7 teacher暂未应用Frail。

### impervious_tampered_30_to_60

岖然不动：将卡面格挡30篡改为60。

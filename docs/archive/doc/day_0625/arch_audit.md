# 架构裂缝检查 (2026-06-25)

全链路通读 obs(`encode.py`) / action(`action_space.py`, `actions.py`) /
model(`model.py`, `pointer_policy.py`) / env(`run_env.py`, `game_env.py`,
`combat_env.py`) / reward(`reward.py`) / 训练(`train.py`)后的裂缝清单,按严重度排序。

判定:**1/2/3 要修**。4/5/6 有效果但麻烦(有用度 5>6>4,麻烦度 6>4=5)。7 待评估。8 故意为之。9 不重要。

---

## 🔴 高危 — 要修

### ~~1. 无 episode 截断,空 mask 会让 vec-env worker 永久卡死~~ done
- `run_env.SlayTheSpireEnv.step` 永不返回 `truncated=True`;`train.make_env` 也没套 `TimeLimit`。
- `game_env.legal_semantic_actions()` 对不支持的事件(`MATCH_AND_KEEP` / `BONFIRE_SPIRITS`)返回 `[]`
  (注释自承 "run stalls there honestly")。
- 链式后果:空 legal → `action_masks()` 全 False → MaskablePPO masked logits 全 `-inf` →
  softmax NaN;采样到任意 id → `step` 判非法 → 返回 `-0.1` 且 `terminated=False` → **永不终止**。
  该子环境从此每步吐 `-0.1`、永不 reset,worker 成死重并向 buffer 灌 NaN/垃圾 GAE。
- 即便不撞空 mask,无限循环战斗(0 per-step reward)目前也只靠 gamma 软处理,无硬上限兜底。
- **修法**:加 `TimeLimit`(per-episode step cap)+ 对空 mask 显式 `truncated=True` 终止。

### 2. 战斗内 SINGLE 选牌:候选牌不进 obs,且 pointer 读错 token
- 战斗内 SINGLE 选牌屏候选索引 `k` 指向**该屏特定牌堆**的列表(`encode_select(k)` →
  `SINGLE_CARD_SELECT(source_idx=k)`)。Armaments 选手牌,但 **Headbutt 从弃牌堆选、
  Exhume 从消耗堆选、Dual Wield/Sentinel…** 候选都不在手牌里。
- obs 侧:战斗 select block 的 `selected` 恒 `[]`,候选牌 id(`select_card_ids`)只在
  overworld `card_select` phase 填充 → **战斗 SINGLE 屏没有任何候选牌 id 进 obs**,
  模型只能看到无序 pile 计数桶。
- pointer 侧(`pointer_policy.py:156-162`):combat phase 下 `PICK_CARD` 一律用 **hand token**
  打分。Headbutt/Exhume 候选不在手牌,`k` 是弃牌/消耗堆下标 → 用"手牌槽 k"给"弃牌堆第 k 张"
  打分,纯噪声。
- 结论:这类卡的选择只能靠 floor reward 盲猜,是**确凿的动作-观测错位 bug**,非单纯表征损失。
- **修法方向**:给战斗 SINGLE 屏补候选牌 token 族,或让 pointer 的 combat-pick 来源按 select
  屏所属牌堆动态切换。

### ~~3. 整局无法使用药水~~ done
- `action_space.USE_POTION` 是 reserved 段,thick 层从不枚举药水动作 → **整个 run agent 永远喝不了药水**。
- action_design §5 明确 deferred,但对胜率影响大(高层 boss 战常靠药水翻盘)。
- 归因 act2 入口停滞时应记一笔:这是已知能力缺口。

---

## 🟡 中危 — 表征缺口(有效果但麻烦,排期)

### 5. deck 升级状态完全不可见(有用度最高)
- `deck_counts` 只按 base card id 计数(`encode.py:317` 丢弃 `(cid, _up)` 的 `up`);CARD_VOCAB 是
  base-card 粒度,升级只走 `hand_upgrades` 这条**仅手牌有**的标量。
- 后果:**全升级牌组与全未升级牌组在 deck 层 obs 完全一样**。Armaments / 火堆 Smith / 升级流构筑
  的价值在宏观视角不可见。
- **修法**:给 deck 加"升级计数桶",或把升级编进 vocab。

### 6. 地图只能看一步,无全图 lookahead
- `game_env._canonical()` map options 只给下一行可达节点 `(x, room_type)`(`MAP_EDGES=3`);整张 act
  地图已用 `SpireMap` 重建,但 obs 只暴露贪心一步。
- 后果:路线规划(绕路拿精英遗物、boss 前找火堆/商店)结构上不可能,只能靠 floor_num 粗感知深度。
- 对 StS 是较强策略限制。

### 4. count 桶 / deck 桶 与 card embedding 解耦(有用度最低)
- pile Linear `(d_model, N_CARDS)` 每张牌一列、从零单学,与 `shared["card"]` 无参数共享,双向浪费:
  card 语义传不进牌堆桶,牌堆统计传不回 token。
- 影响范围:draw/discard/exhaust/deck 计数桶 vs hand/offer/select/shop token。
- 优先级最低:只要 deck 计数可见,宏观构筑信号在(虽仍有损)。
- **修法**:把 pile Linear 换成"共享 card embedding 加权求和" `Σ_c count_c · E_card[c]` + pile-role 向量。

---

## 🟢 低危 / 设计取舍

### 7. 卡牌奖励只 token 化前 3 张(待评估,见下)
- `REWARD_CARD` 段宽 5,但 `offer_card_ids` 只有 `OFFER_SLOTS=3` token。Question Card / Prismatic
  时第 4-5 张候选既不进 obs,pointer 也只填前 3 个 id、后两个用过时 flat head 值。
- 默认铁甲很少触发,影响小。

### 8. HP / 死亡 reward 权重为 0(故意为之)
- `RewardConfig.hp_delta=0.0`、`death=0.0`。HP 只通过"能否到下一层"间接影响回报。已确认有意。

### 9. 遗物动态状态不可见(不重要)
- `relics` 只是 id 列表,Ink Bottle 计数 / Nunchaku 等内部状态未编码。多数遗物无所谓。
</content>
</invoke>

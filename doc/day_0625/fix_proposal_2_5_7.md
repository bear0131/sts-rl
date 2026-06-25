# 架构裂缝修复方案 — item 2 / 5 / 7 (2026-06-26)

承接 `arch_audit.md`。前提:**这一批一起动 obs 结构,旧 ckpt 全部报废**。
优先级:2 必修,5 重要,7 好改。6/4 缓做(不在本批)。

---

## 0. 贯穿前提:版本与"一次多改"

三项都只动 **obs**,不动动作 id 表:

- `OBS_VERSION` 6 → **7**(一次性涵盖三项所有 obs 形状变化)。
- `ACTION_VERSION` **不变**:不新增/重排任何动作 id。item 7 的 `REWARD_CARD[3..4]`、
  item 2 的 `PICK_CARD` 段都早已存在,thick 层 mask 也早就放行(见各项)。
- 旧 ckpt 全部报废(已接受)——正是"一次多改"的窗口。

关键省事点:
- `model.py` 的 token 槽位数全部由 obs 形状自动推导(`self._n(...)`)。
- `pointer_policy.py` 指针段宽由 `scores.shape[1]` 自动跟随。
- → 加宽 obs 槽位时,模型/指针多数**零改动**。

建议实施顺序 **7 → 5 → 2**(风险递增);7 先把 pipeline + OBS_VERSION bump 跑通一遍。

---

## 1. Item 7 — 奖励牌补满 5 张(最好改)

**根因(已核实)**
- `encode.OFFER_SLOTS=3`,`offer_card_ids` 只填前 3(`encode.py:329`),但 `REWARD_CARD` 段宽 5。
- thick 层**早已枚举全部** reward card:`game_env.py:255-256`
  `n=len(get_card_reward()); reward_card(j) for j in range(n)` → Question Card / Prismatic
  的第 4–5 张**动作可选、mask 放行**,只是 obs 没 id、指针后两列退化成 flat head。
- 附带坑:`OFFER_SLOTS` 还被 `boss_relic_ids` 复用(`encode.py:163`),但 boss 永远 3 张
  → 直接调 5 会让 boss_relic 白填 2 个 PAD,需**解耦**。

**改法**
1. `encode.py`:新增 `BOSS_RELIC_SLOTS = 3`;`OFFER_SLOTS` 3 → 5。
   - `offer_card_ids` 形状随 OFFER_SLOTS 自动 5;`_encode_options` 的 `[:OFFER_SLOTS]` 自动吃 5。
   - `boss_relic_ids` 改用 `BOSS_RELIC_SLOTS`;`_encode_options:364` 的 `[:OFFER_SLOTS]` → `[:BOSS_RELIC_SLOTS]`。
   - `_empty()` 两数组形状跟改。
2. `model.py`:`offer`/`boss_relic` 槽数由 `self._n(...)` 自动推导 → **零改动**。
3. `pointer_policy.py`:REWARD_CARD 指针写 `scores.shape[1]` 列,自动变 5 → **零改动**,
   只需更新 line 51-52 那条 "first 3" 注释。

**影响面**:仅 obs 形状(offer 3→5,boss 仍 3)。风险极低,无新动作、无新分支。

---

## 2. Item 5 — deck 升级可见(重要)

**根因(已核实)**:`encode.py:317` `for cid, _up in g["deck"]` 显式丢 `_up`,
`deck_counts` 只按 base id 计数;升级仅有的通道 `hand_upgrades` 只覆盖手牌
→ 全升级 deck 与全未升级 deck 的 obs 完全相同。

**改法(最小且完整解决 deck 维度)**:加一条平行向量 `deck_upgrade_counts`
(N_CARDS,每个 base id 的已升级份数)。
1. `encode.py`:`obs_space()` + `_empty()` 加 `deck_upgrade_counts: scal((N_CARDS,))`;
   `_encode_global`:
   ```python
   for cid, up in g.get("deck", []):
       idx = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
       o["deck_counts"][idx] += 1.0
       if up:
           o["deck_upgrade_counts"][idx] += 1.0
   ```
   (Searing Blow 可多级,这里按"是否升级"计 1;base 牌 95% 只 0/1 级,按份数即可。)
2. `model.py`:`pile_keys` 末尾追加 `"deck_upgrade_counts"` → 走同一条 `pile` Linear,
   自动多一个 pile token + pos 槽,`token_pos["pile"]` 计数自动 +1 → 其余零改动。
3. 指针不涉及(pile 非指针目标)→ 零改动。

**取舍**:复用同一条 `pile` Linear 会把"升级份数"和"持有份数"两种语义喂进同一权重
(正是 item 4 的解耦问题),但 pos embedding 不同、模型可区分。item 4 缓做,这里**接受共享 Linear**。

**可选加挂**(几乎免费,默认不做):给 `draw/discard/exhaust` 也补 `*_upgrade_counts`,
战斗内也能看到牌堆里哪些是升级版。本批只做 deck。

---

## 3. Item 2 — 战斗 SINGLE 选牌错位(必修,最复杂)

**纠正 audit 一处措辞**(核到 C++ `Action.cpp:480-536`):候选来源按 task 分四类,
`source_idx` 是对应 pile 的下标——

| 来源 pile | task |
|---|---|
| **hand** | ARMAMENTS, DUAL_WIELD, EXHAUST_ONE, FORETHOUGHT, WARCRY |
| **discard** | HEADBUTT, LIQUID_MEMORIES_POTION |
| **exhaust** | EXHUME |
| **draw** | SECRET_TECHNIQUE, SECRET_WEAPON |
| **生成数组**(`cardSelectInfo.cards`) | DISCOVERY, CODEX |

- **hand 源(含 Armaments)**:指针现用 hand token,`source_idx` 正好是手牌下标 → 碰巧对
  (audit 说 Armaments 也错,偏严)。
- **discard/exhaust/draw 源(Headbutt/Exhume/Secret*/Liquid Memories)**:`source_idx` 是
  非手牌堆下标,却用 hand token 打分 → **确凿错位,真 bug**。
- **生成数组源(Discovery/Codex)**:候选不在任何已暴露 pile → 跨角色(Defect/无色),**本轮缓做**。

**两个利好**
- hand/discard/exhaust/draw 四 pile,`normalize_sim` 早已全部产出(`combat_state.py:254-257`),
  且 `bc.to_dict()` pile 顺序与 `setupCardOptionsHelper` 迭代顺序一致
  → **不需重编译 C++**,Python 侧即可把候选 id 摆到正确 `source_idx`。(仅 Discovery/Codex
  需新 binding,缓做。)
- `arg_step` 已能区分单/多选:MULTI 走 `pick_hand`,SINGLE/overworld 走 `pick_card`
  (`run_env.py:205-208`;`vocab` 里两者都有)→ 指针路由**不用新增 obs 字段**。

**改法(三处)**

1. **暴露候选 id(Python,无需重编译)** — `combat_state.py` 加 task→pile 映射;
   `combat_env._obs_canon`(`combat_env.py:171-172`)挂到 select 块:
   ```python
   # combat_state.py
   SELECT_TASK_PILE = {
       "ARMAMENTS":"hand","DUAL_WIELD":"hand","EXHAUST_ONE":"hand",
       "FORETHOUGHT":"hand","WARCRY":"hand",
       "HEADBUTT":"discard","LIQUID_MEMORIES_POTION":"discard",
       "EXHUME":"exhaust",
       "SECRET_TECHNIQUE":"draw","SECRET_WEAPON":"draw",
   }
   # combat_env._obs_canon
   if self.bc.input_state == _CARD_SELECT:
       sel = combat_select_options(self.bc.select_info())
       pile = combat_state.SELECT_TASK_PILE.get(sel["context"])
       if pile:
           sel["cards"] = [(c["id"], c["up"]) for c in d.get(pile, [])]
       d["options"] = {"select": sel}
   ```

2. **encode 侧填 `select_card_ids`** — `_encode_select`(`encode.py:381`)追加
   (overworld 路径不动,它走 `_encode_options`,互不重填):
   ```python
   for k, (cid, _up) in enumerate(sel.get("cards", [])[:G_MAX]):
       o["select_card_ids"][k] = CARD_INDEX.get(cid, CARD_INDEX[EMPTY])
   ```

3. **指针路由按 arg_step 切换**(`pointer_policy.py:153-162`):
   "combat → hand token" 改成 "**仅 `pick_hand`(MULTI)→ hand token**,其余→ select token":
   ```python
   # __init__: self._pick_hand_step = V.ARG_STEP_VOCAB_INDEX["pick_hand"]
   pick_hand = obs["arg_step"].long().reshape(-1) == self._pick_hand_step
   pick = sel.clone()
   pick[:, :k] = torch.where(pick_hand.unsqueeze(1), hand, sel[:, :k])
   ```
   `_logits` 改成吃 `pick_hand` 标志(或直接收 obs)。MULTI 仍走 hand token + `select_hand_mask`
   (不变);战斗 SINGLE 改吃 select token(含 hand 源的 Armaments,统一且正确)。
   `self._combat_phase` 这条 PICK_CARD 路由可弃用。

**影响面**:`combat_state.py`、`combat_env.py`、`encode.py`、`pointer_policy.py`。无新动作 id。

**风险 / 待验证**
- ⚠️ **唯一硬假设**:`bc.to_dict()` 各 pile 顺序 == C++ `setupCardOptionsHelper` 迭代顺序
  (即 `source_idx` k 对得上 `select_card_ids[k]`)。**必须写测试**:跑一场带 Headbutt/Exhume
  的战斗,断言每个 legal `SINGLE_CARD_SELECT` 的 `source_idx` 处 `select_card_ids` 的牌 ==
  该 pile 该位置的牌,且 == sim 实际选中的牌。
- discard 可能 >G_MAX=64?典型远小于此,溢出部分本就被 mask 排除。
- **遗留缺口**:Discovery/Codex(生成数组)仍盲选——跨角色,本轮不做;audit 里 item 2
  可标"Ironclad 范围已修,生成牌待 binding"。

---

## 实施顺序与验证

1. 改完 `OBS_VERSION` 一次性 bump → 7。
2. Sanity:`encode.encode(canon)` 各 phase 跑通、形状对得上 `obs_space()`;
   模型能 `forward` 一个 dummy batch(token 槽位自洽)。
3. Item 2 专项测试(上面那个 source_idx 对齐断言)——整批最易静默错的点。
4. 跑一小段训练确认不崩、mask 与 obs 不打架(尤其 select/multi 两条 arg_step 分支)。

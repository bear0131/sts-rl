# 决策:打牌头预训练 = 随机配置 combat gauntlet + 冻结主干接全局 (2026-06-25/26)

## 背景

当前全局 run 的模型非常弱,连"该打该防"都分不清,远够不到"构筑"。卡牌奖励的
信号噪声根源是:**"某张牌好不好"完全取决于"模型会不会用它"**——打牌头半随机时,
卡牌选择的奖励就是噪声。先把战斗子策略 bootstrap 到能打,全局训练的卡组评估才有意义。

战斗在 STS 里是一个**可分离的子 MDP**,占一局里绝大多数 per-step 决策,适合单独、
低成本地大量采样训练。

## 方案:combat gauntlet(车轮战)

一个 **episode = 一套固定的随机 loadout 连续打一串随机遭遇**:血量跨场结转、不回血,
**每清一场 +1**,**死了 episode 结束**。

之所以用 gauntlet 而非"单场战斗一个 episode":把"血量的价值"从**奖励整形**层面挪到
**episode 结构**层面。血量决定还能再赢几场,"少掉血/该防就防"通过"多活几场 = 多拿 +1"
**内生**出来——不用手调 `hp_delta` 权重,也不用死亡负奖励(死的代价 = 被 gamma 折现的、
被截断的未来 +1 流)。这一阶段没有任何奖励魔数。

### loadout 以 level L 锚定(in-domain 但宽随机)

每个 episode 采 `L ~ uniform[1,50]`,act / 怪池 / 牌组规模都由 L 推出,所以 loadout 是
"真实 run 在第 L 层的一个(随机化的)快照"。**L 在整个 episode 内固定**(给 critic 一个
稳定的 floor 特征)。`act(L)= 1(≤16)/2(17–33)/3(≥34)`。

| 项 | 均值(L) | f50 | 实际采样 |
|---|---|---|---|
| 加牌次数 | `15·(1−e^−0.06L)` 饱和曲线 | ~14.3 | `uniform[0, 2·均值]` |
| 删牌次数(删初始 Strike/Defend,留 Bash) | `0.10·L` | 5 | `uniform[0, min(2·均值,9)]` |
| 普通遗物数(`_RELIC_POOL`) | `0.20·L` | 10 | `uniform[0, min(2·均值,池)]` 无放回 |
| boss 遗物数(`_BOSS_RELIC_POOL`) | `act(L)−1` 确定 | 2 | 无放回 |
| 初始血量 | `uniform(0.2,1.0)·max_hp` | — | — |

- 加牌曲线:早期斜率 ~0.85/层(开局必抓牌),越深越有取舍,act3 几乎不加,总均值到
  50 层 ~15。act1 加 ~9、act2 加 ~3.7、act3 加 ~1.4。
- **~5% 概率给牌组塞一张随机普通诅咒**(`_CURSE_POOL`,对应事件/诅咒之钥)。
- **~10% 概率用一张随机 boss 遗物换掉初始"燃烧之血"**(对应 Neow 的换 boss 遗物)。
- boss 遗物池 curate 过:只放**战斗内/开局生效**的(史尼克之眼、卢恩金字塔、痛苦面具…),
  排除"拾取时重铸牌组"(潘多拉魔盒/星象仪)和纯局外的——sandbox 只挂遗物、不跑拾取逻辑。
- 每场遭遇独立 roll 类型 `normal/elite/boss = 0.75/0.20/0.05`,从 `act(L)` 池抽一个。
  **不给 boss 额外 reward**:遭遇是强制的、没有"挑/避 boss"的决策,bonus 不改变 fight 内
  策略梯度方向,只会扭曲 value target 并重新引入魔数。

### 战力评估(白送)

gauntlet 里 `V(s) = 期望折现后续清场数`,**本身就是**"按卡组+层数估自己还能打几场"的战力
评估。obs 的 global 块已喂了 `act/floor/deck/relics/hp`,value 头能 condition。**宽方差**
loadout(同一层有强有弱的牌组)正是让 critic 把"层数→怪更硬"和"牌组更厚→更强"两个边际
效应**解耦**的前提。接全局时冻结 encoder,这套战力表征正好被带进全局 value。

### 接全局

用预训练权重 warm-start 全局训练,**冻结 encoder(主干)**,只训各 head。

## 关键取舍:handoff 选了"简单冻结"

- 独立子网络:扔掉共享主干带来的迁移(战斗的单卡 literacy 反哺卡组评估),还复制一份昂贵
  encoder——否决。
- 简单冻结:最小实现;第一阶段 encoder 冻结后不可能忘战斗。**采纳**。
- **已知局限**:纯战斗信号对"层数/跑图进度/构筑连贯性"没有梯度,冻结会把主干在这些维度锁成
  "战斗近视"。**升级路径**:将来发现全局 value 对 floor/进度不敏感时,切到"解冻 encoder +
  在 batch 里掺战斗 rollout 做 rehearsal"。战斗 sandbox 届时已在手,升级近乎免费。**故现在
  不把冻结写成永久方案。**

随机牌组 = "识字课本"不是"实战":只练单卡 literacy,练不到 synergy/构筑——但现在模型
够不到构筑,这正是要的。精细构筑(按共现概率自举)、消耗类/局外收益推迟到后续精细 RL。

## 落地(已实现并跑通)

**关键修正(早先踩的)**:`SimCombatEnv`(`combat_env.py`)用更窄的 **71-id 战斗动作空间**,
而 `StSPointerPolicy` 输出 **149-id 整局空间**,对不齐、权重无法迁移。所以预训练走 thick
`SimGameEnv` 裁剪到单场 + thin `SlayTheSpireEnv` 这条路,与全局**完全同一动作头**。

1. **`env/core/combat_backend.SimCombatBackend(SimGameEnv)`**:gauntlet 语义——`reset` 直接
   随机 loadout + `create_battle`;`_settle` 在**胜利且没死**时把 bc 终局 HP 同步回 gc、起
   下一场随机遭遇(floor 不变);`is_over` 只在**死亡**为 True;`combats_won` 计数器。
2. **`factories.random_gauntlet_factory`**:返回 `(make_gc, next_encounter)`。`make_gc` 采 L
   并经 `_build_gauntlet_gc` 造 L 锚定 loadout;`next_encounter` 按类型概率从 `ACT_POOLS[act]`
   抽。辅助:`_act_of` / `_mean_added` / `_around` / `_BOSS_RELIC_POOL`。
3. **`encounters.ACT_POOLS`**:act 1–3 的 normal/elite/boss 遭遇名(act2/3 由 subagent 从
   sim `MonsterEncounters.h` 补齐并验过 `encounter_names()` 成员 + create_battle 出怪)。
4. **`rl/reward.CombatGauntletRewardWrapper`**:读 backend 的 `combats_won`,每新清一场 +1,
   死亡 terminated 不罚。
5. **`rl/train_combat.py`**:`--floor-lo/--floor-hi`(默认 1/50),指标
   `combat/cleared_per_episode`,step cap 4000(一个 episode 跨多场,无回血必死)。
6. **`rl/train.py` 的 `--init-from <combat.zip>` + `--freeze-encoder`**:只 `load_state_dict`
   权重(step 从 0)+ 对 `core.encoder.parameters()` 置 `requires_grad=False`。

## Sanity(本机已验证)

- loadout 缩放贴合目标:加牌均值 L=1/8/16/25/45/50 → 1.0/5.6/9.7/11.9/14.5/14.5(target
  0.9/5.7/9.3/11.7/14.0/14.3);删牌 0→4.6;普通遗物 0→9.8;boss 遗物 0→1.1(act2)→2.1(act3)。
- 遭遇按固定 floor 的 act 抽(L=5 act1 / 25 act2 / 45 act3),floor 全程不变。
- 随机策略 gauntlet 跨三幕跑通,boss 遗物挂载不崩;HP 跨场单调结转(80→55→44→35→10→死)。
- `python -m rl.train_combat`(dummy/CPU/256 步)跑通,`cleared_per_episode == ep_rew_mean`,存档。
- `--init-from <ckpt> --freeze-encoder`:warm-start + 冻结 108 个 encoder 张量,训练继续。

## 启动命令(远端正式训练)

    python -m rl.train_combat --steps 2000000 --n-envs 8 --floor-lo 1 --floor-hi 50 --wandb
    python -m rl.train --arch transformer-pointer --init-from <combat>/final.zip \
        --freeze-encoder --steps 10000000 --n-envs 8 --wandb

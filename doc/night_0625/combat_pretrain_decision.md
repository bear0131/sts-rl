# 决策:单独预训练打牌头(combat-only) + 冻结主干接全局 (2026-06-25 night)

## 背景

当前全局 run 的模型非常弱,连"该打该防"都分不清,远够不到"构筑"层面。
卡牌奖励的信号噪声根源是:**"某张牌好不好"完全取决于"模型会不会用它"**——
打牌头半随机时,卡牌选择的奖励就是噪声。先把战斗子策略 bootstrap 到能打,
全局训练的卡组评估才有意义。

战斗在 STS 里是一个**可分离的子 MDP**:一场战斗内的决策(出哪张/打谁/顺序/
何时结束回合)与元游戏(选牌/走图/商店/篝火)耦合很弱,且占了一局里绝大多数
per-step 决策。适合单独、低成本地大量采样训练。

## 方案

1. **Combat-only 预训练**:用 `SimCombatEnv`(一场战斗=一个 episode)训练,
   gc_factory 随机化:
   - 牌组:在起始牌组上**按合法方式随机加牌**(`gc.set_deck`)。
   - 遗物:随机一批(`gc.set_relics`)。
   - 初始血量:随机(`gc.cur_hp` / `gc.max_hp`),让模型学会读自己血条。
   - 药水:**v1 暂不做**(需加 C++ 绑定,见落地成本),用 HP 维度先教会打防。
   - 遭遇战:随机 act-1 遭遇(已有 `random_act1_factory`),后续可扩到精英/boss。
2. **Reward**:已有的 `HP delta + 终局 ±1`,把**死亡的终局负奖励放大**
   (现为 -1.0)以强化"别莽死"。
3. **接全局**:用预训练权重 warm-start 全局训练,**冻结 encoder(主干)**,
   只训各 head。

## 关键取舍

- **随机牌组 = "识字课本",不是"实战"**:随机合法牌组只能练**单卡 literacy**
  (在大量上下文里见到每张牌),练不到 synergy/构筑——但现在模型够不到构筑,
  这正是我们要的。精细构筑(按"两张牌共现概率"加权采样自举)、消耗类/局外
  收益("打出此牌时整局 xx +xx")**全部推迟到后续精细 RL 阶段**。

- **handoff 选了"简单冻结",而非独立子网络 / warm-start+rehearsal**:
  - 独立子网络:扔掉了共享主干带来的迁移(战斗的单卡 literacy 反哺卡组评估),
    还复制一份昂贵 encoder——否决。
  - 简单冻结:最小实现;第一阶段 encoder 冻结后不可能忘战斗,省心。**采纳**。
  - **已知局限**:纯战斗 reward 对"层数(floor)/跑图进度/构筑连贯性"没有梯度,
    冻结会把主干在这些维度锁成"战斗近视"——而全局 value 恰恰需要它们。
    **升级路径**:将来发现全局 value 对 floor/进度不敏感时,切到
    "解冻 encoder + 在 batch 里掺战斗 rollout 做 rehearsal"。因为战斗 sandbox
    届时已在手,这个升级几乎免费。**故现在不把冻结写成永久方案。**

## 落地成本(已实现)

**一个关键修正**:`SimCombatEnv`(`env/core/combat_env.py`)用的是更窄的
**71-id 战斗动作空间**(`actions.N_ACTIONS`),而 `StSPointerPolicy` 输出的是
**149-id 整局动作空间**(`action_space.N_ACTIONS`)——两者对不齐,直接拿
SimCombatEnv 预训练的权重无法迁移(实测 logits 149 vs action_dim 71 直接 shape
报错)。149-id 的战斗处理逻辑住在 thick `SimGameEnv` 里、由 thin `SlayTheSpireEnv`
驱动。所以预训练必须走 thin 层这条路,才能和全局共享同一个动作头。

已实现(全部跑通,见下方 sanity):

1. **`env/core/combat_backend.py`(新)**:`SimCombatBackend(SimGameEnv)`,
   把整局 thick backend **裁剪到单场战斗**——`reset()` 直接随机 loadout +
   `create_battle`,`is_over` 在战斗结束即 True,`_settle` 不再进 reward/走图。
   战斗的 `legal_semantic_actions`/`step`/`obs` 全部**原样复用** SimGameEnv,
   因此 thin 层给出的 149-id 动作空间 + mask 与全局**完全一致**,权重可迁移。
2. **`factories.random_combat_factory`(新)**:采样起始牌组 + N 张随机 Ironclad
   牌(`set_deck`)、Burning Blood + 随机遗物子集(`set_relics`)、随机初始血量
   (`cur_hp`)。卡池/遗物池硬编码在 `factories.py`(sim enum / stable id)。
3. **`rl/reward.CombatRewardWrapper`(新)**:HP delta(`/HP_SCALE`)+ 胜局
   `+win` + 死亡 `death=-5`;从 `thick.combat_state()` 读 bc 的 hp/outcome
   (`gc.cur_hp` 在战斗中是 stale 的)。镜像 `ShapedRewardWrapper`。
4. **`rl/train_combat.py`(新)**:thin `SlayTheSpireEnv(backend=SimCombatBackend)`
   + `CombatRewardWrapper` + `TimeLimit`,复用同一 `StSPointerPolicy`,存 .zip。
   带 `combat/win_rate` 监控。
5. **`rl/train.py` 加 `--init-from <combat.zip>` + `--freeze-encoder`**:前者只
   `load_state_dict` 权重、step 计数从 0 开始(区别于沿用计数的 `--resume`);
   后者对 `core.encoder.parameters()` 置 `requires_grad=False`(Adam 自动跳过)。

sim 绑定已现成:`gc.set_deck`/`set_relics`/`cur_hp`/`deck_ids`/`relic_ids` →
随机化纯 Python,无需改 C++。

**潜在 C++ 改动(已推迟)**:随机化药水需要新增绑定(`set_potions` /
`obtain_potion(name)` + 暴露 `Potion` 枚举)并重编译。v1 不做。

**踩到的本机坑**:本机 `slaythespire.cp311-*.pyd` 是 06-25 00:44 的旧构建,缺
今天加进 `slaythespire.cpp` 的 `bc.legal_potions()` 绑定 → `game_env`(全局也走
这条)在本机会崩。已 `bash sim/build_local.sh` 重新编译修复。autodl 远端若是新
构建则不受影响。

## Sanity(本机已验证)

- `random_combat_factory`:牌组 18/27 张、遗物随机子集、HP 27/56/80,
  胜局 reward ≈ +0.9、死亡 ≈ -5.3。
- `python -m rl.train_combat`(dummy/CPU/256 步)跑通,`combat/win_rate` 正常记录,存 ckpt。
- `python -m rl.train --arch transformer-pointer --init-from <ckpt> --freeze-encoder`:
  warm-start 成功、冻结 108 个 encoder 张量、训练继续并存档。

启动命令(远端正式训练):

    python -m rl.train_combat --steps 2000000 --n-envs 8 --pool normal --wandb
    python -m rl.train --arch transformer-pointer --init-from <combat>/final.zip \
        --freeze-encoder --steps 10000000 --n-envs 8 --wandb

# `env/` — Gymnasium 战斗环境

包装 `sim/`(sts_lightspeed)的强化学习环境层。**是一个真正的包** —— 从仓库根目录用模块方式跑脚本(`python -m env.scripts.play`),核心逻辑用相对导入(`from env.core import encode`)。

## 设计铁律

1. **`env` 不许 import `rl`**(算法层)。env 只认 sim,这样才能随时换算法 / 挂 SB3 / 给别人跑 baseline。
2. **纯模块(`encode` / `actions` / `render` / `encounters` / `cards_vocab`)不许 import `slaythespire`**。它们只跟 canonical dict 和抽象 action id 打交道,是 backend-agnostic 的,没编译 sim 也能 import(本地/CI 测试就靠这个)。唯一碰 sim 的是 `combat_env.py` 和 `factories.py`。
3. 这条缝的回报:将来做 `RealCombatEnv`(经 bridge 接真机做 inference)时,encode/actions/render **原样复用**,只换掉后端。policy 不知道自己跑在 sim 还是真机上。

```
        canonical 状态 ─► encode ─► obs ─► policy ─► action id ─► actions.decode ─► 后端
             ▲                                                                       │
   ┌─────────┴─────────┐                                              ┌──────────────┴────────┐
 normalize_sim   normalize_real                                 bc.execute()          CommMod 命令串
 (训练/sim)        (inference/真机)                              (SimCombatEnv)        (RealCombatEnv,未来)
```

## 布局

```
env/
  core/                 环境核心逻辑(包)
    cards_vocab.py      全集卡词表(canonical id 空间,checked-in,append-only)   [无 sim]
    vocab.py            生成的 categorical 词表(monster/move/relic/...)         [无 sim]
    intents.py          move 名 → INTENT_CATEGORY 派生                          [无 sim]
    encode.py           canonical dict → 结构化 Dict obs(OBS_VERSION=11,布局真相源) [无 sim]
    actions.py          战斗内 71-id sts↔id 映射                                [无 sim]
    action_space.py     薄层整局 181-id autoregressive 面(ACTION_VERSION=6)       [无 sim]
    game_actions.py     backend 无关的语义动作面                                [无 sim]
    render.py           canonical 状态/动作 → 人读字符串                        [无 sim]
    encounters.py       act-1 遭遇战名单(normal/elite/boss)                     [无 sim]
    spiremap.py         RunMap:重建 SpireMap 枚举可达节点                      [需 sim]
    factories.py        gc_factory 构造器(random_act1 / fixed)                 [需 sim]
    combat_env.py       SimCombatEnv(单场战斗)                                  [需 sim]
    game_env.py         SimGameEnv(厚层整局状态机)                              [需 sim]
    run_env.py          SlayTheSpireEnv(薄层整局 gym.Env)                       [需 sim]
    __init__.py         sys.path bootstrap:把 sim build 目录 + bridge/ 挂上
  scripts/              可执行入口(不被 core import)
    play.py             人类交互式玩一场战斗
    play_run.py / play_run_v2.py   人类交互式玩一整局
    test_pure.py        纯层冒烟测试(不需要 sim)
    test_combat.py      单场战斗回归(需要 sim)
    test_game.py        厚层整局回归(需要 sim)
    test_run_v2.py      薄层整局回归(需要 sim)
```

## 当前规格

- obs:`encode.OBS_VERSION = 11`,结构化 `spaces.Dict`(见 `doc/day_0623/obs_design.md`)。战斗/战外字段由模型白名单隔离，完整主卡组按每张实体牌及升级数编码。
- 动作:薄层 `action_space.N_ACTIONS = 181`,`ACTION_VERSION = 6`(整局 autoregressive id 面,见 `doc/day_0623/action_design.md`)。`actions.py` 的 71-id 平面只作战斗内 sts↔id 映射被复用。

### obs 稳定性怎么保证

模型输入一旦变就等于重训,所以 obs 靠四条纪律保持稳定:

- **定长 + 全集 + 预留零槽**:手牌不变长(`HAND_SLOTS` 个槽)、敌人铺到 `MAX_MONSTERS=5` 行零填充；主卡组最多 96 张，每张实体牌一个 token，并保留升级数；全部持有遗物进入 208 个预留槽。
- **版本号 `OBS_VERSION`**:只有改布局(键集合/某字段形状/词表顺序/scaling)才跳号 —— 跳号 = 旧 checkpoint 作废的信号。**把一个预留零槽填上真实数据,不算改布局,不跳号。**
- 词表(`cards_vocab.py` + 生成的 `vocab.py`)一律 **append-only**,绝不重排/删除。

### 已经在 obs 里 / 还没在的

- ✅ 敌人这回合的 `base_damage` × `hits` —— `to_dict()` 已导出。
- ✅ **敌人意图**:`monster_move`(真实 `MonsterMoveId`,obs v2 起 C++ 导出 `moveHistory[0]`)+ `monster_intent_cat`(纯 Python 从 move 名派生,见 `intents.py`)。v1 的「预留零槽 + 暂未导出」已落地。
- ✅ **药水使用**:`USE_POTION` 段已接入厚层与动作 mask；战斗内外都读取实时药水槽。

### 手牌编码

手牌是 `hand_ids`(每槽一个 `CARD_VOCAB` index,categorical)+ `hand_upgrades`(每槽升级数 scalar)。模型侧 embed 这些 id —— **不是** v1 那种「单标量 id 塞进 float 向量」(那会臆造虚假序结构)。policy 选牌靠 `action_space` 的 id+mask。

## 战斗怎么生成(reset)

`SimCombatEnv(pool=...)` 默认用 `factories.random_act1_factory`:每个 episode 随机抽 `seed`、`floor`、`encounter`,用 vanilla 铁甲 `GameContext`(自带初始套牌 + 燃烧之血)强制开一场 act-1 战斗。`pool` 取 `"normal"|"elite"|"boss"`。`reset(seed=...)` 可复现整条 episode 流。

要打固定战斗(测试/过拟合)用 `factories.fixed_factory(seed, encounter, floor)`,或给 env 传自定义 `gc_factory`。

> ⚠️ 只有**战斗内** RNG(`Random(seed+floor)`)和真机对齐;地图/事件 RNG 没对齐。所以"随机 encounter+seed"是 seed/floor/名字三者独立组合,不还原真机地图 —— 训练要的是战斗分布覆盖,这没问题。遭遇战名必须用 `monsterEncounterStrings` 拼写(见 `encounters.py` 注释)。

## reward(占位)

目前 = 每步 hp 变化 / `HP_SCALE`,终局 ±1(胜/负)。胜负读 `bc.to_dict()["outcome"]` 的**战斗侧 `Outcome` 枚举**(1=胜 2=负),**不是** `sts.GameOutcome`(两者数值相反,混用会把败局判成胜局)。reward shaping 以后再细化。

## 跑

```bash
# 从仓库根目录:

# 交互式玩一局(需要 sim 已编译):
python -m env.scripts.play                          # 随机 normal 战
python -m env.scripts.play --pool elite --seed 42
python -m env.scripts.play --encounter "Cultist" --seed 7

# 纯层测试(encode + actions + render),不需要 sim:
python -m env.scripts.test_pure

# 单场战斗回归(需要 sim):随机合法策略跑过 act-1 三类池
python -m env.scripts.test_combat

# 整局交互玩 / 整局回归(需要 sim):
python -m env.scripts.play_run_v2     # 薄层 SlayTheSpireEnv 跨屏幕手动玩
python -m env.scripts.test_run_v2     # 薄层整局 masked-random 回归
python -m env.scripts.test_game       # 厚层整局回归
```

> sim(`slaythespire` 的 `.so`/`.pyd`)需先在本机编译,见 `sim/DEVNOTES.md`。当前 mac 上未构建;训练机(Linux/py3.12)上已验证整体跑通。

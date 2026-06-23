# `env/` — Gymnasium 战斗环境

包装 `sim/`(sts_lightspeed)的强化学习环境层。**平铺,不作为可安装包** —— 用的时候把 `env/` 加进 `sys.path`,直接 `import encode` / `import actions` / `from combat_env import SimCombatEnv`。

## 设计铁律

1. **`env` 不许 import `rl`**(算法层)。env 只认 sim,这样才能随时换算法 / 挂 SB3 / 给别人跑 baseline。
2. **`encode.py` / `actions.py` 不许 import `slaythespire`**。它们只跟 canonical dict 和抽象 action id 打交道,是 backend-agnostic 的。唯一碰 sim 的是 `combat_env.py`。
3. 这条缝的回报:将来做 `RealCombatEnv`(经 bridge 接真机做 inference)时,encode/actions **原样复用**,只换掉 `combat_env.py` 这个后端。policy 不知道自己跑在 sim 还是真机上。

```
        canonical 状态 ─► encode ─► obs ─► policy ─► action id ─► actions.decode ─► 后端
             ▲                                                                       │
   ┌─────────┴─────────┐                                              ┌──────────────┴────────┐
 normalize_sim   normalize_real                                 bc.execute()          CommMod 命令串
 (训练/sim)        (inference/真机)                              (SimCombatEnv)        (RealCombatEnv,未来)
```

## 文件

| 文件 | 作用 | 依赖 |
|---|---|---|
| `cards_vocab.py` | 全集卡词表(371 项,**canonical id 空间**,checked-in)。append-only。 | 无 |
| `encode.py` | canonical 战斗 dict → 定长 `float32` obs 向量。obs 布局的唯一真相源。 | numpy |
| `actions.py` | 固定整数 action 空间 + `decode` / `encode_*`。 | 无 |
| `combat_env.py` | `SimCombatEnv(gym.Env)`,把 sim ↔ encode ↔ actions 接起来。 | slaythespire, gymnasium, numpy, bridge.schemas |
| `test_pure.py` | 纯层冒烟测试,**不需要编译 sim** 就能跑。 | numpy |

## 当前规格(v1)

- `encode.OBS_SIZE = 1288`,`encode.OBS_VERSION = 1`
- `actions.N_ACTIONS = 71`,`actions.ACTION_VERSION = 1`

### obs 稳定性怎么保证

模型输入一旦变就等于重训,所以 obs 靠四条纪律保持稳定:

- **定长 + 全集 + 预留零槽**:手牌不变长(`HAND_SLOTS` 个槽)、敌人铺到 `MAX_MONSTERS=5` 行零填充、卡组堆用全集计数向量。加内容不改长度。
- **版本号 `OBS_VERSION`**:只有改布局(reorder/resize/改 scaling)才跳号 —— 跳号 = 旧 checkpoint 作废的信号。**把一个预留零槽填上真实数据,不算改布局,不跳号。**
- 词表 `cards_vocab.py` 与 buff 词表(`encode.py` 里 `PLAYER_BUFFS`/`MONSTER_BUFFS`)一律 **append-only**,绝不重排/删除。

### 已经在 obs 里 / 还没在的

- ✅ 敌人这回合的 `base_damage` × `hits` + `is_attack`(派生自 `base_damage>0`)—— `to_dict()` 已导出,v1 就能看见。
- ⏳ **敌人意图类别 one-hot**(`INTENT_SLOTS`):v1 里是**预留零槽**。sim 暂未导出 intent/move_id(见 `sim/DEVNOTES.md`),补一个 cpp 绑定后填进这个槽,**obs 长度不变、版本号也不必跳**。
- ❌ **药水**:不在 v1 的 action 空间(sim 没干净地枚举药水动作)。id 空间尾部留好了,将来 append。

### 手牌编码的临时方案

手牌槽位现在存的是 **card index 标量**(占位)。单个标量 id 是糟糕的网络输入(给无关的牌强加了序数结构),v1 先这么跑通循环。将来应改成每槽 one-hot / embedding —— 那次改动会动 `OBS_SIZE`,届时 `OBS_VERSION += 1`。注意:policy 选牌靠的是 `actions` 那套 action id,**不是**靠这个标量。

## 两个挂账(见 `combat_env.py` 注释)

1. **`reset()` 收一个 `gc_factory`**:vanilla `GameContext` 造不出真机战斗(Neow→进图会把 miscRng 推到位,见 DEVNOTES),所以调用方得提供一个推进到战斗前状态的 gc + 遭遇战名。可参考 `bridge/replay.py` 的 `_sync_gc`。真正接训练时再补。
2. **reward 是占位**:目前只有 胜/负 ± hp 变化。

## 跑测试

```bash
# 纯层(encode + actions),不需要 sim:
python env/test_pure.py

# combat_env.py 需要先在本机编译出 slaythespire(.so / .pyd),见 sim/DEVNOTES.md。
# 当前这台 mac 上 sim 尚未构建;训练机(Linux)上构建后才能整体跑。
```

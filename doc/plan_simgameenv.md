# 计划:SimGameEnv（厚层整局环境）+ 薄层 SlayTheSpireEnv 预留

## 目标与决策（已确认）

- 两层抽象：**厚层 backend 专属**（`SimGameEnv` / 将来 `RealGameEnv`）+ **薄层 backend 无关**（`SlayTheSpireEnv`，gym.Env，模型唯一接口）。
- 第一版范围：**窄垂直切片** —— 打通 act-1 一条完整链：Neow → 战斗 → 卡奖 → 篝火/事件（极简）→ … → act-1 boss。商店/复杂事件先 stub（只能跳过/保守选项）。
- 时机：**先厚层**。先让 `SimGameEnv` 的语义 canonical + 语义动作面落地、能用脚本手动玩通一局，确认接口形状稳了，**薄层 encode/mask 随后**（避免 obs 设计返工）。

## 已验证的关键事实（远程探针）

1. `transition_to_map_node(x)` 进 MONSTER 房后 `screen_state` 自动变 BATTLE，需手动 `create_battle`（同 `replay.Driver.start_battle`）。
2. **从同 seed 重建的 `sts.SpireMap(seed, asc, act, burning)` 与 gc 内部地图一致**：可达边 `has_edge(x,y,x2)` 和落地房型 `get_room_type` 都和 `gc.transition` 实际结果吻合。→ env 用重建的 SpireMap 枚举"当前节点可达哪些下一节点"，**不改绑定、守住黑盒**。这是唯一硬风险，已排除。
3. `replay.py` 的 `Driver` 已是一套完整 overworld dispatch（Neow/map/rewards/event/shop/rest/chest/boss-relic），且上个 commit 已"act-1 全程跑通 floor1→18 打穿 Guardian"。`SimGameEnv` 是把它从"读 log 回放"改成"枚举合法语义动作→接受 agent 选择"。
4. Neow 选项可能开 CARD_SELECT 子屏，需在循环里消化到 MAP_SCREEN 才算一个决策点结束。

## 分层文件落点

```
env/core/
  game_env.py        # NEW  SimGameEnv：整局状态机（唯一 import sim 的整局文件）
  spiremap.py        # NEW  薄封装：从 seed 重建 SpireMap，枚举当前可达下一节点
  game_actions.py    # NEW  语义动作面（与 backend 无关的“整局动作”定义 + 编解码骨架）
  combat_env.py      # 复用：BATTLE 阶段委托给现有 SimCombatEnv（已验证可训、RNG 对齐）
bridge/schemas/
  combat_state.py    # 扩展：normalize_sim_overworld 增加“当前屏幕选项”字段（见下）
env/scripts/
  play_run.py        # NEW  整局交互脚本（仿 play.py，但跨屏幕；手动验证用）
  test_game.py       # NEW  sim 整局回归：随机策略能从 Neow 跑到 act-1 boss 终局
```

薄层 `SlayTheSpireEnv` 本计划**不实现**，只在 `game_env.py` 注释里固定它将消费的语义 canonical / 动作面契约，留待下一阶段。

## SimGameEnv 设计

### 1. 屏幕 → 语义阶段（phase）映射

env 不直接 reset 成 gym 那种“一个 episode”，整局 = 一个 episode。每个决策点按 `gc.screen_state_name` 分派：

| sim screen            | 语义 phase     | 合法语义动作                                   | 选项来源 |
|-----------------------|----------------|------------------------------------------------|----------|
| EVENT(Neow)           | `neow`         | `("neow", i)` i∈0..3                           | `gc.info.neowRewards`（已有 choose_neow_option）|
| MAP_SCREEN            | `map`          | `("map", x)` x∈可达下一节点                    | **重建 SpireMap 枚举可达边**（核心） |
| BATTLE                | `combat`       | 委托 SimCombatEnv 的 71 动作面                  | `bc.legal_actions()` |
| REWARDS               | `combat_reward`| `("claim", kind, idx)` / `("reward_card", j)` / `("proceed",)` | `gc.get_reward_info()` + `get_card_reward()` |
| CARD_SELECT(grid)     | `card_select`  | `("select", k)`                                | `gc.get_select_cards()` |
| REST_ROOM             | `rest`         | `("campfire", opt)` opt∈{rest,smith,...} / `("proceed",)` | 固定下标，按 relic 过滤（dig/recall 视遗物） |
| EVENT_SCREEN(非 Neow) | `event`        | `("event", i)` i∈0..n-1                        | **事件选项数缺口**，见风险 |
| TREASURE_ROOM         | `chest`        | `("chest", open)` / `("proceed",)`             | 固定 |
| SHOP_ROOM/SHOP_SCREEN | `shop`         | 第一版 **stub**：只 `("proceed",)` 离开         | —— |
| BOSS_RELIC_REWARDS    | `boss_relic`   | `("boss_relic", i)`                            | 固定 3 选项 |

战斗结束：`bc.is_battle_over` → `bc.exit_battle(gc)`（写回 hp/gold/deck/rng 并开 REWARDS）→ bc=None，回到屏幕分派。

### 2. SpireMap 封装（`spiremap.py`）

```
class RunMap:
    def __init__(self, seed, ascension, act, burning): self._m = sts.SpireMap(...)
    def first_row_nodes(self): [x for x in 0..6 if room(x,0)!=NONE]
    def reachable(self, x, y): [x2 for x2 in 0..6 if has_edge(x,y,x2)]   # boss 行特判
    def room_type(self, x, y): str
```
- act 切换 / burning 取值：act 从 `gc.act` 读；burning 第一版取 `False`，但**精英 buff**需要像 replay `_sync_gc` 那样在进精英前把 `cur_map_node_x/y` 坐到 burning 节点（用 `gc.burning_elite_x/y`）。第一版若 burning elite 落在路径上，复用该逻辑；否则不影响。
- 首屏 node=(-1,-1)：用 `first_row_nodes()`；之后用 `reachable(cur_x, cur_y)`。
- boss 行（y=14 的边指向 row15 boss）：`transition_to_map_node(cur_x)` 任意 x 均可（replay 已用此法）。

### 3. 语义动作面（`game_actions.py`）

- 定义语义动作的**枚举/元组协议**（如上表），以及 `SimGameEnv` 内部 `apply(gc/bc, semantic_action)` 的 dispatch。
- **暂不**定义定长 id / 全局 mask —— 那是薄层的事。厚层只暴露 `legal_semantic_actions() -> list[tuple]` 和 `apply(action)`。这样薄层后续可以独立决定怎么把可变长语义动作集压成定长 id+mask（沿用 combat 的 actions.py 思路），不锁死。

### 4. canonical 扩展（`combat_state.py`）

`normalize_sim_overworld(gc)` 现在只有 floor/hp/gold/deck/relics。扩展成包含**当前屏幕的决策上下文**：
```
{ ...现有..., "phase": <screen→phase>, "options": <该屏可选项的语义描述列表> }
```
保持 `normalize_real_overworld` 同形（真机侧 options 来自 `choice_list`，下一阶段对接）。`diff_overworld` 已可用于将来 sim↔真机整局差分。

### 5. gym 接口（厚层，先不强求 gym.Env 合规）

第一版 `SimGameEnv` 以**朴素 step/reset/render/legal API** 为主（不急于塞进 gym Box/Discrete，因为动作是可变长语义元组）：
```
reset(seed) -> 起一局到第一个决策点；返回 canonical
legal_semantic_actions() -> list[tuple]
step(semantic_action) -> (canonical, reward, terminated, info)
render() -> str  # 跨屏幕的人读视图（扩展 render.py）
```
reward 第一版：整局 = 终局 ±（赢一个 act / 死）+ 过程 hp delta，占位，细化留后。

## 测试与验证

1. `test_game.py`：固定若干 seed，随机合法策略从 Neow 跑到 act-1 boss 终局；断言：每个决策点 `legal_semantic_actions()` 非空（除非终局）、`apply` 不抛、最终能到 `act==2` 或玩家死亡、deck 在卡奖后确实增长（验证“能拿初始手牌外的牌”）。
2. `play_run.py`：你手动跨屏幕玩通一局（同 play.py 体验，扩到整局）。
3. 远程跑（本地无 sim）：rsync → conda base → 跑 test_game + 手动 play_run。

## 已知风险 / 缺口（第一版如何处理）

- **事件选项数没从 sim 暴露**：`choose_event_option(i)` 能调，但"当前事件有几个合法选项"无 getter。第一版策略：act-1 事件较少，先用 `gc.cur_event_name` 查一张**手写的事件→选项数小表**（仅覆盖 act-1 常见事件），覆盖不到的事件保守只允许"离开/第 0 项"。记为后续补绑定的 TODO（不在本计划改 C++）。
- **商店内容读取**：`shop_buy_*` 有，但"货架有哪些卡/遗物/价格"在 sim 侧没有 list getter（replay 靠真机日志）。第一版 shop 直接 `shop_proceed()` 跳过，记 TODO。
- **burning elite buff**：非 burning 路径不受影响；若路径经过 burning elite，复用 replay `_sync_gc` 的坐标设置逻辑。
- 这些缺口都**吸收在厚层**，不影响薄层 obs/action 契约 —— 符合分层纪律。

## 不做的事（守约束）

- 不改 sim 绑定 / C++（黑盒）。事件/商店缺口走 Python 侧手写表或 stub，记 TODO 待将来补绑定。
- 不动 combat 的 OBS_VERSION/ACTION_VERSION；薄层这次不写。
- env 不 import rl。

## 提交粒度

1. commit A：`spiremap.py` + 探针固化为 `test_game.py` 的地图部分（先证地图脊梁在回归里）。
2. commit B：`game_env.py` + `game_actions.py` + canonical 扩展（整局状态机跑通随机策略到 act-1 boss）。
3. commit C：`play_run.py` + render 扩展（手动玩通）。
每个 commit 都在远程验证通过再落。

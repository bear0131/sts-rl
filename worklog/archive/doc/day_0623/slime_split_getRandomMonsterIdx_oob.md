# 史莱姆分裂 → monstersAlive 计数漂移 → 战斗不结束 + 日志刷屏

**状态**：✅ 根因已修（0624，见下「根因修复」）。兜底/对账三处保留作 defense-in-depth。
远端（autodl）重编验证：1600 场随机对战（Slime Boss/Large Slime/Lots Of Slimes/Small Slimes ×400, A20）
全部正常终结，0 hang / 0 crash / 0 诊断 stderr；**临时关掉对账兜底后**仍 0 hang、`RECONCILE_WOULD_FIRE` 计数为 0，
证明 monstersAlive 已能靠正常 `die()` 路径归零，兜底不再 load-bearing。

## 根因修复（0624）

真正的病根不是"加减某一步多/漏"，而是 `largeSlimeSplit` 的**槽位冲突**：它把两个分裂子代写到
`placeIdx` 和 `placeIdx+2`、死尸放 `placeIdx+1`，**假定这两格是空的**——只有当大史莱姆是数组里最后一只时才成立。
SlimeBoss 战里两只大史莱姆位于 idx0 和 idx2（对照 `reference/cardcrawl` 的 drawX smart-positioning：
[SpikeSlime_L@0, deadBoss@1, AcidSlime_L@2]）。左边那只先分裂时，`placeIdx+2=2` 直接**覆盖掉还活着的 AcidSlime_L**，
于是真实存活数 -1 而 `monstersAlive++` 照加 → 计数卡在 >0 → 永不胜利 → 引擎对尸体派伤害刷屏/越界崩。

修法（`MonsterSpecific.cpp`）：`largeSlimeSplit` 改为像真机 SpawnMonsterAction 一样**把尾部整体右移 2 格**腾出空间，
不再覆盖活怪；若右移会超出 5 格定长数组，先**压实掉已有死尸**（死/halfDead/逃跑的怪在所有 alive-only 路径与
`getRandomMonster` 里都被过滤，丢弃尸体不影响 RNG 选怪与活怪相对顺序）。`slimeBossSplit` 同步把 boss 尸体保留在 idx1
（旧码直接 `arr[0]=Monster()` 把 boss 丢了）。net +1 的计数语义不变（父死 -1、两子 +2）。

字节对齐（bit-exact）回头再细校；当前对 RNG 选怪/胜负计数已对齐。

## 现象
SubprocVecEnv 多 worker 训练时，某场带史莱姆分裂的战斗：
- stdout 刷出上百万行 BattleContext dump（`was dead when damaged`、`count:5 alive:1`、整个战斗状态）。
- 严重时某 worker 段错误死掉 → 主进程在 socket 永久等它 → 全员死锁挂起。

## 根因（已坐实）
关键证据是 dump 本身：三只史莱姆 `hp` 全 0，但 `monstersAlive: 1`。
- 胜负判定完全依赖 `monstersAlive`：`Monster::die()`（Monster.cpp:293/334）在它归零时置 `PLAYER_VICTORY`。
- 史莱姆分裂（`largeSlimeSplit`/`slimeBossSplit`，MonsterSpecific.cpp）对 `monstersAlive` 的加减与实际存活数失配，使它卡在 >0。
- 于是战斗永不结束，引擎继续对**已死的尸体**派伤害/动作 → 每次触发 `was dead when damaged` dump → 刷屏；并最终在 `getRandomMonsterIdx` 找不到活怪而越界读野内存 → worker 崩。

## 为什么"打印却不崩/不停"
`sts_common.h:8` 无条件 `#define sts_asserts`，所以所有 `#ifdef sts_asserts` 的 cerr 都编进去了；
但 Release build 带 `-DNDEBUG`，`assert()` 是 no-op，拦不住任何东西。
`BattleContext.cpp` 的 loop 溢出分支 assert 后还**没有 break**，于是每循环一次 dump 一次 → 百万行。

## 修复（三处，均不碰健康局 RNG，保留全部诊断 dump）
1. **病根对账** `BattleContext::executeActions` 主循环：队列空闲且 `arr[]` 里
   没有任何怪还"在场"（`isAlive() || isHalfDead()`，halfDead 是觉醒者/暗影重生窗口，
   不能误判胜）时，把 `monstersAlive` 拉回 0 并判 `PLAYER_VICTORY`。健康局永不进入此分支。
2. **loop 溢出分支**（BattleContext.cpp:728）：**保留** seed+状态 dump 用于诊断，
   但 assert 后补 `outcome=PLAYER_LOSS; break` —— 失败时只 dump 一次就退出，不再刷屏。
3. **getRandomMonsterIdx 越界 guard**（MonsterGroup.cpp）：`i >= monsterCount` 时
   退回线性扫描最后一个活怪返回，防野内存读。`#ifdef sts_asserts` 的诊断 cerr 保留
   （只打一次，紧接着 guard return）。

`Monster.cpp:409/468` 的 `was dead when attacked/damaged` dump **原样保留**——它们是定位本 bug 的功臣，有了第 1 项后不该再触发；若触发会 dump 一次帮助诊断。

## 待办
拿到原版反编译源码后，核对分裂时 `monstersAlive` 的正确维护方式（哪一步多减/漏减），
修真正的加减逻辑，再视情况移除对账兜底。

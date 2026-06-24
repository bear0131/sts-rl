# 史莱姆分裂 → monstersAlive 计数漂移 → 战斗不结束 + 日志刷屏

**状态**：已加兜底/对账修复，可继续训练。字节对齐（bit-exact）回头拿到原版反编译源码再校。

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

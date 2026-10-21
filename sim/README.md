# 原生模拟器

`sts_lightspeed/src/game` 是整局规则，`src/combat` 是战斗状态/动作，`src/sim/search` 是原生搜索；严格 Full Gumbel/NOSL 的主要实现是 `Nosl.cpp`。
Python binding 位于 sts_lightspeed/ 的绑定源码；第三方 json/pybind11 保持原目录，不纳入 Python 测试搬迁或全仓格式化。

按 ../tests/README.md 从本仓库源码构建。用绝对 STS_LIGHTSPEED_BUILD_DIR 选择 .pyd/.so；测试需要 STS_ENABLE_RULE_TRACE=ON。不要把已存在的 build-gumbel-current 目录当成“当前源码已重建”的证据。
改规则需同时验证 canonical/packed 观测、合法动作与结算边界；一次训练 smoke 不能验证全部游戏规则。

## 生命范围勘误：monster-hp-ranges-v2（SIM10）

`sts_lightspeed/include/constants/MonsterIds.h::monsterHpRange` 修正三个生成规则：

| 怪物 / 适用进阶 | 旧规则 | 当前游戏 / 修正规则 |
|---|---|---|
| BanditBear，A0–A6 | 38–52 | 38–42 |
| Centurion，A7+ | 76–83 | 78–83 |
| AwakenedOne，A9+ | 300–320 | 固定 320 |

65 个物种的低/高生命范围及进阶门槛已对照参考源码；当前实际游戏 jar 的 204 个
HP 常量与参考源码一致，以上三处另核实际构造字节码。该核对是静态来源审查，
不是行为测试或全规则验收。现有 checkpoint 的观测/动作几何与原生 API 不变，
没有重训或改权重；新原生的实际模型加载/运行证据须与旧版本分开记录。

已发布的 `codex/simulator-handoff-20261021`（b5e92dc / 94187a48 原生）保持不可变，
仍含上述三个旧规则，不应当把它描述为已经包含本次勘误。本次修正的独立本机构建
位于 `sts_lightspeed/build-real-search-hp-ranges-v2`；不得覆盖已导入旧扩展的进程。
当前最终 TRAIN47 的 A0 范围只涉及第一项：旧模拟器可能把这一事件生成得偏难，
不能据此推算总胜率变化或声称权重不可用。真实桥接先按实际 HP 恢复根，不应把
构造模板的这一差异直接当作出牌/目标映射错误。

## Python 环境交接

Python 的 `env/`、`bridge/schemas/`、模型适配器和原生扩展必须来自同一源码版本，
不能只复制 `.pyd` 覆盖其他分支的旧构建。启动新进程前显式设置
`STS_LIGHTSPEED_BUILD_DIR`；Windows 还须提供编译时使用的 MinGW DLL，放在扩展旁或
由 `STS_LIGHTSPEED_DLL_DIRS` 指定。不要依赖 `build-gumbel-current` 等目录的默认选择，
也不要给已经导入扩展的进程热换文件。

整局 Gym 入口为 `env.core.run_env.SlayTheSpireEnv`，厚层语义入口为
`env.core.game_env.SimGameEnv`，单场入口为 `env.core.combat_env.SimCombatEnv`。
新建战士 AR 环境时使用真实原生枚举，而非字符串角色名：

```python
from env.core import _BUILD  # 先完成显式扩展路径与 DLL 注册
import slaythespire as sts
from env.core.run_env import SlayTheSpireEnv

env = SlayTheSpireEnv(
    character=sts.CharacterClass.IRONCLAD,
    ascension=0,
    combat_action_contract="autoregressive-v2",
)
```

这是新环境示例，不是历史 checkpoint 的通用构造参数。加载旧权重继续走
`rl.policies` / `rl.evaluation` 的已有空间与观测兼容适配；不能猜测旧模型的
观测开关、动作头或搜索配置，更不能跨协议恢复 optimizer/replay。
当前配套合同及选择来源见 [env/README.md](../env/README.md)，构建依赖和跨平台重建见
[tests/README.md](../tests/README.md)。具体交接的源码、二进制、依赖与验收边界记在
对应 `docs/YYYY-MM-DD/` 交接记录，不用某个可变的构建目录名标记版本。

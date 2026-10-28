# 原生模拟器

`sts_lightspeed/src/game` 是整局规则，`src/combat` 是战斗状态/动作，`src/sim/search` 是原生搜索；严格 Full Gumbel/NOSL 的主要实现是 `Nosl.cpp`。
Python binding 位于 sts_lightspeed/ 的绑定源码；第三方 json/pybind11 保持原目录，不纳入 Python 测试搬迁或全仓格式化。

按 ../tests/README.md 从本仓库源码构建。用绝对 STS_LIGHTSPEED_BUILD_DIR 选择 .pyd/.so；测试需要 STS_ENABLE_RULE_TRACE=ON。不要把已存在的 build-gumbel-current 目录当成“当前源码已重建”的证据。
改规则需同时验证 canonical/packed 观测、合法动作与结算边界；一次训练 smoke 不能验证全部游戏规则。

## 战斗状态复制与纯模拟器基准

`ActionQueue` 只拥有待执行回调：弹出时移动所有权，清空/战斗胜利取消时释放捕获，
复制时仅复制有效环区。选牌中断留下的回调必须独立复制，胜利后保留动作的顺序和
`clearOnCombatVictory` 标记不变；不能把整个非空队列视为临时数据丢弃。
内部所有权/环形边界回归在 `tests/integration/test_action_queue.py`。

纯引擎基准用原生循环，避免现有 Python 单步探针的绑定、观测编码开销：

```sh
cmake --build BUILD --target simulator-benchmark --parallel 2
BUILD/simulator-benchmark clone 5000000 42101 60
BUILD/simulator-benchmark playout 100000 42101 60
BUILD/simulator-benchmark search 512 42101 60
BUILD/simulator-benchmark playout 400000 42101 60 4
```

Windows 可执行文件后缀为 `.exe`。参数为模式、固定次数、种子和执行阶段秒数上限。
可选最后一个参数是线程数（1–32，默认1）。总工作量在各线程间按索引分配，每个
线程拥有独立状态和搜索器；固定线程数比较工作量和摘要，不能跨线程数比较摘要。
`seconds` 含搜索器构造和线程退出，不含线程创建；外部同时记录进程端到端时间。
超时返回非零。模式分别计量战斗复制、随机完整动作、128次模拟/深度8的无模型PUCT。
使用95个左右的A0初始牌组初始/中途状态，固定六类Act1遭遇，随机战斗最多256动作，
上限命中单独记录。JSON区分建状态与计时，给出实际动作/模拟次数、结果摘要和状态大小。
这是性能工作负载，不替代完整规则正确性验证，也不是生产神经搜索rollout。
新旧构建必须使用相同编译器/选项、seed/次数并核对work/digest，交替ABBA、记录资源；
并行测试须核对实际进程CPU占用，仅设置亲和性不证明已使用多核。
固定CPU核心的结果不能外推到四台远端吞吐。另用 `tools.validation.bitwise capture-search`
和 `compare --allow-native-rebuild` 比较真实叶子编码和搜索树字段。

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

# 模拟器环境交接：战士、无色卡与共用规则

## 本次交付

用户要求："sl 相关的先别改了，你先搞一版收尾的模拟器环境，其他任务需要用了"。
本次冻结已集成的模拟器，不再扩展 SL，不启动新一轮真机、不改权重或训练配置，
也不改正在使用的 `D:/sts-rl` 主工作区及其运行任务。

- 交接分支：`codex/simulator-handoff-20261021`。仅发布本次交接，后续开发留在原桥接分支。
- 运行时源码基线：`528d8aa1428fcd9f176830a581e500cd2e5db2f5`。
- 本次交接提交仅增加说明和更新 TODO / worklog，运行时与上述基线完全相同。
- 本机独立交接目录：`D:/sts-rl/runs/local/clone-resume-worktree/runs/local/simulator-handoff-20261021-01`。
- 其中 `runtime/` 为完整的 25 个非 `runs` 根、1,991 个 Git 文件，`source.zip` 为相同源码归档；
  `native/` 包含明确的 Python 3.11 / Windows x64 扩展和三件配套 MinGW DLL，
  `manifest.json` 记录来源、接口、版本、SHA256 和验证范围。

二进制、本机运行输出和 checkpoint 不提交 Git。远端使用完整交接分支源码；
Linux / Python 3.12 等环境必须从同源重建，不能使用 Windows `.pyd`。
本机快照不是新的 Git checkout，也不包含历史真机日志或游戏反编译参考源码。

## 本机直接使用

从新的进程使用冻结目录，先设置路径再导入环境。不要继续以其他分支的 cwd 或
已导入的 `env` / `schemas` 模块启动，否则仍可能消费旧 Python 代码。

```powershell
Set-Location 'D:/sts-rl/runs/local/clone-resume-worktree/runs/local/simulator-handoff-20261021-01/runtime'
$env:STS_LIGHTSPEED_BUILD_DIR = 'D:/sts-rl/runs/local/clone-resume-worktree/runs/local/simulator-handoff-20261021-01/native'
$env:STS_LIGHTSPEED_DLL_DIRS = $env:STS_LIGHTSPEED_BUILD_DIR
# 后续已有评测 / 训练入口使用此解释器；本段不自动发起训练或游戏：
# D:/sts-rl/.venv-eval-gpu/Scripts/python.exe
```

整局 Gym 入口是 `env.core.run_env.SlayTheSpireEnv`，角色参数是
`slaythespire.CharacterClass.IRONCLAD`，不是 `"IRONCLAD"` 字符串。
厚层为 `env.core.game_env.SimGameEnv`；原生路径加载方式及新建 AR 环境例子见
[sim/README.md](../../sim/README.md)。历史权重继续使用既有模型 / 评测适配器恢复
真实观测和搜索配置，不能把默认环境参数当成 checkpoint 合同。

## 配套版本和兼容范围

| 项目 | 本次固定值 / 边界 |
|---|---|
| 原生扩展 | `slaythespire.cp311-win_amd64.pyd`，SHA256 `94187a4857e72c15ca10a9f838cd36c35cd7bd77f3b6bab7916c2b5d8812cb34` |
| 编译环境 | GCC 14.2.0、CMake 3.30.2、Ninja、Release、C++17、`STS_ENABLE_RULE_TRACE=ON` |
| 状态接口 | OBS20 / ACTION8 / native compact layout9 / combat card snapshot3，必须配套 |
| 动作接口 | 保留 legacy / native-v1 / joint-v2；TRAIN47 沿用 autoregressive-v2 的 198 列 |
| 本机 Python | `D:/sts-rl/.venv-eval-gpu/Scripts/python.exe`，3.11.15，Windows AMD64 |
| 实际本机依赖 | torch **2.11.0+cu128**、numpy2.4.6、gymnasium1.0.0、SB3 / sb3-contrib2.9.0、psutil7.2.2 |
| 仓库锁定依赖 | pyproject / uv.lock 另行声明 torch **2.12.0**，CUDA extra 为 **cu132**；不是本机已使用环境的版本证明 |

本次不修改共享 venv，也不把已有真机证据当作 torch2.12 / 新建 Linux 环境或多机训练验收。
需要复现近期本机运行时使用上面的已有解释器；需要新环境时按
[tests/README.md](../../tests/README.md) 和同源锁文件独立构建验证，不能在运行任务上直接 sync。
原生扩展自身不链接 Torch DLL，Python 3.11 ABI 和随包三件 MinGW DLL 仍须一致。

最终 TRAIN47 checkpoint / 冻结卡向量不修改、不重复打包，仍在
`D:/sts-rl/checkpoints/final/train47-step-000688914432-eval31/`；模型 SHA256
`693670e4405f73a2d41d1b92002d520331fb8720b2298e29b5e839b355369e3a`，向量 SHA256
`5b1689cd201562a99a312171edf97a6e0ee5538a4d89924b5bed0a44daf5d4da`。
旧 checkpoint 可由保留的推理适配器消费当前环境，但不能因此宣称历史 optimizer / replay
可以跨 OBS / 动作合同继续训练。修复后的规则也不保证逐步复现旧错误规则下的轨迹或胜率。

## 收录内容与未关闭边界

本次收录此前集中修复的战士、无色卡及其共用牌堆 / 费用 / 成长 / 复制 / 药水 / 遗物、
事件奖励与怪物行为 / 生命周期 / 搜索边界链。包括 SIM05–08 的 Chomp 双击、Wither 意图、
牌堆派生缓存与零计数能力 presence；开场排队、怀表补抽、固定招式历史 / RNG、
未知遭遇 / 招式显式拒绝等近期修复也已在同一源码 / 原生构建中。
未知身份或缺失必要信息仍明确拒绝，不用 PAD / 首合法动作 / 假终局遮盖问题。

这是一版可供其他任务开发和接入的冻结环境，不是“全游戏再无错误”的认证。
唯一持久任务表仍是根 [TODO.md](../../TODO.md)：

- BRIDGE10（SL 搜索身份持久化）按用户最新要求暂不修改；不声称已修复。
- BRIDGE28 / BRIDGE37 等完整待结算、特殊选择 / 复活 / 动态槽位边界继续保留，
  不能由普通路径成功推断全部组合已覆盖。
- 其他职业和 Act4 未扩大验收；前三幕旧权重仍按既有钥匙兼容策略使用。
- BRIDGE170 是桥接 CLI 回滚路由修复，代码包含在基线，但新一轮真机验证尚未进行；
  它不改变本次原生模拟器产物。用户改变优先级后没有创建 / 启动 `resume200202-06`。

## 本轮检查与已有真实证据

1. 当前源码增量构建命令：

   ```powershell
   $env:PATH = 'D:/Dev-Cpp/MinGW64/bin;' + $env:PATH
   cmake --build sim/sts_lightspeed/build-real-search --target slaythespire --parallel 2
   ```

   exit0，Ninja `no work to do`。从原生构建提交
   `dda9970ed294bea6c6efc50823eef8fb1dc39a4e` 到冻结基线，`sim/`、`env/`、
   `rl/policies/`、`rl/search/` 无源码差异；未把旧目录名当重建证明。

2. 固定基线 Git 归档、提取目录均为 1,991 个文件；逐文件比对 ZIP 字节 SHA256 通过，
   原生和三件 DLL 的复制逐 SHA256 通过；`objdump -p` 核直接及三件 DLL 的传递依赖。
   `source.zip` SHA256 为 `263ad07c9b9375fabf5998fe836f568e4007695be8b215103fbfb36158ebbd22`。

3. 最近实际真机 `resume200202-05` 的运行时源码是 `4c37157`，使用同一原生 SHA94187a48。
   已独立核对实际模型子进程加载路径 / 完整二进制 hash，TRAIN47 / 原128搜索成功，
   从未修改的 F50 自然档通关至 F51，56/90HP、750分；67 个原件已逐 hash 恢复。
   本轮只复用其有限证据，不把它说成基线最新 CLI / 所有卡牌 / 全训练生命周期验收。

4. 遵从之前“不做测试”的要求，本轮没有 pytest、离线模型 / 环境行为测试、训练、
   replay、合成状态或新一轮真机。构建 / 静态检查和已有真实证据分别记录，未混算通过数。
   20 个既有种子仍为19胜1负，多个修复版本 / 续档不作为固定环境正式95%评测。

关联：TODO SIM09；详细执行和检查结果追加在 `worklog/2026-10-21.md`。

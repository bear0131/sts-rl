# sts-rl

在 [sts_lightspeed](sim/) 模拟器上训练杀戮尖塔（Slay the Spire）智能体，并把策略接回模拟器做人类可读的实时回放。

> 这份 README 只做**地图、入口和"为什么"**。具体接口/参数/枚举以代码为准（flag 看 `-h`，组件细节看各自的 README），这里不复述，免得变成第二份会过期的真相。

## 仓库地图

| 目录 | 是什么 | 更多 |
|---|---|---|
| `sim/` | C++ 模拟器（`slaythespire` pybind 模块），RNG 对齐真机 | `sim/sts_lightspeed/README.md` |
| `env/` | Gymnasium 环境层：canonical 状态 ↔ obs/action 编码，backend 无关 | `env/README.md` |
| `rl/` | 训练 / 评估 / 实时回放（MaskablePPO） | 各脚本 docstring |
| `bridge/` | sim ↔ 真机差分测试（replay + diff），共享 `schemas/` 规范化 | — |
| `doc/` | 设计笔记与决策背景（obs/action 设计、gameflow、审计） | — |

所有脚本都从**仓库根目录**用模块方式跑：`python -m rl.train`、`python -m env.scripts.play` 等。

## 本地怎么跑（任意开发机）

开发机可能不止一台、不同 OS。共同约定：**本地一律用 Python 3.11**（原因见下），每台机器各自编自己的 sim `.pyd`、各自建 `.venv`，二进制都不进库。

**1. 建 3.11 dev 环境**：

```bash
uv sync
```

> `pyproject.toml` 的 torch 已按平台切换：N 卡的 Windows/Linux 机装 CUDA cu132 版，macOS 装默认 PyPI 版（CPU/MPS）。所以这步在哪台机器都能跑。N 卡机器可自检 `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`。

**2. 编 sim**（自动探测平台：Windows 用 MinGW g++ 并拷运行时 DLL，mac/Linux 用系统编译器）：

```bash
bash sim/build_local.sh
```

- 前提：装好 `cmake` + 一个 C++ 编译器（Windows 用 MinGW `g++`，建议再装 `ninja`；mac/Linux 用系统 clang/g++）。
- 产物 `slaythespire.cp3XX-<平台>.pyd`/`.so`（XX = 你的 Python minor，3.11 即 cp311），落在 `sim/sts_lightspeed/build/`，二进制不进库。

**3. 实时看模型打牌**：

```bash
bash rl/pull_latest_ckpt.sh                  # 从 autodl 拉最新 checkpoint 到本地同 run-id 路径
uv run python -m rl.watch <拉下来的.zip>      # 清屏滚动回放；-h 看 --step / --delay / --stochastic
```

## 训练在哪跑

训练在远端 autodl（py3.12，自带镜像），依赖见 `rl/requirements.txt`，不复用本地 `pyproject.toml`。Checkpoint 频率/落盘看 `rl/train.py` 的 `--save-every`（默认每 ~10 万 transition 一个，落在 `rl/runs/<时间戳>/checkpoints/`）。

## 为什么本地要能编 sim + 钉 Python 3.11

> Status: Active · 2026-06-25

历史上 sim 只在远端编、开发机当无 sim 跑纯层测试。现在开发机也编，硬理由是 **sim ↔ 真机差分测试（`bridge/`）只能在装了游戏 + CommunicationMod 的开发机上做**，顺带也让本地能跑 viewer / eval。

pybind 模块按 Python 版本绑定（`cp311` 的 `.pyd` 用 3.12 解释器导不了），所以同一台机器上的 `.pyd`、`.venv`、要在本地加载的 checkpoint 必须同一个 Python。开发机统一钉 **3.11**，这样换机器时同一份 checkpoint 都能直接 load；`pyproject.toml` 的 `requires-python = "==3.11.*"` 把这条交给 `uv` 强制，用错版本直接拒绝，不靠记性。远端训练是另一套（3.12 / `.so` 是 cp312），两边二进制不通用，各编各的——正常，别混用。

# 杀戮尖塔 2 训练环境

本模块使用 [Spire Mentor](https://github.com/tfriedel/spire-mentor) 的 Rust
`sts2-py` 引擎，不依赖 Godot、游戏进程或一代 `slaythespire` 扩展。
固定来源见 [backend.json](backend.json)，面向游戏 v0.107.1。
这是社区规则实现，**尚未完成本仓库对正版游戏的差分验收**，不能声称完全一致。
上游代码和二进制只放 `runs/local/`，不纳入本仓库。

## 构建

使用 Python 3.11、Rust 1.96.0；Windows 需要 MSVC C++ Build Tools。
依赖复用根目录 `pyproject.toml` / `uv.lock`，使用独立虚拟环境防止影响一代训练。
下列命令在 Git Bash 中执行，Linux 将 `Scripts/python.exe` 换成 `bin/python`。

```bash
UV_PROJECT_ENVIRONMENT=runs/local/sts2-venv uv sync --locked --extra cpu --group test --no-dev --python 3.11
export GIT_LFS_SKIP_SMUDGE=1
git clone --filter=blob:none --no-checkout https://github.com/tfriedel/spire-mentor.git runs/local/spire-mentor
git -C runs/local/spire-mentor sparse-checkout set sts2-rs
git -C runs/local/spire-mentor checkout 32742bfc11130f5ac9f1bff6c5122a3e0784d87b
rustup toolchain install 1.96.0 --profile minimal
runs/local/sts2-venv/Scripts/python.exe -m tools.validation.sts2 build --source runs/local/spire-mentor --output runs/local/sts2-backend --timeout 1200
export STS2_BACKEND_MANIFEST="$PWD/runs/local/sts2-backend/manifest.json"
runs/local/sts2-venv/Scripts/python.exe -m tools.validation.sts2 doctor
```

`build` 检查固定 commit、上游工作树、编译退出码，记录 source hash、二进制 SHA256。
加载时检查 manifest 和二进制；缺少扩展会失败，不会自动换成近似 Python 实现。
本机已构建的上游 checkout 是 `runs/local/spire-mentor-review`。

## 环境契约

`Sts2Env(EnvConfig(task="combat" | "run"))` 提供 Gymnasium reset/step。
初期支持 Ironclad、A0–A10、第一幕 Overgrowth；其余幕由上游整局状态机推进。
`combat` 用真实新开局的第一个战斗作为 episode，前置选择固定为首个合法项；
`run` 从开局起由策略选择全部阶段。药水、出牌、结束回合、多选牌都由策略控制。

- `state` 是公开状态的 1024 维 signed hash；`candidates` 为每个候选的 256 维 hash。
  hash 有碰撞，是可训练基线表示，不能作实体 ID 或精确状态序列化。
- 原生动作字典与候选逐项对应，目标怪物槽位保留；抽牌堆顺序、RNG 和怪物私有 AI 不进入观测。
- 多选枚举完整子集并保留原始候选索引；超过 `max_actions=512` 显式失败，不静默裁剪。
- 自然胜利 +1、死亡 -1、其余 0；步数上限为 truncation，保留末状态估值。
  默认 gamma=0.99，因此是折扣胜负目标，不等价于无折扣通关率。
- `clone()` 使用原生深拷贝，供未来搜索使用。`snapshot/restore` 保存 seed 和精确动作轨迹并重放，
  校验公开状态、完整战斗 wire、RNG 游标及合法动作指纹；不能拿 lossy wire 直接恢复引擎。
- checkpoint 严格绑定环境、观测、动作、奖励、引擎二进制与训练配置；一代 checkpoint 不兼容。

## 验证

```bash
runs/local/sts2-venv/Scripts/python.exe -m pytest tests/unit/test_sts2_encoding.py tests/integration/test_sts2_environment.py -q
```

原生测试验证真实动作、clone 隔离、重放、自然终局与截断；独立编码测试验证隐藏信息边界。
完整多幕通关、所有卡牌/遗物/事件、正版差分和 GPU 性能不是这些检查的覆盖范围。

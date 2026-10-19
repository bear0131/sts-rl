# 开发与验证

本文件是维护中的构建/测试说明；工作报告和阶段结果放 docs/日期，日常进度放 worklog。
`unit/`、`integration/` 是唯二 pytest 收集目录，fixture 在 conftest.py。已知缺陷以 TODO 的稳定问题 ID 标识。

## Runtime and native build

先安装 Git、uv 和支持 C++17 的 GCC/Clang 编译器。Windows 使用 Git Bash 和兼容 MinGW；下面的 Bash 命令适用于 Linux 与 Git Bash。json/pybind11 源码已随仓库提供，无需从工作目录外复制模拟器源码。CMake/Ninja 由 test 依赖组安装；必须使用锁定的 CMake 3.x，现有 json 子项目的旧 policy 声明不兼容 CMake 4。

依赖只由根 pyproject.toml 声明，uv.lock 记录解析结果。不再分别维护 CI/远端 requirements。
CI 原生回归用 Python 3.11；元数据允许 3.12，切换解释器时必须重建并验证原生扩展。

    uv sync --locked --extra cpu --group test --no-dev --python 3.11

激活刚创建的环境，使 `python`、`cmake`、`ninja` 来自同一个安装：Linux/macOS 用 `source .venv/bin/activate`，Windows Git Bash 用 `source .venv/Scripts/activate`。所有后续命令从仓库根目录执行；不要误用系统 Python。

GPU 开发用 `--extra cuda --group test --python 3.11`，保留默认 dev 组中的 W&B/TensorBoard/Plotly。
cpu/cuda 互斥，只选择 wheel 来源；训练/评估仍使用同一套运行时版本。macOS 继续使用 PyPI Torch。
后续命令使用该环境的 `python`（或 `uv run --no-sync`），避免运行命令隐式重选安装配置。
不选择 extra 时使用 PyPI 来源；需要复现时必须显式选 cpu/cuda。
已有运行中的实验环境应在停机维护或独立环境验证后再 sync。

From the repository root:

    cmake -S sim/sts_lightspeed -B sim/sts_lightspeed/build-ci -G Ninja -DPYBIND11_FINDPYTHON=ON -DSTS_ENABLE_RULE_TRACE=ON -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
    cmake --build sim/sts_lightspeed/build-ci --target slaythespire --parallel 2
    export STS_LIGHTSPEED_BUILD_DIR="$(pwd)/sim/sts_lightspeed/build-ci"
    python -c "from env.core import _BUILD; import slaythespire; print(_BUILD, slaythespire.__file__)"
    python -m pytest -q

Set STS_LIGHTSPEED_BUILD_DIR to that absolute build directory before running
Python. If reusing a build directory from another interpreter/compiler, use a
new build directory instead of trusting its cached configuration. On Windows use Git Bash via gbash, a compatible MinGW compiler and its
runtime DLLs (beside the extension, or via STS_LIGHTSPEED_DLL_DIRS). The original
CMake setup targets GCC/Clang flags; this is not an MSVC build recipe.

Windows 若加载 `.pyd` 报找不到依赖 DLL，将 `STS_LIGHTSPEED_DLL_DIRS` 设为 CMake 输出中所选 `g++.exe` 的目录，例如 Git Bash 中 `export STS_LIGHTSPEED_DLL_DIRS='C:/Strawberry/c/bin'`（替换为实际编译器目录）。Python 需要在 import 前读到此变量；不要混用另一套 MinGW 的 DLL。

## Correctness layers

    python -m pytest -q
    python -m pytest tests -q
    python -m pytest tests/integration/test_training_smoke.py -q
    python -m tests.integration.test_combat
    python -m tests.integration.test_game
    python -m tests.integration.test_run_v2 --episodes 20 --seed 0

The main suite collects tests/unit and tests/integration, including the relocated
legacy tests. Native fixtures
require the native extension: a missing native build is an error, not a skipped
success. pytest-timeout bounds each test to 120 seconds; native deadlocks can
terminate the test process, so CI also has a job timeout.

`tests/integration/test_relic_pool.py`, `test_event_callbacks.py`,
`test_reward_transactions.py`, `test_shop_prices.py`, `test_shop_eggs.py`,
`test_shop_restock.py`, `test_deck_upgrades.py` and `test_bottle_transforms.py` use the shared
fixture to build the test-only CMake target `game-context-contract` inside the
explicit build directory. This gives isolated subprocess coverage of private pool
boundaries, copied event callbacks, reward transactions, shop pricing and bottle selection without
adding debug APIs to the Python runtime. The first case allows up to 180 seconds for compilation; each
native case has a 10-second limit. Missing CMake, a build from another checkout,
or a failed native child is a test failure, not a skip. A separate 200-purchase,
20-second Courier case exercises the loaded Python extension itself.

Known defects have strict xfail markers restricted to their expected assertion
type and a TODO ID. XPASS fails CI: remove the marker when fixing the contract.
Do not expand an xfail to hide setup/import failures.

CPU bitwise tests use one Torch thread, identical software/native binaries,
fixed seeds, real PPO gradients, forced autoresets/truncation, full model and
Adam state comparisons. Save/load checks compare both optimizer tensors and
predictions. They do not promise cross-platform/GPU equality or exact training
resume from a checkpoint without environment/RNG state.

Packed/native leaf tests compare the complete consumed observation arrays for
matching deterministic requests, not just shapes or total search visits. Equal
implementations can share a bug, so separate game semantic counterexamples are
also retained.

## Bounded training and resource benchmark

    python -m rl.benchmarks.train --output runs/local/cpu-small.json --device cpu
    python -m rl.benchmarks.train --output runs/local/gpu-medium.json --device cuda --n-envs 8 --n-steps 64 --batch-size 128 --accumulation 2 --epochs 2 --updates 3 --d-model 128 --layers 4 --heads 8 --timeout 180
    python -m rl.benchmarks.train --output runs/local/gpu-subproc.json --device cuda --vec-env subproc --n-envs 8 --n-steps 64 --batch-size 128 --d-model 128 --layers 4 --heads 8
    python -m rl.benchmarks.train --output runs/local/nosl-cpu.json --device cpu --search puct --n-steps 64 --updates 4 --timeout 180

Each command supervises a separate worker with a wall-time budget; timeout exits
124 and kills only that worker's process tree. Reusing an output filename fails
instead of overwriting evidence. The runner uses WholeRun wrapper ordering,
split-pointer policy, normal or hierarchical PPO, and supports existing cached
subprocess/grouped vector environments. --search puct/full/strict enables native
NOSL self-play and hierarchical combat HP, with depth=0 and an explicitly bounded
replay (default 2,048 rows). Search runs fail if no completed-fight labels reach
replay: merely running the overworld PPO must not masquerade as an AZ update.
The training tests enter short fights through the real run state machine and
verify native PUCT/strict Full Gumbel targets reach the learner. Neither these
tests nor benchmarks cover real multi-machine synchronization or long-term convergence.
The `training` marker also includes a real replay/Adam comparison between the
local learner and a single-rank loopback ring adapter; CI selects the complete
marker group so new training regressions are not omitted by a fixed file list.

JSON records configuration, commit/dirty state, actual native path/hash, dependency
versions and hardware, setup/warmup/process/measurement time, rollout and
optimization time, actual transitions/s, buffer/replay/parameter storage, process-tree
RSS and CPU, system available memory, and GPU device telemetry plus Torch
allocated/reserved/peak bytes. NVIDIA device samples include other processes;
100% process-tree CPU is one logical core and RSS can count shared pages twice.
Short runs and sampling can miss peaks. Missing metrics are null or reported
errors, not zero usage. Different microbatch sizes are different PPO statistical
configurations until F12 is resolved; comparisons are not pure speed ablations.

CI only checks this path runs, terminates and emits evidence. Compare speed,
memory and utilization on identical hardware with repeated runs and matched
workload. Exclude warmup from sustained throughput, but retain process startup
for end-to-end cost. Use a sufficiently long run for utilization conclusions.

## Across-revision bitwise gate

Use the same interpreter and explicit absolute STS_LIGHTSPEED_BUILD_DIR in both
checkouts. Record both exact revisions and keep the baseline checkout available.
The baseline must contain the capture command; otherwise apply the same harness
without changing the baseline runtime and record that change in the evidence.
Do not compare compressed checkpoint bytes: zip metadata is not model semantics.

    # In the baseline checkout:
    python -m tools.validation.bitwise capture --output runs/local/before.json
    # In the candidate checkout (substitute ABS_BASELINE with the absolute path):
    python -m tools.validation.bitwise capture --output runs/local/after.json --restore ABS_BASELINE/runs/local/before-checkpoints
    python -m tools.validation.bitwise compare ABS_BASELINE/runs/local/before.json runs/local/after.json

Each capture has a 180-second default timeout and refuses an existing JSON output.
Four modes each train 64 transitions, then restore the baseline checkpoint and
evaluate eight steps. CPU tensors, Adam, RNG, rollout, initialized replay slots,
evaluation observations/actions/values and native search outputs are compared by
dtype/shape/raw-byte SHA256. The short-combat fixture grants Strength to complete
fights; this is a correctness workload, not a gameplay quality experiment.

For a real checkpoint, run a bounded strict-search evaluation in both
checkouts (no remote operations, same 8-simulation seed-219 fixture):

    python -m tools.validation.eval_revisions --baseline-repo ABS_BASELINE --candidate-repo . --checkpoint CHECKPOINT --output-dir runs/local/real-eval

The evaluator requires the legacy evaluation entrypoint in the baseline
(`rl.evaluation.nosl.eval_full_gumbel_bash_clash`). JSON bytes must match. The output directory must be new.
An absolute shared native build is required. This validates CPU inference/search
for this fixture, not four-rank SSH training, bf16 GPU equality, or resume-state
continuity of a live remote experiment.

    python -m tools.validation.locate_legacy --verify

Archive verification hashes each file against the pre-move manifest; archived
user material deliberately retains original whitespace via .gitattributes.

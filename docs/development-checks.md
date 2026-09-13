# Development checks and performance evidence

These are persistent entry points, not a progress report. See AGENTS.md, TODO.md
and worklog/ for working conventions and unresolved defects.

## Runtime and native build

Use Python 3.11 and an isolated environment. Install a Torch wheel suitable for
your platform; CPU CI installs torch==2.12.0 from the official CPU wheel index.
Then install requirements/ci.txt. The tested developer machine uses
torch==2.12.0+cu132; do not replace an existing GPU environment just to run CI.

From the repository root:

    python -m pip install -r requirements/ci.txt cmake ninja
    cmake -S sim/sts_lightspeed -B sim/sts_lightspeed/build-ci -G Ninja -DPYBIND11_FINDPYTHON=ON -DSTS_ENABLE_RULE_TRACE=ON
    cmake --build sim/sts_lightspeed/build-ci --target slaythespire --parallel 2

Set STS_LIGHTSPEED_BUILD_DIR to that absolute build directory before running
Python. Pass -DPython_EXECUTABLE=<your-interpreter> if CMake finds the wrong
Python. On Windows use Git Bash via gbash, a compatible MinGW compiler and its
runtime DLLs (beside the extension, or via STS_LIGHTSPEED_DLL_DIRS). The original
CMake setup targets GCC/Clang flags; this is not an MSVC build recipe.

## Correctness layers

    python -m pytest -q
    python -m pytest tests -q
    python -m pytest tests/test_training_smoke.py -q
    python -m env.scripts.test_combat
    python -m env.scripts.test_game
    python -m env.scripts.test_run_v2 --episodes 20 --seed 0

The main suite includes existing rl/ and env/scripts/ tests. New tests/ fixtures
require the native extension: a missing native build is an error, not a skipped
success. pytest-timeout bounds each test to 120 seconds; native deadlocks can
terminate the test process, so CI also has a job timeout.

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

    python -m rl.benchmarks.train --output artifacts/local/cpu-small.json --device cpu
    python -m rl.benchmarks.train --output artifacts/local/gpu-medium.json --device cuda --n-envs 8 --n-steps 64 --batch-size 128 --accumulation 2 --epochs 2 --updates 3 --d-model 128 --layers 4 --heads 8 --timeout 180
    python -m rl.benchmarks.train --output artifacts/local/gpu-subproc.json --device cuda --vec-env subproc --n-envs 8 --n-steps 64 --batch-size 128 --d-model 128 --layers 4 --heads 8

Each command supervises a separate worker with a wall-time budget; timeout exits
124 and kills only that worker's process tree. Reusing an output filename fails
instead of overwriting evidence. The runner uses WholeRun wrapper ordering,
split-pointer policy, normal or hierarchical PPO, and supports existing cached
subprocess/grouped vector environments. It does not benchmark NOSL self-play,
Ray, remote ring synchronization or long-term convergence.

JSON records configuration, commit/dirty state, actual native path, dependency
versions and hardware, setup/warmup/process/measurement time, rollout and
optimization time, actual transitions/s, buffer/parameter storage, process-tree
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

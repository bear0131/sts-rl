# 战斗搜索

`native.py` 是 rollout、eval 和交互观看共用的 native 搜索驱动。NativeSearchConfig 描述参数，complete_searches 支持本地或 VecEnv RPC 回调，在全部 active worker 间合批叶子推理；search 是单环境适配。环境/模型生命周期与训练探索由调用方管理。

`env/search/nosl_sampler.py` 派生 NOSL nonce，`env/search/nosl_is_az.py` 提供 worker 内无模型的 Python 同步/session 搜索；`nosl_search_methods.py` 是 random/beam/MCTS 对照；`combat_value_search.py` 是 checkpoint value 驱动搜索。

当前大实验用的严格 Full Gumbel 树在 C++ `sim/sts_lightspeed/src/sim/search/Nosl.cpp`，经 env/core/run_env.py 的 begin/backprop/packed leaf 接口调用。不要把这里的 Python 对照实现当作当前 native 算法的唯一真相。
输入合法动作、先验、nonce 和预算；输出完整动作/访问分布/value。搜索不得修改 live env。depth=0 是按新节点展开而非零次搜索。

`complete_searches_streaming` 支持两种跨独立树的 GPU 合批策略。`fixed_wait` 是旧兼容模式：达到目标
batch 或最老 leaf 超过固定等待时间即前向。`adaptive` 以配置的 batch size 为上限、从半上限开始，
只在固定桶之间根据 fill、ready backlog、queue delay 和实际 leaves/s 带滞回调节；每批目标还受实际
活跃树容量限制。所有 worker 都已返回时必须立即 flush，时间参数在该模式只作为按 worker round-trip
延迟自适应的 straggler watchdog 下限。两种模式都保持每棵树只有一个 pending leaf；不要用增大
`leaf_batch_size` 冒充跨树合批优化。
# Leaf inference acceleration

`inference.CombatLeafGraphs` is an opt-in CUDA backend for compact combat leaves.
It uses an LRU cache of at most 32 observation shapes/relic widths. Exact batch
shapes are the default. `--leaf-graph-batch-multiple 32` optionally rounds the
batch dimension up, duplicates valid independent rows and discards their outputs.
Relic width, observations of real rows and the per-tree search budget are unchanged.
Model parameters retain their storage
and AMP casts run inside the graph, so in-place optimizer updates are visible.
Replace the backend after replacing parameters, model structure, device or AMP
mode. It requires eval mode and is never serialized in policy checkpoints.

Distributed training selects it with `--leaf-inference cuda-graph`; eager remains
the default for other models/devices. GPU tests in
`tests/integration/test_leaf_graphs.py` cover input/weight changes, LRU eviction,
FP32/BF16 same-shape bitwise equality for eager graphs and checkpoint exclusion. Changing batch
shape may select different GEMM kernels: padded-vs-original floating-point outputs
are not guaranteed bitwise, and search trajectories can consequently diverge.
CPU CI cannot validate CUDA.

Unified-upper FP32 inference uses this cache with `device_outputs=True` to keep
logits and values on the GPU for its existing action distribution. Returned
tensors own their storage across replay, eviction and collection cleanup. The
completion event still protects pinned input reuse; device clones record their
stream before graph pools can be released. This mode supports non-joint FP32
cores with independent pools only. Default combat leaf outputs remain owned
NumPy arrays, and autoregressive shared-scratch graphs cannot use device outputs.
No action masking, probability calculation or per-game sampling moves devices.

`--leaf-amp-mode fp16` selects FP16 only for autoregressive entity/outer-return
leaf graphs. The default follows `--amp-mode`; learner, root inference and the
frozen upper teacher retain their existing precision. Other cores reject this
override. FP16 uses full-precision GEMM partial sums and uncached weight casts;
its reduced range requires finite-output and FP32 error checks on the retained
checkpoint family. Changing precision can change search trajectories. Treat it
as an explicit inference experiment until full-collection and strategy checks
have passed on the target machines.
No-grad CUDA FP16 entity attention also keeps Q/K/V projection in batch-first
layout; BF16 and FP32 inference retain their established projection layout.

Autoregressive EntityCombat BF16 leaves fuse the whole forward with
`torch.compile(fullgraph=True, dynamic=True)` before CUDA graph capture. The
FP16 path additionally uses `mode='max-autotune-no-cudagraphs'`; this tunes
kernel choices while leaving graph capture and memory ownership to this runtime.
Compilation and tuning belong to the reported cold warmup budget. The callable
is owned by this runtime; training, root prediction, legacy contracts
and FP32 retain their eager path. The card table is an explicit input whose
stable storage is refreshed before replay after weight changes. Fused BF16
reductions have different intermediate rounding: production-geometry tests
bound logits, legal-action probabilities and utility separately (not bitwise).
First-use compilation is included in graph capture time; report cold startup
and steady collection time separately. Dynamic batch/relic dimensions reuse a
compiled program where supported, but dimension-one specialization can still
require another compile. Replacing the backend resets its callable as well.
Entity cores represent zero active relics with one masked slot. The runtime
normalizes that width before compilation and supplies a positive-width guard
so cold width growth can compile without a symbolic `Max(1, width)` ambiguity.
Other cores retain their existing width contract.

Inductor's disk cache also records compiled kernels and autotuned launch choices;
it is separate from the in-process CUDA graph LRU. A fresh compile can choose
different configurations and produce different floating-point outputs even with
the same checkpoint, input, GPU model and dependency versions. For a strict
before/after comparison, preserve each rank's compiler cache together with its
environment identity, compare fixed leaf inputs first, then validate a complete
collection after warmup, full-state restore and graph-memory release. Matching
fixed inputs alone does not establish identical search trajectories. Clearing
the cache is not an accuracy fix, and copying it across dependency/device changes
does not establish compatibility. Cache artifacts belong in ignored run storage,
not policy checkpoints or Git.

`--leaf-graph-profile-interval N` samples CUDA stream H2D/replay/D2H intervals;
zero disables event sampling. Capture/copy/output wall times and cache counts
are recorded per collection under `nosl_is_az/leaf_graph_*`. Stream event times
include host submission gaps; output wall time includes waiting for GPU work.
These overlapping measurements must not be added as independent kernel costs.

Pure autoregressive entity forwards share one scratch pool within a cache.
Their inputs and refreshed card table are allocated outside capture; each graph
copies its result to its own pinned host buffer inside capture and retains no
device output. Thus no captured allocation carries state between graphs, and
independent shapes can replay in any order. Only one submission may be pending
per cache. This specialization must not be used for graphs with device outputs
or intermediate state consumed by other replays; those require the normal
capture/replay-order rules for shared CUDA pools. Other cores keep private pools.

Each graph captures its cuBLAS workspace inside its pool. Warmup workspaces must be cleared
before capture and the external cache released afterward; otherwise graph
eviction/stream reuse can free memory referenced by a different live graph.
`test_graph_owns_workspace_after_blas_cache_is_released` reproduces that lifetime
failure independently of a long game rollout. This uses the CUDA runtime's
`torch._C._cuda_clearCublasWorkspaces` binding (validated on Torch 2.12.0+cu132).
Production recovery directories receive CPU leaf NPZ/JSON evidence on a graph
runtime error; a poisoned CUDA context is never silently reused.
The distributed entity trainer clears captured leaf graphs after each collection
and before learner backward, then recaptures them on the next collection. The
compiled callable remains; this releases graph private pools when the full
learner microbatch needs GPU memory.

Grouped native search also reports `nosl_is_az/worker_*` timing. Each worker
response carries the C++ leaf call, Python leaf conversion, backpropagation,
begin setup, grouped method, group packing, parent receive/unpack and roundtrip
durations. The corresponding `worker_packet_count` gives the denominator for
mean latency per group response. Worker durations are summed across parallel
processes and therefore must not be added to `collection_seconds` or treated
as serial wall time. Roundtrip includes scheduling and time before a ready
response is collected; its residual is not a pure IPC measurement.
Within one grouped worker reply, compatible packed leaves can share a single
NumPy observation decode. Each search tree retains its own root-global fields,
request IDs, legal mask, and ordered response slice. A single leaf and the
canonical fallback use the original decoder. This changes only worker-side
observation materialization; native tree expansion and backup are unchanged.

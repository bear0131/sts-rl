# 战斗搜索

`native.py` 是 rollout、eval 和交互观看共用的 native 搜索驱动。NativeSearchConfig 描述参数，complete_searches 支持本地或 VecEnv RPC 回调，在全部 active worker 间合批叶子推理；search 是单环境适配。环境/模型生命周期与训练探索由调用方管理。

`env/search/nosl_sampler.py` 派生 NOSL nonce，`env/search/nosl_is_az.py` 提供 worker 内无模型的 Python 同步/session 搜索；`nosl_search_methods.py` 是 random/beam/MCTS 对照；`combat_value_search.py` 是 checkpoint value 驱动搜索。

当前大实验用的严格 Full Gumbel 树在 C++ `sim/sts_lightspeed/src/sim/search/Nosl.cpp`，经 env/core/run_env.py 的 begin/backprop/packed leaf 接口调用。不要把这里的 Python 对照实现当作当前 native 算法的唯一真相。
输入合法动作、先验、nonce 和预算；输出完整动作/访问分布/value。搜索不得修改 live env。depth=0 是按新节点展开而非零次搜索。
# Leaf inference acceleration

`inference.CombatLeafGraphs` is an opt-in CUDA backend for compact combat leaves.
It uses an LRU cache of at most 32 exact observation shapes/relic widths; it does
not pad batches or change search ordering. Model parameters retain their storage
and AMP casts run inside the graph, so in-place optimizer updates are visible.
Replace the backend after replacing parameters, model structure, device or AMP
mode. It requires eval mode and is never serialized in policy checkpoints.

Distributed training selects it with `--leaf-inference cuda-graph`; eager remains
the default for other models/devices. GPU tests in
`tests/integration/test_leaf_graphs.py` cover input/weight changes, LRU eviction,
FP32/BF16 bitwise equality and checkpoint exclusion. CPU CI cannot validate CUDA.

Each graph captures its own cuBLAS workspace. Warmup workspaces must be cleared
before capture and the external cache released afterward; otherwise graph
eviction/stream reuse can free memory referenced by a different live graph.
`test_graph_owns_workspace_after_blas_cache_is_released` reproduces that lifetime
failure independently of a long game rollout. This uses the CUDA runtime's
`torch._C._cuda_clearCublasWorkspaces` binding (validated on Torch 2.12.0+cu132).
Production recovery directories receive CPU leaf NPZ/JSON evidence on a graph
runtime error; a poisoned CUDA context is never silently reused.

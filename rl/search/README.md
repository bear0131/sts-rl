# 战斗搜索

`env/search/nosl_sampler.py` 派生 NOSL nonce，`env/search/nosl_is_az.py` 提供 worker 内无模型的 Python 同步/session 搜索；`nosl_search_methods.py` 是 random/beam/MCTS 对照；`combat_value_search.py` 是 checkpoint value 驱动搜索。

当前大实验用的严格 Full Gumbel 树在 C++ `sim/sts_lightspeed/src/sim/search/Nosl.cpp`，经 env/core/run_env.py 的 begin/backprop/packed leaf 接口调用。不要把这里的 Python 对照实现当作当前 native 算法的唯一真相。
输入合法动作、先验、nonce 和预算；输出完整动作/访问分布/value。搜索不得修改 live env。depth=0 是按新节点展开而非零次搜索。

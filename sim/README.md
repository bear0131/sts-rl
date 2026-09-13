# 原生模拟器

`sts_lightspeed/src/game` 是整局规则，`src/combat` 是战斗状态/动作，`src/sim/search` 是原生搜索；严格 Full Gumbel/NOSL 的主要实现是 `Nosl.cpp`。
Python binding 位于 sts_lightspeed/ 的绑定源码；第三方 json/pybind11 保持原目录，不纳入 Python 测试搬迁或全仓格式化。

按 ../docs/development-checks.md 从本仓库源码构建。用绝对 STS_LIGHTSPEED_BUILD_DIR 选择 .pyd/.so；测试需要 STS_ENABLE_RULE_TRACE=ON。不要把已存在的 build-gumbel-current 目录当成“当前源码已重建”的证据。
改规则需同时验证 canonical/packed 观测、合法动作与结算边界；一次训练 smoke 不能验证全部游戏规则。

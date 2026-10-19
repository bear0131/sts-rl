# 原生模拟器

`sts_lightspeed/src/game` 是整局规则，`src/combat` 是战斗状态/动作，`src/sim/search` 是原生搜索；严格 Full Gumbel/NOSL 的主要实现是 `Nosl.cpp`。
Python binding 位于 sts_lightspeed/ 的绑定源码；第三方 json/pybind11 保持原目录，不纳入 Python 测试搬迁或全仓格式化。

按 ../tests/README.md 从本仓库源码构建。用绝对 STS_LIGHTSPEED_BUILD_DIR 选择 .pyd/.so；测试需要 STS_ENABLE_RULE_TRACE=ON。不要把已存在的 build-gumbel-current 目录当成“当前源码已重建”的证据。
改规则需同时验证 canonical/packed 观测、合法动作与结算边界；一次训练 smoke 不能验证全部游戏规则。

主牌组的实际取得效果集中在 `Deck::obtain`：御守先消耗挡诅咒次数，剩余实际取得的诅咒才触发黑石护符最大HP/回血及陶瓷小鱼金币。`obtainRaw`、`set_deck_state`等恢复接口不重放取得效果。最大生命变化复用GameContext的HP入口，减最大HP最低1；Bloom只阻止回血。药贩子重复取得突变剂转为Circlet，首次计数1、重复递增。A0/A20与公开星盘路径回归见 `tests/integration/test_bottle_transforms.py`，需要从当前源码重建原生扩展。

巫毒娃娃的开战力量与`gc.relic_state()`公开counter由当前主牌组全部诅咒数派生，包含AscendersBane、CurseOfTheBell和Necronomicurse。简化恢复接口接受历史counter输入，但其值不会覆盖派生计数；取得遗物与恢复牌组的顺序不影响结果。用于事件可移除资格的`Deck::hasCurse`保持原语义。回归见`tests/integration/test_duvu_doll.py`；观测shape和词表不变，旧扩展需要重建才能获得该修复。

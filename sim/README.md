# 原生模拟器

`sts_lightspeed/src/game` 是整局规则，`src/combat` 是战斗状态/动作，`src/sim/search` 是原生搜索；严格 Full Gumbel/NOSL 的主要实现是 `Nosl.cpp`。
Python binding 位于 sts_lightspeed/ 的绑定源码；第三方 json/pybind11 保持原目录，不纳入 Python 测试搬迁或全仓格式化。

按 ../tests/README.md 从本仓库源码构建。用绝对 STS_LIGHTSPEED_BUILD_DIR 选择 .pyd/.so；测试需要 STS_ENABLE_RULE_TRACE=ON。不要把已存在的 build-gumbel-current 目录当成“当前源码已重建”的证据。
改规则需同时验证 canonical/packed 观测、合法动作与结算边界；一次训练 smoke 不能验证全部游戏规则。

主牌组的实际取得效果集中在 `Deck::obtain`：御守先消耗挡诅咒次数，剩余实际取得的诅咒才触发黑石护符最大HP/回血及陶瓷小鱼金币。`obtainRaw`、`set_deck_state`等恢复接口不重放取得效果。最大生命变化复用GameContext的HP入口，减最大HP最低1；Bloom只阻止回血。药贩子重复取得突变剂转为Circlet，首次计数1、重复递增。A0/A20与公开星盘路径回归见 `tests/integration/test_bottle_transforms.py`，需要从当前源码重建原生扩展。

巫毒娃娃的开战力量与`gc.relic_state()`公开counter由当前主牌组全部诅咒数派生，包含AscendersBane、CurseOfTheBell和Necronomicurse。简化恢复接口接受历史counter输入，但其值不会覆盖派生计数；取得遗物与恢复牌组的顺序不影响结果。用于事件可移除资格的`Deck::hasCurse`保持原语义。回归见`tests/integration/test_duvu_doll.py`；观测shape和词表不变，旧扩展需要重建才能获得该修复。

地图玩家动作使用`GameContext.choose_map_node(x)`，检查地图画面及原生当前候选后进入房间。`map_node_options()`返回当前位置的候选列（调用方负责画面条件），与C++ GameAction共用`getMapMoveCost`。飞靴初始3次，只在有出边的当前位置飞向未直接相连的下一层实际节点时扣次，耗尽为-2；普通连线、首行及Boss入口不扣。`set_relic_state`恢复时须显式给counter，`set_relics`按新飞靴初始化。`transition_to_map_node`仍是脚本事件/受控状态使用的底层房间入口，只检查边界，不作为玩家动作验证；SecretPortal使用该内部路径。

Python RunMap保存物理DAG，`full_view(next_columns=...)`使用原生当前候选设置choice/reachable，不修改children。SimGameEnv、replay和Watch同步使用此合同；新Python必须配合重建的native map_node_options/choose_map_node接口，无旧扩展静默回退。观测shape、词表与缩放不变；现役局部三槽预览的限制见BRIDGE40。回归`tests/integration/test_wing_boots.py`覆盖取得、消耗、耗尽、原生分支及公开动作/观测，不代表实机确认或完整真实遗物恢复（RELIC02）。

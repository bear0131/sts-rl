# 环境与 worker 搜索

`core/game_env.py` 是厚层整局状态机，`core/run_env.py` 是薄层 Gym 接口；`core/combat_env.py` 是单场战斗环境。它们调用显式选择的 slaythespire 构建。

| 模块 | 契约 |
|---|---|
| core/encode.py、vocab.py、cards_vocab.py | canonical 字段到固定观测；字段/形状/词表版本以源码为准 |
| core/action_space.py、actions.py、game_actions.py | 整局/战斗动作与候选映射，mask 和 native action 必须一致 |
| core/factories.py、encounters.py、spiremap.py | 初始场景、遭遇与地图构造 |
| core/run_env.py | reset/step、缓存、fork、native begin/backprop/packed leaf |
| [search](search/README.md) | 无模型的 Python NOSL session/nonce，供 worker 与学习侧共同使用 |
| scripts/ | 人工 play、binding 探针与接口文档生成；不是 pytest 目录 |

纯 encode/vocab/action 模块不加载 simulator；原生环境与工厂加载 simulator。env 不依赖 rl/apps/tools/tests，依赖方向由 tests/unit/test_repository_layout.py 检查。
共享状态规范化在 bridge/schemas；模型额外观测裁剪在 rl/policies/model.py。不要因当前模型不使用某字段就改变环境实际状态或动作语义。

测试在 ../tests。怪物观测槽位修复及剩余pending来源、候选区域问题见根TODO F01/F02/F03；gauntlet通过native commit_battle_state保存持久状态，同时禁用胜利回血与奖励界面回调。

怪物观测第i槽对应当前后端的动作target i。原生INVALID预留槽在canonical保留为PAD，数值与语言字段全零；死亡实体保留其原槽。C++ packed leaves遵守相同合同，`nosl_compact_layout().monster_slot_version == 1`才可启用；旧扩展退回canonical编码，需重建本checkout扩展才能恢复packed路径。OBS_VERSION=16的shape、词表和缩放不变，这是修正既定槽位合同；旧checkpoint可加载，但在有空槽的战斗上输入分布改变，历史指标需保留原代码版本，不能直接当作等价评测。回归 `tests/integration/test_monster_slot_identity.py`。

## 完整战斗动作契约 native-v1

新策略可显式创建 `SlayTheSpireEnv(combat_action_contract='native-v1')`。
默认仍为 `legacy`；71-id combat codec、181-id whole-run mask、神经观测和旧
checkpoint 的布局及已有索引含义均未重映射。整局181动作的单选直接使用原生
候选索引0..95，经SELECT语义执行，不再经过71动作的十槽适配；普通执行、搜索
分支及原生搜索共用这个映射。固定181契约遇到超出95的候选明确报错，不能溢出到
MAP_NODE等动作段。独立71动作接口仍只有十个选牌槽；候选观测缺失仍属F03。

- `native_combat_actions()` 返回完整原生 uint32 动作的 tuple，保留 source/target
  索引，包含卡牌、单选、原生 `rule_actions` 支持的全部 MULTI 子集、结束回合和
  可饮用药水。弃药不在此版本契约内。依赖显式 native 构建提供 `rule_actions`。
- `step_native_combat(bits)` 先验证当前完整动作集合，再复用原生执行、结算、奖励
  和终局路径；非法动作或未完成的薄层 target/multi 选择抛错，不推进状态。
- `combat_selection_candidates()` 返回 `SelectionCandidate`，包括版本、context、
  zone、native_index、action_bits 和卡牌 `(native_id, upgrades)`；CODEX skip
  的 card 是 None。只暴露当前单选界面的合法候选，draw/discard/exhaust 使用真实
  区域，绝不按手牌十槽截断或悄悄压缩索引。MULTI 通过完整 bitset 动作表达。
- `Decision(None, source, native_action=bits)` 由共用 evaluation runner 执行，
  与 thin action 互斥。完整动作不经过 thin mask；wrapper 必须显式提供该执行
  方法并承担自己的 transition 处理，runner 不绕过 wrapper。info/result row 的
  `combat_action_contract` 及 strategy 同名属性的 manifest 记录所用契约。

native-v1 仅在完整合法集合也为空时将战斗判作 dead_screen。因此当前 thin mask
可以全 False，而完整候选仍可执行；旧模型必须继续使用 legacy 默认。分支复制保持
候选身份。**现有 NOSL redeterminize 会重排 draw**：
已打开的 draw-select 不允许将原环境索引带入重排后的副本；新 planner 应先采样，
再用 `remap_draw_selection_from` 按原生实体身份映射选择。不能先执行选择，否则排队的
Dark Embrace 等效果可能读到真实未来。native_index 是执行句柄，不能用来
推断后续抽牌顺序。选牌mask覆盖扩大，但没有补齐旧神经观测的候选表示。

回归入口：`tests/integration/test_native_action_contract.py`、
`tests/integration/test_combat_selection_indices.py`。秘密技法/秘密武器的长候选
测试已改为正常断言；Headbutt候选观测仍保留F03严格xfail。长候选执行、分支隔离、
固定预算搜索及71/181 checkpoint接入均独立验证，不把mask修复称为完整F03验收。

### Native Heart runs

The fixed policy codec remains ACTION_VERSION=6 (181 ids). To play Act 4, use
`SlayTheSpireEnv(backend_factory=lambda: SimGameEnv(ascension=20, stop_after_act=4), combat_action_contract="native-v1")`.
The default backend still stops after Act 3. Keys must be obtained naturally:
`env.thick.legal_key_rewards()` lists offered `sapphire`/`emerald` keys and
`env.step_key_reward(key)` claims one through the native side interface, recorded
in the environment replay journal. Sapphire forfeits its linked chest relic;
claiming that relic instead removes the key offer. Ruby uses the existing recall
campfire action. Fixed-codec policies have no key slots and cannot choose these
side-interface actions; checkpoint shapes and observations are unchanged. This
requires a native extension with key-reward bindings and the Act 4 map fix.

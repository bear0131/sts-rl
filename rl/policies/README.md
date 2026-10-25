# 观测到动作/价值

`model.py` 管 tokenizer、编码器与 `observation_for_policy` additive-field 兼容。`pointer_policy.py` 是单 policy；`split_policy.py` 管共享表征和 combat/overworld 两组 actor-critic。`semantic_reasoning_policy.py` / `wholerun_reasoning.py` 是可选 reasoning 路径。RLlib checkpoint 不再支持。

参数名称和 tensor shape 是 checkpoint 契约。旧类模块路径由 rl/_compat 保留；不能靠改 key 名“整理”state_dict。mask/动作目标语义归 env 管，策略不自行压缩怪物槽。

`checkpoint.py` 是 SB3 `.zip` 的共享加载入口。策略 checkpoint 与 card encoder `.pt` 是不同文件：某些模型只保存外部 encoder 路径，移动文件后需要显式提供训练时同一份 encoder。eval CLI 用 `--card-vector-checkpoint PATH`，其他消费者可以设置 `STS_CARD_VECTOR_CHECKPOINT`。loader 先读 metadata 解析依赖，再加载模型；支持 SB3 的省略 `.zip` 写法。现役实验模型不保证存在于新 clone 中，先跑不需要 checkpoint 的训练冒烟可验证安装。

自动探测会跳过另一台机器上不存在或无权访问的历史路径，继续尝试本地候选。显式参数或环境变量具有优先级：指定文件不可访问时直接报错，不能悄悄换用另一个 encoder。
## Small Combat MLP

`SplitStSPointerPolicy(combat_arch="mlp")` keeps phase routing,181 atomic logits,
the legal-action mask and the root/leaf actor-critic interface. Only the combat
specialist changes; overworld/shared checkpoint tensors retain their original
geometry. The default remains `combat_arch="transformer"`, including old checkpoints.
Supported experiment: inner-only NOSL with the existing native collector, eager
or CUDA-graph leaf inference and normal save/load/learner recovery. Standalone
`CombatStSPointerPolicy` pretraining does not accept the MLP option.

`combat_mlp.py` uses all combat numeric fields plus small categorical embeddings.
Hand/monster/potion slots remain explicit. Piles and relic IDs are pooled;
status/intent language is a masked mean embedding, not a sequence encoder.
For the current production non-language schema the input width is2149 and the trainable
combat model has716566 parameters: shared256/256 ReLU trunk, separate128-unit
policy/value heads,181 logits and an unbounded scalar critic. The optional language
schema builds2781 inputs/900022 parameters; enabling it is a different experiment.
This is a lossy
architecture experiment, not semantic equivalence to the Transformer.

Frozen-upper conditional-value profiles can specify `lower_geometry` with
integer `width`, `layers`, and `heads`. This is supported for a fresh random
lower network under the unscaled conditional objective. It changes the combat
encoder, context, actor and value heads while keeping the frozen source upper
network and card vectors intact. Omitting it retains the source geometry and
historical checkpoint identity. Explicit geometry is saved in policy constructor
data and replay/checkpoint identity; states from a different geometry cannot be
resumed into the new model. Count all `core.combat.parameters()` when reporting
the lower parameter ratio, excluding the frozen teacher and nonparameter buffers.

To replace only the combat model while preserving a source's frozen upper:

```bash
python -m tools.checkpoints.replace_combat --source saved-model.zip --out mlp-step0.zip --seed 0
```

The operation is CPU-only, accepts split-pointer inner-only NOSL sources, keeps
their algorithm/search settings and frozen tensors, and resets combat weights,
Adam, replay and progress. Do not resume an old learner bundle into this new
architecture. Start a distinct run with the existing training preset/launcher.

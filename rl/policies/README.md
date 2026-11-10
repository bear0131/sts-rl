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

To replace only the combat model while preserving a source's frozen upper:

```bash
python -m tools.checkpoints.replace_combat --source saved-model.zip --out mlp-step0.zip --seed 0
```

The operation is CPU-only, accepts split-pointer inner-only NOSL sources, keeps
their algorithm/search settings and frozen tensors, and resets combat weights,
Adam, replay and progress. Do not resume an old learner bundle into this new
architecture. Start a distinct run with the existing training preset/launcher.

## Cached deck context and additive combat inputs

`deck_latent.DeckLatentPolicy` / `--arch deck-latent` is a fresh architecture,
not compatible geometry for SharedWholeRunPolicy weights. The first executable
stage uses the same PPO objective/reward with lambda1; the observation includes
pre-reward anchors even before combat returns are truncated there. The model
requires `deck_latent_context` observation v1: public map during combat,
permanent card growth, and all-ID upgrade counts/sums for draw/discard/exhaust.
Old observation and checkpoint schemas stay unchanged. The current reference
projection uses the canonical native state; native vector root packing and the
old shared-Transformer CUDA graph are deliberately unavailable for this schema
until their complete output contracts have passed parity tests.

`RunContextEncoder` encodes deck, permanent metadata, static relic identities
and public map/room context. HP, gold, potions and live relic counters are not
cache keys. Inference caches only one bounded row batch and invalidates on input,
parameter version, device or shape change. Training deduplicates equal contexts
within the physical batch and recomputes with gradients. `SearchContext` stores
128-wide combat and 256-wide anchor projections once per root; callers share or
index those tensors instead of reapplying the large projections in every lane.
Permanent branch changes require refreshing the corresponding context.

`combat_summary.AdditiveCombatInputs` defines context-independent per-card and
identity-conditioned counter contributions. `CombatSummary` stores unnormalized
pile sums/counts and a relic sum. Moves subtract/add contributions; upgrades and
counter changes replace old contributions; nonlinear context mixing and count
normalization happen afterwards. A fork clones only the small accumulators.
It does not serialize simulator state, masks, hidden order or pending actions.
Exact entities/physical slots remain in the action head. Pooling is learned,
lossy compression. Native event emission and CPU export are a subsequent
integration boundary, not a demonstrated performance result of this reference.

The initial tested path is FP32 eager collection/update, native Gym reference
observations, packed PPO and full checkpoint/Adam/environment restoration.
Anchor predictions consume branch player HP/maxHP (not stale run_hp), live
resources and cached run context, and never reward-page offers. Search auxiliary
training and the readiness stage controller are not yet connected here.

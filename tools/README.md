# 工作工具

`analysis/` 放诊断/绘图/实验分析，`benchmarks/` 放低层性能探针，`checkpoints/` 放 checkpoint 迁移/向量合并，`validation/` 放有预算的跨版本数值检查。
运行方式：仓库根目录 `python -m tools.<package>.<command> --help`。这些命令允许依赖运行时，rl/env 不反向导入 tools。诊断结果不是源代码，写入 runs/local；需要长期复用的实验参数放 configs。

`python -m tools.checkpoints.replace_combat --mode entity-material --source checkpoints/final/train47-step-000688914432-eval31/model.zip --card-vectors checkpoints/final/train47-step-000688914432-eval31/frozen-card-vectors.pt --profile configs/train/combat_outer_return_876_4x5090.json --out runs/local/combat01-preparation-20261021/initialization`
导出COMBAT01初始化材料，默认仍保留原`--mode mlp`行为。新模式先核对固定SHA及源配置，再用Torch的weights-only读取policy；分别复制上层actor/critic和下层actor/encoder，丢弃旧下层distribution/transition头，不导入Adam、replay或进度。输出目录须不存在，失败不发布完整manifest。只需要CPU/Torch，不导入模拟器或分配rollout。
输出有逐tensor/文件SHA、卡表依赖和原checkpoint的space JSON；这些space仍是原SB3序列化记录，仅供验SHA后的可信加载，本工具不执行其反序列化。`constructor.json`保留原架构信息，不是新loss配置。材料明确`launch_ready=false`，不是可以交给现有trainer的模型；运行时冻结、新标量critic、完整搜索结算和新目标训练仍按[COMBAT01设计](../docs/2026-10-21/frozen-upper-combat-design.md)实现/验收。权重放忽略目录，不能上传到Git。

`python -m tools.analysis.probe_nosl_updates BUNDLE --batch-sizes 1024 --updates 10 --max-seconds 120 --output runs/local/update-probe.json`
对不可变的完整恢复包做有预算的 **克隆 learner 拟合诊断**（默认 CPU，可选 CUDA），复用真实 NOSL learner、保存的 Adam/replay；不启动训练环境、W&B、SSH 或保存训练 checkpoint。
这一入口独立于纯推理 HP CLI，因为它会在私有模型副本上执行 optimizer；不能把这种操作混进日常只读推理接口。
各 batch 分支重新加载同一模型/Adam，固定 seed 和 HP 分层 audit rows，逐次记录额外 update 后的偏差及 policy KL。
这里用均匀、冻结的单 rank replay，无新 collection，也无四机梯度平均；不是生产 batch/update 设置的等价重放。
`--low-hp-fractions 0 .25` 可在每个 batch 设置下对比原生均匀抽样与25% HP1–5 配额；每个组合重新加载同一源包。
配额影响 policy 和 value 两种监督，不做重要性校正；其余样本来自非 HP1–5 的全部 replay（包括终局），分区不足时允许重复抽样。
输出记录各分区可用数量，缺少任一分区则报错；固定 audit 行仍来自原始 replay，不能把低 HP 改善当作整体改善。
audit 行来自训练 replay，结果必须标为 in-sample，不能用它替代独立验证来决定上线。
源 bundle 校验 SHA、输出拒绝覆盖源目录或已有文件；时间预算在 update 间检查，外层应再加进程超时。
FP32/train-mode 与所选设备的 BF16 推理控制用于分辨实现差异；CPU BF16 不代表 CUDA BF16，BF16 推理也不代表 BF16 optimizer 对照。
`--seed` 控制 learner 抽样并在模型加载后设置；`--audit-seed` 默认5311，跨 learner seeds 保持审计输入固定。
输出带工具/源文件 SHA；`state=budget_exhausted` 保留部分曲线且 CLI 退出码2，不能称为完成的全部分支比较。

Update/batch 对照可用 `--device cuda --micro-batch-size 2048 --batch-sizes 6144 24576 --updates 40`。
`--hp-corpus` 和 `--terminal-corpus` 接收 HP 评测入口保存的不可变 NPZ，在初始及 `--eval-updates 5 10 20 40` 时评测；输出保留完整双方 HP 曲线。
`--audit-bundle` 可指定同一 run/rank 的更晚完整包，审计只选 sample ID 不早于训练源 `total_added` 的 policy roots；后续 replay 只做审计，绝不进入 optimizer。
这比 in-sample 更严格，但仍受时间和采样策略分布变化影响，不代表独立游戏 seed 的泛化测试。
CUDA 记录同步后的 update 耗时和 Torch 峰值 allocated；并行实验的耗时含设备竞争，不能当独占吞吐。
默认仍是 CPU 且关闭 AMP；CLI 关闭 TF32。CPU/GPU 间不承诺 bitwise，生产 BF16/四 rank/fresh replay 需另行验证。
`--device cuda --amp bf16` 可单独对照训练精度；默认 `--amp off`。两者的 audit/HP 曲线仍使用 FP32 推理，避免混淆训练与评测精度影响。
`--learning-rate-scales 0 .25 .5 1` 在每个 batch/配额组合内重新加载源状态，分别缩放保存的 optimizer 学习率及 schedule，保持 Adam moments。
逐次输出实际各参数组学习率，不能仅按 CLI 值推断生效。默认1保留原步幅；0是无权重更新控制（Adam moments仍推进），用于检查固定语料评测漂移，不是候选训练设置。
降低学习率若只是接近不更新的基线，不能据此宣称学得更好；需联合观察相对源模型的误差改善与跨seed波动。

`python -m tools.analysis.probe_nosl_updates BUNDLE --audit-only --per-stratum 64 --max-seconds 180 --output runs/local/replay-supervision.json`
只读检查保存的 replay：按 policy root/value-only、编码中有无存活敌人及 HP 区间统计覆盖，
固定 seed 抽样报告预测误差、偏差和负值，并验证源包未被改写。没有 optimizer 或 rollout；
这是训练 replay 上的校准，不是独立验证。历史 replay 没有保留 `sample_kind`，因此
“无存活敌人的 value-only”只是编码分组，不能等同于精确 terminal 分类或推断回血是否合法。
该模式不执行训练参数矩阵，推理始终 FP32；预算不足以 `budget_exhausted`/退出码2返回部分结果。

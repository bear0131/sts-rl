# 工作工具

`analysis/` 放诊断/绘图/实验分析，`benchmarks/` 放低层性能探针，`checkpoints/` 放 checkpoint 迁移/向量合并，`validation/` 放有预算的跨版本数值检查。
运行方式：仓库根目录 `python -m tools.<package>.<command> --help`。这些命令允许依赖运行时，rl/env 不反向导入 tools。诊断结果不是源代码，写入 runs/local；需要长期复用的实验参数放 configs。

`python -m tools.analysis.probe_nosl_updates BUNDLE --batch-sizes 1024 --updates 10 --max-seconds 120 --output runs/local/update-probe.json`
对不可变的完整恢复包做有预算的 **CPU 克隆 learner 拟合诊断**，复用真实 NOSL learner、保存的 Adam/replay；不启动环境、W&B、SSH 或保存训练 checkpoint。
这一入口独立于纯推理 HP CLI，因为它会在私有模型副本上执行 optimizer；不能把这种操作混进日常只读推理接口。
各 batch 分支重新加载同一模型/Adam，固定 seed 和 HP 分层 audit rows，逐次记录额外 update 后的偏差及 policy KL。
这里用均匀、冻结的单 rank replay，无新 collection，也无四机梯度平均；不是生产 batch/update 设置的等价重放。
audit 行来自训练 replay，结果必须标为 in-sample，不能用它替代独立验证来决定上线。
源 bundle 校验 SHA、输出拒绝覆盖源目录或已有文件；时间预算在 update 间检查，外层应再加进程超时。
FP32/train-mode 与 CPU BF16 控制用于分辨实现差异；CPU BF16 不代表 CUDA BF16 的实测。
`--seed` 控制 learner 抽样并在模型加载后设置；`--audit-seed` 默认5311，跨 learner seeds 保持审计输入固定。
输出带工具/源文件 SHA；`state=budget_exhausted` 保留部分曲线且 CLI 退出码2，不能称为完成的全部分支比较。

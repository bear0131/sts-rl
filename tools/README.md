# 工作工具

`analysis/` 放诊断/绘图/实验分析，`benchmarks/` 放低层性能探针，`checkpoints/` 放 checkpoint 迁移/向量合并，`validation/` 放有预算的跨版本数值检查。
运行方式：仓库根目录 `python -m tools.<package>.<command> --help`。这些命令允许依赖运行时，rl/env 不反向导入 tools。诊断结果不是源代码，写入 runs/local；需要长期复用的实验参数放 configs。

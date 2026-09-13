# 历史冻结数据输入

这里原有 tokenizer/小样本被历史 checkpoint 或启动参数按路径引用；一阶段保留文件名和字节，避免将路径清理混成 checkpoint 迁移（TODO F13）。
这里是数据而非 Python 运行时。新数据集写仓库根 data/ 或外部存储，生成程序在 rl/rules/data。旧路径全部迁出需先做保留 checkpoint 家族的依赖审计。

# 输出与历史模型

`local/`：忽略的当前测试/基准/eval 输出，不进 Git。
`archive/`：从根目录、docs/、wandb/ 迁出的既有原始结果，按来源保存，字节不变。旧路径/hash 见 docs/path-migration.json。
其余已有 checkpoint 目录：保留原路径，避免破坏实际实验和序列化依赖；此次没有清理 Git 大文件历史。后续外置模型需配套 manifest 和下载/校验机制，见 TODO F13/F17。

新的 checkpoint、log、截图、CSV/JSON 都写 local/ 或外部 artifact store，不直接添加大文件。

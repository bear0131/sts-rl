# 运行结果

训练 checkpoint、采样/评估结果、日志、profile、截图和验证证据统一放这里。
- `local/`：当前本地验证；忽略，不进入 Git。
- `legacy/`：历史散落的输出，保留原始内容。
- 其它历史 run 目录：从 artifacts/、rl/runs/ 迁入的 checkpoint，名称和字节不变。

已有受版本管理的文件保留跟踪；新的运行结果默认忽略。需要提交的小型回归输入放 tests/fixtures；报告放 docs/日期。旧路径查 tools/validation/migrations.json。

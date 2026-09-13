# 实验启动环境

`launchers/` 保存机器/路径/通信/W&B/后台进程设置；训练数值参数已从三个 NOSL 脚本分离到 configs/train。其它历史 launcher 保留原有参数，明确属于部署预设。
当前实验启动路径为 `bash experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh RANK`；原 rl/launch_nosl_* 路径通过薄 shell 转发保留。

脚本保留既有远端地址、run id 和 `pgrep` 防重复逻辑。它们会实际启动/SSH/排队，阅读文件不是运行授权。运行前应由操作者确认目标与身份；此次重构只进行了本地解析与 bash -n。
实验进度放 worklog，结果放 runs/local；历史实验材料归 docs/archive/input，引用时给出具体文件名，不把快照当作当前启动指令。

# 实验启动环境

`launchers/` 保存机器/路径/通信/W&B/后台进程设置；训练数值参数已从三个 NOSL 脚本分离到 configs/train。其它历史 launcher 保留原有参数，明确属于部署预设。
当前实验启动路径为 `bash experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh RANK`；原 rl/launch_nosl_* 包装已删除，统一使用 experiments/launchers。

脚本保留既有远端地址、run id 和 `pgrep` 防重复逻辑。它们会实际启动/SSH/排队，阅读文件不是运行授权。运行前应由操作者确认目标与身份；此次重构只进行了本地解析与 bash -n。
实验进度放 worklog，结果放 runs/local；历史实验材料归 docs/archive/input，引用时给出具体文件名，不把快照当作当前启动指令。

## Windows 到远端的二进制传输

在 Windows PowerShell 5.1 中，不要用 `Get-Content -Encoding Byte | ssh ...` 传输 tar、
checkpoint 或其它二进制文件。PowerShell 管道会把字节作为对象传递给原生命令，可能
发生格式化或转码，远端得到的内容不是原文件；`tar -tf` 会报
`This does not look like a tar archive`。这也适用于看起来只是在做 stdin 重定向的类似写法。

使用 `scp -B` 或 SFTP 传输二进制文件。解包或加载前，必须在本地和远端比较完整文件的
SHA256；tar 包还应先运行 `tar -tf`，确认成员列表后再解包。文本命令需要通过 SSH 发送时，
可以使用 UTF-8 文本或 base64 包装命令本身，但不能把这个做法扩展为未经校验的二进制
artifact 传输。

从 PowerShell 调用 SSH 时，包含 Bash 的 `$()`、反引号或 `$name` 的远端命令也不要直接
拼在双引号字符串中；它们可能先被本地 PowerShell 展开。优先复用 launcher 中
`Invoke-Remote` 的 UTF-8/base64 命令封装。临时只读命令若不能复用该入口，应先把完整
命令编码后发送，再在远端解码交给 Bash；不要靠多层反斜杠猜测两边 shell 的转义顺序。

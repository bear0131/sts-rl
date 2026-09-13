# NOSL 叶推理加速、训练恢复与在线实验交接

## 范围和阅读顺序

这次改动针对四台 RTX 5090 上的 NOSL/AZ 战斗训练：环境在 CPU 做树搜索，主进程合批运行网络，为完成的战斗生成监督标签，再从 fresh/replay 样本更新网络，四个 rank 同步梯度。首要瓶颈是大量小 batch 推理的 CPU 调度，其次是环境 IPC；训练反向传播不是当前主要耗时。

第一次接手先读根 README 和 [训练模块](../../rl/training/README.md)，再读 [搜索后端](../../rl/search/README.md)、[基准](../../rl/benchmarks/README.md)、[运维协议](../../tools/operations/README.md)。本文是实施快照，持久问题以 [TODO](../../TODO.md) 为准。

用户先要求保留旧 run，发现旧 trainer 数小时未保存且不保存 replay 后，明确授权放弃旧 run 并重新开始。旧健康 W&B run 是 `2733k5b8`，附件中的 `vqidwlbz` 是更早的失败启动。旧日志保留；新实验没有继承旧 run 的训练进度。

## 改动和回归防护

- 接纳远端的 `avg_grad_sec=0` 修复。梯度平均合并进同步后，W&B payload 还引用已删除字段，首次上报便失败。CI 现在经真实 PPO/NOSL 入口跑到采样、更新、指标构造、checkpoint 和恢复；仅替换外部 W&B 服务。
- `rl/search/inference.py` 缓存精确 observation shape 的 CUDA graph，最多 32 个 LRU 条目，不补齐 batch、不改搜索顺序。输入通过可复用 pinned/GPU buffer 传送；权重转换在 graph 内，原地 optimizer 更新生效。模型结构、参数存储、设备或 AMP 变化须重建后端。默认 eager；四机 launcher 显式选择 graph。
- role embedding 直接取权重行，移除每层重复创建 CPU tensor 并送 GPU 的工作；值和梯度均有逐位合同。
- `rl/training/environments.py` 为生产和基准共用 checkpoint 环境适配，避免基准测另一条路径。
- checkpoint 历史模块别名优先解析到 canonical policy，防止部署目录遗留旧 flat 文件覆盖现役类；shadow-package 回归覆盖这一真实部署问题。
- 每个 rank 保存 learner bundle，并通过 ring 在更新边界协调 STOP 与保存完成。吞吐的 `iteration_wall_sec` 包含聚合和保存屏障；仍保留旧阶段字段。W&B 客户端随后 enqueue 不在该计时内。
- SSH socket 目录用固定长度 run 哈希，避免长实验名超过 Unix socket 上限；过长自定义 root 在调用 SSH 前明确拒绝。

## 性能证据与适用范围

相同 rank 0 RTX 5090、Torch 2.12.0+cu132、BF16、初始 d_model=256 / 8 层 / 8 heads checkpoint；24 env，单线程 BLAS，seed=42。checkpoint SHA256 `6c38a9b7f3f66f1d75134b991666785d99df20386aeee3f3abc06ead8f443392`，native SHA256 `88bc7d6f7ecdda4893c83d8a733da5483d511ad1c2ef08e9cc50d27b0954edbf`。

| 固定工作量 | eager | CUDA graph | 提速 |
|---|---:|---:|---:|
| 24 叶 × 300 批，单批平均 | 3.831 ms | 0.589 ms | 6.51× |
| 128 simulations，384 transitions | 19.227 s | 12.060 s | 1.59× |
| 512 simulations，384 transitions | 61.941 s | 42.396 s | 1.46× |

512 对照先运行 graph 再 eager，包含 graph 首次 capture，输出摘要均为 `926356413e0aa37ec25b92b70640beedca01803b950b99f69cff1afd2f7bccdc`。摘要覆盖最后一个 rollout 的观测、动作、奖励、value、log-prob 和完整战斗监督；本次每次只测一个 rollout。不是跨平台逐位承诺，也不是完整训练收敛或多随机种子的统计结论。

512 对照进程树 CPU 平均 608%→838%（100% 为一个核），采样 RSS 峰值约 15.79→15.81 GB；Torch peak allocated 93→357 MB、reserved 180→583 MB。CPU RSS 累加会重复计共享页；设备 VRAM/GPU util 是设备级采样，原始 JSON 有每 0.5 秒记录。不要把利用率数字单独当速度目标。推理 graph 用更多持久 buffer 换取更少调度。

每 worker 放 4 env 的方案将 RSS 降至约 5.7 GB，但 128-simulation workload 变慢至 24.010 s，未采用。后续效率工作优先实测 IPC/packed leaf 路径，不能靠增加并行参数猜测收益。

原始 JSON、trace、算子表：rank 0 `runs/local/profile/`，本地下载在 `runs/local/gpu-performance/`（忽略目录）。复现使用 `python -m rl.benchmarks.train --workload rollout --checkpoint MODEL --search-config SEARCH.json --device cuda --amp bf16 --n-envs 24 --n-steps 16 --updates 1 --vec-env subproc --leaf-inference eager --output runs/local/eager.json --timeout 180`，另一份只改后端与输出文件。SEARCH.json 使用结果中的有效 `search_config`；小模型相关 CLI 默认值在 checkpoint workload 中不决定实际模型架构。

## 新正式 run 与恢复

传输身份 `nosl-rootpaper-graph-20260914`，W&B ID `rpgr0914`，名称 `nosl-s512-q3-cov04-rootpaper-replay-cudagraph`。四个 rank 位于 `connect.bjb2.seetacloud.com` 端口 12009 / 17944 / 13500 / 44892，仓库均为 `/root/autodl-tmp/sts-rl`。凭据使用用户提供的私有附件，不入仓库。

初始配置 `configs/train/nosl_rootpaper_replay.args`：512 simulations、q scale 3、coverage .4、Gumbel scale 0、End-Turn leaf、非 strict paper；每 rank 24 env × 128 steps、40 次 learner updates、replay 131072、旧样本上限 .9、LR 1e-4、BF16。目标 10,000,000 global env steps。每完整 collection 增加 12,288 global steps；不是每个 optimizer update 单独保存。

统一入口 `experiments/launchers/launch_nosl_replay.sh RANK`，设置 `NOSL_RUN_ID`、`NOSL_WANDB_ID`，线上用 `WANDB_MODE=online`。日志 `runs/nosl-rootpaper-graph-20260914/rankN.log`；PID 同目录。首个长名启动因 socket 限制在训练前退出，日志前段保留；重试使用同一身份。

每轮存到 `recovery/rankN/step-XXXXXXXXXXXX/`：`model.zip` 为策略/Adam，`learner.pt` 为有效 replay、父进程 RNG/global step，最后发布 `manifest.json` 与 SHA 校验。保留三代，`latest` 是指针。远端代码按本次 delta 部署，文件 hash 见 `runs/local/deployed-source.json`，覆盖前副本在 `runs/local/deploy-backup/`；部分机器无可靠 Git 元数据，不能用其 git HEAD 代替部署身份。

恢复：核对四 rank 完整 generation 的 step、身份、SHA；以 `NOSL_RESUME_STEP=STEP` 和同一 ID 启动四个 rank。`NOSL_OVERRIDES_FILE` 只保存本 run 有意的参数实验。需要停止时在 run 目录创建 `STOP`，等待四 rank 更新边界保存退出；确认后才能清除该 STOP 并恢复。

**恢复边界**：支持 learner 恢复，不保存活的 simulator 状态；未结束战斗重置且丢弃其未完成监督。已完成 replay、Adam、父进程 RNG 和 global step 保留；不能宣称重启前后完整环境轨迹逐位接续。保留干预前 checkpoint 副本，避免三代轮换覆盖归因证据。

## 低频监测与实验归因

每半小时检查四 rank 进程、日志前进、最新 generation 和 layer 5 曲线。rank 0 执行 `python -m tools.operations.monitor_nosl --log runs/nosl-rootpaper-graph-20260914/rank0.log --state runs/nosl-rootpaper-graph-20260914/watch.json`，读取日志中的 `game/floor_won/05`（W&B 对应 `game/layer_5_win_rate`）。

判断规则和逐参数回退顺序只维护在 `rl/training/monitoring.py` 与运维 README：足够数据后明显持续下降先 Gumbel scale→1；每次干预后至少六个新 collection 且一小时才允许下一步。单个低点和起步低值不触发。非有限数、进程退出或不再前进要先诊断运行故障，不能直接归咎超参。完成或有实质问题才通知。

参数变化前统一保存停止，备份干预前 generation，四机写一致覆盖参数，同 ID 恢复并确认继续上报，再记录 intervention。在线恢复只说明时间关联；定位参数作用需固定干预前 checkpoint/seed 的成对复测。最终 strict 配置来自 `configs/train/nosl_paper_replay_fresh10_20260913.args`，不会一次全部切换导致无法区分参数。

## 验收记录

本地完整 suite 244 passed、2 GPU-only skip、8 既有 strict xfail；新增 SSH 边界与真实生命周期 focused 4 passed。GPU 独立合同 3 passed，覆盖 FP32/BF16 graph 输入、权重变化及缓存。

真实四机小预算 smoke 从 0→512→2816→3328：每 rank 自有 replay 保留，ring 真实平均梯度；仅 rank 0 发 STOP 后四 rank 同步保存退出。3328 时四机权重 SHA `ff8ba0e316294e93f5c6c5a30d0c5f444275b6feda2eeb150237c2a07ff355aa`，Adam SHA `53d6bf919ac8a25ab5422921d9166476ae30b8eefb1e39e1438b36d81e6b0f85` 完全相同。W&B smoke 为 offline，不把它声称为云端 resume 验收。

关键提交：dc126d9 生命周期回归；06ac056 推理；edc8b43 learner 恢复；1402fcf 旧模块遮蔽；5f83e57 运维/监测；20b120e socket 路径。正式启动结果、最终 CI 和后续干预追加工作日志。

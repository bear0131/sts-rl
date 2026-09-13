# Agent 工作约定

这些约定服务于可复现、易审查和安全演进，不追求形式上的“规范”。用户当次明确指令优先；遇到冲突应说明取舍，不能用本文件顶替用户决定。

## 开始工作

- 先读本文件、TODO.md 和 worklog/ 最近的相关记录；检查分支、工作区、remote 和实际运行入口。
- 先写下用户的原始 intention、范围和验收条件，再实现。区分修复、迁移、实验和性能优化；不要把它们混成一个未说明的改动。
- 不根据 author 名称猜人、不根据旧文档猜当前实现。以固定 commit、实际代码、可复现命令和观测为依据。
- Windows 优先使用 gbash 进入 Git Bash；需要 Windows API 时显式使用 PowerShell。

## 工作日志与长期文档

- 进度只追加到 worklog/YYYY-MM-DD.md（日期以当前工作时区为准）。同一天多件事使用小节，保留已有记录。
- **每件事完成，或做到一半需要暂停/交接/结束一轮工作时，必须更新日志。** 记录用户原始意图（关键原话或忠实引用）、实际范围、变更、验证命令和结果、未决问题、下一步、关联 commit/分支。日志和代码一起提交；当前 commit 可写“本节随当前提交”，避免为填写自身 hash 制造递归提交。
- TODO.md 是唯一持久仓库任务清单：记录问题 ID、状态、适用路径和验收条件。日志引用 TODO，不复制多份进度表。
- 只有需要持续复用的资料才独立建文档：例如构建/测试入口、接口契约、数据集/实验协议。一次性进度、临时诊断、启动结果放 worklog；原始大输出放忽略的 artifacts/local/，日志记录路径和摘要。
- 不把密钥、完整环境变量、私人凭据、游戏反编译源码或大 checkpoint 放进日志/Git。外部参考记录位置、版本/hash 和必要的事实，不复制整个文件。

## 主动提交与审查粒度

- **一个有意义的阶段结束后主动进行本地 commit。** 不要等用户催促，更不要积累数千行无关修改到一个“update 日期”提交。
- 每个提交围绕一个问题或一个可验收边界；附上相关测试和 worklog。大型重构先提交测试/契约，再移动代码，再改行为；每个阶段保持可定位。
- 提交前看 git diff --check、diff 和 git status，只 stage 本次负责的文件，不把用户已有改动一并提交。提交消息说明行为/目的，不以“misc/update”代替说明。
- 用户质疑拆分时，解释小提交对回滚、二分定位和审查的实际价值，并提出合并粒度建议；保持直接、尊重，不争吵。用户明确要求不提交/暂停时遵从。
- 本地 commit 不代表有权 push、部署、改远端训练或改写共享历史。没有明确授权不做这些动作。

## 正确性与兼容

- 状态/动作/奖励/搜索是不同契约。改动其中之一必须沿 simulator → observable state → action codec → policy/search → learner → evaluation/bridge 检查消费者。
- 实体 ID、怪物槽位和选择候选需要显式映射。优化不能静默删除空槽、丢掉 pending 来源、把不同候选区域统一当手牌，或把不同条件终局混用成同一 label。
- 规则数据的 teacher 不自动等于游戏 oracle。先核对数据来源、卡牌守恒和真实 transition，再讨论 token accuracy。
- observation/action/reward 或冻结依赖变化须明确版本和兼容范围。至少验证保留 checkpoint 家族的加载、一步预测、环境接入；不要只修 dashboard。
- 性能优化先过语义等价测试。要求 bitwise 时必须规定同设备、同依赖/编译器、确定性设置及比较对象；浮点误差测试使用有理由的容差，不伪称跨 GPU/平台 bitwise。
- 测试预期行为而不是复制实现。新增回归放 tests/，使用共享 fixture；测试集中在 tests/unit 与 tests/integration；不得在运行时代码旁新增 test_*。已知未修缺陷可以 xfail(strict=True)，但必须绑定 TODO.md 问题 ID、具体失败断言和原因，不能用 skip 掩盖支持路径。
- 原生测试必须加载明确的扩展构建。缺少扩展或可选依赖与测试通过是不同结果；核心 CI 不允许整批静默跳过。
- 不跑无预算训练。冒烟、基准和实验都需固定 seed、步数/超时、退出码、结果位置；意外耗时应记录并调查。

## 性能评测

- 先定义工作量：环境 transition、完整原子动作、搜索 simulation/leaf、有效训练样本和 optimizer update 不能混为一个“step”。
- 报告端到端时间，并区分 warmup、采样、编码/IPC、神经前后向、同步和保存；GPU 时间测量前后同步。
- 同时记录进程树 CPU 时间/利用率、RSS（标明共享页重复计数）、系统可用内存、Torch allocated/reserved/peak、设备 VRAM 和 GPU util。不可用指标填 null 并说明，不能填 0。
- GPU util 是设备级采样，可能含其他进程；CPU 进程利用率可超过 100%。报告硬件、线程/env 数、模型、batch/accumulation、AMP、搜索预算、原生路径/commit、版本和采样间隔。
- 改 microbatch、phase 权重、搜索算法、replay 分布、终局/截断定义时，视为可能改变统计方法的实验，不把吞吐提升直接归因于等价优化。
- 通用 CI 验证可运行和结果格式，不设跨硬件 FPS 硬门槛；性能回归在同硬件受控基线比较。

## 组织边界

- 尽量复用一个 codec、一个观测兼容适配层和一个 outcome 定义，避免训练、GUI、真实 bridge 各复制规则。
- 新功能先说明支持的后端/配置组合，不无限添加无测试的布尔开关。复杂实验模式应有配置对象和入口校验。
- 避免全仓自动格式化、重命名或顺手大重构。清理与行为修复分开；不引入只有一次调用、没有复用价值的框架。
- 仓库规范和测试入口一起维护；报告区分已复现、静态发现、设计取舍和未验证，不把本机冒烟写成多机/GPU/完整游戏验收。


## 一阶段之后的文件归属

- 核心运行时按 rl/README.md 的包边界放置；worker 无模型搜索归 env/search。新代码不得引用 rl 的旧 flat module，_compat 只服务历史调用方。
- train/eval 参数优先 configs/*.args 或纯数据模块；部署地址、run 身份与启动操作归 experiments/launchers；不要将有执行算法的 .py 当作 config 掩盖职责。
- 临时输出写 artifacts/local，日期进度写 worklog/YYYY-MM-DD.md。历史记录在 worklog/archive，原始输出在 artifacts/archive；按 path-migration.json 查旧路径，不再向归档写新结果。
- 文件迁移同步检查 import、python -m、__file__ 相对位置、shell sibling 路径、checkpoint 模块名、文档命令和 CI。先比较跨版本行为，再移除兼容名。
- 同一阶段继续保留有意义的小提交；大目录移动用 git diff --find-renames 审查，不夹带格式化或算法调参。

# 旧模块兼容层

此目录是 rl.__path__ 的第二个搜索目录；Python 正常 import 机制按需找到旧名文件。文件仅将 sys.modules 中的旧名指向 canonical 模块，同一个类/全局变量只有一份。`python -m rl.旧名` 用 runpy 转到新入口。

为何不删：旧 SB3/cloudpickle checkpoint 保存了模块名，现有实验脚本也用旧 CLI。全部旧 flat module 显式登记于 docs/module-migration.json。新代码禁止从旧名导入；tests/unit/test_repository_layout.py 守住依赖边界。移除兼容层须先盘点保留 checkpoint 并另做迁移验收。

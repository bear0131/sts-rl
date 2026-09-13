# 数据输入

新小型、可分发 fixture 放这里；大型 corpus/tokenizer 数据集使用外部存储与版本/hash manifest。生成逻辑属于 rl/rules/data，实验配置属于 configs。
历史 rl/data 暂留原路径供旧 checkpoint 使用；数据输出不是源代码，不要把训练生成的百万行 JSONL 直接加入 Git。

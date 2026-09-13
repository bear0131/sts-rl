# 规则学习

现役入口为 `training/train_rule_vectors_v7.py`；token 规则模型及旧向量 trainer/eval 已删除。`rule_vector_model` 与 `vector_rule*` 管当前向量表示、编码和流；v7 仍消费的底层 v6 record 编码保留，并由测试覆盖，不能按文件名删掉。

`data/` 提供当前生成算法，`visualize_card_vectors.py` 提供共享向量抽取/PCA。旧实现从 Git 历史恢复，不维护两套入口。

程序不是游戏 oracle：近似 teacher 的正确性见 TODO F04。数据 schema、tokenizer vocabulary 与模型配置一起构成加载契约。数据文件归仓库 data/，参数预设归 configs/；不要在本包写入实验结果。

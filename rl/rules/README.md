# 规则学习

`rule_data` / `rule_model` 是 token 规则路径；`rule_vector_model` 与 `vector_rule*` 管向量表示、v6/v7 编码与流。`data/` 是可复用的生成/采集算法（部分带 CLI），`training/` 是规则专用训练编排，`visualize_card_vectors.py` 提供共享向量抽取/PCA。

程序不是游戏 oracle：近似 teacher 的正确性见 TODO F04。数据 schema、tokenizer vocabulary 与模型配置一起构成加载契约。数据文件归仓库 data/，参数预设归 configs/；不要在本包写入实验结果。

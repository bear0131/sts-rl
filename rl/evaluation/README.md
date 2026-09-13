# 评估协议

- `eval.py`：whole-run policy 评估，胜负/统计/CSV。
- `eval_combat.py`：combat task。
- `nosl/`：corpus replay、搜索方法比较、Bash/Clash 固定场景、HP 敏感性。包含真实执行逻辑，不是单元测试，也不只是 config。
- `rules/`：规则模型 accuracy、生成、场景解释；shared fixture 尚有私有函数复用，后续按协议抽取。
- `baselines/`：leftmost/starter 等独立基线，环境限制和玩法不等价于普通 checkpoint eval。

参数预设放 ../../configs/eval，结果放 runs/local；新增场景优先配置化，新增 solver 必须明确终局、预算和 state key。此次未优化吞吐、未统一不同 solver 的统计口径（TODO F16）。

# v6 卡牌规则向量 PCA

本图使用 v6 wide checkpoint 的 `CardRuleEncoder` 提取匿名卡牌规则向量。
模型输入是由 `card_program` 编译出的规则 token；卡牌名称只用于图中中文标注，
没有作为 token 或 ID embedding 输入模型。样本为 75 张未升级战士卡牌。

- checkpoint：`.tmp-remote-v6-wide\last.pt`
- 训练 step：`7500`
- 卡牌向量维度：`160`
- PCA 1 解释方差：`21.9164%`
- PCA 2 解释方差：`15.9779%`
- 两个主成分合计：`37.8943%`

## 类别统计

攻击牌：32 张；技能牌：29 张；能力牌：14 张。

PCA 是无监督投影，坐标轴方向本身没有固定语义；类别颜色用于观察规则表示是否自然形成分组，
不能单独作为模型泛化能力或卡牌强弱的结论。

## 生成文件

- `card-representations-v6-wide-step7500-pca.png`：带中文标签的二维图。
- `card-representations-v6-wide-step7500-pca.json`：每张卡牌的坐标、类别、向量和 PCA 方差。

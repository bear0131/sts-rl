phase 0
用 mod 链连好 java 真游戏，做一个 python（吗？）orchestrator，把 state 拉出来，action 传进去。验收：能靠 json 交互打完一局游戏，就说明我们对游戏大致结构的理解没问题。

phase 1
开始搭 cpp simulator。并行开发，1. 打控制台 mod，接 java 端，在 python 侧（？）实现一个控制台 api，支持调试修改。2. 实现 ironclad 局内战斗，回合轮转、抽牌洗牌、选牌出牌、掉血回血、buff debuff、输赢判定，支持少量的牌（效果简单的基础牌，剑柄打击耸肩无视铁斩波这些）和少量的敌人（机制简单，用以上系统能描述的敌人），先不做遗物、药水系统。验收：枚举卡组和敌怪配置，输入进模拟器，java 端用控制台改出相应配置，打牌差分测试通过。

phase 2
并行开发，1. 开始搭建基础的 rl，先做打牌机，输入 state 输出 policy，用简单的卡组做最简单的训练。2. 模拟器侧开始写领取奖励、选牌、选路线、事件、商店逻辑，支持更多牌和更多敌人。验收分两路：1. rl 看起来跑通，真机上能以初始卡组几乎无伤地打赢一层弱怪池，这件事只需要知道“它打我我就防御，它不打我我就打它”就行，即我们不要求效果只判断看上去跑通了 2. 同样用控制台随机配置，模拟器差分通过“给定只剩一滴血的怪，一张牌把它打死，之后进入选牌页面选牌加入卡组，进入下一层”和“在事件选择选项”和“在商店买牌删牌”两件事。同样不动遗物药水。

phase 3
继续并行。1. rl 那边开始进行带完整游戏流程的训练。具体来说，选路线收益低，暂时不做，让我们的模型（或者模型系统，至少是多个判别头）支持输出选牌、事件选项、商店买哪些牌删哪张牌。从这里开始调效果。 2. 模拟器进一步完善。写遗物、药水、支持几乎完整的 ironclad 卡组和几乎所有敌人和 boss。 验收可以分为，1. 模型能在模拟器上以非零概率见到第一个 boss 2. 模拟器在不含有复杂机制的敌人、遗物、卡牌、事件的野生随机情形下通过差分测试 3. 模型 模拟器 真机 联调。理论上如果sim2real gap 为 0 那么模型直接上真机理应没有任何问题，但实际肯定还是有问题的。这一步只要能录下模型见到第一个 boss 的 demo 就算成功。
fallback：如果模型暂时学不会这个，那就接一个大语言模型让它写 json 做决定。我知道这是额外的工作量，但我们团队（即，寝室里的室友们）对于 llm agent 相比于基础 rl 来讲十分熟练，而且这部分工作也很有趣，肯定不会浪费。

phase 4
往后就比较自由了。rl 侧可以开始实验不同的架构和算法，加训练量；模拟器侧可以慢慢查漏补缺以及开始写其它角色。最终模型能够在 A0 通关游戏，即视为项目初步成功。


```
sts-rl/
  cpp/
    sim/              # C++ 高速模拟器核心
    tests/            # simulator 单测 / 回归测试
    bindings/         # pybind11 / nanobind Python bindings

  bridge/
    java-mod/         # 真机 Slay the Spire mod / bridge
    client/           # Python/C++ 侧 bridge client
    schemas/          # State/Action/Result 协议定义

  python/
    env/              # Gymnasium/RL env wrapper
    train/            # PPO/MCTS/offline RL 训练代码
    models/           # policy/value/network 定义
    eval/             # 评估、真机测试、benchmark

  data/
    wiki/             # 本地 wiki dump / parsed json
    fixtures/         # 真机录下来的 diff test cases
    seeds/            # 固定测试 seed

  tools/
    extract_wiki/
    record_oracle/
    replay_diff/

  docs/
    design.md
    state_schema.md
    roadmap.md
```
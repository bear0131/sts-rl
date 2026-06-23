你需要知道的领域常识

1. 整个 mod 生态的分层

杀戮尖塔主流 mod 生态大概是这样：

ModTheSpire 是 mod loader，用来在不改游戏本体文件的情况下加载外部 mod；它的手动安装模型是把 ModTheSpire.jar 放到游戏目录、建 mods 文件夹、把 mod 的 jar 放进去，然后从 ModTheSpire 里勾选运行。开发侧要求 JDK 8 和 Maven。

BaseMod 是更高层 API，帮你注册卡牌、遗物、角色、药水、事件、怪物、关键字、文本、hook 等，不然你会大量写底层 bytecode patch。BaseMod 官方 wiki 明确说它的目标就是让 modder 更容易 hook 进游戏功能，而不用写很多字节码补丁。

StSLib 是常用可选依赖，提供通用关键词、机制、遗物 hook、Power hook 等，避免每个 mod 重造轮子；很多复杂 mod 会依赖它。

所以你的心智模型可以是：

能用 BaseMod 注册的内容，先用 BaseMod。
BaseMod 没覆盖的行为，再考虑 ModTheSpire patch。
常见复杂机制先查 STSLib，别急着自己 patch。

2. 技术栈有点“老派”

你需要准备：

Java 8 / JDK 8，不要用现代 Java 当默认运行环境。

Maven。

IntelliJ IDEA 或你喜欢的 IDE。

能读反编译 Java 代码。

能接受“文档不完整，源码就是文档”。

BaseMod 的 README 也列了 Java 8、Maven、ModTheSpire 作为开发要求，并提醒 Java 9 不适合 ModTheSpire。

3. 游戏内部对象模型

你需要逐步熟悉这些类和概念：

模块	你要理解什么
Card	AbstractCard、费用、伤害、格挡、魔法数、升级、预览、标签
Action	游戏效果通常不是直接执行，而是塞进 action queue
Power	buff/debuff、回合开始/结束、抽牌/受击/造成伤害等触发
Relic	战斗开始、回合开始、打出卡、获得金币、拾取等触发
Potion	使用时产生 action 或即时效果
Character	卡池、遗物、能量、动画、选择界面、起始牌组
Monster / Event	复杂度更高，建议后面做
Strings	卡牌/遗物/Power 文本通常走 JSON，方便本地化
Assets	图片路径、尺寸、命名、资源打包

杀戮尖塔 mod 的核心不是“写一个普通 Java app”，而是理解：游戏在什么时候调用你、你什么时候注册内容、你什么时候把效果放进 action queue。

4. Hook 和 patch 的区别

BaseMod 使用 subscription/hook 模型：你的主 mod 类通常实现若干 subscriber，在初始化时 BaseMod.subscribe(this)，再在对应 hook 里注册卡牌、遗物、字符串等。BaseMod wiki 也建议一般只在带 @SpireInitialize 的主 mod 文件里使用 subscriber，避免重复订阅。

你可以把两类扩展分清：

注册内容： 新卡、新遗物、新 Power、新关键字、新文本、新角色。
优先 BaseMod hook，比如 EditCards、EditRelics、EditStrings、EditCharacters。

修改原版行为： 改抽牌逻辑、改奖励生成、改商店、改怪物 AI、改事件流程。
通常需要 @SpirePatch，更容易和其他 mod 冲突，也更依赖反编译源码。
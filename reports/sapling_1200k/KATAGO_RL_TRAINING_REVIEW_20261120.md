# Sapling 的训练信号该从哪里来

2026-11-20。复核对象是 Sapling 对仓内原版 Master3s2 的训练。本文核对了 KataGo 论文、作者的训练代码及强化学习原始文献，并把其中的方法和本项目的实测逐项比较。

**建议优先修正从「学生怎样出令」到「训练怎样评价这条令」的链路，保留现有网络作对照。** 目前有直接证据表明，网络学得更像高手、以及从高手回放中挑出短期现金增长快的动作，都可能给出错误的优化方向。现有试验尚未分离出参数量的影响。[对原版的整场记录](MASTER3S2_90.md)

## 先读哪几份资料

| 要回答的问题 | 原文里值得看的内容 |
|---|---|
| 怎样节省自博算力，同时保留高质量策略标签？ | Wu 的 KataGo 论文修订版 §3.1–3.2。19 天主实验初期随机抽 25% 的步做 600 次搜索，其余做 100 次。仅把完整搜索步写成训练样本。快步让相同预算走完更多棋局；强制探索带来的部分访问量在生成策略标签时被扣回。§3.4 和 §4.1 再说明对手续手、终分和归属怎样训练共享网络。[论文](https://arxiv.org/pdf/1902.10565v5#page=4) |
| 作者后来怎样实现？ | 官方训练说明把产局、打乱数据、训练、导出、比较候选串起来。作者后来还实现了可选的重分析，在原有快慢混合对局结束后，从便宜步中抽一部分，恢复历史并用完整搜索重算策略目标。该选项默认关闭，源码给出实现，单独的胜率收益仍待实测。[官方训练说明](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/SelfplayTraining.md#L5-L32) · [重分析源码](https://github.com/lightvector/KataGo/blob/d91ea855110dae533f0aada947b2b7d78cc8a4e1/cpp/program/play.cpp#L1363-L1478) |
| 能否让学生学会纠正自己的失误？ | DAgger 用学生自己走到的局面向专家询问动作，缓解只看专家顺风局造成的状态分布偏移。Expert Iteration 用规划器在这些局面产生可学习的改进策略。两者合读时要追问一句：给定学生的后续水平，老师改选的那条令到底值多少？[DAgger 原文](https://proceedings.mlr.press/v15/ross11a.html) · [Expert Iteration 原文](https://arxiv.org/abs/1705.08439) |
| 没有可靠搜索教师时，神经网能学什么？ | PPO 从当前行为策略采样，用实际动作在新旧策略下的概率比和优势估计更新网络。OpenAI 的教程给了伪代码，适合逐行与训练器对照。若主要目标是赢一个已知强敌，策略空间响应方法也给出明确方向：对冻结对手学习响应，以其他对手独立检验泛化。[PPO 论文 §3–5](https://arxiv.org/abs/1707.06347) · [Spinning Up 教程](https://spinningup.openai.com/en/latest/algorithms/ppo.html) · [PSRO 原文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html) |

你给的围棋史文章里，最适用于这场比赛的是它比较棋力的方法。权重、搜索预算、硬件和双方执色共同决定一次读数。Master3s2 的搜索按墙钟截止，同样的源码在拥挤主机上可能只走到较浅层。比较 Sapling 的父子代要用同机、同图、换席的原版直打，并保留考官实际搜索深度与超时。围棋里扩大 Transformer 得到的 Elo 差不能用来估 Carbon 的训练量。[围棋史原文](https://read19.com/guides/go-ai-sota-history)

## 三份改变决策的仓内证据

1. **学得更像，实战可以更弱。** 用前五对局训练的模型在六张留图上都提高了动作命中率。在另外四张新图上双方换席，它和原版 Master3s2 打了 0 胜 8 负，场均官方分差 −2143；同场父代也是 0 胜 8 负，场均 −1537。前五回放仍适合学习动作、局面与对手风格。要判断新动作是否值得学，需要看到学生执行以后，真实对手如何响应。[16 场对照及双方收入](master6_actor_dev/strategic_destination_learning/elite_direct_policy/development/README.md)
2. **短期现金标签会漏掉投资和对手反应。** 同一批 234 张训练图、1,167 个局面，按未来 20 步现金给示范加权后，这 20 步的双方相对净现金增长提高 102.29，紧接着 40 步反而低 74.57。这是改变同一批回放的权重所得，无法解释成新网络打出了该分差。另有一条实证：在地图 621758540 第 69 步，网络看到眼前有 60.672 碳，选择让空载工停下采集。对方同时派空载工进入同格，双方消失，实际采集为零。同一根的等权对照向下撤出，避开碰撞。两条八步干预分别打到终局，分差为 −608.71 和 −0.95。后续动作与考官搜索已经分化，不能把全部 607.76 分都归给这次撤退。[跨图核账](master6_actor_dev/strategic_destination_learning/midgame_economic_policy/label_horizon/README.md) · [同根两条完整续弈](master6_actor_dev/strategic_destination_learning/midgame_economic_policy/native_response/FINDINGS.md)
3. **训练器以为改得很小，上场动作却变了。** 一次真实终局回报更新中，训练器只在父代提供的两个选项上算，平均 KL 散度为 0.00292。重新从发布包解码，288 个登记时点里有 67 个失去至少一条原先可选的整队令，30 个连当局实际执行的命令都已发不出来。两图换席对原版，新网和父代各 0/4，平均分差分别为 −1511 和 −1230。另做过固定两案的重打分，在六张图的留一验证中仅两图改善。简单缩短步长、再训练同一个二选一头，都已有反例。[整队动作概率与整场结果](master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/README.md) · [按步长核对](master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/physical_support.json)

## 建议只推进的一条主线

为复核第 3 点，本次把这两个已发布的包放回父代对原版的 24 条训练轨迹（六图、双席、每席两次），从开局按相同历史输入重算。**24/24 条旧轨迹都在第 4–38 步首次出现新包绝不会发出的原实发命令，中位数第 22 步；23/24 首次出现在普通确定性步。** 共复演 556 个前缀步，遇到第一个零概率即足以确定整轨概率为零。这个结果说明，给整条旧轨迹重加权无法从这批数据估算新包胜率；它不排除使用行为策略真正采样的单步数据来做小幅 PPO 更新。新包离开旧轨迹之后会好还是会坏，只能由自己的后续对局回答。[逐局出令概率与复算](master6_actor_dev/strategic_destination_learning/physical_policy_audit/README.md)

旧训练局还能回答一个更直观的问题：学生有没有机会尝试打劫或防守？在有两条不同队令的 275 个时点，用当时原版实际发出的对手命令各推进一拍。换选项能改变本拍种树的只有 2 处，能改变本拍正常交仓的 7 处，能救下载货工的 1 处。改变本拍夺树的是 0 处，从敌采集工手里实际夺货的也是 0 处。走位仍可能在几拍后创造机会，这张表只测当拍。下一版探索应允许学生依据局面在「返航／躲拦截」「采集／截击」「种树／转移」之间作出能改变实际行动的选择，并记录每条完整命令的采样概率。[275 次同根对照](master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/menu_coverage/README.md)

**以真实对手响应训练 Sapling 的策略网络。** 先沿用现有参数规模、推理预算与守卫。对当前候选先用旧轨迹复演，核对实际发送的完整命令在新旧包中的概率。特别检查前序工人改变后，其他工人的合法选项、资金和占位是否随之重算。若同一条完整命令可由多条路径或守卫回退产生，概率要把这些路径合并。关键验算是从最终单文件实际采出的动作频率能否复现训练器记录；一个固定菜单会系统性漏掉的动作，也要能被记录。已经失去支持的历史命令保留其零概率，不以很小的假数掩盖。

**随后只对可复现的策略做一次父子对照。** 前五回放提供模仿初值和训练时的对手预测目标。用于策略梯度的局则由学生自己采样，面对保持原版搜索和温度的 Master3s2 打到官方终局。每局记录双方胜负、含灭绝的官方分差及实际行动概率。双方每拍依据各自当时可见的历史发令；真实敌令、未来收益和隐藏碳图只进入事后监督。训练图同时覆盖被压制和领先时刻，在部署时学生自己决定采集、回防、封路或撤出。已有六图 24 场全部落败，只给出了败局之间的分差排序，无法从中辨认何种更新提高获胜概率。发现训练图太少或父子两包实际行为相同，应先解决这个具体问题。[已有同图采样的局限](master6_actor_dev/strategic_destination_learning/return_training_scope/official_objectives/README.md)

KataGo 式搜索仅在有证据的地方提供额外标签。在现有固定两案试验中，四张历史留图、28 个有真实终局对照的局面，教师选中较高分差的一臂 **14/28**。逐图等权的分差改善为 +86，直接固定改用备选令为 +131，教师仅在一图较原令有正增量。六个有第二条或第三条响应流的特选局面中，四个还出现符号翻转。当前证据不足以批量标注。今后若检验另一位教师，先在学生实际访问且双方存在有意义选择的局面上，比较被蒸馏的整份动作分布与父代，沿各分支让原版自然应答到终局。分开保留随机覆盖的局面和刻意多抽的危险局面，按来源图统计。策略标签只覆盖实测候选；原轨迹的终分可监督该行为策略的价值，替代动作要有自己的终局依据。[固定候选教师验收](master6_actor_dev/strategic_destination_learning/fixed_menu_teacher/README.md) · [正式 q 收益与抽样合同](../STAGE1_TEACHER_GATES_20261113.md)

值得保留的辅助任务是让共享骨干从己席当前信息预测对手下一拍会占哪格、未来一段时间双方各能交仓多少。它允许网络学习“这步留下采集，敌人可能进这格”之类的局面判断。仓内已有对手预测头，曾把行动预测命中率从 57.32% 提到 59.78%，但在 86 个时点中一次也没改变两套队令的首选。一个辅助头能否改善主决策，应由实际改令与终局共同验证。若需要把终局奖励提前分摊，势函数法在折扣和终止口径一致、所有终局势为零时，能保持整局目标的排序；它也无法补出未尝试动作的反事实收益。[已有攻防头](master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/opponent_contact/README.md) · [KataGo 对手续手目标 §3.4](https://arxiv.org/pdf/1902.10565v5#page=7) · [奖励塑形原文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

## 算力怎么花，强度怎么认

仓内另有可直接对照的 Echo9／Echo10 实例。策略组记录的 58 场败局中，47 场达到其搜索纠错条件，共保留 963 行、隔离 72 行。这里可以复用逐行局号、原动作和纠正动作的溯源做法。要将这些动作迁移为 Sapling 的训练目标，仍需验证它在自身能走到的局面和自身续弈策略下能否获益。[原始定版核验](../VERIFY_SEARCH_CORRECTION_20261120.md)

- L20 用来拟合网络。CPU 优先给原版的自然应答、真实终局与部署包计时。仓里曾测得另一教师在 Studio 单 worker 每局面 5.55 秒，119,600 个局面全标约 184.4 worker 小时，标 10% 约 18.4 小时。两数都未含真实续弈、产局、训练及复测。固定两案的搜索打分另有 108 根合计 20.75 秒的记录，两者执行的任务不同。[旧预算原始口径](../STAGE1_TEACHER_GATES_20261113.md) · [两案计时范围](master6_actor_dev/strategic_destination_learning/fixed_menu_teacher/README.md)
- 单个新版本先在预登记的开发图上对原版做同机父子换席对照。独立单位按地图计算，保留每场失利及墙钟。单文件 SHA、1.5 核 / 512 MiB、冷启、含通信的逐步耗时和考官搜索深度共同约束读数。首胜包在十张开发图换席取得 **2/20，平均官方分差 −1164.13**。这还不足以估计有多快能达到目标。[全部二十场与墙钟](master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/fresh_original/incumbent_history/README.md)
- 本次复核保留原定两图三臂 12 场开发赛的登记，暂缓派发。正式考卷继续封存，冻结候选后才在预先固定的 100 张新图换席；同一批 200 局须至少 180 胜且官方平均分差至少 +500。资料复核和单根续弈都未新增独立胜场。[开场前冻结的开发赛协议](master6_actor_dev/strategic_destination_learning/midgame_economic_policy/development/protocol.json) · [正式目标与台账](MASTER3S2_90.md)

# Sapling 对 Master3s2 的胜局证据与策略案例

复核基线为 main `53d4332b`。**我方可运行的两个首选参照是 PPO it50 和 Echo4**。两者在历史登记考场均超过六成，但那批逐局记录缺少考官每回合实际搜索的计数。在能审计搜索的近期实战里，单 it50 在同一张开发图换席两局均负。公开平台上，现有我方 Echo 得分卡也都输了。因此，适合用这两包产生不同的候选动作，与 Sapling、其他对手和真实终局回报比较。已有证据尚不足以把其中任何一包定为最优教师。[历史逐局复算](master3s2_references.json) · [近期原版 it50 双席](tree_exam_it50/results.json)

## 考官身份和计分

仓内公开原版的注册 ID 为 `d679ed45ab595e8b7bb1ea51c3451154`，文件 SHA-256 为 `65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5`，MD5 为 `57f9a594cce4f2ec9e3573d8d9d26ca1`。原作者发布形态含信念跟踪、0.58 秒墙钟搜索和温度 1 的非固定采样。旧考场 wrapper 的 MD5 为 `8d209ddb46faafae151aae16b7e17d55`，随后登记的 wrapper 为 `b4cde262695d38270e16bde33875a443`。实测强制 PyTorch 无法导入时，同一原版仍会返回合法动作，8 次调用搜索 0 次。因此这里把“源码和包装器一致”“搜索开关打开”“该局实际搜索了多少次”分开记账。平台得分卡提供公开注册名、座次和终局得分，没有提供当场运行文件哈希或每步搜索计数。[来源和注册 ID](../../sources/master3s2/README.md) · [考官源码](../../sources/master3s2/master3s2.py) · [依赖探针](master3s2_dependency_probe.json)

## 我方可运行策略及其成绩

均按候选自己的胜负和分差列出。前三行历史固定集每个种子打一局、按奇偶换席；it50 的 300 图包含在另两包的 500 图内。分母不能跨行相加当成独立地图。历史行可核双方分数、座次、选手文件和考官身份、双方错误数；不能由此补出搜索深度。

| 候选及取得方式 | 登记为完整原版的历史考场 | 直接源代码实战或公开天梯 | 使用边界 |
|---|---|---|---|
| [PPO it50 单文件包](../../code/packs/ppou2_m_it50.py)，MD5 `c35305af7f8a7bc58afb9cfb4918518c` | 187/300，0 平，平均 +262.12。席 0 为 99/150，席 1 为 88/150 | 早期 i5 研究场报 1/4、−374。近期有搜索审计的图 620098101 为 0/2，席 0 −1,549.34、席 1 −529.59 | 包内含量化网络与生产包装，现有双网构建脚本能读取其策略权重。单独的 it50 vnet 是评论家 |
| [Echo4 单文件包](../../code/packs/echo4_int8.py)，MD5 `819250bfc82632786063dabd46a616bd` | 313/500，0 平，平均 +197.74。席 0 为 163/250，席 1 为 150/250 | 早期本机对完整考官 0/4、−1,433。公开天梯同名考官 0/4、−1,612.13 | 可作为另一个策略分布，单独记录守卫改子和本体动作 |
| [Echo6 单文件包](../../sources/echo6/echo4_search_int8_d1_homefix.py)，MD5 `25f144f3d9ecc87be9073d135d3ab8c8` | 287/500，0 平，平均 +124.52。席 0 为 150/250，席 1 为 137/250 | 公开天梯 `claude-opus55-cloud-Echo6` 0/4、−1,119.86 | 可检验回家与终局节奏；在该历史场上，分差低于 Echo4 |
| [Echo4 借用 Master3s2 搜索的实验包装](../../code/exp/hybrid_m3s2search.py)，MD5 `49cabee62c8761b102c5f66f47e32777` | 166/300，0 平，平均 +158.45 | 早期研究场报 2/4、+294。对应消融另有 300 局对 Echo4 约 9 万步未改子 | 直接调用考官内部搜索和策略候选，只用于解释历史实验。对手定向训练需遵守单独登记的信号范围 |

上述大场原件位于冻结归档分支。[it50 逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/abD_it50_3s2.jsonl) · [Echo4 逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/f2_echo4_3s2.jsonl) · [Echo6 逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/f1_comp_3s2.jsonl) · [实验包装逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/abF_hybrid_3s2.jsonl) · [早期研究场原报告](../REVIEW_WALL_MASTER3S2_CLAUDE_20261116.md) · [完整和关闭搜索消融](../CLOSURE_ABLATION_STEP1_20261116.md) · [包权重读取](build_it50_v3.py)

Sapling 自身应独立记。it50 75%＋Sapling V3 25% 的首次三图双席为 0/6、均差 −397.84。随后 D1 在另一组两图双席为 0/4、−806.95，同图同席亲本 0/4、−1,197.57，改善 390.62 而未取得胜场。原始协议中另有 leg-1 学生 `e6f62c8b` 16/200、−885.85，是外部蒸馏线的进度；环 2 学生 `162d6142` 对原版为 0/200、−2,294.96。前者在 main 有逐局证据，其原名候选包未在该树找到。这些记录不替代 Sapling 正式胜率。以 Master2、m3s2r1 替身、关闭搜索考官或练习规则对手得到的胜场均未计入本表。[双网三图汇总](IT50_SAPLING_HYBRID.md) · [D1 与亲本](d1_sapling_results.md) · [leg-1 原始行](../../data/k0_calib/preview_leg1_vs_master3s2_live.jsonl) · [环 2 正式判读](../VERDICT_BUDGET_RACE_B_20261119.md)

## 平台上的外部胜轨

按冻结 main 的 `data/*_matches.json` 全部 70 个家族索引取并集，以“赛事 ID＋game”去重；同卡的两份来源必须在物理座次和得分上一致。精确匹配原版公开注册名 `claude-AlphaCarbon-Master3s2` 得 338 张卡。Master3s2 本人的两个索引并不收齐所有对手侧卡，例如 it720-preview 的第九场来自其他索引。名字含 `Krypton-M3s2-EarlyHome` 等衍生对手均不混入。公开登记名与仓库源文件的关系由来源说明支持，每张得分卡无法单独证实部署字节。[Master3s2 家族索引](../../data/m3s2_matches.json) · [第二个 3s2 索引](../../data/master3s2_matches.json) · [it720-preview 索引](../../data/it720preview_matches.json) · [Master6 索引](../../data/master6_matches.json)

| 对手的公开注册 | 对 Master3s2 胜/局 | 自方平均分差 | main 的 data 目录中，按赛事 ID 可直接定位的逐步文件 |
|---|---:|---:|---:|
| RyanX-AIPowercore-it720-preview | 9/9 | +1,588.89 | 2 |
| ClaudeFable5.2-v7 | 9/9 | +782.46 | 8 |
| ClaudeFable5.2-v8 | 11/11 | +1,078.34 | 8 |
| RyanX-AIPowercore-M9XJ-630 | 7/8 | +837.97 | 0 |
| claude-AlphaCarbon-Master6 | 1/1 | +2,120.03 | 0 |

这五行是第三方参赛策略，得分卡及部分动作记录可核。没有以同一原作者参数哈希核验的可运行策略权重可交付；近似名 `*.npz` 或复刻包不能代替。平台我方其余对应记录为 Echo3 0/1、−1,899.73 和 Echo5 0/4、−1,415.48。Master6 的单场为 `1cdd726ef54f66516e0d315693c85bf1-1`，得分 7,889.25 对 5,769.22，样本量只支持描述本局。[Fable v7 全部得分卡](../../data/fablev7_matches.json) · [Fable v8 全部得分卡](../../data/fablev8_matches.json) · [M9XJ-630 得分卡](../../data/m9xj630_matches.json)

## 三个可直接重放的案例

下面三局的原始卡逐项核对了双方座次、得分、种子和连续 299 次联合推进。差值均为胜方现金减对方现金。为避免事后挑时段，用全局现金差**单步增长最大**的一次作为转折，并给出转折前五步的起点。这是事后研究窗口，尚未证明某个动作单独造成胜利。

| 胜方、物理席和种子 | 官方终局得分（胜方先） | 实际观察 | 回放文件 |
|---|---:|---|---|
| it720-preview，席 0，1863859933 | 10,344.58 / 8,704.30 | 第 60 步树数 0 比 11，现金落后 111.49。第 80、84、85 步现金差依次为 −430.0、−414.0、+39.9；84→85 胜方全队带货 749.4→294.9。用这一段检查交货准备和运输，而非给“少种树”加奖 | [f79352e6，压缩件 SHA-256 前缀 0ec5e961f721](../../data/it720_corpus/f79352e66e4925297b8b2dfff0303d93-1.json.gz) |
| Fable v7，席 1，1810398867 | 8,472.70 / 7,669.82 | 第 60 步树数 7 比 4，现金已领先 450.86。第 213、217、218 步现金差为 +355.2、+470.2、+836.0。第 240 步树数反为 2 比 9，现金仍领先 909.55。留作领先时守住交付和评估树回本的样本 | [74daa2ee，压缩件 SHA-256 前缀 a3d8b0536215](../../data/fablev7_corpus/74daa2ee9dcdd4cdb5acf4f4c31573a9-1.json.gz) |
| Fable v8，席 0，744327590 | 7,806.64 / 6,650.91 | 第 165 步，单步规则重放显示对方 `player-1-worker-29` 的 132.94 货被胜方存活采集工截获；这笔货不能直接记成银行现金。最大跃升在第 276 步，胜方全队带货从 845.7 降到 452.5，现金差从前五步的 +1,154.4 经上一步 +992.3 到 +1,406.1。两个事件相隔 111 步，需分别归因 | [2accb93f，压缩件 SHA-256 前缀 1a54cc2a9fcd](../../data/fablev8_corpus/2accb93f01450a1680e79fd9512f4fb4-1.json.gz) |

下一轮可分别考查运输兑现、树投资与防守、截获后持货归属三类选择。先在自有训练地图上让 Sapling、it50 和 Echo4 对同一公开状态提出合法动作，用相同续弈比较终局 U，保留本来也该维持 Sapling 动作的负例。观察案例的原图及这轮读过的真身开发图继续按已曝光处理。现有 Master3s2 定向 RL 登记指明 m3s2r1 起跑这条线，只从学员自己的终局结果学习。它不得读取 Master3s2 步级动作、搜索分布或内部价值作监督；对固定陪练的成绩只支持对手定向的结论。[原定向 RL 登记](../RULING_RL_UNFREEZE_3S2_OPPONENT_20261119.md)

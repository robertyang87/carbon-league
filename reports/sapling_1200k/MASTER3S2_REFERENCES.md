# 我方历史模型对 Master3s2 的实战基准

从 `sync/h29` 的逐局记录复算，Echo4、it50、Echo6 在各自历史固定集上均赢过半数。能直接取用的两个较强参照是 Echo4 和 PPO it50。它们对训练路线的价值是提供一个能赢真身的比较对象。Sapling 的胜率需用自己的冻结权重直接取得。[原始比赛记录](https://github.com/robertyang87/carbon-league/tree/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live) · [原注册](../REGISTRATION_ABLATION_STEP1_FULL_20261116.md)

| 冻结参照 | 席 0 胜场 | 席 1 胜场 | 合计胜率 | 自方平均分差 |
|---|---:|---:|---:|---:|
| Echo4 `819250bf` | 163/250 | 150/250 | 313/500，62.6% | +197.74 |
| PPO it50 `c35305af` | 99/150 | 88/150 | 187/300，62.3% | +262.12 |
| Echo6 `25f144f3` | 150/250 | 137/250 | 287/500，57.4% | +124.52 |

三场均无平局。程序逐行核了双方原始分数、胜负、种子唯一性、席位、双方包和 harness 的 MD5，以及双方错误计数。每个种子打一局，席位按奇偶交替。it50 的 300 张地图包含在另外两场的 500 张里，三场合计 1,300 行，共 500 张独立地图。这些经常复用的地图适合参考和开发。[复算件](master3s2_references.json) · [Echo4 原包](../../code/packs/echo4_int8.py) · [it50 原包](../../code/packs/ppou2_m_it50.py) · [Echo6 原包](../../sources/echo6/echo4_search_int8_d1_homefix.py)

## 考官搜索与机器时间

原场登记的考官源码 MD5 是 `57f9a594`，wrapper 是 `8d209ddb`。后来 Sapling 千局考试沿用同一源码，wrapper 改为 `b4cde262`。原场每行的考官统计只有 `other_errors`，全为零；逐回合完成搜索的次数与深度无法从旧档重建。F0 二十局冒烟另报考官平均 16.6 ms/步。这是当时的计时记录。[旧场身份与计时](../RECORD_COMPOSITE_SUBMISSION_DEVIATION_20261116.md) · [后续考官身份](../EXAM_SAPLING_VS_MASTER3S2_20261119.md)

源码在加载 PyTorch 或搜索依赖失败时允许退回策略动作；搜索开启时先根据已经耗去的墙钟和实测成本选候选数，每轮再按剩余时间决定是否继续。默认 `_S_ABORT=0.58` 秒，深度上限 8。在后来直接导入源码的 Sapling 四局微测中，考官对 V2 平均 450.7 ms/步，对 V3 平均 448.5 ms/步。每臂各调用 1,196 次，其中各 1,194 次完成搜索，走到深度 8 的次数分别为 893 和 972。机器、候选对手和局面均不同，两批耗时不能换算为强度差。要复验历史参照的可迁移性，应在同一台当前考场让原包直打，留存考官 `searched`、`single`、`changed`、深度、逐局结果和双方计时。[考官源码](../../data/master3s2_source.py) · [V2 直接对局](cash_exam_v2.json) · [V3 直接对局](cash_exam_v3.json)

## 真身作为 RL 陪练时的身份核验

2026-11-19 的另一条 PPO 线以 `m3s2r1` 为起点。Master3s2 在联赛训练池中的局数占比封顶 25%，训练只取学员自己的终局 U、胜负与分差。该线对 3s2 的 35/50/65/80/90 各档，是针对这个特定对手的目标测量；独立的 Echo6 守门尺检验迁移。五档依次使用 966200–967199 的互斥区块，**每档 200 张图各打一局、席位随种子奇偶交替**。这与 Sapling 的 100 张图 × 双席设计分别登记。上述提交登记了用途和种子，本次审查未得到该 PPO 线正在运行的具体对手路径或机器上的搜索计数。[RL 用途登记](https://github.com/robertyang87/carbon-league/blob/0e560247/data/k0_calib/rl_opponent_m3s2_registration_20261119.json) · [分档区块](https://github.com/robertyang87/carbon-league/blob/50125f6f/data/k0_calib/rl_staged_verify_seed_blocks_20261119.json) · [联赛与双尺](https://github.com/robertyang87/carbon-league/blob/6d932300/reports/SUPPLEMENT_RL_LEAGUE_POOL_20261119.md)

对仓内实际入口做了一个可复跑的反例。固定原版源码 `57f9a594`，经 `eval_v53.load_agent_fn` 加载，只在一臂模拟 PyTorch 不可导入。关闭依赖时 8 次调用全正常返回动作字典，其中 7 次有非空指令，`_SEARCH_OK=False`，`searched=single=0`；同设置正常导入时 `_SEARCH_OK=True`，8 次均正常返回，本次实际完成 7 次搜索、另有 1 次根上只有一个候选。两臂的 `agent.last_stats` 都为空。旧 `8d209ddb` 的部署时三件套有文档留证；在本次查阅的 main 和八条 `sync/h*` 分支尖端未找到该 wrapper 的独立同名字节。探针检验的是原版源码在依赖失效时的行为，无法据此追认旧考试当时的执行模式。[降级探针](master3s2_dependency_probe.py) · [两臂实跑](master3s2_dependency_probe.json)

`rl2_ppo_rollout.make_opponent` 对 `.py` 对手沿 `load_bot → eval_v53.load_agent_fn` 将其当作普通可调用策略；常规 `bigeval` 从 `agent.last_stats` 汇总内部错误。要把分档胜率归到开启搜索的 3s2，训练来源与考试各自应保留**实际使用的 wrapper 和底层源码哈希**，考试显式检查底层 `_SEARCH_OK`、`_TEMP`、`_SEED`，逐局记录 `searched`、`single`、`aborted`、完成深度和调用次数。无搜索且正常出招的局里，W/L 和 `err=0` 仍可能看起来完好。比赛因证据不足时，另记为对未验证搜索形态的读数。[PPO 对手入口](../../code/rl2_ppo_rollout.py) · [考试加载器](../../code/eval_v53.py) · [逐局计分](../../code/bigeval.py)

## 90% 的验收口径

当前预备终验的主指标是 **100 张新地图各打两席，200 局总体至少 180 胜**。冻结候选包、考官与规则后才抽图。平局不计胜；两席共用一张地图，总体区间按地图计算。分席另报：如果追加“两席各自实测至少 90%”的主张，则每席需至少 90/100；如果追加“两席的真实胜率均高于 90%”且要求联合 95% 置信度，则每席取单侧 97.5% Clopper–Pearson 下界，需各赢至少 96/100（下界 0.9007）。开发或选权重过程中看过的图单列，未完成、异常降级、搜索计数缺失按预先约定的完整性规则报告。[当前终验程序](tree_win_gate.py)

从仓库根目录复算历史三场和两批计时。脚本只读取钉死的 Git 对象，输出可与同目录 JSON 逐字比较。[复算程序](master3s2_references.py)

```bash
git fetch --no-tags --depth=1 origin refs/heads/sync/h29
python reports/sapling_1200k/master3s2_references.py > /tmp/master3s2_references.json
cmp /tmp/master3s2_references.json reports/sapling_1200k/master3s2_references.json
```

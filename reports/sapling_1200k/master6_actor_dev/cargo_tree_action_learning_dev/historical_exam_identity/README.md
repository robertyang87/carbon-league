# 历史 it50 的 187/300 和当前混合模型的 1/8

历史格 D 的 187 胜可以重算。能排除一个具体解释：按原注册的排期与进程数，格 D 不可能让考官平均每步花上当前父代实战的 437 毫秒。耗时差的具体原因仍缺格 D 自己的考官运行状态。把 62.3% 作为同一搜索强度下的 Sapling 起点会高估训练起点。[计算结果](measured.json) · [复算程序](verify.py)

| 完整批次 | 胜/局 | 己方均分 | 考官均分 | 官方平均分差 | 考官耗时或搜索记录 |
|---|---:|---:|---:|---:|---|
| 历史单 it50，300 张图各一席 | 187/300 | 8,517.79 | 8,255.67 | +262.12 | 整批按注册 299 步推导上界 72.46 ms/步 |
| 当前混合父代 09b41c24，四图双席 | 1/8 | 8,102.83 | 9,581.66 | −1,478.83 | 实测 436.91 ms/步，2,392 步中 1,922 步到深度 8，0 次零深回退 |
| 同期首胜候选 8da4b465，相同四图双席 | 1/8 | 8,229.66 | 9,430.33 | −1,200.67 | 实测 390.80 ms/步，1,723 步到深度 8，263 次零深回退 |

两份现役包分别赢了图 621758346 的席 1，另三图全部失利。父代与候选分别消耗考官公开入口累计 0.2903 和 0.2597 小时。历史和近期的地图、考生、对局采样、实现环境均有变化，表中考官得分的差不能归为某个网络的训练效果。[历史 D 逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/abD_it50_3s2.jsonl) · [首次双席](../direct_original/README.md) · [独立复测](../replicate_original/README.md)

## 从原始时钟能推到哪里

格 D 的五波出分分别在 20:34:50–51、20:35:14–16、20:35:38–42、20:36:03–07、20:36:26–28，每波恰好覆盖连续种子的 62、62、62、62、52 行。关搜索格 C 的五波节奏相近。该时钟是父进程在收到结果后写的本机时钟。首项出分时，首批比赛已经完成，不能把 98 秒写成整批用时。[历史 C 逐局](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/abC_it50_3s2nosearch.jsonl) · [记录器](../../../../../code/bigeval.py)

注册安排同机顺跑 F→B→D→C→E，每格 62 个 worker。前格 B 最后出分 20:33:17，D 最后出分 20:36:28；兼顾时钟舍入，D 全批最多 192 秒，即最多 11,904 worker·秒（3.307 worker·小时）。单一 worker 内双方顺序执行，故可扣除 D 的考生耗时 60.263 ms/步，且逐行向下放宽 0.005 ms 的舍入误差。按 300 局各 299 步，剩余可分给考官的上界为 72.451 ms/步，该数还把模拟、装载和等待统统让给了考官。即使只认 300 行双方均留有正数现金的第 270 步快照，上界也只有 86.705 ms/步。两个界依赖注册所述的单机顺排及 62 worker；没有将其写成格 D 实测耗时。近期父代按逐拍收据实测为 436.909 ms/步。[前格 B 末记录](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/bigeval_live/abB_echo4_3s2nosearch.jsonl) · [原注册 §3–4](../../../../REGISTRATION_ABLATION_STEP1_FULL_20261116.md) · [顺序调用及第 270 步快照](../../../../../code/eval_v53.py)

另有同一考场的独立 F0 证书。考生是原文件 exp/master3s2_src.py，20 个子进程读回 MD5 57f9a594，对手是 opp_master2r.py。5,980 次调用含通信均值 16.6 ms、最大 20.3 ms、子进程峰值 62.6 MiB。证书绑定了不同入口、对手和资源配额，因此不能拿 16.6 填作 D 格的考官单步实测。[F0 原始证书](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/m3s2_timing_cert.json)

## 源码、依赖、温度与样本

- D 的 300 行全是考生 c35305af、考官 wrapper 8d209ddb、harness 354bcd1b，源码 57f9a594 由机上登记支持。关搜索 C 的 wrapper 是 f5d7b7ee，与仓内 code/exp/opp_m3s2.py 全 MD5 一致。D 的包装器完整字节、底层实际导入路径和进程内状态均未在本次查得。考生、C/D 的包装身份可以区分，D 内部具体如何调用底层仍需当趟工件。[机上三件套](../../../../RECORD_COMPOSITE_SUBMISSION_DEVIATION_20261116.md) · [已验字节的 C 包装器](../../../../../code/exp/opp_m3s2.py)
- bigeval 使用 ProcessPoolExecutor，子进程不单独传 env，eval_v53 对两席各调用 agent(observation, CFG)。考官源码默认温度 1、随机种子 None。C 包装器允许从环境覆盖 M3S2_TEMP、M3S2_SEARCH、M3S2_HOME；bigeval 的启动记录只筛 EVAL_* 和三项具名环境变量，遗漏这三个覆盖项。certify 则用 sys.executable 启动自己的子进程并继承各自父进程的环境。D 与 F0 的父解释器和环境没有逐进程绑定，D 的温度 0 或 1、主动关搜索与否都不能从成绩判定。D 的 300 个 bst 字段只含 other_errors=0。[加载与计时](../../../../../code/eval_v53.py) · [环境记录](../../../../../code/bigeval.py) · [certify 子进程](../../../../../code/certify_pack.py) · [原版策略](../../../../../data/master3s2_source.py)
- 物理机为 C1 **spot-34**；sync/h29 是归档名。11 月 14 日的“无 Torch”记录属于 spot-29 的 .venv 和 spot-31 的 .venv。spot-34 在 11 月 15 日另购，部署的是从 spot-30 提取的 fdd877ba 镜像，验收项确认 NumPy 2.5.4。spot-34 同日另一场 it50 对 Echo4 的脚本使用 /root/carbon/.venv/bin/python，该脚本不是格 D。格 D 缺同进程解释器路径、Torch 导入测试、配置和搜索计数。原版遇 Torch 导入失败会让搜索直接返回策略动作；现有强制失败探针可复现，历史是否实际走此分支仍未证实。[spot-34 场地](../../../../REGISTRATION_ABLATION_STEP1A_20261116.md) · [旧机器依赖](../../../../war_room_log/LOG_20261114.md#L8110) · [新机部署](../../../../C3_LOGISTICS_LEDGER_20261116.md#L330) · [镜像来源](../../../../war-room/spot_bootstrap/golden.md5) · [同机另一考试脚本](https://github.com/robertyang87/carbon-league/blob/d599ec679718dd4407a718aa6ad810e35e2d3983/synced/h29/scripts/it50-exam.sh) · [降级探针](../../../master3s2_dependency_probe.json)
- 同地图同行对读 D 与 C，D−C 官方分差均值 −5.47，按地图配对的 t 区间 [−111.04, +100.11]；考官均分变化 −15.87，区间 [−118.23, +86.49]。采样轨迹各自独立，该区间衡量本组同地图结果的差，无法反推两臂搜索开关相同。当前父代由 it50 及单独训练的 Sapling 按 75/25 联合出令。在当下真身运行中，也有完全相同的历史单 it50 原包在一张独立开发图上双席皆负，分差 −1,549.34、−529.59；这一张图不足以定量归因到混合策略。[原版 it50 两席收据](../../../tree_exam_it50/results.json) · [当前父代训练与发布](../../deployed_mix_policy_gradient_dev/full_episode/README.md)

复算须在能读取已归档 Git 对象 d599ec67 的 checkout 内运行。本脚本只查询四条固定 Git 路径及已落盘的现役收据，不开新比赛。

```bash
python reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/historical_exam_identity/verify.py > /tmp/sapling-historical-exam.json
cmp /tmp/sapling-historical-exam.json reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/historical_exam_identity/measured.json
```

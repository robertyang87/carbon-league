# Sapling（未训练初始网）vs Master3s2 考试记录

- 时间：2026-11-19 00:06–00:08 CST（冒烟 00:06:04，主考 00:06:59–约 00:08:10）
- 座位：spot-72（carbon-shanghai-spot-72，c8ae 64 核，100.100.94.84），独立目录 /root/sapling_exam，未触碰 accept72 环 2 验收套件；考后座位空闲（bigeval 进程 0）
- 考官注册口径：REGISTRATION_MASTER3S2_EXAMINER_20261116 §5 固定集，分布读数（margin 为主尺、U 仅描述；3s2 采样 _TEMP=1.0/_SEED=None，不作逐种子配对主张）

## 身份三件套（机上直读 md5）

- 考生包：sapling_init_int8.py，md5 8a397c59022624d7cb4c5f06c923fb33，837,152 B；打包口径 cnet2_pack（仓 main e3a0ae09 的 code/cnet2_pack.py 119e9bb9），teacher=2、shield/nocollide/endgame/pshield/rguard 全关；pack_src_sha256 1cdaa40c12c4c623193307408ee0a568d18a503facab7e8d328fd43b8d96cc86
- 初始权重：init.npz（L20 训练 venv torch 2.15.0+cu132、CPU 生成，seed 551187），sha256 fae95a7240198a2e25582096a530dc8f3e4216a3a22724b1b78e9142370bedb3，md5 e798c6b06ee3f8c737b6687e3f9331a2，2,228,662 B；回执 params=551187、cfg={C:64, NB:4, TA:0, D:96, NH:4, L:1, E:32, G:8}、untrained=true。原件留 L20 /root/sapling_init_tmp/init.npz 与本机 ~/workspace/sapling-exam/init.npz
- 考官：master3s2.py 57f9a594cce4f2ec9e3573d8d9d26ca1 ＋ wrapper opp_master3s2.py b4cde262695d38270e16bde33875a443（README 3644794db7a8edd78e19945caf7b2aaa）
- harness：bigeval.py 354bcd1bc2dc8bc188c813f53d28c893（钉版）；环境 PYTHONPATH=accept72/k0_dist/pkg/code（eval_v53 45792dfb、cnet2_feat 2a45b08b、cnet2_np d8c15ad4，与仓 main 逐字合）

## 开场闸

- 冒烟 20 局（20291101–20291120）：err=0、FAIL=false、双边内部错误 0；考生单步 mean 4.3 ms、worst 12 ms（闸五 1 秒基线/3 秒上限内）
- 正对照 loop1 学生包（445c0c98）vs 3s2 同块 4 局：mean −419、1 胜、dep [8597, 8651]、plants [33, 25]——考场与考官健康，与已知 −750 量级相符
- 自博对照 Sapling vs Sapling 4 局：全 0 分、零产出——零行为是考生自身属性，非考官压制或接口故障

## 主考结果（固定集 20291101–20292100，1,000 局，10 分片×100，EVO_WORKERS=6/片）

- 完整性：1,000/1,000 局，err=0，十片 FAIL 全 false，a/b 内部错误总计 0
- **平均分差 −4,819.8（sd 3,156，se 99.8，95% CI [−5,015, −4,624]），中位 −4,495.6，最好单局 −288.8，最差 −13,884.5**
- **胜/平/负 = 0 / 0 / 1,000**；U 均值 0.0411（描述口径）
- 双席拆分：席 0（500 局）−4,544.2（se 139.3，U 0.0462）；席 1（500 局）−5,095.4（se 142.0，U 0.0360）
- 行为账（每局均值，考生/考官）：存入 0 / 4,036.9；树收入 0 / 4,716.2；种树 0 / 41.4；死亡丢货 0 / 14
  - 逐局台账抽查：考生基地从不招募种树工（rec60_p=0）、采集工从不采收（cargo 恒 0、现金恒 20）——未训练网没有形成任何生产闭环，不是「打得弱」，是「没在打」
- 成本：考生单步 mean 6.14 ms（p95 6.7，单步最坏 14 ms）；1,000 局全场墙钟约 85 秒（60 worker 并行）

## 参照

- 已训学生对同一考官的历史 margin：loop1 父代环 0 定标约 −754（200 局）、快线学生 preview −774（200 局，989000 块）。Sapling 未训练基线 −4,820 约为其 6.4 倍深——此数即后续训练环对 3s2 外考的起点基线。
- 结论：工程链（生成→打包→钉版考场→考官）在注册固定集上全通、零错误、成本可忽略；棋力为零产出基线，首环训练的首要目标是建立「招募种树工＋采收存入」的生产闭环（对应 SAPLING_551K 报告问题 1 的开局招募读数）。

## 证据

- 本机归档：~/workspace/sapling-exam/sapling_exam_results.tgz（md5 ad60478c408d5d2ada1cf960952d9fda；含十分片 bigeval.jsonl 汇总、逐局 live 行、冒烟/对照日志、考生包）
- 72 原件：/root/sapling_exam/（shards/s0–s9、harness、stage）

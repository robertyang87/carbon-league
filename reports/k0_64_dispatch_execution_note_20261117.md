# §6.4 派单执行注记（DGX，2026-11-17）

应件：DISPATCH_K0_64_PRODUCTION_20261117（commit c0e70148）＋ RULING_K0_NLOCK_20261117（commit 150ff93d，N_games＝80）。本注记落派单 §2／§4 要 DGX 落字的三件：build 入仓版引用、模式关回归闸状态、测量局种子域指定与零交集复验。

## §A build 入仓版（派单 §2）

- 入仓版：repo main `code/cnet2_build.py`，commit **4f21d3baabff4efabc172c889a36a3ac2e06ec25**，blob 4d48b46bd5bd52399c50fcfaf87a58982feae158，文件 md5 **5c2dd58e734ccab8d71c5a6fce51dde9**（84,051 B，API 读回验合；该 commit 只含此一文件）。
- 内容：按 RULING_LADDER_BUILD_SCAN_20261117（33f8c6e3）加 `CNET2_LADDER_FRAME=1` 显式模式（默认关；非 ours 分支 sides=[0,1]；与 CNET2_WEIGHTED 同开 die；模式状态写入代 meta 与 gen_key）。源＝钉版 e0051524，diff 只落 cnet2_build.py 一文件。
- cnet2_feat 一字未改，复验 md5 2a45b08b6f97508541e62867e408748e。DGX 生产副本仍 e0051524；§6.4 开工只认本入仓版。
- 闸①扫描级证据（试点 200 副本，局集与注册表 keys sha256 67a3a318… 全等）：钉版基线 0/200（no_teacher_side 200）→ ladder 模式 200/200、missing 0、filtered {}、sides 全 [0,1]；main 冒烟 3 局 samples 1794、teacher_sides 全 "other"、meta ladder_frame=true。

## §B 模式关回归闸（派单 §2／裁定 §4 闸②）——已过（替代口径），原字面前提纠正待策略组改字入册

原字面「模式关扫 shard1 与 gen-5f47630f 在册一致」对钉版自身不可满足：钉版 e0051524 模式关扫 gen1r_shard1 ＝ 0 局、no_teacher_side=200，与在册（200 局、sides 全 [0,1]、filtered {}）不符——该生产代非现行钉版 _scan 在现行 index 上产出。替代证据（已报策略组 b5b88778）：

- (a) 默认路径未触碰：新版模式关 ≡ 钉版模式关，三语料逐项全等——shard1（同 0 局、filtered 同）、m2s_corpus（入选 8 局，sides 映射 sha256 da3b3b5e… 两版同，filtered {min_steps 1, no_teacher_side 198} 同）、top5（入选 10 局，sides 映射 sha256 e49e6ff7… 两版同，filtered 同）。
- (b) 正向复现生产代：ladder 模式扫 shard1 ＝ 200 局全 [0,1]、filtered {}，局集 sha256 35339389… 与 sides 映射 sha256 cd931f5f… 与 gen-5f47630f 在册 meta 逐字全等。

## §C 测量局种子域（派单 §4）——DGX 指定并入册

- **种子域：game id 991050–991129（80 局，连续块）**，即窗口裁定预留的测量生产起点 991050 起首个 80 局块；父代 selfplay 新产、无教师调用。
- 指定时逐 id 机械复验（脚本 DGX /home/arm/junyang/check_seed_domain.py，2026-11-17 01:1x CST 直跑）：与 shard0（994000–994199，200）／shard1（994400–994599，200）／Stage B 池 1,525／train 子集 1,202／test 子集 323／标定根 991000–991049（50）／K0 留出 987000–987031（32）／试点注册表 200 逐 id 交集全 0；DGX 数据树内无任一候选 id 的现存回放文件。总命中 0，判定 ZERO-INTERSECTION。
- 生产前（2026-11-18 生产窗开工时）按派单 §4 同口径再复验一次，复验结果随生产开工三件套报。

## §D 其余执行项状态

- 腿一分区登记＋逐局 sha256 清单（派单 §3）：生产窗（2026-11-18–19）前由 DGX 发出登记，各机收后验、首片回 DGX 验过放行后续波次——待发（不在本注记内）。
- 测量 duel 跨机切分（派单 §4）：待 S／C 训后，照 RULING_SIGMA_DUEL_SPLIT 五条件类推，分区表先入册。
- S 臂开工裁定不在本件（派单 §5／§6 照原口径）。

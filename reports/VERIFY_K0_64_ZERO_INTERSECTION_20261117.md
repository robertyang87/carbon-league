# §6.4 测量种子域零交集复核回执（2026-11-17 20:14 CST，DGX 线）

## 口径与来源

- 依据：DISPATCH_K0_64_PRODUCTION_20261117 §4（生产前机械复验、逐 id 不交、零命中才许产）；N 锁 80（RULING_K0_NLOCK）；种子域 991050–991129（FRAMEWORK_POOL_SWAP_RULING_20261117）。
- 本复核原定 2026-11-18 第一件事，作战室 2026-11-17 20:08 令提前到今晚，与 2026-11-17 15:55 复核同口径重跑。
- 机器记录：data/k0_calib/k0_64_measure_seed_domain_20261117.json（本回执同批入仓）。

## 结果：PASS（六组对照，全部零命中）

| 对照集 | 规模 | 值源 | 命中 |
|---|---|---|---|
| shard0（gen1r_shard0） | 200（994000–994199） | DGX carbon/data/gen1r_shard0/replays 文件名普查 | 0 |
| shard1（gen1r_shard1） | 200（994400–994599） | 仓 data/k0_consumed_gen1r_shard1_registry_v2.json（md5 781cdf00）；与机上目录普查逐一相等 | 0 |
| Stage B 池 1,525 | 1,525（train 1,202＋test 323） | 仓 stageB_split_N323_train.json（md5 f7b50267）＋stageB_split_N323.json（md5 fd4cf144）；池 md5 d6916c80；按 id 原串与 selfplay- 前缀两种形态查 | 0 |
| 标定根集 | 50（991000–991049） | 区间＋DGX k0_dist/diag_split_registry.json 键集（50 键） | 0 |
| K0 留出 | 32（987000–987031） | 在册留出种子域 | 0 |
| 附带全普查 | 810 | DGX carbon/data 下全部 selfplay-*.json.gz 种子普查：域为 987000–987031／988000–988127／991000–991049／992000–992199／994000–994199／994400–994599 | 0 |

测量域 991050–991129 恰在标定根集（止于 991049）与下一在册域（992000 起）之间，与全部在册域逐 id 不交。DGX 上 selfplay-991050 文件不存在（未产）。

## 测量生产包（已备，明日窗口待令即产）

DGX `/home/arm/junyang/k0_dist/k0_64_measure_pkg/`：

- `SEEDS_991050_991129.json`（md5 71ed8987）——80 种子登记，状态 registered-not-produced。
- `ZERO_INTERSECTION.json`（md5 a4dd71d9）——本复核机器记录。
- `selfplay_gen.py`（md5 bd62ead9）——与 k0_calib_holdout identity 的 code_md5 同代，源 DGX carbon/code/producer/。
- `PRODUCTION_PACKAGE.json`（md5 278ad6b1）——代码集 md5（照标定 identity 六件）、标定 config 模板（config_hash 0b559cee，仅作模板）、产出目标 carbon/data/k0_64_measure/（replays 目录已建，布局同 k0_calib_holdout）。
- 待生产令定：父代 weights 身份、OPEN_SEED 基、执行座与分区。本包未自定这四项。

## 结论

零交集复核 PASS，种子域可用于 2026-11-18–19 §6.4 腿二测量生产；生产包已备齐，待作战室生产令。

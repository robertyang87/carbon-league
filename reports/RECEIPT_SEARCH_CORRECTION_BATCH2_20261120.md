<!-- 落仓头（工程组，2026-11-20）：执行人收据原件（~/workspace/lossprobe3/batch2/RECEIPT_BATCH2.md）逐字转录；主代理独立核验见 reports/VERIFY_SEARCH_CORRECTION_BATCH2_20261120.md（重算 0 不符，判读通过）。 -->

# 输局旁路 ③ 第二批（Echo11 败局）终版收据

执行人：工程组。范围：补充件 `reports/SUPPLEMENT_SEARCH_CORRECTION_BATCH2_20261120.md`
（commit ab0d96aa）冻结的 Echo11 36 局中 11 局败局；设计口径沿第一批
`reports/DESIGN_SEARCH_CORRECTION_20261120.md`（80566708）。

**本收据为执行人交付件；我不落仓——主代理独立复算后落仓并入 Gen-5 报批件。**

## 1. 冻结范围与盘点核对

- 清单 `data/echo11_corpus.manifest` 36 行，md5 `f46aea516ca8479bea8510c4b4c2a836`；
  matches 36 局全 done，清单 id 与对局 id 逐字相等。
- 胜负按名取座：冻结 36 局＝**25 胜 / 11 败**。11 局败局：
  `e4921a31`、`d15d0dd9`、`f3dc9e3c`、`2d8b2e5d`、`d94fc61c`、`4cab29be`、
  `a0299775`、`9d8ce33a`、`c4bf549f`、`7791ec33`、`cbb0000b`（全 id 见 `games_echo11.json`）。
- 座位规则复核：`order.index(Echo 侧)` 与生产转换日志逐局对，7/7 有生产行局全合；
  终局分差与转换日志 `final_margin` 7/7 全合。
- 天梯交叉核对：10:50 直读 DGX `ladder.py mine` 为 Echo11 40 局/27 胜（天梯续打中；
  冻结 36 局 25 胜/11 败不变）。
- 语料：11 局回放仓内字节与 DGX 镜像字节逐件 md5 全合；Studio 11 件已落位。
  披露：清单逐件 md5 栏与文件字节及解压后内容均不合——Echo9/10/11 三份清单皆如此，
  是侦查兵线清单格式的既存问题（非本批文件错）；本批以「仓内＝座上字节同＋保真对生产行」为准绳。
- 冻结后新局（`9bbcb1f5…`、`3f512224…` 及之后）未进本批。

## 2. 检查点与包（定版包，裁定 1a6e45f0）

- 检查点 gen4：`/home/arm/l20_katago_mirror/work0/gen4/ckpt.npz`，
  md5 `42888eabe142d688de3e0110dcda48e2`（DGX 直读）。
- 定包 `gen4_fixed_u.py`：md5 `b4b55da0b1df7db49eae929cef15af16`，10:42 DGX 建成
  （打包器 57c75d84＋底盘 4b3423a0＋布局 6877bc73＋桥 f31b07dc，四件开建前 DGX 逐件 md5 核合）。
- 仪表化包 `gen4_fixed_u_lp3.py`：md5 `5b2f5dcf2356a4c323ed65b706099844`，
  16 补丁全上（P0/P1/P2a–m/P3），语法过、`perf_counter` 0 处。两包双座均 md5 读回核合。
- 包 `--verify`（PID 2625484，10:42–11:25:30，DGX 争用下 43 分钟）：
  前向 err **0.0000**（tol 0.25）PASS；确定性 1 局×2 跑 **IDENTICAL**
  （终局现金 [8848.52, 8478.03]）；步时 mean 1.669s／p95 3.708s／max 23.745s
  （0.75s 线）FAIL。裁定 1a6e45f0：步时项只报数、不参闸（0.75s 是天梯单步时限，
  定额纠正包从不去天梯），触发条件前向 PASS ∧ 确定性 PASS 已过，**开搜**；包无需修。
- 工具：driver 88773086、classify_issue 9ee2f972、test_driver_logic 73bfef98、
  instrument_pack 66b20d4c——原件一字未动。两处不得不改的钉死常量（已在开工三件套披露）：
  instrument_pack 的 `EXPECTED_PACK_MD5` 与 classify_issue 的 `CKPT_MD5` 写死 gen3 值，
  各建一批次副本：`instrument_pack_b2.py`（528d4d97，diff 仅 2 行）、
  `classify_issue_b2.py`（3e22d87a，diff 仅 1 行：CKPT_MD5→gen4），逻辑零改动。

## 3. 保真与双座验证

- 保真先行（DGX，10:57）：2 局（`e4921a31`、`2d8b2e5d`）×299 位置×11 字段
  对生产转换行逐字节零差。
- 双座验证局 `e4921a31`：Studio 份与 DGX 份逐字段比对——299/299 行，
  除 `sec` 外 21 字段**零差**；候选 npz（24 行）与基行 npz 均位级同。
  采 Studio 份（第一批同款做法）。DGX 份终局分差 −1134.73、issued 24，与 Studio 逐字同。

## 4. 搜索执行

- Studio 三 worker 11:26 起跑、11:30 前全部落盘（w1 4 局／w2 4 局／w3 3 局）；
  11/11 局各 299 行、四件齐，本地 11:30–11:32 拉回。
- DGX 验证局 11:26–11:45:44（PID 3637713，争用下 19:44 墙钟；
  CPU 01:00:29；DGX Σsec 1,137.2 s、均值 3.8034 s/位置，远高于第一批的 1.1761，
  属争用读数）。
- 种子 999，阈值 adv_u>0.15，obs-carbon fog，包 5b2f5dcf（四座 run_manifest 均核合）。

## 5. 四数（终版）

| # | 数 | 值 |
|---|---|---|
| ① 分类 | 错输局 / 被压制局 | **11 / 0**（11 局全为错输局，无被压制局） |
| ② 纠正行数 | 错输局内超阈行 | **221 行**（占全部位置 6.72%；错输局内占比同 6.72%，因 11 局全为错输局） |
| ③ 被压制候选 | 隔离行 | **0 行**（无被压制局；`rows_correction_suppressed.npz` 为 0 行空件，同 11 键，md5 见 §7） |
| ④ 塌方步分布 | 全步 adv_u 分布（Echo11 无 ① 塌方清单，取全步分布代之） | 全 3,289 位置：mean 0.0428、median 0.0191、p75 0.0665、p90 0.1254、max 0.8864；超阈 221 行：mean 0.2514、median 0.2059、p90 0.4111、max 0.8864；超阈占比 0.0672 |

- 总位置 11×299＝**3,289**；一致率 **0.2402**（790/3,289）。
- 步时分计：Studio Σsec **593.1 s**、均值 **0.1803 s/位置**（与第一批 Studio 均值逐字同）；
  DGX 验证局 Σsec 1,137.2 s、均值 3.8034 s/位置（争用读数；验证局为 Studio 局的复本，
  其 sec 不计入生产总秒数——生产总秒数 593.1 s）。

逐局分类（N_over／S_over／max_adv，线 N≥10 ∧ S≥2.0）：

| 对局 | N | S | max_adv | 类 |
|---|---|---|---|---|
| `2d8b2e5d` | 52 | 16.3846 | 0.8156 | MISTAKE |
| `9d8ce33a` | 36 | 12.0242 | 0.8864 | MISTAKE |
| `e4921a31` | 24 | 5.3782 | 0.4968 | MISTAKE |
| `c4bf549f` | 19 | 3.7453 | 0.3971 | MISTAKE |
| `a0299775` | 16 | 3.4264 | 0.4235 | MISTAKE |
| `d94fc61c` | 14 | 2.8297 | 0.4099 | MISTAKE |
| `f3dc9e3c` | 13 | 2.7964 | 0.3077 | MISTAKE |
| `4cab29be` | 12 | 2.4096 | 0.2676 | MISTAKE |
| `d15d0dd9` | 12 | 2.3301 | 0.3251 | MISTAKE |
| `7791ec33` | 12 | 2.1589 | 0.2518 | MISTAKE |
| `cbb0000b` | 11 | 2.0749 | 0.2281 | MISTAKE |

最贴线者 `cbb0000b`（N=11、S=2.0749）与 `7791ec33`（N=12、S=2.1589），均在双线之上；
无一局贴近 N 线下沿（最小 N=11）。

## 6. 核验（执行人自验；主代理独立复算待办）

- 阈值四向复算：逐位置重算 (searched ∧ ¬agree ∧ adv_u>0.15)，与 logged issued 标志、
  summary.n_issued、候选行数、分类 N_over 四向全合（11 局／3,289 位置，0 不符）。
- 分类工具内断言：每局候选行数 == N_over（11/11 过）；合并时逐局
  `rows_correction.npz` 与候选件逐数组位级同。
- 深度断言：3,289 位置全合深度规则（step≤285→9、step≥286→299−step），0 违规；
  超阈 221 位置同 0 违规。
- 纠正行格式：合并 221 行逐点 tw/tb 和 ∈{0,1}；行和 tw 3–10（多单元动作，
  与第一批 2–10 同口径）、tb 恒 2.0；searched 全为 1。
- 合并精确性：合并 npz 与分类表顺序逐局拼接逐数组位级同（落盘后重读复核）。
- 处置：11 局全 (d) 纠正、未纠正步保持原权重；被压制局 0，故 (b) 路本批无局；
  被压制降权档（policy ×0.5、value 不动，029c3d2b）本批未触发。

## 7. 交付文件（本地 `~/workspace/lossprobe3/batch2/run/merged/`）

| 文件 | md5 / 内容 |
|---|---|
| `rows_correction.npz` | `08571aaefc1b5f4d80940d48ebe5cecb`，221 行，3,691,383 B，11 键（x/sc/tw/tb/wmask/bmask/carbon/margin/aux_delta/win/searched） |
| `rows_correction_games.json` | 221 行来源（逐局 start_row／n_rows，分类表顺序） |
| `rows_correction_suppressed.npz` | `db4eaa54cc41f87cf8d817fcf1e3bf39`，0 行空件，同 11 键 |
| `rows_correction_suppressed_games.json` | rows_total 0，games [] |
| `classification_table.json`／`.csv` | 11 行，全 MISTAKE，ckpt 42888eab／pack 5b2f5dcf |
| `disposition_table.json` | 线＋11 行处置（全 (d) 纠正） |
| `four_numbers.json` | §5 四数 |
| `manifest_check.json` | 四座 manifest 全过（阈值 0.15／pack 5b2f5dcf／种子 999／fog） |
| `depth_assertion.json` | 0 违规 |
| `collapse_adv_distribution.json` | 全步 adv_u 分布（§5④） |
| `run_manifest.json` | 合并 manifest（games 11，merged_from 四座） |
| `seat_manifests/` | 四座 run_manifest＋progress 原件 |
| `echo11/<11 局>/` | 逐局 positions.jsonl（299 行）＋summary＋基行／候选 npz＋纠正 npz |
| `../merge_batch2.py` | 合并＋核验脚本（可复跑） |

另：`run/run_studio_w1|w2|w3/` 为三 worker 拉回原件；DGX 验证局原件在
`/tmp/dgx_e4921a31_positions.jsonl`（＋候选／基行 npz 同名）。

## 8. 时序

- 包 verify 完成 11:25:30；搜索 11:26 开跑；Studio 11 局 11:30 前落盘；
  DGX 验证局 11:45:44 落盘；双座比对＋分类＋合并＋收据 11:48 完成。
- Gen-5 训练不等本批；本收据交付时刻 11:48（CST）——纠正行在开训前落仓即并入 Gen-5，
  否则进 Gen-6（以主代理落仓时刻为准）。

# RECEIPT — KataGo 新探针 Part 2 补趟：价值叶仿射校准（策略组 2026-11-15 12:42 裁定）

**状态：三步交付完成。仿射拟合（同一留出集）a ＝ 3687.9177926610573、b ＝ −638.5907028300794，叶注册映射候选＝仿射对（a, b）；根级质量屏重跑判读＝过——改子根 61 根、ΔU 负占比 0.4262 ≤ 0.5（判读线钉死不变）。** 如实出数：未调参、未重跑、未换集换口径。实现＝工程组，验收＝策略组；产物全为诊断件，未入 main、未开任何对局场、未用注册种子块。

补趟缘由（在册）：Part 2 注册的过原点形式 a·z 表达不了实测加性偏移（零交叉 z≈+0.28、带截距诊断斜率 3687.9／截距 −638.6），系注册函数形式错误；本趟按裁定改仿射形式 M_eq＝a·z＋b，同一留出集拟合、同口径重跑质量屏、判读线（负占比 ≤0.5 过）不变。

---

## §1 联合拟合（同一留出集，禁碰 rootset32）

- 数据：Part 2 留出集逐根 z＋margin 存档 `heldout_z.npz`（DGX part2 侧，md5 实测 `17f26aa926f9a686e34ab6fba3bc331e` 合在册值；本地镜像同值）。集合口径同 Part 2 回执 §2.1：ds_m3s2 文件名序 [32:64] 共 32 局、步 [200,299)、双席、6,336 根；z 为值头预激活（Part 2 fit_z.py 管道提取）、margin 为 ds meta 终局 scores 按根席位定向。
- 管道自检：同数组复算过原点拟合，a＝3347.515730530027，与 Part 2 注册值逐字合（残差分位／均值／标准差／MAE／RMSE 亦全合），方跑仿射拟合。
- 方法：OLS，margin ＝ a·z ＋ b（`numpy.linalg.lstsq`，float64）。脚本 `scripts/fit_affine.py`（md5 `a9daffde0293ceec844652c76ee9569d`），结果 `data/fit_affine_results.json`（md5 `bccbf8ed28dc77d0cafe25cd15dc83b2`）。

| 量 | 过原点（Part 2 注册，本趟同数组复算） | 仿射（本趟） |
|---|---|---|
| a | 3347.515730530027 | **3687.9177926610573** |
| b | —（钉 0） | **−638.5907028300794** |
| R² | 0.8270 | **0.9111** |
| 残差分位 5/25/50/75/95 | −1858.3 / −972.2 / −402.4 / −132.1 / +231.9 | −1204.5 / −305.7 / +101.4 / +395.6 / +848.1 |
| 残差均值／标准差 | −579.6 / 652.3 | −1.2e-12（≈0）/ 625.5 |
| MAE / RMSE | 638.9 / 872.6 | 468.1 / 625.5 |
| Pearson(z, margin) | 0.9545 | 0.9545 |

附注：仿射零交叉 z＝−b/a＝0.1732；a 对解析锚 3000 的偏差为 +22.93%——此数与 Part 2 的 10% 线不可直接比：注册形式已改，偏移由截距 b 承担，斜率的角色随之改变（Part 2 §2.3 的结构归因——零点偏移＋负侧压缩——正是本形式要吸收的）。拟合优度与残差系统性（中位 −402→+101、均值 −580→0）的改善即形式修正生效的实测。

## §2 仿射变体（叶注册映射候选）

- 变体：`part2b/modules/cnet2_search_vleaf.py` md5 **`7f283b15e64f1ee0337e832288eea419`**——探针②冻结件 `4a3b5f1b45fde9b659a383c1c3fba87f` 的模块拷贝（冻结件本身未动、md5 复验仍 4a3b5f1b），**仅**改价值叶映射：注释行改写为仿射式说明＋`return 3687.9177926610573 * z_ + (-638.5907028300794)`（a、b 全精度字面量）。diff 全表存 `part2b/data/variant_diff.txt`（md5 `bf7450c985873702ff32a0bc4b0a84f5`），两处 hunk 均在 leaf 块内（:688–695 注释、:701 返回行）；py_compile 通过。
- 其余四模块与冻结集逐字节同（feat 2a45b08b／np d8c15ad4／search 94f7fb64／fastsim aefe6c83，DGX 侧复验同值）。
- **叶注册映射候选＝仿射对（a, b）＝（3687.9177926610573, −638.5907028300794）**，拟合证据＝本回执 §1＋`fit_affine_results.json`（md5 `bccbf8ed28dc77d0cafe25cd15dc83b2`），待策略组验收入册。

## §3 根级质量屏重跑（同口径）

### §3.1 消融

- rootset32（md5 `c0871c50…`）、构造关同口径（gate_driver.py `2858ea72` 原样、三臂同根同种子、冻结 DEFAULTS），DGX 16 worker：searched 6,006/臂、errors 0；**changed(value@仿射 vs heur) = 61 根**；附带 changed(value@仿射 vs const) = 45、changed(const vs heur) = 18；wall 975.9 s；单根均时 value 0.7535 s／const 0.7243 s／heur 0.7377 s。
- 负载注记：本趟 DGX 发车（12:45 CST）load 44.1/54.6/60.2，消融收尾（13:04）降至 19.2/38.0/55.1，单根均时 0.7535 s（Part 2 趟为 1.0468 s／wall 1343.8 s）；底座预算是墙钟口径、改子集对机器负载敏感，且本趟叶映射与 Part 2 不同——61（本趟）与 100（Part 2 趟）的改子计数**不可直接比**，本验判读只立在本趟 61 根的续局上。

### §3.2 续局仪器（显著声明）

**两分支同仪器、对称使用**：改子分支与默认子分支的续局玩家同为 Echo3 部署包本体（md5 `7b7620499411a088cf4dc17c0e929bd7`，生产 CONF、SEARCH=False），双席皆然——包括根步的对手动作（在每个分支内由同一包在 pristine 根状态上算出）。构造与计分照 Part 2 cont_driver.py（`bdd7b44f`，原样复用）口径：FastCarbon.load_observation 重建根状态、tid +500；步限终局＝终局现金差（经记录局校验口径）；ΔU 经 `_u_of_margin` 换算。对称性实证：61 根两分支的根步对手动作逐根断言相等，**61/61 全等**（continuation_summary.json 内 opp_root_action_symmetric_all＝true、逐根 continuations.jsonl 同）。

终局形态：61 根 × 2 分支续局全部步限终局（limit 61／extinct 0）——F01 灭绝分支本趟亦未触发，其口径靠代码引证、未经实测局校验（与 Part 2 同）。

### §3.3 结果与判读

改子根 61 根的 ΔU 分布（格式同 Part 2 回执 §3.3）：

| 量 | 值 |
|---|---|
| 均值 | −0.01159 |
| 下四分位（Q1） | −0.00163 |
| 中位 | 0.00000 |
| 上四分位（Q3） | +0.00059 |
| **负占比** | **0.4262（26/61）** |
| 零占比 / 正占比 | 0.2459（15/61） / 0.3279（20/61） |

**判读：负占比 0.4262 ≤ 0.5——按钉死判读线，过。** 事实附注（不改判读）：分布主体贴零（中位恰 0、Q1 仅 −0.0016），但均值 −0.0116 为负——由尾部拉动：26 个负根合计 −0.980，其中单个最深负根 −0.5786 占负和 59%；≤−0.01 的负根 10 个、≤−0.1 仅 1 个。判读线只裁负占比，均值形态如实并报，供策略组验收 weighing。

### §3.4 限定（底座差异与口径）

- 本趟消融与变体坐在探针②的 DEFAULTS/teacher=1 墙钟底座上；Part 3 的对齐底座是 Echo3 CONF＋sims 时钟、TEACHER=2，两者不同——本验只校量纲形式与根级质量，不预判 Part 3 场效（与 Part 2 §3.4 同）。
- 拟合侧 z 的特征化口径与消融里叶的 z 路径之别，同 Part 2 §3.4 声明。
- 续局仪器自根步起以全新状态起算（Feat 历史自根起积累），双分支对称、已声明。

## §4 工件 md5 全表

| 工件 | md5 |
|---|---|
| 变体 part2b/modules/cnet2_search_vleaf.py | `7f283b15e64f1ee0337e832288eea419` |
| 冻结件 impl/modules/cnet2_search_vleaf.py（未动） | `4a3b5f1b45fde9b659a383c1c3fba87f` |
| Echo3 包（续局仪器，DGX 侧 echo3_pack.py＝本地 echo3_ref.py） | `7b7620499411a088cf4dc17c0e929bd7` |
| gate_driver.py（原样复用） | `2858ea720d9139d26f0feef9f0560c63` |
| cont_driver.py（原样复用） | `bdd7b44fa7b72a29940c792e32a293aa` |
| fit_affine.py | `a9daffde0293ceec844652c76ee9569d` |
| fit_affine_results.json | `bccbf8ed28dc77d0cafe25cd15dc83b2` |
| heldout_z.npz（留出集 z＋margin 逐根，Part 2 存档复用） | `17f26aa926f9a686e34ab6fba3bc331e` |
| variant_diff.txt | `bf7450c985873702ff32a0bc4b0a84f5` |
| out_gate_aff/summary.json | `518e7274c0aae39d95340aa5f826cca7` |
| out_gate_aff/results.jsonl | `d035b435aa15d9ad1c1b44df0fb64531` |
| out_cont/continuations.jsonl | `d046746505faf0b9d916b2340ca81d3f` |
| out_cont/continuation_summary.json | `63a8951600c9d4c9c086f3ffd6f86b4f` |
| 权重 cnet2_m3s2r1L.npz | `b6db08c9b076257c54817377bb93130d` |
| rootset32.json | `c0871c50d0cc1eea7e07f8aed6587838` |

工件位置：本地 `~/workspace/katago-probe2/part2b/`（modules/、scripts/、data/ 含拟合结果、消融 summary、续局全量 jsonl、variant_diff）；DGX `/home/arm/junyang/katago_probe_scratch/part2b/`（含 out_gate_aff／out_cont 全量与 run_gate_aff.sh／run_cont.sh）。

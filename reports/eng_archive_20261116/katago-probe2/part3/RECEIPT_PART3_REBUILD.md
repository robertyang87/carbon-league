# Part 3 A 臂改基重建回执 — 状态：COMPLETE（pack 空间重拟合＋双臂同值重建＋闸三复审 33/33；未开场、未跑对齐质量关、未消耗任何种子）

依据：策略组 2026-11-15 13:34 验收裁定 §3 处置 (iii)＋补充口径（MARGIN′ 为双臂共享机制参数）。基底＝A 包 `ea20a271`；留出集＝Part 2 同集 [32:64]、6,336 根（rootset32 未碰）。

**Headline 数字**：a′＝3106.795995527259、b′＝+567.9139352517781、σ_pack＝825.419098466563、**MARGIN′＝825.4**（≠625.5 → 双臂同改，B 臂仅 MARGIN 行）；新 A 臂包 md5 **`c6a48797cff10cbd41dc6c3ac5c7dc5c`**；新 B 臂包 md5 **`3d48742fb489caee2f41a7fc0db3d67c`**；已验收旧臂 `d8d7ebc8`／`fbe9beff` 原样未动。

## 1. pack-z 重提取（第 1 步）

- 计算路径＝部署叶自身路径，逐项同 test_leaf.py T2 口径：A 基底包内嵌 feat（`2a45b08b`）的 Feat 在根观测（gate_driver 拼装口径：step/players/trees 取席 0 条目、carbon 取本席条目）上 observe；包内嵌 fastsim（`61859708`）以席 0 完整观测 load_observation 建 sim；`feat_copy` 分支副本 observe `sim.obs(seat)`；包内嵌 np（`1433b5ac`，经包 search 模块 `encode_b`/`stack_x`）＋包解码 int8 权重（载荷 md5 `444cc54c`）的 `pol` 算 ctx；z_pack＝ctx·value.weightT[:,0]＋value.bias，教师 T＝2（新 agent 无 T_now，与部署叶 `st.get("T_now", T)` 一致）。
- 管道验证（先于全量）：边界用例步 297/298/299 双席共 6 点，本路径 z_pack 与构造回执 §3／`leaf_tests.json` 的 z_pack **逐位全等（|diff|＝0.0）**。
- 输入完整性：32 局回放自 DGX 拉回，md5 与 `heldout_manifest.json`（`de0ab6d8`）逐项全合；每根 margin 与 manifest 记分差交叉核对 **6,336/6,336 合**；配对键 (game_key, seat, step) 与 `heldout_z.npz`（`17f26aa9`）行序一致。
- 产物：`part3/data/heldout_pack_z.npz`（md5 `a4c426ff`）——逐根 z_pack 与 z_fp、margin 配对存盘。

## 2. Δz 全集分布（第 2 步）——「近恒定平移」不成立

Δz＝z_pack − z_fp，6,336 根：

| 量 | 值 |
|---|---|
| 均值 | −0.3559548260817593 |
| SD | 0.1684695515958609 |
| min / max | −1.0313024067377188 / +0.07592380219297579 |
| 分位 5 / 25 / 50 / 75 / 95 | −0.6436475518229359 / −0.45805828285419325 / −0.3459621569762939 / −0.24202738750546643 / −0.0947606951631478 |
| Pearson(Δz, z_fp) | 0.33992734637636773 |
| Pearson(Δz, z_pack) | 0.5686762928391894 |
| Pearson(z_pack, z_fp) | 0.9668878996141805 |
| Δz 对 z_fp 的 OLS | 斜率 0.10546319935886225、截距 −0.3742165722452194、残差 SD 0.15843747612066755、R² 0.11555060081447965 |
| 距均值 ±0.05 内根数占比 | 0.2528 |
| 距均值 ±0.10 内根数占比 | 0.4667 |

判读：平移只是分布的中心，不是逐根事实——SD 达 |均值| 的 47%，四分位距 [−0.458, −0.242]，5–95% 区间宽 0.549，另有弱 z 依赖（斜率 0.105、相关 0.34）。构造回执 §3 的三态样本（−0.3585/−0.3891/−0.4052）恰落在分布中心附近，与全集相容但不具代表性。结论：把 fp 仿射对平移 Δz 均值后部署的做法不可行，pack 空间重拟合是唯一正路——已按裁定执行（§3）。

## 3. 双空间拟合对照（第 3 步）

OLS margin＝a·z＋b（float64 lstsq，口径与 Part 2B fit_affine.py 逐项同）。自检：由配对数组复算 fp 侧过原点 a＝3347.515730530027（差 <1e-6）与 Part 2B 仿射对＋残差 SD（差 <1e-9），全合后才报 pack 数。

| 量 | fp 空间（Part 2B 登记） | pack 空间（本趟重拟合） |
|---|---|---|
| a | 3687.9177926610573 | **3106.795995527259** |
| b | −638.5907028300794 | **+567.9139352517781** |
| R²（centered＝uncentered） | 0.9110997759705017 | 0.8452112690984608 |
| 残差 SD | 625.5417075364155 | **825.419098466563** |
| 残差分位 5 / 25 / 50 / 75 / 95 | −1204.468 / −305.732 / +101.430 / +395.568 / +848.052 | −1338.987 / −576.821 / +20.701 / +536.336 / +1427.808 |
| MAE | 468.11642649667715 | 661.8664314820455 |
| Pearson(z, margin) | 0.9545154665957818 | 0.9193537236007271 |
| a 对解析锚 3000 偏离 | +22.93% | +3.56% |
| z 均值 / SD | 0.1731575210545294 / 0.543008442445717 | −0.1827973050272299 / 0.6208329324469208 |
| 零交叉 z（−b/a） | 0.17315752105452908 | −0.1827973050272316 |

读法（事实陈述）：部署管道的 z 整体负移且展宽，对 margin 的预测噪声大于 fp 管道（R² 0.845 对 0.911、σ 825 对 626）；在部署空间重拟合后斜率回到解析锚 3000 附近（+3.6%），截距翻正以补偿 z 的负移——测的叶与部署的叶自此为同一个叶。残差展宽对 Part 3 对齐质量关（pack 空间、A 基底）的影响由该关自身读数裁，本趟未跑该关。

## 4. MARGIN′（第 3 步尾）

- 规则（已验收）：MARGIN′＝1.0 × pack 空间残差 SD，0.1 取整；尾部归质量关负占比线，不上调至 √2σ。
- **σ_pack（未取整）＝825.419098466563 → MARGIN′＝825.4**。
- MARGIN′ ≠ 625.5 → 按补充口径触发双臂同改（§5）。

## 5. 重建与闸三复审（第 4 步，含补充口径）

- 构建：`build_variant_refit.py` 重放 build_variant 登记编辑集（终局修正＋叶接线自 BV 导入、build 逐行镜像），仅换字面量：A 臂叶块仿射对 → (a′, b′)、出处注释改指 `fit_pack_results.json`（md5 `36665219`）；双臂 DEFAULTS MARGIN → 825.4。产物为新文件 `echo3_part3_a_pack_v2.py`／`echo3_part3_b_pack_v2.py`，已验收旧臂文件不覆写（复审已验其 md5 仍为 `d8d7ebc8`／`fbe9beff`）。
- 复审：`audit_variant_refit.py` **33/33 全过**，要点：
  - 双臂骨架（_SRC 行外）／CONF 逐键／权重载荷／内嵌 feat·np·fastsim 与基底逐字节同；DEFAULTS 除 MARGIN（20.0→825.4、GUARD_CASH 40 保留）外逐键同；
  - 构造性复现：基底源＋改基登记编辑集逐字节复出双臂内嵌源；
  - 逐行分类 other＝0（A：MARGIN 2／leaf 21／terminal 35／leaf-wiring 6；B：MARGIN 2／leaf 2／terminal 35／leaf-wiring 6）；
  - **A′↔B′ 单变量复证**：A′ 内嵌源去掉改基叶块后与 B′ **逐字节相等**；双臂同载 MARGIN′＝825.4，无双变量形态；
  - **B′ 对已验收 B 的差异恰为 MARGIN 一行**（新旧两行、均 MARGIN 行），叶块逐字节未动；
  - A′ 对已验收 A 的 19 行差异全部落在叶块（新旧）与 MARGIN 行内。
- 叶功能核验（T2 口径闭环）：A′ 包实建 agent，边界步 297/298/299 的部署叶返回值与 a′·z_pack＋b′ **逐位全等（|diff|＝0.0）**（−1356.157070／−1356.829642／−1312.252314）。
- 新臂身份：A′ 包 `c6a48797cff10cbd41dc6c3ac5c7dc5c`（4,580,557 B；内嵌 search `a4ef042db11738fbd081ed47aee79db0`）；B′ 包 `3d48742fb489caee2f41a7fc0db3d67c`（4,579,223 B；内嵌 search `1a966c64ca00c2f19500fb8655acff9d`）。

## 6. 进度口径与检查点（所有者 2026-11-15 13:41 新规；本趟规前开工、按节补记）

| 阶段 | 工件位置 | 已完成量/总量读法 | 中断续跑点 |
|---|---|---|---|
| ① 回放拉取 | `part3/work/refit/replays/`（32 文件、7.5 MB） | 文件数/32；逐个 md5 对 manifest | 缺哪局补拉哪局（scp 单文件） |
| ② pack-z 提取 | `part3/work/refit/shards/<game>.npz`（32 片、每片 198 根）；日志 `extract_rem0.log`／`extract_rem1.log` | **shard 数/32**（＝局数进度）；根数＝shard 数×198/6,336 | 重跑 `extract_pack_z.py` 自动跳过根数合的 shard、只补缺片；双 worker 以 `--mod 2 --rem 0/1` 奇偶分片、无竞态 |
| ③ 组装 | `part3/data/heldout_pack_z.npz` | 文件在位＋md5 `a4c426ff` | 全 shard 在位时重跑组装（秒级、全恢复模式） |
| ④ 拟合 | `part3/data/fit_pack_results.json` | 文件在位；内含 fp 自检通过标记 | 直接重跑 `fit_pack.py`（秒级、确定性） |
| ⑤ 重建 | `part3/modules/echo3_part3_{a,b}_pack_v2.py`＋`work/refit/build_manifest_refit.json` | 清单在位 | 重跑 `build_variant_refit.py`（确定性、覆写 v2 自身） |
| ⑥ 复审 | `audit_variant_refit.py` 标准输出（33/33） | 断言计数 | 任意时点只读重跑 |

不可测缺口（如实注）：提取以局为分片粒度，局内 198 根无逐根外部计数——局内中断至多重算 1 局（实测约 2 分钟/局·双进程分摊口径）；其余阶段均为秒级确定性重跑，无黑箱长跑。本趟后续若再启长子跑，一律沿此分片口径。

## 7. 工件 md5 全表

| 工件 | md5 |
|---|---|
| scripts/extract_pack_z.py | 9cff7bbc080fb3c2bcc303da06323918 |
| scripts/fit_pack.py | 87404b6745c6822a868f4c6fea0dc1bb |
| scripts/build_variant_refit.py | adf39edae559168503e256765729038f |
| scripts/audit_variant_refit.py | ebcedda15f51b37b0e7dd1b852bb4ead |
| data/heldout_pack_z.npz | a4c426ff919cc17d2c8cf0d074cbe90f |
| data/fit_pack_results.json | 366652198441e292af64c01d77b84898 |
| modules/echo3_part3_a_pack_v2.py（新 A 臂） | c6a48797cff10cbd41dc6c3ac5c7dc5c |
| modules/echo3_part3_b_pack_v2.py（新 B 臂） | 3d48742fb489caee2f41a7fc0db3d67c |
| work/refit/A/cnet2_search.py（A′ 内嵌源） | a4ef042db11738fbd081ed47aee79db0 |
| work/refit/B/cnet2_search.py（B′ 内嵌源） | 1a966c64ca00c2f19500fb8655acff9d |
| work/refit/build_manifest_refit.json | c0486df78956444b82b64e443fd95a09 |
| modules/echo3_part3_a_pack.py（已验收 A，未动） | d8d7ebc82a564c9510e9aeae015ce4cd |
| modules/echo3_part3_b_pack.py（已验收 B，未动） | fbe9beffd1683ddd7995a1282aafc572 |
| modules/echo3_search_pack.py（基底 A，未动） | ea20a2714209e1a58b3e435b41d943bb |
| part2b/data/heldout_z.npz（输入，未动） | 17f26aa926f9a686e34ab6fba3bc331e |
| part2/data/heldout_manifest.json（输入，未动） | de0ab6d83c2f3b05f09a360aa51930dd |

声明：未开场、未跑 Part 3 对齐质量关、未消耗任何种子；Part 3 开场双闸（Part 1 判读非负格＋Part 3 构造验收）不在本趟范围。

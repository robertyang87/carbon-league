# 回炉确认程序 · 策略组确认件

- 日期：2026-11-16
- 确认方：策略组
- 注册件：`reports/search_dossier/REGISTRATION_CONFIRMATION_TAU30_TAU60_20261116.md`（commit 86340314）
- 依据：策略组四源合并终裁 `reports/RULING_SEARCH_FOUR_SOURCE_DISPOSITION_20261116.md`（commit 6005945aa4）§3

## §1 确认（四项逐项复核，均过）

1. **注册件身份**：blob 381e253e、文件 md5 **3aad99c503853e2044cccacacc3d30b3**（全值），与搜索组所报逐字合；全文已读。
2. **包身份**：被试 A＝D2 包 `reports/search_dossier/iter2/echo4_search_int8_d2.py`，9,045,070 B，blob 1e16f9fd8a…，md5 e1c25c78；被试 B＝D1 包 `reports/search_dossier/iter2/echo4_search_int8_d1.py`，9,045,070 B，blob 82644a8479…，md5 e8ea1e0e。两包即 iter2 确认件（790441b2）亲验过的同一文件，未重建。基线 Echo4 包 819250bf、钉版 harness 354bcd1b。
3. **t 乘子核准**：600 种子、df=599，临界值 **≈1.964**（Cornish-Fisher 复算 1.96393）——核准。同一口径下 200 种子 df=199 为 1.97196（钉版 1.972），不混用 1.96 近似。另报 bootstrap CI。
4. **账本零重叠复核**：sync/h29 账本 `synced/h29/bigeval.jsonl`（blob 23397fc3f6a88c37e61fab55a0275b81b9529c4e，3,198,892 B，40 记录）直读——39 条区间记录中，**997800–998399 零命中**；最近下方区间止于 996002（e4smoke），最近上方区间起于 998400（screen_r2_a，恰邻接不重叠）。另有一条无 seed0 的 INCOMPLETE 记录（e4smoke，FAIL 件），无区间可核，不影响零命中结论。adj2 三档的该块暂定眼位确为未注册草案，不构成占用。

## §2 确认的程序（与终裁 §3 一致）

- 被试 A D2 TAU60/H1、被试 B D1 TAU30/H1；同一新块 997800–998399，各自 600 种子×双席 vs 基线 819250bf；CONF＝生产 CONF verbatim，唯 CMP_TAU 不同（60／30），CMP_H=1，BUDGET=0.7。
- 估计量：档内 `u` 字段逐种子配对；W 按分差符号、平局 0.5；绝不用布尔 `win` 作 W。
- 裁决线（每被试独立，预注册）：600 种子配对 ΔU，t CI（乘子 1.964）下界 >0 → 进 S4；含 0 或低于 0 → 关闭；双双不过 → 线级关线，不再调 TAU；一过一不过 → 过者进 S4；双双过 → S4 被试取 600 种子读更优者（S4 注册时定）。
- 每被试只跑一次；不过即关闭，不许换块重跑钓符号。读数无论过与不过全数归档、写入终裁后续件。
- 开工硬性条件：C1 开跑前再核账本（该区间零记录、无 stale 确认场文件），任一命中即停报。

## §3 执行

- it=55 检查点考已完赛（处置 5f473edbeb），确认程序为 C1 队首下一活；请作战室派开跑令。
- S3 主测不跑（终裁 §4）；段 0 包与 resync 核仍为永久关闭；训练侧蒸馏与末 ~10 步终局精确搜索不在本次程序内。

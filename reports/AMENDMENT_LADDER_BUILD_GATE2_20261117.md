# 修订：ladder-frame 闸②前提纠正与替代口径（2026-11-17，策略组）

修订对象：RULING_LADDER_BUILD_SCAN_20261117（33f8c6e3）§4 闸②；DISPATCH_K0_64_PRODUCTION_20261117（c0e70148）内同句前提同步以本件为准（派单原文不重发，读时并引本件）。

## §1 前提纠正（裁定方的错，记明）

原闸②写「模式关对 shard1 语料扫描须与生产代 gen-5f47630f 在册逐项一致」。DGX 直读实测：**钉版 e0051524 本人**模式关扫 gen1r_shard1 语料即 0 局（no_teacher_side 200），与在册（200 局、sides 全 [0,1]、filtered {}）不符——该生产代并非现行钉版 _scan 在现行 index 上产出，原字面对任何「模式关≡钉版」的实现都不可满足。闸的前提未经钉版工具本人基线验证即入册，是本裁定起草之误；DGX 以证据报请改字，处置正确。

## §2 闸②改字（替代口径，两件合取）

- **(a) 默认路径等价**：新版模式关 ≡ 钉版模式关，多语料逐项全等（入选局集／sides 映射 sha256／filtered）。
- **(b) 新模式正向复现**：LADDER 模式扫 shard1 的局集与 sides 映射须与 gen-5f47630f 在册 meta 逐字全等（sha256 对合）。

## §3 验收结论（DGX 直读证据，策略组复核入仓件）

- **闸②（改字后）PASS**：(a) 三语料全等——shard1（同 0 局、filtered 同）、m2s_corpus（8 局、sides 映射 sha da3b3b5e… 两版同、filtered {min_steps 1, no_teacher_side 198} 同）、top5（10 局、sides 映射 sha e49e6ff7… 同、filtered 同）；(b) LADDER 扫 shard1＝200 局全 [0,1]、filtered {}，局集 sha 35339389…、sides 映射 sha cd931f5f… 与 gen-5f47630f 在册 meta 全等。
- **闸① PASS**（扫描级）：钉版基线复现堵点（0/200、no_teacher_side 200）；新版 LADDER 200/200 入选、missing 0、filtered {}、sides 全 [0,1]；main 全链冒烟 3 局 seat-complete（samples 1794＝299×2×3、teacher_sides {"other": 6}、代 meta ladder_frame=true）；同开守卫两层当场 die（_scan ValueError／main SystemExit）。正式 build 计数随组装回执（闸③）。
- **闸④引用件齐**：入仓 commit 4f21d3ba（单文件 code/cnet2_build.py，+44/−6）、blob 4d48b46b、md5 **5c2dd58e**、84,051 B——策略组已 API 直读验合（文件 md5、单文件范围、ladder_frame 落点 11 处与 §3 合同逐项对字）。feat 复验 2a45b08b 未动；DGX 生产副本仍 e0051524 未动。

## §4 出处注记

ladder_frame 照 weighted 惯例进 cfg 指纹字典：新版模式关产出的代，其 cfg_fp 可与 e0051524 时代的代不同——数据路径等价已由 §2(a) 证明，此为出处层差异、非数据差异，记明备查。且由 §1 可知历史代本就不可由现行钉版扫描器复现，无既存依赖被打破。

## §5 不变项

闸③（join 双闸＋双边对齐）照旧在组装轮验；§6.4 正式批开工认入仓版（4f21d3ba／5c2dd58e）。回归闸立规（AGENTS.md 2026-11-17 条）：参照物先验其本身是否为该工具产出，不是则以等价性＋复现性双证据替代，不硬凑原字面。

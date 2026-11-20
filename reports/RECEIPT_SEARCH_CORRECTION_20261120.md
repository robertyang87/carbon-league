# 输局旁路 ③ 全量重搜 — 定版回执（第二段）

执行人：工程组第二段执行人。钉版依据：`reports/DESIGN_SEARCH_CORRECTION_20261120.md`
（commit 80566708，md5 bf1e869d）＋驱动四件（instrument_pack 66b20d4c／search_driver
88773086／classify_issue 9ee2f972／test_driver_logic 73bfef98）。深审
`reports/REVIEW_SEARCH_CORRECTION_20261120.md`（0b506765）判无作废级问题，本跑有效。

## 1. 口径与钉死项核对（全过）

- 范围：58 局（Echo9 36＋Echo10 22），逐局全步 299 位置，共 **17,342 位置**。
- 检查点 gen3 md5 `708f8cca57f5674d00d1e5e56758e14c`；仪表化包
  `gen3_fixed_u_lp3.py` md5 `9f915f00`（两座开跑前逐件读回核对；Studio 侧驱动
  search_driver.py 开跑前测得旧版 82cebdd4，已先同步至钉版 88773086 再跑；
  设计件与驱动其余各件两座全合）。
- 阈值 adv_u＞0.15、分类线 N_over≥10 ∧ S_over≥2.0、K=12/TTA=4、seed 999、
  obs-carbon fog——逐项按钉版执行，无一改动。
- **阈值逐局比对（深审第 3 点，§11）**：仓内 6 份座上 run_manifest 全部读数
  threshold_u＝0.15／pack_md5＝9f915f00／seed 999／fog（manifest_check.json）；
  另对全部 58 局逐位置复算（searched ∧ ¬agree ∧ adv_u＞0.15），与日志 issued
  标记、summary.n_issued、候选行数四方全合（17,342/17,342）——每局有效阈值
  均为 0.15，无一局触发 §11 作废。注：run_studio_w1/w2/w3 无 run_manifest
  （manifest 为跑末产物，三会话在第 14 局后被本机服务重启杀死，见 §3），
  其逐局证据以 progress.json＋上述逐位置复算为准；fix1/fix2 的 manifest
  games＝1 是逐局调用覆写所致。
- **depth 机械断言（深审 F2）**：全部 17,342 位置断言 depth＝计划深度
  （step≤285 为 9；末 13 步 step≥286 为剩余步数 299−step）——**0 违例，PASS**
  （发行位置子集同 PASS；depth_assertion.json）。

## 2. 结果

- **分类：错输局 47 局、被压制局 11 局**（分类表 classification_table_headed.*
  表头明记线值：adv_u 阈值 0.15、N_over≥10、S_over≥2.0——深审 F1 已落实；
  原始表 classification_table.* 为 classify_issue 原样输出，一并保留）。
- **纠正行：963 行**（rows_correction.npz，仅错输局 47 局）；分类前候选共
  1,035 行，被压制局 72 行候选隔离于 rows_correction_suppressed.npz（11 局）。
- **四数（分座）**：
  - Studio（50 局，14,950 位置）：一致率 0.2308；单位置成本 0.1803 s
    （Σsec 2,696.0 s）；纠正行以全局计（见下）。
  - DGX（8 局，2,392 位置）：一致率 0.2308；单位置成本 1.1761 s
    （Σsec 2,813.3 s；DGX 全程与他线作业争用，单价为争用下读数）。
  - 合计：总位置 17,342；总成本 Σsec 5,509.3 s；一致率 0.2308
    （与设计件试点口径一致：试点不一致率 78.9%）。
- **替换占比**：全步口径 963/17,342＝**5.55%**；错输局内口径
  963/14,053＝**6.85%**。
- **塌方步优势分布（方案 (d) 存活检验，51 局切片）**：① 塌方步处 adv_u
  分布——n＝51，均值 0.0435，中位 0.0255，p75 0.0631，p90 0.1409，
  max 0.3098；**超 0.15 阈者仅 4/51（7.84%）**。即在 ① 认定的塌方步上，
  同局面搜索绝大多数时候并不认为实走着法差出阈值——与 ① 的归因
  （塌方以降以对手得分与局势重估为主、己方着法仅 15.7%）方向一致。
  逐局明细见 collapse_adv_distribution.json。
- 逐局处置表 disposition_table.{csv,json}：§6.2 全字段无空项；被压制局
  降权档填「policy ×0.5（提议档、待报批钉定）」；错输局未纠错步处置＝
  原权重（设计件 §6.3）。

## 3. 执行记录

- 切分（assignment.json 为凭据）：Studio 50 局／DGX 8 局。依据实测单价
  （开跑前同局同口径验证跑：Studio 0.141 s/位置、DGX 1.354 s/位置争用下），
  按局切、局内不拆。验证局 0c567ea37 两座各跑一份，**全 299 位置逐字段
  逐位一致（Δ＝0，sec 除外）**，定版采用 Studio 份。
- 时间线（CST）：2026-11-20 09:36:47 DGX 两片 nohup 开跑；09:37:20 Studio
  三 worker 开跑；DGX 10:02 前后 8/8 落地收尾。
- **事故与恢复**：09:49 前后本机服务重启，Studio 三条会话被 SIGTERM：
  杀死时点已落 36 局安然无损（逐局落盘）；远端进程其后又各落 2 局
  （至 14/17、14/16、14/16）才退出；剩余 7 局以逐局独立会话补跑
  （run_studio_fix1/fix2），09:57 前全齐。DGX nohup 不受影响。
  Studio 定版 50 局＝w1/w2/w3 各 14＋fix 7＋验证局 1。
- 产物三处在册：本目录（~/workspace/lossprobe3/run/）、DGX scratch
  （lossprobe3_scratch/work/run_dgx_{a,b}）、Studio scratch
  （lossprobe3_scratch/work/run_studio_*）；本地另存 run_pull/ 原始拉回件。

## 4. 记账注记（钉版文档不改，回执纠正）

1. 设计件 §10 切分行「Studio 36／DGX 17」为 53 局草稿数；实际切分以
   assignment.json 逐局分配表为准（Studio 50／DGX 8，58 局口径）。
2. 设计件 §9「新补 2 局」实为 **7 局**无 ① 塌方条目（Echo9：2e38577a、
   cb791c39、cb9cf4c4；Echo10：234319e1、6ebdfc68、8cfc3f8f、e0be1986）；
   其塌方步分布以全步 adv_u 分布代，逐局明细在
   collapse_adv_distribution.json 的 games_without_collapse_entry。
3. 设计件 §6.1「全 53 局」同为旧数残留，现行范围为 58 局。

## 5. 消费隔离注记（深审第 5 点）

被压制局候选行靠文件名隔离：**消费方只认 rows_correction.npz＋分类表**；
rows_correction_suppressed.npz 是隔离件，**不得被任何下游当纠正行消费**。
被压制局的 (b) 降权走行权重通道（② 线），不在本包内。

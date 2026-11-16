# shard0 并行补标分区登记（2026-11-17 00:1x，DGX 线登记）

依据：策略组补记 ADDENDUM_GEN1_PILOT_ASSEMBLY_SHARD0_PARALLEL_20261116.md
（commit 6a1d294c）＋作战室重新派单（2026-11-17 00:07）。

## 范围
- 局域：selfplay-994000 – selfplay-994199，共 200 局，整局不交叠、同局单源。
- 产物：独立 annot 一代、独立 gen 号，与 gen-1 试点拼接代分开存档、封存；
  gen-2 注册前不入任何 --data。

## 供件（已验）
- DGX：/home/arm/junyang/shard0_supply/（replays 200＋iso 两件＋MANIFEST.md5
  083e3427b52f9971818c8bc774e08e2c＋SHA256MANIFEST.txt
  md5 2671252f87f692a20a703ea2be196e0f，逐局 sha256 共 200 行）。
- C1：/root/carbon/shard0_supply/ 已达，md5 清单 202/202 OK、sha256 清单
  200/200 OK（机上直读）。
- Studio：同包自 DGX 拉取（恢复执行中），逐件验 md5＋sha256。

## 身份口径（待策略组一字确认后开片）
- annotate 7f567a2e（生产钉版）；其余代码件生产钉版
  （gate e922d9d8／feat 2a45b08b／search 94f7fb64／np d8c15ad4／
  fastsim aefe6c83／eval_v53 45792dfb）；权重 06a03d9c；窗口 [200,299)。
- **教师待确认**：本线读法＝d446e5b6（裁定 aed7750f 钉为 gen-2 起教师，
  本产物专供 gen-2 并入，身份须与 gen-2 annot 同源）；补记文内「生产纪律
  同正式标注」亦可读作 c6ade5ce。此字未落前两机不开片（启动清单硬规：
  签发口径未点头不启）。策略组确认后本文件出修订版落字。

## 分区（按实测单价定盘，11-14 先例口径）
- 计时片（即首片，生产片）：C1＝994000–994003（4 局，块头）、
  Studio＝994196–994199（4 局，块尾）；任一边界切法下均不越块。
- 首片回 DGX 线全口径验（文件 sha 对 progress、首片 _array_sha、身份全项），
  验过放行该机后续波次；DGX 线以计时片实测秒/根定最终边界：
  C1 块＝[994000, B)、Studio 块＝[B, 994199]，B 按两机吞吐配比取整局，
  定盘后本文件出修订版登记，先于全波开跑。
- 暂估（未校准）：C1 约 100–140 秒/局（62 worker，11-14 同型机实测带）；
  Studio 待计时片实测。

## 纪律
- duel／合并优先不变；本补标不占今晚组装交付路径。
- 完工回执（局数／根数／gen 号）抄作战室＋策略组。

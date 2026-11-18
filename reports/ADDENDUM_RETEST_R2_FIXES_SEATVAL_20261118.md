# 补记：R2 复测冻结集座位验证（spot-72）与 R2-06 离线链路径修正

日期：2026-11-18 22:27 CST。关联：`reports/RETEST_R2_FIXES_20261118.md`（commit 239494dd，计数更正 7b1152e2，R2-09/10 补述 770b0021）。

## 1. 座位验证经过

S3 开测门第 2 项（verify3 力学座位验证）按作战室令在 spot-72（c8ae，carbon-shanghai-spot-72，100.100.94.84）执行，验证包 `verify3_seatval.tar.gz`（冻结 17 件）。

- **首跑 FAIL（22:15–22:16，墙钟约 0 秒）**：冻结 md5 核验 17/17 全合；`tests_s1_fix_r2.py` 9/10（G1–G9 全 PASS，G10 FAIL）；`tests_r2_fix.py` 在 import 时 `fetch_pack()` 即崩，11 项 0 跑。失败已当轮单独报作战室（未与后报捆绑）。
- **重跑 PASS（22:27:11–22:27:15 CST，墙钟 4 秒）**：冻结 17/17 全合；`tests_s1_fix_r2.py` 10/10；`tests_r2_fix.py` 11/11；overall PASS。`seatval_result.json` 已自座上拉回读过（seat carbon-shanghai-spot-72，started 1795012031，ended 1795012035）。

## 2. 首跑根因：R2-06 离线 checkout 候选路径差一级（冻结代码潜伏缺陷）

`code/search_proto/smoke_real_s1.py` 的 `CHECKOUT_PACK_CANDIDATES` 写的是脚本目录上两级
`../../packs/echo4_int8.py`，归一化后为 `<checkout>/packs/echo4_int8.py`；但钉版包实际在
`code/packs/echo4_int8.py`（`PACK_REPO_PATH` 自证，仓内无根级 `packs/`）。自 `code/search_proto/`
起正确相对路径是 `../packs`。

本地复测 21/21 能过是被掩盖了：本机 `/tmp/echo4_int8.py` 缓存（md5 819250bf，已验）在，`fetch_pack`
根本没走到 checkout 候选；且本机 `sys.path` 有 `gh_api` 可走在线通道。72 座上三者皆无（无缓存、
无 gh_api、候选路径不存在），按设计 fail-closed。座位本身无问题；chunk04 分片全程未动。

## 3. 修正

- 代码：`smoke_real_s1.py` 一行 `../../packs` → `../packs`（注释同步改），提交
  **97b52b8d93d173372a4ae9080ccba2073a11433c**；主干读回 md5
  `0fbc6016751369980bbd8aea6dde9281`（原 `2425f34066c1b2b56208bdd4a2b7ac10`），其余 16 件冻结值不变。
- 验证修正本身：本地缓存移开后重跑两套 **21/21**（确证走通修正后的 checkout 路径，非缓存/在线
  掩盖）；重打验证包 `59b59c61150d63b4bab1eee6f5ff7a2d`（3,579,829 B，干跑 17/17、无 `__pycache__`）
  上座重跑通过（§1）。
- 教训：凡声称「离线链可跑通」的回归，必须在**无缓存、无在线通道**的干净环境里跑一遍才算证过；
  缓存/在线命中会把候选路径错误完全掩盖。G10 自身用覆盖候选路径的方式测 fallback，故测不到默认
  候选写错——默认候选须另有一条不覆盖的直通验证。

## 4. 对复测报告的更正

- 复测报告 §1 冻结表：`smoke_real_s1.py` 一行更新为 `0fbc6016…`（其余不变）；正文凡称
  「checkout 在场时完整离线链可跑通」处，改为「checkout fallback 默认候选路径曾差一级，
  2026-11-18 由座位验证暴露并修正（97b52b8d），修正后离线链在无缓存、无 gh_api 的座上跑通」。
- 结论不变：六项修复复测全过；座位验证 PASS 后，S3 开测门第 2 项完成。正式 S1 battery2 全量重跑
  （§5 第二半）仍待作战室排正式座位，与本项无关。

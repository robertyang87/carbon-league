# Gemini ApexCarbon-V2 — 归档记录（2026-11-19）

## 来源

- 所有者于 2026-11-19 16:03 CST 直接上传代码件，说明为「Gemini 做的优化第二版」（第一版见 reports/GEMINI_APEX_V4TTA_20261119.md，已提交天梯为 claude-opus55-cloud-Echo7，策略 id 9b76a7a5319f2b3970776c57bae58cde）。

## 入仓文件

- `code/apexcarbon_v2.py` — 与上传件逐字节一致，未改一行
  - 大小：2,904,798 B（1,876 行）
  - md5：`129fb44af9b30811f8c6451767ed9eec`
  - sha256：`2f5931c68103234a8b7e4b4a1aed6a7d07969d7df4ed772a14c8bb0e75f2f141`
  - 入口：`agent(observation, configuration)`

## 与第一版的差异（静态比对）

- 内嵌权重两行与第一版逐字相同（`_WEIGHTS` 2,797,861 字符、`_VALUE` 32,327 字符）——网络权重未换，改的是搜索/决策层代码。
- 代码行数（去权重行）：1,588 → 1,875 行，相似度 0.79。
- 文件头自述新增：Physics-Exact Economic & Tactical Logit Refiner（树 NPV ROI 闸、抢树优先、敌树税收规避、机会击杀、重载存碳助推、招募配比平衡）、Exact 1-Ply Physics Pre-Evaluation、Neural Critic + Exact Economic Residual Fusion。

## 我方核验（本 VM）

- AST 语法解析：PASS
- 导入冒烟：PASS（2.14 秒，agent 可调用）
- 静态扫描：无网络 / 文件 / 子进程调用（`eval(` 命中 2 处均为自有函数名 `_econ_eval`，非内置 eval）

## 考试

- 所有者令：考官用 Master 3s2。试点结果见本目录后续记录（Studio 钉版考场，考官件 opp_master3s2.py md5 b4cde262695d38270e16bde33875a443）。

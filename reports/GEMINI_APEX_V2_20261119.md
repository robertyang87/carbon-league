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

## 考试（2026-11-19，Studio 钉版考场，考官 Master 3s2）

- 考官件：opp_master3s2.py（md5 b4cde262695d38270e16bde33875a443，依赖同目录 master3s2.py md5 57f9a594cce4f2ec9e3573d8d9d26ca1）；harness bigeval.py md5 354bcd1bc2dc8bc188c813f53d28c893；4 worker 并行，种子 975100–975103（与第一版 Echo6 试点同种子段）。
- **第二版（tag gemini_apexv2_m3s2_pilot4r2）：1 胜 3 负，平均分差 −1,007（SE 574，最差 −2,540），err=0、FAIL=false，单步最差 347 ms。** 逐局：−1,148／+133／−2,540／−475。台账：存入 7,562 vs 考官 8,748、树收入 3,512 vs 3,530、死亡丢碳 276 vs 204、种树 28 vs 31。
- **第一版对照（tag gemini_apexv1_m3s2_pilot4，同考官同种子）：2 胜 2 负，平均分差 +146（SE 549，最差 −783），err=0、FAIL=false，单步最差 578 ms。** 台账：存入 8,165 vs 8,572、树收入 4,071 vs 3,250。
- **配对差（v2−v1，逐种子）：−1,729／+916／−1,829／−1,973，均值 −1,154，SE 692。** 4 个种子 3 个退步、聚合台账两项（存入 −603、树收入 −559）同步变差，方向一致指向退步；但 n=4 未达 2SE 定罪线，定论需扩样配对。
- 与 Gemini 自述对照：其自称第二版对 3s2「4 胜 0 负、平均 +688」在我方考场复现失败（实测 1 胜 3 负、−1,007）；其自称第一版 +456 与我方实测 +146 同量级（噪声内）。Gemini 自测口径（种子 401–402、0.42s 预算）与正式考场不一致，自测数字不可作判读依据。
- 逐局台账在本机 ~/workspace/gemini-apex-v2-20261119/exam_pilot4_m3s2/。
- 状态：仅归档＋试点，**未提交天梯**；是否扩样定论或回退，等所有者裁定。

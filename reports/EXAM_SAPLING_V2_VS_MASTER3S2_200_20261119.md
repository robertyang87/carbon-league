# Sapling V2（120 万参数·前五互打）整场外考 200 局（对 Master3s2）

日期：2026-11-19 09:19–09:40 CST · 机器：Studio（Mac Studio M2 Ultra，独立目录 ~/sapling_exam_studio/v2exam200；spot-72 已下线）

## 考生与身份

考生包取自 `robertyang87/carbon-league` 提交 `c057e30e` 的 `data/weights/sapling_1200k/top5_h2h_v2_agent.py`（1,674,060 B，MD5 `8299bbaf060c95febd6ac9382a3b99bf`，SHA-256 `dc20c132…ce29`），配置 teacher=2、shield/nocollide/endgame/pshield/rguard 全关，与 V1 两场外考同口径。Studio 侧（作战室）09:18 机上直读复核：十个分片考生包 MD5 全同 8299bbaf，对手 wrapper `b4cde262`、harness `354bcd1b` 与钉版逐字合。

## 考场

注册口径同前：考官 Master3s2（包 `57f9a594cce4f2ec9e3573d8d9d26ca1`＋wrapper，_TEMP=1.0、_SEED=None）；钉版 harness bigeval `354bcd1b…`；钉版 pkg 代码整套（七件 MD5 全合，PYTHONPATH=pkgcode_full；Studio 常驻 eval_v53 为 aa4c5bb6，未使用）。固定集前 200 局：种子 20291101–20291300，10 分片×20 局，EVO_WORKERS=2。err=0、双边内部错误 0，考生单步 5.88 ms。每局墙钟约 122 秒（考官搜索成本随场面密度上升），200 局约 21 分钟。

## 结果（同 200 种子对比）

| 候选 | 平均分差 | 95% CI | U（描述） | 胜/平/负 | 本方得分≈ | 存入/局 | 考官得分≈ |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 initial（72） | −4,815 | ±436 | 0.0408 | 0/0/200 | 0 | 0.0 | 4,933 |
| V1 sqrt（72） | −9,154 | ±331 | 0.0041 | 0/0/200 | +25 | 68.7 | 9,385 |
| V1 sqrt（Studio） | −9,679 | ±346 | 0.0058 | 0/0/200 | +20 | 66.5 | 9,900 |
| **V2（Studio）** | **−9,633** | **±273** | 0.0014 | 0/0/200 | **+2,181** | **2,327.3** | **11,781** |

V2 最好单局 −323，最差 −15,011；存入覆盖 200/200 局。

## 判读

- V2 已是真产出型策略：存入 2,327/局（sqrt 的 34 倍）、本方终局得分 ≈+2,181（三代轨迹：initial 0 → sqrt 约 +20 → V2 +2,181），与研究侧 2 局真身实战（席 0 交付 1,454）同量级互证。
- 但主尺分差与 sqrt 的 Studio 读数统计上无区别（−9,633 vs −9,679，差 46 « CI）：考官在 V2 的高产局里同步把得分抬到 ≈11,781（存入 9,249/局）。第二轮外考的模式在更高水平重演——总经济膨胀，考官捕获增量的效率更高。
- 因此单看 margin 会系统性低估 V2 的进步；后续报告应固定同时报本方得分与考官得分。本轮未跑满 1,000 局（所有者令先跑 200），余 800 局种子未跑。

## 证据

逐局台账在本机 `~/workspace/sapling-exam/v2/results200/`（10 分片 bigeval.jsonl＋bigeval_live）；Studio 上 ~/sapling_exam_studio/v2exam200 留全量工作目录。被停的 1,000 局版（v2exam）跑了约 70 局后按所有者令终止，其分片不作正式读数。

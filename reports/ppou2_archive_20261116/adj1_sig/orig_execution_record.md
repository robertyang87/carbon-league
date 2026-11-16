# Echo2 / Master2 真身补测 · 执行记录（adj2 设计输入）

目的：adj1 Tier-1 关闭后的补充测量——用 adj1 同款固定集口径量 Echo2 与 Master2 两个真身的完整签名（产能／转化／对抗三轴），作 adj2 注册的转化定型输入。非门控、不抢主臂。

## 执行

- 驱动：L20 `/root/carbon/run_sig_orig.sh`（md5 7c59479f8f4b2a270e9092c66586116a），2026-11-16 14:25:34 起跑，15:31:25 驱动退出。
- 测量器：`/root/carbon/signature_measure.py`（md5 5c4ea68e5e41deef3b3ec79823cf88ff，与 adj1 同版）。
- 聚合：`/root/carbon/agg_orig.py`（md5 a7deb76f4c4052fa423f071ca9c3514e，agg_adj1.py 仅换 glob 为 orig_*.jsonl）＋ `agg_orig_output.txt`。
- 环境：nice 10、线程钉 1、CPU；与主臂（至 15:21 停机）及仪表包络腿并行。优先级纪律：主臂 pace 受扰先停本测量——未触发。
- 冒烟（约 14:24，每体 4 局通过）：Echo2 seed 20291101 treesteps 1290、opp_dead 23；Master2 treesteps 552、opp_dead 13。

## 包身份（2026-11-16 15:52 机上直读）

| 角色 | 包 | md5（全值） |
|---|---|---|
| 被测 Echo2 | /root/carbon/probe_m2/code/exp/echo2.py | 15b6d754e6e21de33870074d3983a16f |
| 被测 Master2 | /root/carbon/probe_m2/code/exp/opp_master2r.py | 2aebfbc1008d3475ece8e178696e292d |
| 对手 Echo4 | /root/carbon/probe_m2/code/exp/echo4_int8.py | 819250bfc82632786063dabd46a616bd |

Echo2 权重 cnet2_m3w5L.npz md5 82fde46b2633950d3cbb29e8cacbfdc2（身份预建记录）；Master2 本体为考场钉版（2aebfbc1）。
行内无包身份字段（与 adj1 仪器同限），身份以本记录＋上表 md5 为准。

## 固定集

- 种子：1,000 个，20291101–20292100，逐种子一局；席位＝seed%2；对手 Echo4；eval_v53 台账。
- 落盘：10 片 ×200 行＝2,000 行（orig_echo2_s0–s4 14:30–14:58；orig_master2_s0–s4 15:05–15:31），无 .tmp 残留；每片 200 行、种子不重、err 0、终局全 299 步（策略组独立验合一致）。

## 聚合终值（n=1,000 各，per-game-ratio 口径）

| 本体 | treesteps | cash/ts PGR（se，95%CI） | opp_deaths | opp_cash40 | recruits | 对 E4 胜率 |
|---|---|---|---|---|---|---|
| Echo2 | 1157.9 | 7.264（0.053，[7.161, 7.368]） | 20.168 | 78.6 | 20.96 | 0.187 |
| Master2 | 603.9 | 17.380（0.308，[16.777, 17.984]） | 14.151 | 110.1 | 17.71 | 0.030 |

## 读数修正（策略组 2026-11-16 复审，我线认账）

- **Echo2 转化过线未确立**：点估计 7.264 在转化线 7.235 之上，但 95%CI [7.161, 7.368] 覆盖该线——我线此前「CONV ✓／过线」的表述改为「点过线、CI 未确立」。adj2 转化定型请按 CI 口径。
- **Master2 的 17.380 是小分母伪影**：其 treesteps 仅 603.9，分母极小把比值抬高，不作转化证据。我线此前「大幅过线」的表述收回。Master2 的有效读数是现金剖面（opp_cash40 110.1 全场最高）与低种树。
- 产能轴：两真身均未过线（1593.0），与 adj1 结论一致。

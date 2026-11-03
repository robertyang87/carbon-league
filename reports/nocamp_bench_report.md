# no-camp 基准核对报告（nocamp_bench，2026-11-04）

目的：核对 OpenAI 侧报告的两个数——「no-camp 对 base +3,278（24 局）」与
「NN 对 no-camp 8 局 6 胜、场均 +409」——与我方存档冠军 aux_seg2_zero 对
base 合并探针 −9,615.8 是否自洽。

## 对手本体

站点原件 `claude-opus-v53-nocamp`（id aa758d5a42687ff7aff528cf694095c9，
作者 cjc_de_ds）未能取回：本地 census 快照标 `is_public: false`，浏览器
抓取被策略拦截且不许换路重试。按工单走本地重构 `nocamp_recon.py`：
base_v53.py（与 reference_claude_opus_v53.py 逐字节一致，md5 1779b453…）
改 3 个参数——soft_w 1.0→0.0、campers 4→1、camp_from 6→12（原值已在源码
274/282 行核实）。diff 存 `nocamp_recon.diff`，来源说明
`nocamp_recon_provenance.md`。

## 协议

与存档探针一字不改：random.seed(7919*seed+104729*side+31)、policy=被测方、
seed 1000–1003 × 双边 = 每组 8 局、FastCarbon、caseb_rl_probe 同款账本、
全程 traces 留痕并登记 traces/INDEX.md「no-camp 基准核对」节（24 局）。
每组首局复跑核对：3/3 终分逐分 MATCH（+0.00）。脚本 `nocamp_bench.py`，
逐局数据 `nocamp_bench_results.json`。

## 结果（被测方口径，均值）

| 组 | 胜负 | 场均分差 ± 样本std | 存款 | 树收入 | 死亡丢碳 | 终局滞留 |
|---|---|---|---|---|---|---|
| nocamp_recon vs base_v53 | 8 胜 0 负 | **+3,351.7 ± 1,683.5** | 9,899.7 | 3,183.3 | 627.4 | 245.4 |
| aux_seg2_zero vs nocamp_recon | 0 胜 8 负 | **−14,414.6 ± 1,543.4** | 2,898.5 | 311.7 | 720.2 | 55.0 |
| fork_v0_1 vs nocamp_recon | 0 胜 8 负 | **−3,024.3 ± 1,890.5** | 7,208.3 | 2,986.1 | 264.2 | 274.6 |

逐局分差：nocamp vs base 全正（+1,313 ~ +5,293）；aux vs nocamp 全负
（−12,108 ~ −16,433）；fork vs nocamp 全负（−298 ~ −5,338）。

## 结论

1. **不自洽的解释在 NN 本体，不在 no-camp 本体、也不在协议。**
   校验组用我方重构＋我方播种协议复现了 OpenAI 的 no-camp 强度：
   +3,351.7 对其报告 +3,278（24 局），方向量级一致——no-camp 本体与
   协议两侧等价，这两项被排除。但存档冠军 aux_seg2_zero 对同一
   no-camp 是 0 胜 8 负、场均 −14,414.6，与 +409 相差约 14.8k，
   约为本组样本 std（1,543）的 9.6 倍，样本噪声不可能弥合。故
   OpenAI 侧那句「我们的 NN」不可能是 aux_seg2_zero：要么是其
   新建/续训的另一份 NN（checkpoint 本体差），要么用的是另一份
   同名策略文件。附带：其 ±285 也不是 8 局分差的样本 std（我方三组
   std 均在 1.5k–1.9k），更像 SEM 或别的口径，不影响结论。
   传递性旁证自洽：aux 对 base −9,615.8、nocamp 对 base +3,351.7，
   两者预示 aux 对 nocamp 约 −13k 量级，与实测 −14,414.6 同带。

2. **我方口径下的真实档位。**
   - aux_seg2_zero：远低于 no-camp 档（天梯约 1,950–2,080 带）——对它
     每局净输约 14.4k，存款被压到 2,899（对方 11,430）、树收入
     312 对 7,564。它是「对 base −9.6k」的存量研究件，不是 2,000 分
     水平，OpenAI 报告里「NN 约 2,000 分」的估算不能套在它头上。
   - fork_v0_1：同样低于 no-camp 档，但量级完全不同：每局 −3,024，
     是我方两件存量里对 no-camp 明显更强的一件；按其对 base +1,184、
     nocamp 对 base +3,352 的相对位置，它仍在 nocamp 之下数百分
     （天梯分）的中段，与此前估位（约 #80–180）不冲突。
   - nocamp_recon 本身经校验合格，可作今后本地评测的强对手基准
     （对 base +3,352/8 局全胜）。

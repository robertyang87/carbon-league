# 验证 manifest（冻结）：标注优化第二批 k0_annotate_queue

冻结时刻：2026-11-18 15:30（spot-70 到货时刻）。依据：设计 `reports/DESIGN_ANNOTATE_BATCH2_20261118.md`（a35ec834）、建成回执 `reports/BUILD_K0_ANNOTATE_QUEUE_20261118.md`（d86ec5fc）、策略组 O-4 裁定（2026-11-18，裁减①单独档）。状态：**VALIDATION-ONLY，生产链不动。**

## 座位与窗口

- spot-70：实例 i-uf6b8qwd75lr4id7mlsm，c8a.16xlarge／M 区，¥2.459176/h，到货 2026-11-18 15:30:00，保护期到约 16:30，公网 47.117.49.49（尾网 IP 后勤确认后补记）。
- 验证总 ETA：座就绪约 16:00（估计）→ 全部完成约 **21:10–22:10（估计）**。分段：备料验合约 0.3 h、框 B 约 0.4 h、框 A 约 0.8 h、框 A 对账约 0.2 h、框 C 约 3.2–3.8 h、终对账成文约 0.3 h。
- 回收敞口：验证窗长于保护期。缓释：闸序前置（框 B、框 A 的等价裁决在座就绪后约 1.5 h 内落地）、分片逐片出机同步、根级 partial 账本开启；框 C（速度）殿后，座亡只损速度腿不损等价结论。

## 身份（起跑逐件在座读回，不合即停）

- 候选：`code/k0_annotate_queue.py` md5 d40e060c74fe9624ab80e2fe742cf3ca（commit cccc0fbc）；教师钉版 `stage0_teacher.py` c6ade5ce；共享六件 search 94f7fb64／np d8c15ad4／feat 2a45b08b／fastsim aefe6c83／gate e922d9d8（＋model 3bb5857f）；权重 06a03d9c。
- 工具：`tools/compare_shards.py` 9839fed9、`tools/rng_stream_probe.py` 034b988a。
- 语料：annot_a 同集 50 局（selfplay-300002..300051）回放＋games_a50.json，自 L20 `loop_pilot` 侧取、在座验 games/replay sha 与 annot_a 身份段合。
- 起跑身份闸：候选 progress 身份段 config_sha256 须＝**3cc2e67e**（annot_a/b/c 生产同参值），不等即停报。
- 参照：L20 `/root/carbon/katago/loop_pilot/annot_a/shards/` 50 片（已验合在册）。

## 链路卡（具名＋触发写死，步步自动衔接）

执行人唯一：工程组。备料上座（触发：尾网 IP 到）→ 身份验合（触发：备料齐）→ 框 B（触发：身份合）→ 框 A（触发：框 B 全等＋阴性对照全变）→ 框 A 对账（触发：12 片齐）→ 框 C（触发：框 A 对账过闸）→ 终对账＋报告草稿回主代理审（触发：50 片齐）→ 入仓转策略组验收。任一闸不过：停在该步、报主代理，不自行变通。

## 框 B（RNG 流，闸）

根集：selfplay-300002 步 {2,3,200,211}×双座＝8 根。钉版路 vs 候选路逐股比对（种子/抽取次数/digest/sha256）；B3 阴性对照 seed_base+1 须全变。不过即停。

## 框 A（等价，闸）

12 局冻结（annot_a 50 局按 feed 序等距抽）：300002、300006、300010、300015、300019、300023、300027、300031、300036、300040、300044、300048。全窗 [0,299]、生产同参、workers 62。逐根对账（compare_shards 对 L20 参照）：priors/q/Q/Q_std max|Δ|、flips 点名、arms/belief/padding 差、逐数组 sha。**闸：max 偏差 ≤1e-6 且 flips 0；构造预期恰 0.0，任何非零先按构造断裂定位、不许 tolerance 放行。**
**深窗处置：目标座补跑——框 A 本身即深窗对账（12 局全窗 [0,299]、深步 200–299 全量在内、逐根对 annot_a 钉版参照）；建成期浅窗样本仅为建成闸冒烟，不作深窗证据。**

## 框 C（速度，无钉版重跑）

候选 50 局整波（argv＝annot_a 生产形态仅换候选脚本）。基线＝annot_a 已录逐局时长（分片 mtime − 08:43:15 起跑；在座口径标注：同档异座、跨座硅差未单独标定）：**Σ 工作量 12,241.4 分、整波墙时 278.9 分（4h38m55s）、单局 min 169.1／max 278.9 分**（逐局表存工程组本地证据 `annota_durations`，报告引用 Σ/max）。读数：候选整波墙时＋逐局墙时比＋聚合根速。

## O-4 与后补项

①单独形态（snapshot 按局 map）经策略组裁定裁减，不入本 manifest。若框 A/C 出现等价或总效应异常，归因自动升必做、另行排期。

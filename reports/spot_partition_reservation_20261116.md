# spot-36/37/38 分区预留表（2026-11-16，DGX 线发 C3 存档）

登记表：gen1_pilot_registry_20261116.json（md5 6c2199a24bbde8950b6d209ce5bbcad1，200 局）。
今晚批现席位：L20＝登记序 1–140；Studio＝尾段 141–200。三台新机为**机动席**，不预占固定局号。

## 每台预留局块口径（36/37/38 同口径）
- 来源池：入网定版时刻登记表**未开工池**（200 局减去 L20／Studio 已开工局），按登记序划块。
- 块大小：按该机计时小片实测单价（秒/根）与批剩余时间定版，由 DGX 线逐机出清单（局号＋逐局 replay sha）随补推同发；先入网者先划。
- 时点分流：批完工（约 00:50，估计）前入网 → 编入今晚批扫尾／下一波（DGX 定盘）；赶不上 → 不硬塞，直进明日 §6.4 标注分区＋gen-2 预备队列。
- 硬规：同局只认一个来源；任何局被两台消费即按双产处置（另一侧分片隔离不进组装）。

## 补推三小包（每台同清单，入网即推，逐件 md5 读回）
1. **教师权重**：cnet2_m3s2r1_heads_v2.npz，md5 06a03d9cfeec239af3065a49fefec382（DGX 预置包 /home/arm/junyang/spot_topup_bundle/）。
2. **分区回放**：该机定版局块的回放文件＋逐件 md5 清单（入网时自登记表回放源现切）。
3. **代码补缺**：钉版七件＋配置——k0_annotate 7f567a2e／gate e922d9d8／feat 2a45b08b／search 94f7fb64／np d8c15ad4／fastsim aefe6c83／eval_v53 45792dfb／carbon.json cd55531b；镜像（spot-34）已有件逐件 md5 读回，只推缺件（源：DGX /home/arm/junyang/k0_dist/pkg）。

## 对接序（每台）
成交 → 作战室铸钥 → C3 跑 bootstrap（user-data v3）→ 入网 → DGX 线逐台验 user-data 非空读回 → 补推三小包（逐件 md5）→ 计时小片 → 首片回 DGX 验（--verify＋身份字段）→ 放全量波次。

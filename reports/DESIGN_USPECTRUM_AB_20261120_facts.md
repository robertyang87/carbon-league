# 价值谱 A/B 事实包（工程组备料，2026-11-20）

全部数字注明出处：仓内文件直读（gh_api，main）／座上实测（L20、DGX 只读探针）／本地精确计算（numpy）。估计标「估计」、未核标「未核」。

上游件：NETEXP 方案 `reports/PLAN_NETEXP_20261120.md`（commit 2d23f1d6）；预注册＋预裁 `reports/PREREG_USPECTRUM_AB_20261120.md`（commit 8072d127）；复判 `reports/REJUDGE_LOSS_COLLAPSE_20261120.md`（commit 7cbefdd5）；桥裁决 RULING_KATAGO_VALUE_SCALE_BRIDGE_20261119（bridge.py 文件头引用）。

---

## 1. NETEXP A 臂配方与执行状态

### 1.1 方案文本口径（PLAN_NETEXP §5 直读）

- A 臂＝gen1r 检查点原样作 init、同数据、同配方重训；B 臂为结构变量（与本线无关）。
- 数据（§5.2）：Gen-2 自博弈 8,000 局分片＋Echo9 天梯行；行数当时为**估计 ~716 万行**（按 Gen-0 的 895.5 行/局外推），Gen-2 集封存登记当时未做（未核）。
- 配方（§5.3）：「train.py 现行配方」——AdamW，lr 1.5e-4 cosine→3e-5，batch 512，epochs 3，weight_decay 0.01，BF16 autocast；**--fast-policy-weight 0**（并明文把 fpw=1.0 限定为 Gen-1 修复轮对 Gen-0 数据专用）；总步数 ≈41,979（估计，随 716 万行假设）。
- 打包口径（§5.1）：fix3 壳、--value-scale u、同时钟补丁档位。

### 1.2 执行实况（L20 只读直读，2026-11-20 09:38–09:42）

- **数据已封存**：`/root/carbon/katago/work0/netexp_ab/seal_manifest.md5`（2026-11-20 08:28 落，2,788 件）：gen2_replay 2,426 片＋gen2_replay_b 354 片＋echo9_rows 8 片。行数由 gen2.log 加载行直读：gen2_replay 3,123,952＋gen2_replay_b 444,015＋echo9_rows 7,176＝**3,575,143 行**（与派单数逐字合；方案的 716 万行估计已被实测取代）。
- **A 臂训练命令尚未登记/未发车**：netexp_ab/ 内只有 seal_manifest.done＋seal_manifest.md5，无命令文件、无训练日志（NETEXP 两臂按作战室令在 Gen-4 训练后接档；Gen-4 正在 L20 跑）。
- **生产训练口径实证（Gen-4 在跑进程 argv，ps 直读）**：`train2_fixed.py --data work0/gen4_replay,work0/gen2_replay_b,work0/echo9_rows_g4,work0/echo10_rows_g4,work0/echo11_rows --init work0/gen3/ckpt.npz --out work0/gen4/ckpt.npz --epochs 3 --device cuda --fast-policy-weight 1.0`。即 L20 生产自 Gen-3 起＝**train2（座上件 train2_fixed.py，md5 06e3c87c932a762dba6a09ee1bd06117，与仓内 train2.py 现行版逐字同）＋fpw 显式 1.0**。train2 的 argparse 默认 fpw＝0.0（train2.py L449），生产靠 argv 覆盖为 1.0。
- Gen-2 本轮实况（gen2.log）：train.py 跑 6,709,560 行（含 gen0 窗口目录）39,315 步收官——步数式＝ceil(行数/512)×3（13,105×3），与 train2.py L484 `steps_per_epoch = math.ceil(len(rows) / args.batch)` 同式。本 A/B 集 3,575,143 行 → 每 epoch 6,983 步、**总 20,949 步**（公式钉死，执行登记时以实跑日志复核）。
- train2 其余默认（train2.py L444–464 直读）：epochs 3、batch 512、lr 1.5e-4、lr-final 3e-5、weight-decay 0.01、n-extra 0、seed 0、feed-workers 8、prefetch 4、compile 关。与 train.py 默认 13/13 逐字同（TRAINOPT 深审 ⑤已验）。

### 1.3 逐 flag 对照（工作底稿；成稿入设计件 §3）

| 项 | 方案文本 | 派单/预注册口径 | 实证 | 判定 |
|---|---|---|---|---|
| 数据 | Gen-2 自博弈＋Echo9（估 716 万行） | 3,575,143 行 | seal manifest 2,788 片；DGX 镜像逐片行数实测合计 3,575,143 | **合**（以封存清单为准） |
| 基座 init | gen1r（短指纹 bd05231b） | 同基 gen1r | 全 md5 两座同值（§4） | **合** |
| 训练器 | train.py | train2 | 生产 Gen-3/4 实跑 train2（06e3c87c） | **文本不合**：方案写于 train2 入役前；执行口径以 train2 为准，A 臂发车登记须读 train2，读 train.py 则借用
...[truncated 7899 chars]
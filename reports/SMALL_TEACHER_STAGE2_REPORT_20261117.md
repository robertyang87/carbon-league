# 小教师蒸馏线阶段② 回执（2026-11-17）

范围：接入前置两闸复核＋延迟原型实测（DGX 座位）＋蒸馏 harness 三件＋接入教师件，全部构造冒烟。
红线遵守：未提交 GitHub、未跑两路真实生成、未跑真实训练、未触生产目录/进程；DGX 侧只在
`/home/arm/junyang/small_teacher_scratch/` 内工作。非实测数字一律标「估计/推断」。

依据：注册 `9bc75b46`、设计 `cb6efbe7`、事实表 `45263fd4`、签核 `5fcdaab4`（commit 28e06112）。

---

## 1. 前置闸 A：cnet2_search 钉版复核 —— 结论：触碰面逐字一致，接入形态不受影响

- **canonical 直读**（DGX 指定路径）：`/home/arm/junyang/carbon/code/cnet2_search.py`
  md5 = `94f7fb6438bb7577b141e388f068e122` ✓ 与钉版一致。
- **交叉验**：`/home/arm/junyang/k0_dist/pkg/code/cnet2_search.py` md5 同为 `94f7fb64…` ✓
  两处一致，无差异发现。本地留存 `refs/cnet2_search_pinned.py`（md5 同上）。
- **与 main 版（c3222eea）逐项对照**：同为 21 个顶层定义，20 个逐字相同，**唯一差异是
  `make_search_agent`**（教师件从不 import/调用它）。教师实际触碰的三名逐字一致
  （文本 md5：`CachePolicy` 41 行 `277be6a8…`、`feat_copy` 13 行 `788243df…`、
  `BankruptcyGuard` 108 行 `ddf77412…`）。
- **签名/语义审计**（对 stage0_teacher c6ade5ce 的实际调用面）：`BankruptcyGuard(cfg_dict)`
  读 LOWW/GUARD_NARROW/GUARD_CASH/GUARD_LOG；`reset_step / apply(x, aux, y, lg0, pol, own_uid,
  teacher, stats=None)`（教师第 8 位置参数 = gtot ✓）`/ snapshot / restore`（原地回填
  guard_base）`/ copy` 全部吻合；guard 重解码走 `pol.enc_for`＋模块级 `decode_b`，两版相同；
  `CachePolicy._sig` 覆盖 x 九字段（R12 纪律）。钉版缺 main 的 `cnet2_comparator` import，
  与教师 import 行（`from cnet2_search import BankruptcyGuard, CachePolicy, feat_copy`）无关。
- **结论落字：教师触碰面在钉版与 main 间逐字一致；接入件形态（仅换权重的主干替换）不受
  任何差异影响，接入件按原形态定稿（非草案）。**

## 2. 前置闸 B：权重生产原件直读 —— 三处原件全合

| 路径 | md5 |
|---|---|
| `/home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads_v2.npz`（生产） | `06a03d9cfeec239af3065a49fefec382` ✓ |
| `/home/arm/junyang/k0_dist/pkg/weights/cnet2_m3s2r1_heads_v2.npz` | 同上 ✓ |
| `/home/arm/junyang/spot_topup_bundle/cnet2_m3s2r1_heads_v2.npz` | 同上 ✓ |

本地参照副本 `refs/teacher_heads_v2.npz`（md5 同上）。

## 3. 延迟原型（DGX 座位可达性读数，非 §4 结论）

学生原型：B 档 cfg（C48/NB4/TA2/D96/L2/NH4/TNH4/E24/G4）随机初始化，按 cnet2_np 布局
直接构造（`build_student_prototype.py`），**参数量 528,391 ≈ 0.53M**（与设计 §1.2 一致），
经生产 `cnet2_np.Policy` 原样加载成功。`refs/student_B_prototype.npz` md5 `6a38cda3…`。

局面来源：封口局 994400 回放单趟重放（双 Feat 每步各 observe 一次，gate 纪律），
步 0 与步 200–207 双座快照共 **18 个真实局面**，循环复用以避开 CachePolicy 单条目缓存；
batch=1、单进程、nice -n 10；每政策 **288 次** 计时调用（≥200 达标）。
座位：DGX（GB10，Cortex-X925＋Cortex-A725，20 核）；测时 loadavg（act 轮）1 分钟约 4.0→5.9、
5 分钟约 19（机器与他线共享，非静座读数）。栈 md5 见结果 JSON（cnet2_search 94f7fb64、
cnet2_np d8c15ad4、gate e922d9d8、stage0_teacher c6ade5ce 全合）。

| 口径 | 教师 mean / p95 | 学生 B mean / p95 | 倍数 mean / p95 |
|---|---|---|---|
| `pol.act`（含 Prefix＋12 步解码） | 29.71 / 31.12 ms | 6.90 / 7.31 ms | **4.31× / 4.26×** |
| `pol.encode`（仅主干，分解读数） | 26.47 / 28.02 ms | 5.32 / 5.46 ms | **4.98× / 5.13×** |

分解：act−encode 固定开销教师 ≈3.2 ms、学生 ≈1.6 ms；倍数被压低的主因不是解码开销，
而是**学生主干本身**：64M MACs 的 numpy 小张量计算被逐算子开销与访存吃掉，有效算力
仅约 24 GFLOP/s（教师约 54 GFLOP/s），11.2× 的 MAC 比无法兑现（此归因为推断，
依据 = 上表分解与有效算力反推）。

**落字口径**：这是 §4 可达性原型读数（DGX 座位）。正式 §4 在 spot-36 由作战室排窗、
归 L20 产出，本报告**不作 §4 达标结论**。但须如实提示：在本座位上，B 档在 act 口径
（6.90 ms / 4.31×）与主干口径（5.32 ms / 4.98×）**均未够到 §4 的 ≤5 ms 且 ≥8×**；
倍数是跨座位较可移植的量，spot 座位即便绝对值不同，8× 的缺口（现约 4.3–5.0×）不太
可能仅靠换座消失（推断）。是否转 C 档、或由策略组另裁 §4 口径/门槛，归策略组裁定；
本线不自行放宽数字。原始读数：DGX `small_teacher_scratch/results/bench_latency_dgx.json`
与 `bench_encode_only_dgx.json`。

## 4. Harness 四件（本地 `~/workspace/small-teacher-distill/harness/`，未提交）

| 文件 | md5 | 说明 |
|---|---|---|
| `distill_datagen.py` | `9d50f3f4ec9ce1c5e379ff5835126f8b` | 两路生成＋排除硬闸＋分片/装配（见 §5） |
| `distill_train.py` | `f5cad75ed10255183d034093fb4e2204` | 设计 §4 配方；≤45 min 检查点＋续跑＋对称自检门 |
| `distill_export.py` | `5abeac4af924707aec7dce2f3be0c0ef` | ckpt→npz（新身份标记）＋torch/np 对拍 |
| `stage0_teacher_small.py` | `c2ac6ff20585dc61589389fc90618b7a` | 接入件（见 §6） |
| `bench_latency.py` / `bench_encode_only.py` | `35374c33…` / `1f45956a…` | 原型测速脚本（act 口径 / 主干分解） |
| `build_student_prototype.py` | `838e3270…` | B 档随机原型构造器 |
| `smoke_datagen.py` | `f39922dc…` | datagen 构造冒烟脚本 |

### 冒烟记录（全构造级，无真实数据/训练）

1. **datagen 排除闸**（DGX，真实 #7 局单只读装载）：局单 md5 验合 `8f946da7…` 后装载，
   排除集 = 切片 50＋附卷 2 = 52；候选过滤记录 n_excluded_present=5、overlap_after=0；
   **错 md5 → 拒跑（exit 2）；--only-games 点名被排除局 → 拒跑**；缺 --slice-md5 → 拒跑。
2. **datagen 分片机制**（DGX）：种子 FastCarbon 双座 2 局面 → np 批打标 → 分片写＋验；
   t_labels 与生产 batch=1 `act()` 逐标签一致；t_u ∈ [0,1]。
3. **装配模式**（本地伪造两片）：身份/分片计划/交叠校验通过后合并 4/8 局（complete=false
   如实），分片文件 md5 复验后复制；**跨片同局交叠 → 拒合**。
4. **train**（DGX CPU，合成 256 局面）：3 步跑通，`ckpt_ep0_end.pt`/`ckpt_final.pt`＋
   `progress.json`（阶段/epoch/step/已见局面/最近验证/检查点路径/墙钟全字段）落盘；
   `--resume` 后 step 由 3 续到 8（计数恢复验过）。注：合成随机输入下验证 logits 损失
   为病理大量级（随机输入＋随机标签所致），机制面已验、数值面不作结论。
5. **export 对拍**（DGX）：冒烟 ckpt 导出 npz（md5 `98606868…`，参数 528,391，cfg=B，
   标记 SMALL_TEACHER_V1）；32 局面 torch 学生 vs cnet2_np 学生
   **max|Δlogits| = 2.03e-06、max|Δu| = 7.26e-08**（浮点实现容差内一致）。
6. **接入件**（DGX）：`stage0_teacher_small` 在钉版栈上 import 成功，STUDENT_MARKER /
   STUDENT_CFG / DISTILLED_FROM 在位，evaluate_root/gen_candidates 可调用。

冒烟中修掉的构造缺陷（均已回归）：合成标签 tok_mask 24 长 vs 12 槽；建筑槽（10–11）
标签域须 ≤2（aemb 仅 9 行，生产同理）；tgt mask 截前 12；续跑时 step 计数未恢复；
分片 meta_source 改逐局面来源（replay/harvest 分开记）。

## 5. 座位口径变更（所有者 08:54 定、08:55 修正）及其落地

- 数据生成默认座位 = **CPU spot 机队共 4 台**（spot-34＋spot-36/37/38；spot-34 考试座
  空闲时入列、**考试优先、可被抢占**）。分片数纯 CLI 参数（派单定时 3 或 4），代码与
  本说明不写死台数。`distill_datagen.py` 已支持 `--shard-index/--shard-count`
  （kept 局单排序后跨步切分，各台同 split＋同排除 → 同一 kept 哈希，manifest 记录
  kept_list_sha256 与分片计划）；分片**幂等＋可断点续跑**：逐局分片 tmp+rename 原子落盘
  ＋manifest 逐局 md5 记录，重跑（含被考试抢占后重跑）跳过 md5 仍合的局，已落局不重算。
- 打标默认 = **cnet2_np 批路径多核**（`--workers` 按局多进程；spot 64 vCPU）。
  设计 §2.2 的 torch GPU 批前向**降为非默认选项**（`--labeler torch-gpu` 保留 flag、
  默认关）；乙路收割同为 CPU 活、与甲路同一分片口径（收割局集在全 kept 局单上先定，
  机队总收割局数仍为 --harvest-max-games，默认 50）。
- 装配/合并在 **L20** 侧：`--assemble` 模式（校验跨片身份/分片计划/局集互斥/逐片 md5
  后复制合并，产出 distill_train 可直接消费的 merged 目录）。训练 harness（L20 GPU）
  与延迟原型（DGX）不变。
- **对 §2.2 成本估计的影响（估计）**：总量约 0.85M 局面。CPU 口径下逐局面打标以解码
  Prefix 的 Python 成本为主（DGX 实测 act−encode 约 1.6–3.2 ms/局面/核，批 encode 另
  约 3–6 ms/局面/核，均为估计），按每核约 8 ms/局面、机队 3–4 台 × 64 vCPU、效率
  30–50% 估，**打标墙钟约 1–2 小时（估计）**；甲路回放采集约 2 核时量级、乙路收割约
  1 核时量级，机队并行后不成长板；**数据生成总墙钟约 2–4 小时（估计）**。对照设计
  §2.2 原 GPU 口径（打标 10–20 分钟级）：CPU 机队路线墙钟更长，但不占 L20 GPU、
  与 §6.4 窗口零争用，且可随派单台数（3/4）伸缩。

## 6. 接入件 stage0_teacher_small.py

c6ade5ce 逐字底本，unified diff 仅两处（`stage0_teacher_small.diff`，36 行，
md5 `a2e484f6…`）：① 文件头 docstring 内小教师身份段（蒸馏来源 c6ade5ce＋06a03d9c、
学生 cfg、代标记、自身代不与现行教师标注混代）；② `TEACHER_VERSION` 后新增常量
`STUDENT_MARKER = "SMALL_TEACHER_V1"`、`STUDENT_CFG`（B 档）、`DISTILLED_FROM`。
其余逐字不变（闸 A 结论支持原形态定稿）。

## 7. 启动说明（L20/机队照抄区）

前提：#7 局单闸已开（策略组已读回验合局单 md5 `8f946da7c6f8c5767459506c5c2b6563`，
排除清单 = 切片 50 局＋附卷 994400/994401）。train 池 1,202 局回放由 DGX 补送至各座，
落点未定——故下列路径参数全为各座必填、无写死值。代码栈：datagen 用钉版包栈即可；
**train/export 的 --code-dir 须另含 `cnet2_model.py`（3bb5857f，标注包内没有）**。

### 7.1 每台 spot（分片生成；N = 派单定 3 或 4，i = 本台序号 0..N-1）

```
python distill_datagen.py \
  --corpus-root <本座回放落点目录> \
  --train-split <train 切分文件> \
  --slice-file <small_teacher_tier2_slice50.json> --slice-md5 8f946da7c6f8c5767459506c5c2b6563 \
  --out-dir <本台输出目录> \
  --code-dir <钉版包 code 目录> \
  --teacher-weights <cnet2_m3s2r1_heads_v2.npz> \
  --shard-count N --shard-index i --workers 48
```

- 四个路径参数（--corpus-root / --train-split / --slice-file / --out-dir）＋ --slice-md5
  为启动必填，缺失或局单 md5 不合即拒跑（fail-closed）。
- spot-34 的分片可被考试抢占：被杀后原样重跑同一命令即续跑（已落分片按 md5 跳过）。
- 先验一道：同命令加 `--check-exclusions-only` 只打排除核对记录、不生成。
- 断点续跑为默认（--no-resume 关闭）；换分片计划/换 datagen 版本续跑会被 manifest
  身份校验拒混。

### 7.2 L20 装配（各台输出收齐后）

```
python distill_datagen.py --assemble \
  --shard-dir <座1输出> --shard-dir <座2输出> --shard-dir <座3输出> [--shard-dir <座4输出>] \
  --out-dir <merged 目录>
```

merged/manifest.json 的 `complete=true` 才表示 kept 全量到齐。

### 7.3 L20 训练（GPU；开工前另过注册 §5 启动清单与检查点纪律）

```
python distill_train.py --data-dir <merged 目录> \
  --code-dir <含 cnet2_model 的代码栈目录> --ckpt-dir <检查点目录> \
  --device cuda --epochs 40 --batch 1024 --ckpt-minutes 45
```

续跑加 `--resume <ckpt.pt>`。对称增强默认关；启用须带 `--sym-augment --teacher-weights
<06a03d9c 权重>`，自检（max_dlogits ≤1e-3 且 label_agree ≥0.999）不过自动禁用并记录。

### 7.4 导出与对拍

```
python distill_export.py --ckpt <ckpt.pt> --out <student.npz> \
  --code-dir <含 cnet2_model 的代码栈目录> --check-positions 512 \
  --report <report.json>
```

产物身份 = (stage0_teacher_small.py md5 c2ac6ff2, student.npz md5)；按注册 §2 其标注
自成一代、永不与 c6ade5ce 代混训。数据生成完成后按签核序先做 50k 样本校准批，
再定全量配方（设计 §5）。

## 8. 未决/移交

- §4 可达性缺口（§3）：B 档倍数约 4.3–5.0× vs 目标 ≥8× —— 归策略组裁（C 档 / 口径 /
  门槛），本线未自行调整任何数字。
- 真实两路生成由机队座位在 harness 交付后执行（本阶段未跑）；训练开工走注册 §5
  前置清单；L20 租约 2026-11-20 23:59:59 到期、§6.4 窗口优先权不变。

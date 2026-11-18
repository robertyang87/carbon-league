# 设计短笺：标注叶子评估批量化（分叉版，阶段 1 设计）

- 派单：作战室 2026-11-18 09:39（所有者令）。定盘依据（Studio 剖析）：单根约 600 次 numpy 前向、净评估占总成本 75–80%、前缀/IO 占 1–3%——优化只打评估段。
- 口径：钉版原件一律不动；本设计只动分叉副本，分叉自成新身份（新 md5 集）。验收预注册：① 对账 ≥36 根（≥2 局×早/中/残三带×双座），priors/q/Q max|Δ| ≤1e-6（目标 0.0）、flips=0；② 测速同座同局同窗单根价、评估段目标 ≥2.5×（x86 窗作战室安排，本阶段不测速、不下可达结论）。
- 钉版身份（本机直读验合）：`k0_annotate.py` 7f567a2e／`stage0_teacher.py` c6ade5ce／`cnet2_search.py` 94f7fb64／`cnet2_np.py` d8c15ad4／`cnet2_feat.py` 2a45b08b／`fastsim.py` aefe6c83。生产参数 CANON：K=16、M=8、H=2（stage1_teacher_gate.py:113），窗 [200,299)，seed_base = seed_teacher(7000) + step（k0_annotate.py:346）。

## 1. 评估调用链定位（逐行直读）

一根 = 一次 `stage0_teacher.evaluate_root`（k0_annotate.py:341 调用；每局根在 worker 内逐根串行，k0_annotate.py:383）。每次 `pol.act` = 一次完整 trunk 前向（cnet2_np.py:175 `act` → :117 `encode`，无缓存跳过；CachePolicy 只缓存供 `enc_for` 复用，不省 act 内的 encode）。全部 trunk 前向调用点：

| # | 调用点（文件:行） | 形态 | 每根次数 |
|---|---|---|---|
| 1 | 对手根动作：stage0_teacher.py:419 `act(..., sample_rng=rng_m, temp=1.0)` | 单条 act（采样） | M = 8 |
| 2 | 候选贪心解码：stage0_teacher.py:224 `pol.act`（gen_candidates） | 单条 act | 1 |
| 3 | 强制前缀偏差解码：stage0_teacher.py:251 `pol.act(forced=fz)` | 单条 act，同一根 x 反复 encode | D（≥A−1，≤6 槽×4 备选=24） |
| 4 | 臂重打分：stage0_teacher.py:274 `pol.act(forced=arm)` | 单条 act，同一根 x 反复 encode | A（最终臂数，≤K） |
| 5 | 续局双座解码：stage0_teacher.py:467/468 `act(x0)/act(x1)` | 单条 act（贪心，无 RNG） | 每分支每层 2，合计 ≤ A·M·2H = 512 |
| 6 | 叶评估：stage0_teacher.py:187 `_leaf_u` 内 `pol.act(want_u=True)` | 单条 act（只取 u，解码结果全弃） | 每非终局分支 1，合计 ≤ A·M = 128 |

**构成式（满臂 A=K=16、D=15、无早终局）**：8 + (1+D+A) + 5·A·M = 8 + 32 + 640 = **680 次/根**，与注册口径逐一合。其中续局+叶 = 640（94%），候选段 32（内含对同一根 x 的 31 次重复 encode），对手段 8。
与「约 600 次/根」定盘的对账：实际计数随 A 与早终局浮动——如 A=14 时 8+(1+13+14)+5·14·8 = 596 ≈ 600，落在构成式区间内；600 是典型值、680 是满臂上限，不是两个口径打架。
**未计数前向（另列）**：BankruptcyGuard 触发时（cnet2_search.py:565 `apply`）走 `enc_for`（:613）+`decode_b` B=1 再解码（:616）——`enc_for` 命中缓存时无新 trunk 前向，未命中时 +1 次 encode；此类前向不入 `n_act_calls`，触发率可由每根结果的 `guard` 计数（guard_moves/guard_recruit）读出，对账冒烟须一并报。

既有批量件（钉版 cnet2_search.py 内，本设计直接复用、不改）：`encode_b/trunk_b/ctx_b`（:156/:179/:162）、`decode_b`（:235）、`stack_x/act_b`（:353/:360）。注：`decode_b` 与单版同走 f16 量化的 uf/mf（cnet2_feat.py:499 `Prefix.feats` 返回 f16；decode_b:308 同），此项不构成差异源。
**点 B 先例（必须正视）**：stage0_teacher_batch.py（d446e5b6）已把续局段按深度拼批（encode_b/decode_b），C1 实测仅 0.930×（约快 7%），且 396 根附录对账 994401 max|Δq| 达 3.42e-05（flips=0）。本设计与点 B 的差别：点 B 未动叶评估（128 次全 act）、未动候选段（31 次重复 encode）、续局按座分批；本设计三段全覆盖（见 §2）。点 B 的测速是在 C1 生产负载（62 worker）下取得——批量化收益随 worker 并行度被内存带宽争用压缩，**测速口径必须固定 worker 数**，低 worker 测速与满载测速测的不是同一个量（见 §4）。

## 2. 分叉方案

分叉文件（2 件新文件，其余钉版件原样引用）：

| 分叉件 | 源 | 改动 |
|---|---|---|
| `stage0_teacher_leafbatch.py` | stage0_teacher.py c6ade5ce | evaluate_root 内部重构（改造点 P1–P6）；签名与结果 dict 字段不变 |
| `k0_annotate_leafbatch.py` | k0_annotate.py 7f567a2e | 仅三处：import 指向分叉 teacher、`code_md5s()` 清单换分叉件名、身份串标注 leafbatch 版 |

`cnet2_search.py`／`cnet2_np.py` **不分叉、不改**：批量原语用钉版现成的 encode_b/decode_b；若需两遍 GN 版 trunk（见数值风险对策），在 teacher 分叉件内自带 trunk 变体函数，不许改钉版 search。

改造点清单（只许根内攒批）：
- **P1 分支构造段：不动。** 信念采样（:398）、克隆构建、guard 快照恢复逐分支顺序与钉版逐字一致；此段无 trunk 前向（对手动作除外）。
- **P2 对手根动作（8 次）：v1 保持逐条 act，不拼批。** 理由：每场景独立 RNG 流（default_rng(seed_base+m)，:408）经 `sample_rng.choice` 逐槽消费（cnet2_np.py act 采样支），decode_b 的 noise 逆 CDF 抽法与之不是同一消费模式，拼批会改对手动作抽样结果＝改搜索语义；且占比仅 1.2%。列为远期可选（须先做 RNG 等价审计），v1 不做。
- **P3 候选段（32 次 → 1 encode + 1 批解码）：** 贪心 act 保持钉版原样（其 encode 经 CachePolicy 留存 (tok, ctx)）；偏差 D 条 + 重打分 A 条 forced 解码合并为一次 `decode_b`（forced 矩阵 [D+A, OWN]，复用贪心 encode 的 tok/ctx，不再重复 encode 根 x）。去重、guard 逐臂 apply、joint_logp、priors 归一的顺序与判据逐字沿钉版（guard apply 在批解码后逐臂行行执行；apply 内 `enc_for(根 x)` 必须命中贪心留存的缓存——批解码不得污染 CachePolicy.last_enc，此为实现硬检查点）。
- **P4 续局段（512 次 → H 批）：** 按深度 lock-step（点 B 已验结构）：每深度把全部存活分支的双座位置堆成一批（行 = 存活分支×2，≤256），一次 encode_b + decode_b（lg_out 回收 logits）；guard 逐分支逐座在批后按原顺序 apply、sim 逐分支 step、终局判定逐分支——顺序与钉版分支主序一致。guard 触发再解码 v1 沿用 guard.apply 原路径（enc_for 未命中则单条重 encode，数值与钉版同路、只是慢），不做显式传 enc 的优化（列后续项）。
- **P5 叶评估（128 次 → 1 次 encode_b，零解码）：** 全部非终局分支的叶位置（各分支以其 fb[me] Feat observe 后）堆成一批，只跑 encode_b 取 ctx，u 逐行按 cnet2_np.py:200-201 同式计算（ctx @ utility.weightT + bias → sigmoid，float64 标量路径逐行）。钉版此处跑全 act 只取 u、解码纯浪费——本点同时省 128 次 encode 的批量差与 128 次解码。
- **P6 计数口径：** `n_act_calls` 按批行数逐行计（点 B 同口径），与钉版同根必须逐一相等，不等即停报（见 §5）。

**批量大小策略**：自然批 = 当前深度存活行数（续局 ≤256、叶 ≤128、候选 ≤~40），不跨深度、不跨根合并。默认上限 128 行/批（与点 B 按座分批同形状；双座合并 256 行作 bench A/B 变体），超限按分支主序切块。内存依据：教师 cfg C=128、N=225、TA=4——trunk 激活约 115 KB/行·层，cell-attn 分数矩阵 [TNH,N,N] 约 0.8 MB/行·层，128 行单层峰值约 100 MB 量级，机队 121 GB 内存、62 worker 并发下每 worker 峰值须 ≤1 GB 量级，128 上限留足余量；切块对逐行结果数值中性（批内各行独立），但批形状影响 BLAS 求和顺序，故**批上限一经选定全程固定**、写入身份。

**语义不变论证（逐点）**：
- 行独立性：encode_b/trunk_b 的 conv/GN/attn 与 decode_b 的 Prefix 状态全部逐行独立，拼批不引入行间耦合。
- RNG 结构不变：拼批段（候选/续局/叶）全部零随机抽取（贪心 argmax 或 forced；RNG_AUDIT 已对续局段实证）；仅有的两组抽取（信念碳采样、对手动作采样）在拼批段之前以按场景播种的独立流完成、且不与拼批段交错。访问顺序改变只发生在确定性求值之间。
- 顺序/平局打破：臂序、去重序、guard apply 序、分支 step 序全部保持钉版主序；argmax 平局取首元（numpy argmax 语义两版同）。
- 唯一共享可变状态 = CachePolicy.last_enc：钉版语义下它只影响 enc_for 命中/重算（性能），不影响数值（未命中即以同一 encode 函数重算）。分叉实现必须维持此不变量（P3 硬检查点）。

**数值风险（最关键一条）与对策**：批前向与单前向的差来自浮点求和顺序——① trunk_b 的 `_gn`（cnet2_search.py:94）用一遍 E[x²]−m² 求方差，单版 `gn`（cnet2_np.py:85）是两遍均值/方差，公式不同；② BLAS 在不同批形状下分块求和顺序不同。叶 u 的差 ΔQ 经 q 的 exp((Q−maxQ)/0.05) 放大 **20 倍**进 q（点 B 附录实测：max|Δq| 3.42e-05 ≈ 20×ΔQ 1.7e-06 量级，与此算术吻合）——即 q 的 1e-6 闸实际要求 ΔQ ≤ 约 5e-08，比字面闸紧 20 倍，这是本任务最关键的数值风险。对策：(a) 分叉件内自带**两遍 GN 版 trunk 变体**（逐行公式对齐 cnet2_np.gn，只向量化行维），把公式差消掉、只剩 BLAS 顺序差；(b) 对账分阶冒烟：先 encode-only 固定位置差探针（Δctx/Δu 直读），再整根比对，差在哪一阶先定位再推进；(c) 批形状全程固定（见上）；(d) 若 (a) 后 Δq 仍 >1e-6，按 §5 停报 escalate，不许放宽闸、不许用「flips=0」单独报完成。

## 3. 编译运行时（本阶段只列候选，不实施）

| 候选 | 预期面 | 数值差风险 | 取舍 |
|---|---|---|---|
| BLAS 线程数（batched GEMM 多线程） | 大批时直接 | 线程分块改求和顺序 → 有差；且生产 62 worker 口径下线程须钉 1 | 仅 bench 变量，不入生产身份 |
| numba njit 重写 trunk/decode 热点 | 中 | 公式逐行复刻 float32 时差小，但环境未装（标注闭包无 numba）、部署包要变 | 远期，先过纯 numpy 批量对账再议 |
| Cython/C 扩展 | 中 | 同上，另加机队镜像工具链未验 | 远期 |
| torch CPU / torch.compile | 大 | 破坏标注闭包无 torch 的既立事实（部署面扩大）、kernel 全异差不可控 | v1 否决 |
| ONNX Runtime | 中 | 未装、kernel 差不可控 | 否决 |
| trunk 转 float64 | — | 与钉版单版 float32 路径差更大，与对账目标反向 | 否决 |

## 4. 对账与测速的可执行计划

**对账选根（36 根恰数）**：语料 = L20 pilot 语料 `katago_pilot_loop_selfplay_150` 只读拉取（本线已有 300002 局样本+index 在手，路径已验），取 selfplay-300002、selfplay-300003 两局；带 = 标注窗内分三带：早 [200,230)、中 [230,265)、残 [265,299)；每局每带取 3 步（带首/中/末附近、避开终局截断 ft）× 双座 = 2 局×3 带×3 步×2 座 = **36 根**。比对走 evaluate_root 直驱（stage1_teacher_gate.extract_scene + 同 guard_snaps 口径，与 k0_annotate._eval_root 同路），钉版/分叉同 seed_base、同权重 06a03d9c，逐根比 arms/priors/q/Q＋n_act_calls＋guard 计数。
**分阶冒烟排程（本机 muse VM，估计）**：① encode-only 差探针（固定 8 位置，约 10 分钟）；② 6 根小比对（约 10 分钟）；③ 36 根全比对（钉版约 25 秒/根 ⇒ 约 15 分钟＋分叉侧更短）。任一阶不过闸即停在该阶报数，不进下一阶。
**x86 bench 包（约 13:00 窗，提前备好并由主代理向作战室喊窗）**：固定根集 manifest（上述 36 根的 game/step/seat 清单＋回放 sha）、双版代码集（钉版七件＋分叉九件身份表）、权重、一键脚本（双版各跑同 manifest、固定 worker 数两档：1 与目标生产档，逐根墙时＋评估段占比＋n_act_calls 输出成表）。测速判读口径先行写明：报数须带 worker 数，评估段定义 = evaluate_root 墙时（与整根价之比分开报）。

## 5. 风险与停报点

**划界（硬）**：
- **只许根内攒批，跨根攒批不做。** 理由：根间独立 RNG 流（seed_base 随 step 变）、独立 guard 历史与身份计数；跨根拼批使一根的输出浮点结果依赖同批根集合（批形状→求和顺序→末位差），产物不可复现、身份不可归因，且与逐根原子落盘/断点续跑的生产形态冲突。
- 不改访问语义：对手动作采样（P2）、信念采样、guard 状态机、sim 步进顺序一律不动；批只发生在确定性网络求值的合并上。
- 试点生产批（65/66/64 在跑）不扰动：分叉件独立目录/独立进程验证，不碰在跑标注器的工作区与产物。

**停报清单（命中任一即停、当轮报数、不自行放宽）**：
1. encode-only 探针 Δu > 1e-7（公式差未消干净）——停，先修 GN 对齐。
2. 小比对或全比对出现任一 argmax flip（候选臂/续局动作）——停，报 flip 根与差分布。
3. 全比对 max|Δq| 或 max|ΔQ| 或 max|Δpriors| > 1e-6——停，escalate 策略组（附逐根差表）；不许以 flips=0 单独报完成。
4. 任一根 n_act_calls 与钉版不等——停，计数语义已破。
5. guard 触发再解码占比高到吃掉批量收益（gtot 统计报出后判读）——性能停报，报触发率与占比，不硬推上线。
6. bench 测速若只在低 worker 达标、生产 worker 档不达标——如实分档报，不许拿低档数冒充生产价。

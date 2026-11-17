# 接口事实表：现行教师 trunk（小教师蒸馏线阶段①，工程组）

依据注册 `reports/REGISTRATION_SMALL_TEACHER_DISTILL_20261117.md`（commit e6530ea3，md5 `9bc75b46d709d013ac9042def0f22e6d`，已读回验合）。
凡数字注明来源：**直读**＝代码/权重原件读数；**推断**＝由代码结构与注册读数推算，未实测；**需实测**＝阶段②/测量阶段以仪器定。

## 0. 审计对象与身份（全部经 gh_api 从 main 读回、逐件 md5 复核）

| 件 | 仓内路径 | md5 | 字节 | 与生产钉版关系 |
|---|---|---|---|---|
| 教师 | `code/stage0_teacher.py` | `c6ade5ce` | 25,186 | ＝钉版 ✓ |
| 教师（点 B 批版） | `code/stage0_teacher_batch.py` | `d446e5b6` | 27,076 | ＝钉版 ✓ |
| 标注器 | `code/k0_annotate.py` | `7f567a2e` | 49,151 | ＝钉版 ✓ |
| 推理网 | `code/cnet2_np.py` | `d8c15ad4` | 14,953 | ＝钉版 ✓ |
| 特征 | `code/cnet2_feat.py` | `2a45b08b` | 22,539 | ＝钉版 ✓ |
| 模型定义（torch） | `code/cnet2_model.py` | `3bb5857f` | 19,720 | ＝钉版 ✓ |
| 模拟器 | `code/fastsim.py` | `aefe6c83` | 36,212 | ＝钉版 ✓ |
| 搜索/缓存/护栏 | `code/cnet2_search.py`（main HEAD） | `c3222eea` | 131,590 | **≠生产钉版 `94f7fb64`**（见 §7 口径注记） |
| 权重 | `cnet2_m3s2r1_heads_v2.npz`（本地 k0pkg 副本直读） | `06a03d9cfeec239af3065a49fefec382` | 13,973,080 | ＝钉版 ✓ |

**trunk 定义位置（结论）**：trunk 不在 stage0_teacher.py 内。该文件只含搜索逻辑（`evaluate_root`／`gen_candidates`／`_leaf_u`／`_terminal_u`／`belief_carbon`）；网络本体在 **`cnet2_np.Policy`**（纯 NumPy 推理）＋其批版 `cnet2_search.encode_b/decode_b/trunk_b`，架构由权重 npz 的 key 形状与 `meta.cfg` 完全决定（`Policy.__init__` 按 key 计数推 NB/L/TA、按形状推 C/D 等，代码内无写死维度）。torch 侧同构定义在 `cnet2_model.CNet2`（cfg 参数化）。k0_annotate 的接线：权重字节在 plan 时读一次并以 md5 冻结（`k0_annotate.py:260-271`），每 worker 以字节构造 `CachePolicy(w)`（`:327-333`），以 `teacher=cfg["teacher_style"]`（生产＝2）、`seed_base=cfg["seed_teacher"]+step` 调 `evaluate_root`（`:341-347`）。

## 1. 输入契约（trunk 的 `act(x, aux, teacher, …)`）

`x` 由 `cnet2_feat.Feat.observe(obs)` 构造（seat-canonical 帧；量化：planes 经 q8、其余经 f16，保证训服逐位同）。`encode()` 恰消费九个字段（与 `CachePolicy._sig` 的签名字段集逐一合，直读）：

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `maps` | [33, 225] | uint8（用时 /255） | P=33 局面平面，N=15×15=225 格 |
| `scal` | [20] | float32 | NS=20 全局标量 |
| `tok_pos` | [24] | int | T=24 token 的格子（己 10 工人＋2 基地、敌 10 工人＋2 基地） |
| `tok_kind` | [24] | int 0–5 | token 类别 |
| `tok_mask` | [24] | bool | token 有效位 |
| `tok_cargo` | [24] | float32 | token 载货 |
| `hist_cell` | [24, 8] | int | H=8 步历史位置（−1＝无） |
| `hist_cargo` | [24, 8] | float32 | 历史载货 |
| `hist_move` | [24, 8] | int 0–5 | 历史动作 |

`aux`（`Feat.aux`，由 `Prefix` 在解码期消费，不入 encode）：`est／en_min／en_max／en_pl／thr／dob／own_base／en_base／own_tree／en_tree／toll／own_adj`，各 [225] 数组。
`teacher`：int 风格索引（`teach.weight[teacher]`，8 行×16 维嵌入进 ctx）；生产恒＝2。
另有 `forced`（教师强制标签 [12]）、`sample_rng`（调用方 Generator）、`temp`、`hook`、`want_u` 五个行为参数（见 §3/§6）。

## 2. 输出契约

`act(...)` 返回 `(labels, logits, value)`，`want_u=True` 时加第四元 `u`：

| 输出 | 形状/范围 | 语义与消费点（直读 stage0_teacher.py） |
|---|---|---|
| `labels` | [12] int64（OWN=12 槽：10 工人＋2 基地；无单位＝−1） | 联合动作标签。消费：`gen_candidates` 贪心臂与偏差臂、`evaluate_root` 续弈双方动作、对手根动作（经 `Feat.to_actions` 转引擎动作） |
| `logits` | [12, 5] float32（非法动作＝−1e9） | 逐槽解码 logits（自回归教师强制口径）。消费：`joint_logp` 算臂先验 P_A、`gen_candidates` 的 top1−top2 gap 排序、guard 的重解码输入 |
| `value` | (−1, 1) float（tanh 头） | **evaluate_root 全路径不消费**（三处调用均以 `_` 丢弃）；仅为 act 签名完整性存在 |
| `u` | [0, 1] float（sigmoid utility 头，E[U]） | 叶值唯一来源：`_leaf_u` 以 `want_u=True` 取 u 作 H=2 后的裸网叶值（终局叶走 `_terminal_u` 的 F01/F04 计分，不调网） |

即：**功能上被搜索消费的 trunk 输出只有 labels／logits／u 三样**；q 的构造（`q ∝ P_A^0.5 · exp((Q−maxQ)/0.05)`）中 P_A 来自 logits、Q 来自 u 的多场景平均——蒸馏目标应对准这两条（见设计草案 §3）。

## 3. trunk 架构（权重原件直读：187 key＝186 数组＋meta；全 float32）

`meta`：`cfg = {C:128, NB:8, D:128, NH:4, L:2, E:32, G:8, ZI:1, TA:4, TNH:4}`，epoch 19，`params: 3,480,130`（文件自记；按数组元素求和＝3,480,775，差 645 未深究，不影响口径）。

编码器 `encode(x, teacher)`：
1. stem：3×3 conv 33→128 ＋ GroupNorm(G=8) ＋ ReLU；
2. NB=8 个残差块（每块 2×（3×3 conv 128→128 ＋ GN）），块 1/3/5/7 后各插 1 层 cell 注意力（TA=4，TNH=4 头，225 格全注意力＋相对位置偏置 [225,4]＋FFN 128→256→128）；
3. token 路：`tok_features` 89 维（kind 6＋cargo 1＋pos 2＋历史 8×10）→ tok1 89→128 → tok2 128→128，与 `h[tok_pos]`（128）拼接经 fuse 256→128；
4. token 注意力 L=2 层（T=24，D=128，NH=4，FFN 128→256→128，相对偏置 [225,4]）；
5. ctx：[h 全图均值 128 ‖ token 均值 128 ‖ sc2(scal 20→64→64) 64 ‖ teach 嵌入 16]＝336 → ctx 336→128 ＋ ReLU。

解码器（自回归，OWN=12 槽顺序）：h0＝lin(ctx)；逐槽以 `Prefix.feats(s)` 给 uf [9]、mf [5,15]、legal [5]，din 输入 [tok 128 ‖ ctx 128 ‖ uf 9 ‖ mf 75]＝340→128，加 prev 动作嵌入 aemb [9,128]，GRUCell(128) 一步，whead 128→5（工人槽）或 bhead 128→3＋补 0（基地槽），再加低秩修正 `V(mf) @ U(h)`（U:128→32、V:15→32）。`value＝tanh(ctx·w)`、`u＝sigmoid(ctx·w)` 各 128→1。权重另含 `margin／wdl／aux` 三头（cnet2_np 的 act 不消费，系 torch 训练侧产物）。

**逐局面计算量（推算自形状，MAC＝乘加）**：

| 段 | MACs | 占比 |
|---|---|---|
| stem conv | 8.55M | 1.2% |
| 8 残差块（16 conv） | 530.84M | 73.9% |
| cell 注意力 ×4 | 169.80M | 23.7% |
| token 路＋token 注意力 ×2＋ctx | 8.09M | 1.1% |
| 解码（12 槽 GRU＋头） | 1.79M | 0.2% |
| **合计/次 act** | **≈719M MACs ≈ 1.43 GFLOP** | 100% |

编码器占 99.8%——**换小网的杠杆全在卷积栈与 cell 注意力**，解码器动与不动对速度无实质影响（但其结构决定 forced/sample 语义，见设计草案 §1）。

## 4. evaluate_root 内 trunk 调用点与逐根计数（生产口径：belief=sample、feats 在、guard on、K=16/M=8/H=2）

| 调用点 | 位置 | 次数/根 |
|---|---|---|
| 对手根动作（先于候选生成；sample_rng＝`default_rng(seed_base+m)`，temp=1.0） | 主循环前 | M＝8 |
| `gen_candidates`：贪心解码 | 1 | 1 |
| `gen_candidates`：强制前缀单单位偏差解码（≤6 槽×≤4 备选，臂满 K 即停；每次尝试各 1 次 forced act，去重丢弃的尝试同样计数） | D，A−1 ≤ D ≤ 24 |
| `gen_candidates`：最终臂逐臂教师强制重打分（S05） | A（臂数，≤16） |
| 续弈：每（臂×场景）H=2 步×双座 act | A×M×4 ≤ 512 |
| 叶值 `_leaf_u`（want_u=True；终局分支改走 `_terminal_u` 不调网，步数提前终局亦减此数） | A×M ≤ 128 |
| **合计通式** | **8 ＋ (1＋D＋A) ＋ 5·A·M** | |

**对账：满臂零浪费（A＝16、D＝15）时通式＝8＋32＋640＝680，与注册/冒烟口径逐一合**（点 B 首根冒烟实测 `n_act_calls＝680`）。A<16、偏差尝试被去重浪费（D>A−1，理论上限可至 689）或分支提前终局时实际值逐根浮动，`evaluate_root` 以 `n_act_calls` 逐根自报——680 是注册的标准口径而非数学上限。
计数口径注记：guard 触发时的重解码走 `pol.enc_for`＋`decode_b`（cnet2_search），**不计入** `n_act_calls`（不经教师侧 counter）；enc_for 在平直路径上命中 last_enc 缓存、通常不重编码，故额外 trunk 成本以解码级为主、量随 guard 触发数（结果 `guard` 字段 gtot 自报）。此为额外于 680 的小量（推断，需实测其触发率）。

## 5. 38–40 ms/局面的成本结构（注册读数；以下拆分均为推断，需实测）

- 注册 §1：trunk 约 38–40 ms/局面且**批不变**（点 B 把双座续弈拼批后总收益仅约 7%）。与 §3 计算量自洽：719M MACs ÷ 39 ms ≈ 18 GMAC/s（≈37 GFLOP/s），系 NumPy BLAS 在 [225×1152]@[1152×128] 量级矩阵上的合理有效吞吐（推断）。
- 批不变的原因（代码推断）：NumPy 路径无核启动开销、权重仅 14 MB 常驻缓存，encode 的 sgemm 本身已计算密集；拼批只摊薄解码/guard 的 Python 开销（占 ~1% FLOPs 与个位数 ms），故总收益小。
- 39 ms 的构成（推断，需实测）：encode 的 sgemm 主体约 25–33 ms；im2col 展开（17 次 conv 各构造 [225, 9C] 矩阵）＋GN/LN 逐元素遍＋Python 胶水约 4–10 ms；解码 12 槽串行（每槽 Prefix.feats 纯 Python＋约十次小矩阵运算）约 1–3 ms。
- 逐根总账（推断）：680 × 39 ms ≈ 26.5 s，占单根 ~37–40 s 墙钟的约 2/3；余为 Feat.observe（逐根约 650 次分支/根 observe，纯 Python 板面循环）、FastCarbon 克隆/步进（≤384 步/根）、guard 与场景构造。此拆分解释 §4「端到端只作报告项」：trunk 压到 5 ms 时端到端约 3.4 s＋固定 ~11–13 s ≈ 15–17 s/根（约 2.4×，推断），与点 A（免前缀重放，压固定项）叠加后另计。

## 6. RNG 与状态性审计（蒸馏替换的硬约束）

- **trunk 评估路径零随机抽取**：`cnet2_np.py` 全文无 `np.random`／`random` 调用（已逐行查）；唯一的随机消费是 `act` 在 `sample_rng is not None` 时对**调用方传入的** Generator 做 `choice`（每被采样槽恰一次，消费量与 logits 数值无关）。批量版 `decode_b` 的 noise/temp 同理走外部流。
- 两条种子流都在 trunk 之外、stage0_teacher.py 内：信念流 `default_rng([seed_base+m, 0xC0])`（`belief_carbon`，每未知轨道恒两抽）、对手流 `default_rng(seed_base+m)`（根动作采样）。**结论行（点 A/B 同口径）：学生 trunk 只要保持「(x, aux, teacher, forced, sample_rng) → (labels, logits, value, u) 纯函数」形态，根集不变、两条种子流的消费点与量不变。**
- 可变状态仅两处，均不在 trunk 本体：① `CachePolicy.last_enc/last_sig`——以九字段全签名键控的性能缓存，`enc_for` 保证返回恰为 (x, teacher) 的编码，语义中性；学生侧保留或去掉都不改数值（去掉只变慢），但若保留须沿用同一签名纪律（R12 教训：签名漏字段曾致错帧编码 0.061544）。② `Feat`（信念/历史追踪）与 `BankruptcyGuard`（lostw 等，snapshot/restore 隔离）——属搜索侧状态，不在替换范围。
- `Policy.__init__` 的预计算（转置、im2col 索引、tattn 相对偏置表）皆为权重的纯函数，无跨调用状态。

## 7. 口径注记与未合项

1. **cnet2_search 版本差**：main HEAD 为 `c3222eea`，生产钉版为 `94f7fb64`。本表关于 CachePolicy/BankruptcyGuard 的行号与行为取自 main 版；`act` 的签名与语义在两版间未见差异（教师件 c6ade5ce 对其的用法仅为 `BankruptcyGuard／CachePolicy／feat_copy` 三名），但阶段②接入前须以 DGX 钉版包（k0_dist/pkg）原件复核 guard 内部调用形态，不许以 main 版推钉版。
2. 权重原件系本地 k0pkg 副本直读（md5 合钉版）；DGX 生产原件未在本阶段直读，阶段②开工清单中补机上直读。
3. `train_distill.py`（仓库内同名件，md5 `bbd5732f`）是旧 v53 两头 MLP 的蒸馏脚本，与本 trunk 无关，不作复用基底；本线 harness 为新写（见设计草案 §4）。

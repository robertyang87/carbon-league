# RECEIPT — KataGo 新探针 Part 3 构造回执（状态卡：构造完成·双臂齐备·待策略组验收；质量关未跑、场未开、无种子消耗；MARGIN=625.5 为 PROVISIONAL 提案、未验收不称注册值）

构造方：工程组子代理（2026-11-15）。注册文本：`~/workspace/katago-probe/PREREG_SEARCH.md` Part 3 节；仿射注册：`~/workspace/katago-probe2/RECEIPT_PART2B_AFFINE.md`（md5 `77a1906aaa8ed34bd1ed9690553cd997`）。

## §0 基底变更登记（先行事项）

- **装配基底＝A 形态**：`~/workspace/katago-probe2/modules/echo3_search_pack.py`，md5 `ea20a2714209e1a58b3e435b41d943bb`（Echo3 CONF 仅翻 SEARCH、原生墙钟 BUDGET=0.7、无 sims 钩子）。来源＝**策略组 2026-11-15 13:22 裁定**：B（sims 包 `8e6f7a65d3d91b363606e711eaea03c2`）永久关闭、「保留 sims 计数点」约束作废、对齐质量关改在 A 形态底座上跑。
- 本趟先前按 sims 基底建成的首版双臂（A 臂包 `1d587818a475c2e357ec9d842a6d5530`、B 臂包 `d26975d5e2f515a583ab4c4e027ba962`）**已作废**，整体存档于 `part3/work/superseded_b_base/`（含其 build_manifest），不作任何用途。本回执全部数值均为 A 基底重建后的结果。
- 变体构成（双臂同底座、仅叶不同，与注册 Part 3 同构）：A 基底包 ＋ 步限终局修正 ＋ MARGIN 提案值；A 臂＝仿射价值叶、B 臂＝原启发式叶（保留为不可达代码，上方有注释标明）。

## §1 五闸

### 闸一 量纲/定义独立复核 —— 过（构造侧）

- 仿射对自注册文本复核：**a＝3687.9177926610573、b＝−638.5907028300794**，与 Part 2B 回执 §1 入册文本逐字一致；拟合证据 `fit_affine_results.json` md5 `bccbf8ed28dc77d0cafe25cd15dc83b2`（同留出集 [32:64] 6,336 根、R² 0.9111、**残差 SD σ＝625.5417075364155**、残差中位 +101.4）。
- 解析锚独立复推成立：M_eq＝3000·z（main `cnet2_build` `e0051524`:390 `tanh(mg/3000.0)`），仿射对系其在留出集上的最小二乘推广，非笔误口径。
- 价值叶读法与 Part 2 管道同源：`feat_copy(fm)` → `observe(sim.obs(me))` → `encode_b(pol, stack_x([xm]), [T])` 取 ctx → `z = ctx @ value.weightT[:,0] + value.bias[0]` → `M_eq = a·z + b`；教师钉 T＝cfg[TEACHER]＝2。接线数值验见闸四 T2（leaf 与独立复算的 a·z+b 在 1e-6 内逐根相等）。

### 闸二 机制分布 —— 未跑（登记阻塞，待另派）

- 深度分布待对齐质量关趟在 **A 形态底座**上补测。对齐质量关一条登记为：**阻塞待策略组重注册裁定（sims 终值无从锁定——Part 1 标定阶梯停报：五档同一计算、单步 p95 14.32s 破 3s 硬顶）**；13:22 裁定后质量关改 A 形态底座、同线（改子根 ΔU 负占比 ≤0.5、双分支同仪器）不变，仍待派场前两闸（Part 1 判读＋本构造验收）齐备后另派。本趟未跑质量关、未开任何场。

### 闸三 包与基底同一性机械 diff —— 过（27/27 断言）

审计脚本 `part3/scripts/audit_variant.py`（程序化断言，非目测）：

- 双臂包骨架（`_SRC = ` 行以外）与基底 A 包逐行全同；CONF 字典与基底**逐键全等**，点名值合注册口径：TEACHER=2、GUARD=True、GUARD_NARROW=True、ROLL="self"、ABORT_DL=0.95、TEACHER_LATE=2、LATE_STEP=240、SEARCH=True、BUDGET=0.7、HARD=0.85、MODE="sweep"；**无 CLOCK/SIMS/FIX_TRUNC 键**（A 形态本无 sims 键）。
- 权重载荷逐字节同基底；解码权重 md5 `444cc54c0d88da5656ff4c7eef1ec8eb`；内嵌三模块（cnet2_feat `2a45b08b`／cnet2_np `1433b5ac`／fastsim `61859708`）逐字节同基底。
- 内嵌 DEFAULTS 与基底仅 MARGIN 一键不同（20.0 → 625.5 提案值），GUARD_CASH=40 保持。
- **构造性复现**：对基底内嵌搜索源重放登记编辑集（`build_variant.build`）逐字节复现双臂变体内嵌源。
- **逐行分类**（difflib 全量比对，类别集＝登记编辑块）：A 臂 {terminal 35、leaf 20、MARGIN 2、leaf-wiring 6}；B 臂 {terminal 35、leaf 2、MARGIN 2、leaf-wiring 6}；**other 类＝0**。
- **A↔B 单变量**：A 臂内嵌源删去价值叶块后逐字节等于 B 臂内嵌源；CONF 与其余内嵌模块双臂全同。
- 双臂均未引入任何 sims 计数钩子（`leaf_evals` 不在源中；基底本无、约束已作废，此为登记性核对）。

### 闸四 边界用例（终局判定）实测 —— 过（48/48）

脚本 `part3/scripts/test_leaf.py`，结果 `part3/data/leaf_tests.json`（md5 `66aa6dc832725b3b2a786ef55bb0af89`）。根集为合成边界局 `boundary_cases.jsonl`（SYNTH-BND777，步 296–300；该局为同权重确定性双代理的镜像局，cash[0]≡cash[1]，故 T1/T3/T6 的现金差恒 0——非退化证据由 T7 注入非对称现金补足）。

| 用例 | 内容 | 结果 |
|---|---|---|
| T1 | 步 300 根、双席：leaf 与 _terminal | 双臂双席 leaf＝精确现金差（0.0000，镜像局）、_terminal＝True；基底 A 包原 leaf 对照同为 0.0000（lft=0＋镜像态，启发式项相消） |
| T2 | 步 297/298/299、双席：非终局判定＋A 臂接线 | _terminal＝False（双臂）；A 臂 leaf 与独立复算 a·z+b 逐根相等（z＝−0.619310/−0.619527/−0.605179，leaf＝−2922.56/−2923.35/−2870.44）；B 臂 leaf 有限 |
| T3 | 步 297 态以原包双代理实打到 300 | 双臂双席 leaf＝终局精确现金差（0.0000，镜像局） |
| T4 | 步 296 态抹除 side 1（灭绝变异） | _extinct(1)＝True、终局；F01 值＝存活方现金 − pen（pen＝296−300−1＝−5）＝**6455.0300**，双席反对称；灭绝方现金 +5000 后 leaf 不变（与灭绝方现金无关） |
| T5 | 灭绝 ＋ step_count＝300 | **F01 优先于步限现金差**：leaf＝6451.0300（现金差口径将为 0.0000） |
| T6 | 存活态 step_count 置 300/299 | 300 → 终局、leaf＝精确现金差；299 → 非终局 |
| T7 | **非退化步限验**：存活态注入 cash＝[5000, 3200] | step_count＝300 时 leaf＝**+1800.0000/−1800.0000 精确**；299 时非终局；步 299 注入后实打一步到 300，leaf＝终局精确差 **1806.6500**（双臂同值） |

终局优先级（构造决定，待验收裁量）：**灭绝（F01）先于步限**——双条件同真时按 F01 计（pen＝step_count−episodeSteps−1、存活方取状态自带现金；本包无 prev_cash 通道，与独立模块 R02 版口径一致）；步限分支仅当双席皆未灭绝时取精确现金差 `cash[me] − cash[1−me]`（按席定向、与对局计分同口径；探针②/Part 2 续局已实测「300 步局末现金与记录 scores 逐字合」）。

### 闸五 单步成本 —— 未测（待质量关趟）

- 待对齐质量关趟在 A 形态底座上按 1.5 核口径补测。本机冒烟（节流伪影口径，仅登记）：出厂 CONF 下双臂 8/8 根 searched 但 8/8 shallow（Part 1 回执 §4 同款本地伪影），不作成本判读。

## §2 MARGIN 重钉提案（注册前置 b）

- **提案值：MARGIN ＝ 625.5（PROVISIONAL，随变体搭载；未验收前不称注册值）。**
- 推导：现行 20.0 系启发式叶现金标度下的旧值，与仿射叶的 margin 标度脱钩。仿射拟合残差 SD σ＝625.5417（闸一）即「价值叶对真 margin 的单根噪声标度」——取 **1.0×σ**、按 0.1 取整得 625.5。叶差噪声口径交叉核：双叶独立时 σ_Δ＝√2σ≈884.6、同源相关 ρ=0.5 时 σ_Δ＝σ＝625.5；625.5 落于该区间下端（双臂同底座同仪器、叶差相关性偏高，下端为合理锚）。U 尺度交叉核：625.5/3000≈0.2085 个 tanh 单位、sigmoid 最陡处 ΔU≈0.078，与 Part 2 质量屏的 ΔU 量级相容。
- 灵敏度表（同推导、不同倍率）：0.5σ＝312.8／0.75σ＝469.1／**1.0σ＝625.5（提案）**／√2σ＝884.6。
- 尾部形态处理（Part 2B 附注）：Part 2 续局均值 ΔU −0.01159 由尾部拉动（单根 −0.5786 占负和 59%；≤−0.01 的负根 10 个、≤−0.1 仅 1 个）。MARGIN 的职能是噪声准入（管中位 ΔU≈0 的那群根的误触发），深尾根的偏离（≈0.58 个 U 单位 ≈ 数千 margin 单位）**非任何合理 MARGIN 可挡**，归对齐质量关（负占比线）管辖——故提案不按尾部放大倍率，维持 1.0σ 锚；若验收认为尾部应由 MARGIN 侧分担，提高倍率至 √2σ 的代价是中段根触发率系统性下降，请策略组在灵敏度表内裁定。

## §3 基底差异发现（验收必读：z 的双管道系统差）

叶级实测中发现并已定位（2×2×2 交叉矩阵：Feat 特征 × encode 实现 × 权重）：

- 同一状态下，**包内 encode（内嵌 cnet2_np `1433b5ac`，部署形态）与 Part 2 管道 encode（独立模块 `d8c15ad4`）给出系统性不同的 z**：边界三态（步 297/298/299）Δz＝z_pack − z_fp＝**−0.3585/−0.3891/−0.4052**，折 M_eq 为 **−1322.3/−1435.0/−1494.5**。
- 定位：Feat 特征 xm 双管道逐字段全同（max|Δ|＝0.0）；权重差异（int8 解码 vs fp32）仅贡献 Δz≤0.0025；**差异全部来自 encode 实现版本**（包内嵌 np 为 v12 期快照、独立模块为现行版）。
- 含义：仿射对 (a, b) 系在 **fp 管道 z** 上拟合入册；变体叶按注册读法「z 走包内路径」取的是 **pack z**，故变体实际计算的 M_eq 相对拟合标定存在系统平移（本三态约 −1300 至 −1500，约为 −2.1σ 至 −2.4σ）。**接线本身精确**（leaf ≡ a·z_pack＋b，1e-6 内，T2）；此为底座版本事实、非构造错误。处置选项归策略组验收裁量：(i) 按现状接受（价值头以部署形态读数＋注册仿射映射）；(ii) 在 pack z 上重拟合仿射对（留出集同口径重跑）；(iii) 连带重议 MARGIN（σ 系 fp 空间量，若重拟合则以 pack 空间残差 SD 重推）。本回执 MARGIN 提案（§2）按注册 σ 推导、未自行改基。

## §4 构造自检汇总

- 叶级实测 48/48（闸四，含接线验与非退化步限验）。
- 加载冒烟（`part3/scripts/test_loads_smoke.py`，结果 `part3/data/smoke_results.json` md5 `521718307948d32ae99ce2724a5dae2b`）：orig／基底 A／双臂四包 py_compile＋import＋工厂构造＋合成根喂入全过，errors＝0。
- SEARCH 接合（合成冒烟、非关）：出厂 CONF 本机 8/8 shallow（墙钟伪影，见闸五）；诊断口径（测试侧覆盖 BUDGET=10/HARD=30、非出厂字节行为）下双臂各 2/8 根到达叶评估（branches＝72、depth_sum＝18、errors＝0），接合成立；该口径下 A↔B 改子 0/8（合成根、仅 2 根接合，如实登记、不作判读）。

## §5 工件 md5 表

| 工件 | md5 |
|---|---|
| 基底 A 包 modules/echo3_search_pack.py | `ea20a2714209e1a58b3e435b41d943bb` |
| **变体 A 臂包 part3/modules/echo3_part3_a_pack.py** | **`d8d7ebc82a564c9510e9aeae015ce4cd`**（内嵌 search `1126c7a3528294f588212ceac4af7f23`） |
| **变体 B 臂包 part3/modules/echo3_part3_b_pack.py** | **`fbe9beffd1683ddd7995a1282aafc572`**（内嵌 search `4338de33adf91c7498b7c8a1bf04c709`） |
| 解码权重（双臂同基底） | `444cc54c0d88da5656ff4c7eef1ec8eb` |
| scripts/build_variant.py | `767a955750a9911d01c62e3a0872ceb2` |
| scripts/audit_variant.py | `a1ab0dc83f6382bec7a92d2bacdde079` |
| scripts/test_leaf.py | `c6b45c46af6e22999af164ddbdcf3e4c` |
| scripts/test_loads_smoke.py | `a3f2fe948532cec33929b9612507641f` |
| scripts/extract_src.py | `f6f66f7c2569eec7d37631d66a5c9ee2` |
| data/leaf_tests.json | `66aa6dc832725b3b2a786ef55bb0af89` |
| data/smoke_results.json | `521718307948d32ae99ce2724a5dae2b` |
| work/build_manifest.json | `1132b87ab2ed0a3696bda62edb8d517e` |
| 作废存档（sims 基底首版双臂） | A `1d587818a475c2e357ec9d842a6d5530`／B `d26975d5e2f515a583ab4c4e027ba962`，存 `part3/work/superseded_b_base/` |

## §6 限定

1. **质量关未跑、场未开、无种子消耗**；Part 3 开场双合（Part 1 判读不在「上界<0」格 ＋ 本构造验收过）未齐，本件仅为构造验收件。
2. **未加 rollout 终局冻结**（R02 freezing）：前置 (a) 按评估侧修正实现（叶在终局态返回真值）；rollout 动力学不冻结属另一变更、超出 (a) 字面，未自行扩围。残余界：灭绝为吸收态、rollout 末态恒终局、叶按真值计分；F01 的 pen 以末态 step_count 计、存活方现金为末态现金，与严格真值之差由 rollout 深度 ≤5 步界定——留验收裁量。
3. MARGIN＝625.5 为 PROVISIONAL 提案（§2），且其 σ 基为 fp 管道空间（§3 (iii)）。
4. 底座为墙钟 A 形态（BUDGET=0.7）；sims 计数纪律随 B 关闭一并退出本线（13:22 裁定）。
5. 全部产物为诊断件：不入 main、不上天梯、不动冻结主体 `4a3b5f1b` 与 Part 2B 变体 `7f283b15e64f1ee0337e832288eea419`。

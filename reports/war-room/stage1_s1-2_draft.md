# S1-2 修订版设计草案（教师延续闸）

- 状态：DRAFT，只供作战室裁决；不改仓库、不改任务书文件。
- 依据：13:42 策略组纠正定版（所有者，五点全收）＋ 正式 gen-1 剩余工作定版。
- 核验基线：main HEAD `aa8055b0`（读回 2026-11-13 13:49 前后）；下文全部文件 md5 均在该 HEAD 下读回。
- 约束重申：教师闸形态仍是 Studio 位置级一致性评测；400 局比赛形态已取消，不得再提；
  ΔU 阈值（≥ +0.02 且 95% CI 下界 > 0）已定死，本草案不改阈值，只改口径与前置项。

## R0. 口径纠正（先修三个引用错误，再谈设计）

1. **“§8.4 ΔU 延续闸”在现行设计文档中不存在。** `reports/katago_architecture_design.md`
   只有 §0–§8，§8 是“待决事项与回填口径”（且注明为截断后重构）；全文检索 `8.4` 在
   `katago_architecture_design.md`、`plan_katago.md`、`CARBONFORMER_RESPONSE_20261113.md`
   中均为 0 处。任务书的 §8.4 引用无法落到权威源，作废弃引用处理，不得再引用。
2. **S1-2 与 §3.4 最终确认是两道不同的闸，任务书把二者拼在了一起。**
   - 教师延续闸＝设计文档 §2 修订口径 L143：“独立质量闸（新场景/新 RNG、真实续弈到
     终局、按根局面成组验 ΔU），不过闸不产正式标签”。这是 S1-2 的正确锚点：
     场景级、按根局面成组、真实续弈到终局。
   - §3.4 第 4 项“最终确认”＝冻结候选后的 200 局整场：熟悉对手新种子确认
     （Carago g1＋未读种子块）与未见对手泛化确认（备忘冻结的未见对手＋另一未读
     种子块）各 200 局、分别判读、两职都过才算通过。这是**蒸馏后冻结裸网**的闸，
     属于 gen-1 序列末尾的“冻结裸网 ΔU 验证”，不是 S1-2。
   - 任务书 S1-2 括号里的“沿用 R2 M03 口径：熟悉对手新种子 200 局＋未见对手泛化
     各 200 局”是类别错误：200 局整场协议无法套在场景级延续闸上。
     本修订：**S1-2 只继承 §3.4 的阈值数值（ΔU ≥ +0.02 且配对 95% CI 下界 > 0，
     两职分别判读的精神——此处两职指 q 收益闸与 ΔU 闸两道，见下），不继承
     200 局整场协议**；200 局整场协议移交给蒸馏后的冻结裸网验证（S1-2b 仅作
     诊断性保留，见 (c) 末尾）。
3. **“54.7 小时”不得再裸引。** 按纠正第④点：54.7 小时只对应约 3.55 万局面
   （35,282 位置 × 5.55 秒/局面 ÷ 3600 ≈ 54.4 小时）；ds_gen1 的 119,600 局面
   串行全标是约 184.4 小时（Studio）。此后一切教师成本必须写明：标注根数、
   单/双席位、抽样率、并行度与机器（本草案 (c) 的成本表即按此格式）。

## 已核实的代码事实（结论＋依据）

| # | 结论 | 依据 |
|---|---|---|
| F1 | 教师当前 U 来源确为旧 tanh 价值头，未接新三头 U 叶 | `code/stage0_teacher.py`（md5 `d17cf360`，7,695 B）：`value_to_u(v)` L38–45（M = 3000·atanh(clip(v, ±0.999))，U = 0.5W + 0.5·sigmoid(M/1000)）；`_leaf_u` L73–79：`_, _, v = pol.act(x, feat.aux, teacher=teacher)` 后 `value_to_u(v)[0]`；文件头 docstring 自述“leaf value comes from the existing value head (tanh(M/3000))” |
| F2 | 新三头 U 叶接口已就绪，教师没调它 | `code/cnet2_model.py`（md5 `3bb5857f`，19,720 B）：旧价值头 `self.value` L160（仍作 tanh 用，L219/285/311）；新头 L161–164：`utility = nn.Linear(D,1)`（E[U]，sigmoid→[0,1]）、`wdl`、`margin`（预测 M/1000）；`heads(ctx)` L208–212 返回 `(u, wdl, margin_k)`；`forward_heads(b)` L214–221 返回 `(logits, value, aux, utility, wdl, margin_k)` |
| F3 | numpy 侧 `Policy.act` 只算旧价值头，不碰新头 | `code/cnet2_np.py`（md5 `d95f6384`，14,162 B）：`Policy.act` L165–207：`value = tanh(ctx @ w["value.weightT"][:,0] + w["value.bias"][0])`，返回 `(labels, logits, value)` 三元组；`utility.*`/`wdl.*`/`margin.*` 全程未出现 |
| F4 | heads v2 产物含全部 6 个新头键，可直接接线 | DGX 直读（只读，2026-11-13 13:49）：`/home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads_v2.npz`，13,973,080 B，md5 `06a03d9cfeec239af3065a49fefec382`；187 个键，含 `utility.weight/bias`、`wdl.weight/bias`、`margin.weight/bias`；旧 `value.weight` 仍在；meta：cfg C=128/NB=8/D=128/NH=4/L=2/E=32/G=8/TA=4，epoch 19，val all 0.8697 / loss 0.3436，params 3,480,130。终值 val 口径（任务书＋DGX 12:28 直读）：val_l_u 0.00817 / val_wdl_acc 0.9933 / val_u_calib_pred_binned 0.04224；ep2 最优 val_l_u 0.00309 / wdl 1.0。注意：npz meta 不含 val_l_u，该数只在训练日志里 |
| F5 | 三头训练口径：只训新头、骨干冻结；margin MSE（F10）；F12 校准按预测 u 分箱 | `code/stage0_train_heads.py`（md5 `14ba9166`，14,138 B）：损失 `l_u + 0.25·l_wdl + 0.25·l_m`；unusable 行排除出价值族损失；`evaluate()` 报 `val_u_calib_pred_binned` 等 |
| F6 | 教师终局叶用现金差近似，未用 F01/F04 计分 | `stage0_teacher.py` docstring 自述“terminal leaves use the cash difference as the margin approximation (extinction penalty not applied)”；`evaluate_root` 终局分支：`mg = cash[me] − cash[1−me]` 后直接算 U。而 F01/F04 正确口径（`code/eval_v53.py` md5 `aa4c5bb6`，L181/L493–494 已核实）：灭绝方得分 `steps_played − 300 − 1`；幸存方取**终局步之前**的现金（`pre_cash`，L181 取终局步前状态）；无灭绝超时才取终局现金 |
| F7 | 教师 H=2 续弈与候选生成全程 guards off；生产侧 GUARD=True | 教师 `evaluate_root` 经 `pol.act` 直接走子，无 hook、无 guard（`stage0_teacher.py` 全文无 guard 字样）；`stage0_guided_match.py`（md5 `fc4bb32`）docstring：“Both arms use the SAME raw Policy (m3s2r1), guards off”；生产侧 `code/producer/selfplay_gen.py`（md5 `95d2e29f`）CONF L19/L160：`GUARD=True, GUARD_NARROW=True`；考试 exp CONF 同值（作战室冻结口径）。即：教师在 guard-off 下算出的 Q，要蒸馏给 guard-on 下走子的生产策略——正是纠正第②点指出的错配 |
| F8 | guard 是有状态的破产守卫，须按局维护状态 | `code/cnet2_search.py`（md5 `10a0674e`，115,279 B）：`bankrupt_guard(x, aux, y, lg0)` L958（闭包，依赖 `st` 的 lostw/prevw/base_fix 与 cfg LOWW=2、GUARD_NARROW、GUARD_CASH=40.0）；DEFAULTS L38：`GUARD=False, GUARD_NARROW=True, GUARD_CASH=40.0`（生产 CONF 覆盖 GUARD=True）；应用点 L875 `if cfg["GUARD"]: y = bankrupt_guard(...)`；重解码路径用 `pol.enc_for`＋`decode_b`（L235）。`enc_for` 在 `CachePolicy(NP.Policy)` L49–86（`make_search_agent` L509 用 CachePolicy）；教师现用裸 `NP.Policy`（`stage0_measure_teacher.py` L46），无 `enc_for` |
| F9 | 生产 CONF 下除 guard 外的规则编辑均不生效 | `cnet2_search.py` 的 MACRO/LATE_PLANT/MASTER_NO_PLANTER 编辑块均由 cfg 键门控，而生产 13 键模板与考试 CONF 均不含这些键。故生产续弈策略＝裸网贪心解码＋破产守卫（仅此两项），不是“考试 agent 全套规则层” |
| F10 | 教师成本两个实测数不一致，须重标 | 任务书：DGX 28.2 秒/局面、Studio 5.55 秒/局面；DGX n=1 guided 试点：teacher 17.8 秒/调用。两者口径疑似不同（是否含 sim 开销、K/M 设置、机器负载均未注明）。本草案一切工时以 Stage A 实测重标为准，不直接采信任一旧数作预算 |

## (a) U 叶接线前置项（S1-2 开跑的硬前置，缺一不可）

### A1. 新 U 叶接入教师（纠正第①点）

- **改 `code/cnet2_np.py` `Policy.act`（L165–207）**：在现有 `value = tanh(...)` 行旁
  加一行 `u = sigmoid(ctx @ w["utility.weightT"][:, 0] + w["utility.bias"][0])`
  （`utility.weight` 为 [1, D] 二维，`__init__` L60–62 已为其建 `.weightT`；
  F4 已验证 heads v2 npz 含该键）。返回方式：加可选参数 `want_u=False`，
  `want_u=True` 时返回四元组 `(labels, logits, value, u)`，默认三元组不变——
  不改现有调用方（search agent、guided_match、measure_teacher 均不受影响）。
  成本：一次 [D] 向量点积＋sigmoid，无额外 trunk 前向。
- **改 `code/stage0_teacher.py` `_leaf_u`（L73–79）**：改调 `pol.act(...,
  want_u=True)` 取 `u` 直用，删 `value_to_u` 调用。`value_to_u`（L38–45）
  在全文件仅 `_leaf_u` 一处使用——接线提交里一并删除，防止误复用；
  文件头 docstring 同步改写（叶值来源改为新 U 叶）。
- **验收检查（接线后、开闸前必须全过，证据口径）**：
  1. 奇偶校验：同一批输入（≥20 个局面，Studio 一致性评测同协议取），numpy
     `act(want_u=True)` 的 `u` 与 torch `forward_heads` 的 `utility` 逐项
     对齐（容差 1e-5），并与旧 `value_to_u(v)` 的输出**不同**（证明走的是新头，
     不是旧值换皮）；记录最大偏差与样本数。
  2. 缺键大声失败：在不含 utility 键的旧权重上调 `want_u=True` 必须抛明确
     错误（KeyError 带键名），不得静默回退到旧价值头——静默回退即重犯纠正
     第①点指出的归因错误。
  3. 范围断言：`_leaf_u` 返回的 u 恒在 [0, 1]（sigmoid 值域），越界即停。
  4. 教师版本戳：接线提交的 commit＋三文件 md5（stage0_teacher.py /
     cnet2_np.py / cnet2_search.py）写入 S1-2 预注册头；闸跑在旧教师代码上
     的数字一律作废。

### A2. 终局叶改 F01/F04 计分（纠正“正确终局”）

- **改 `code/stage0_teacher.py` `evaluate_root` 终局分支**：现行 `mg =
  cash[me] − cash[1−me]` 改为与 `eval_v53.py` L493–494 逐字同口径：
  灭绝方得分 `steps_played − 300 − 1`；幸存方取终局步**之前**的现金；
  无灭绝超时取终局现金；再由 margin 算 W 与 U（U = 0.5W + 0.5·sigmoid(M/1000)，
  与 `value_to_u` 的 U 公式同式、不同 M 来源）。
- **实现要点**：H=2 循环与根步进前必须像 `eval_v53.py` L181 那样先取
  `pre = clone.get_state()`（终局步前状态），否则幸存方现金无从取得；
  灭绝判定沿用现有的 `_extinct_state`（L62，自述与 `rl2_ppo_rollout._extinct`
  同规则）——接线时逐项核对镜像是否精确（核对项，不确定的写未核实）。
- **验收检查**：手算 3 个终局算例（灭绝胜／灭绝负／超时）与 `eval_v53.py`
  同输入逐项对数；另在 20 个真实终局局面（从 m3s2 语料取）上教师终局 U
  与 eval_v53 终局 U 逐项一致（容差 1e-9，纯算术，不应有浮点外差异）。

### A3. guard 配置统一（纠正第②点）

- **统一配置（生产口径，定死）**：`GUARD=True, GUARD_NARROW=True,
  GUARD_CASH=40.0, LOWW=2`。即生产 CONF 与考试 CONF 的既有值，不新发明。
  适用范围：教师候选生成、H=2 续弈、贪心对照臂、q 收益闸与 ΔU 闸的真实
  续弈、以及 guided_match 诊断——**同一配置贯穿始终**，不得一处开一处关。
- **实现方式**：把 `cnet2_search.py` L958 的 `bankrupt_guard` 从
  `make_search_agent` 闭包抽成可复用的有状态守卫（例如 `BankruptcyGuard`
  类，持 `lostw/prevw/base_fix` 状态＋cfg），`make_search_agent` 改调它
  （行为不变，奇偶校验：同一批局面、同一 cfg，抽取前后守卫输出逐项一致）；
  `stage0_teacher.py` 导入它。教师须把 `pol` 从裸 `NP.Policy` 换成
  `CachePolicy`（`cnet2_search.py` L49，同数值，带 `enc_for`），因为守卫的
  重解码路径要 `pol.enc_for`（F8）。
- **状态分支规则（关键，不可省）**：守卫状态是博弈状态的一部分——根局面
  用真实对局历史算出的守卫状态快照；每个候选臂／每个场景分支各持一份
  快照副本，分支内随步演进、分支间不串状态；新根局面重新快照。不得用
  全局单例守卫跑多分支（会把一臂的 lostw 漏进另一臂）。
- **不走的路（已否决，记原因）**：不把教师续弈改道经 `make_search_agent`
  （SEARCH=False）——那会连带吃进 MACRO/LATE_PLANT/MASTER_NO_PLANTER
  等规则编辑，改变的就不只是 guard（F9：这些编辑在生产 CONF 下本就不生效，
  改道等于悄悄改了续弈策略）。
- **影响面说明**：守卫只在破产风险态触发（已有丢子＋现金 < 40），大多数
  局面不动作——统一的预期数值影响小，但配置必须声明统一，因为被认证的
  ΔU 是生产 ΔU，不是 guard-off 的 ΔU。
- **验收检查**：守卫抽取前后奇偶一致；教师 guard-on/off 对照（同批 50 个
  根局面，q 分布与 Q 均值差异列表，差异只报不判）；S1-2 预注册头写明
  guard 配置四键。

### A1–A3 完成标准（缺一不可，否则 S1-2 不得开跑）

1. A1 奇偶校验通过（numpy u ＝ torch utility，容差 1e-5；与旧值不同）。
2. A2 终局算例与 eval_v53 逐项一致。
3. A3 守卫抽取奇偶一致；教师 guard-on 模式可跑通（50 根局面冒烟）。
4. 三文件改动入 main、md5 读回；预注册头写明权重 md5（heads v2
   `06a03d9c…`）＋三文件 md5＋guard 配置＋种子块（见 (c)）。

## (b) q 收益闸预注册草案（小规模、独立；阈值建议，作战室定）

> 定位：纠正定版序列的第二步，排在 U 叶接线之后、稀疏标注与蒸馏之前。
> 评的是**实际用于训练的 q 分布**（`evaluate_root` 输出的软目标 q），不是
> 一致率/熵/最佳动作——那三项只是诊断列。

- **场景集来源与数量**：m3s2 语料 2026-11-12 新增的 34 局（清单 118→152；
  ds_m3s2 的 35,282 样本来自原 118 局）——**前提核查项**：采样前按对局 id
  逐项确认这 34 局不在 ds_m3s2 的对局清单里（未核实前不得采样；若有重叠，
  改用重叠外的局，不改总数）。34 局对半分：q 收益闸用其中 17 局，
  ΔU 闸用另外 17 局，**两闸对局集不交**，不许位置级混用。
  位置采样：步数 40–240，按阶段分层（40–120 / 120–200 / 200–240），
  每局每层随机抽 1 个局面；**N_q = 150 个局面**（小规模，见成本）。
  种子：采样 RNG 种子预注册写死（建议 `995300` 段，与一致性评测的 995200+
  段不重）。
- **评的 q 分布定义（训练用 q）**：每局面 `evaluate_root` 输出的 q
  （q ∝ P_A^0.5 · exp((Q−maxQ)/0.05)，K≤16、M=8、H=2、新 U 叶、统一 guard，
  teacher=1）。同时记每臂 Q、greedy 臂、熵。
- **报告指标（全报，不挑）**：
  1. **Q 收益（主指标）**：mean[Q(argmax q) − Q(greedy)]，U 单位。
     即：若蒸馏策略照 q 的 argmax 走，相对贪心每步期望多拿多少 U。
  2. **严格改进率**：P(argmax q ≠ greedy 且 Q-gap > 0)。
  3. q 熵均值＋有效臂数 exp(熵)（诊断）。
  4. 一致率 agree_greedy（诊断，与 S1-1 同口径）。
  5. **独立锚点（防教师自评）**：30 个局面子集（从 150 中随机抽，种子写死），
     q-argmax 臂与 greedy 臂各从该局面真实续弈到终局（统一 guard、对手＝
     裸网贪心、同种子配对、F01/F04 计分），报实现 ΔU 点估计＋95% CI
     （n=30，CI 必宽，只作符号检查，不作闸）。
- **阈值建议（作战室定，本草案只建议）**：
  - 通过：Q 收益均值 ≥ **+0.01 U**，且严格改进率 ≥ **25%**，且独立锚点
    实现 ΔU 点估计 > 0。
  - 不通过：Q 收益均值 ≤ 0，或严格改进率 < 10%（教师无收益或有害，停）。
  - 不定：其间——场景集翻倍（同协议、种子段顺延一次）再判一次；仍不定
    则按不通过处理，不许无限扩样。
  - 建议理由：+0.01 是 ΔU 闸阈值（+0.02）的一半——蒸馏只能拿到教师优势
    的一部分（策略容量、联合 CE 拟合损耗），q 收益是上限估计；25% 保证
    收益是普遍的，不是由少数局面驱动；独立锚点防 Q 本身失真（教师给自己
    的 Q 打高分）。
  - 换算参考（W 不变时）：+0.01 U ≈ +80 终局分差点（dU/dM ≈ 1.25e-4/点，
    M≈0 处）；W 翻转一局即 +0.5 U，故 +0.01 也可来自约 2% 的局面翻转胜负。
- **成本（纠正后算法：150 根局面、单席位、抽样率 150/约10,000 位置池、
  并行度写实测）**：150 次 `evaluate_root` 调用；Studio 串行 150×5.55 秒
  ≈ 14 分钟（并行 22 worker 约 1 分钟）；DGX 串行 150×28.2 秒 ≈ 70 分钟
  （并行 64 worker 约 2 分钟；28.2 待 Stage A 重标）。独立锚点 30 局面：
  guided 臂 30×~160 步×单调用成本，并行后 Studio 约 20 分钟、DGX 约
  35 分钟（28.2 口径；重标后更新）。
- **与 S1-1 的关系**：S1-1 的一致性评测（agreement/entropy/slot-match）
  是教师质量的**诊断**，q 收益闸是教师价值的**决策**——两闸用的同一
  教师、不同场景集（S1-1 用 40 局/229 位置、种子 995200+；q 闸用上述
  17 局新对局），数字不混用。

## (c) 修订版 ΔU 闸预注册（S1-2 主闸；阈值沿用已定值）

> 锚点：设计文档 §2 L143 独立质量闸（新场景/新 RNG、真实续弈到终局、
> 按根局面成组验 ΔU）。200 局整场协议属于 §3.4 最终确认（蒸馏后冻结
> 裸网），不在本闸内（见 R0-2）。

- **场景集来源与数量**：q 收益闸之外的另外 17 局新对局（同 34 局池，
  两闸对局集不交；前提核查同 (b)）。位置采样：步数 40–240 按阶段分层
  （40–120 / 120–200 / 200–240），另加开局层（步数 0–40，占 10%）
  以覆盖开局教师价值；每局每层随机抽。**两阶段定 N**：
  - Stage A 试点：N = 30（分层抽），测两件事——① 配对 ΔU 的样本标准差
    σ̂；② DGX 与 Studio 上新 U 叶 `evaluate_root` 的实测单调用成本
    （各 20 局面，写明机器与并行度）。
  - Stage B 定 N：按预注册公式锁定 N = ceil((140·σ̂)²)，即单侧 α=0.025、
    功效 80%、检出 ΔU=+0.02 的样本量（(1.96+0.84)·σ/0.02）；下限 N ≥ 150、
    上限 N ≤ 600。若公式 N 超上限，跑上限并**明文报告在该 N 下的实际功效**
    （σ=0.25 时 N=600 的功效仅约 50%——功效不足只会让真效应更难过闸，
    不会造成假过闸，方向是 fail-safe）；若公式 N 低于下限，跑下限。
  - 种子：采样 RNG 与续弈 RNG 种子块预注册写死，与 q 闸、S1-1 均不重。
- **双方走子方式（预注册，跑前冻结）**：
  - Guided 臂：每步调 `evaluate_root`（K=16、M=8、H=2、seed_base 由
    局面 id 派生、teacher=1、新 U 叶、统一 guard），走 argmax-q 臂。
  - Greedy 臂：同权重裸网贪心（`pol.act`，teacher=1、统一 guard）。
  - 对手：裸网贪心（同权重、统一 guard），两臂同一对手、同一对手种子
    （按根局面＋对手种子配对；对手身份即生产自博弈设定）。
  - 终局计分：F01/F04（A2 口径），U = 0.5W + 0.5·sigmoid(M/1000)。
  - 分组统计：ΔU 按根局面成组（每局面一对 guided/greedy 续弈差值），
    再平均；95% CI 用配对 t 区间，bootstrap（10,000 重采样、百分位法）
    作敏感性检查并行报告、不得替代 t 区间。
- **通过阈值（已定死，抄录不改）**：配对 ΔU 点估计 ≥ **+0.02** 且配对
  95% CI 下界 > 0。不通过即不产正式标签（设计文档 §2 原话），不许事后
  在均分/胜率/+1 SE 之间挑口径改判（§3.4 禁令同样适用）。
- **成本表（纠正后算法；单调用成本以 Stage A 实测为准，下表用旧数作估计带）**

  | 项目 | 根数/规模 | 串行工时 | 并行墙钟（估计） |
  |---|---|---|---|
  | Stage A 试点（30 局面） | 30 根、单席位、抽样率 30/约10,000 | Studio 30×160×5.55 秒 ≈ 7.4 小时·核；DGX 30×160×28.2 秒 ≈ 37.6 小时·核 | Studio 22 worker ≈ 20 分钟；DGX 64 worker ≈ 35 分钟 |
  | Stage B（N=400） | 400 根、单席位 | Studio ≈ 99 小时·核；DGX ≈ 502 小时·核 | Studio ≈ 4.5 小时；DGX ≈ 7.8 小时 |
  | Stage B（N=600，上限） | 600 根、单席位 | Studio ≈ 148 小时·核；DGX ≈ 754 小时·核 | Studio ≈ 6.7 小时；DGX ≈ 11.7 小时 |
  | 全量标注对照（不开） | 35,282 位置 | Studio 串行 ≈ 54.4 小时；DGX 串行 ≈ 276 小时 | Studio 22 worker ≈ 2.5 小时；DGX 64 worker ≈ 4.3 小时 |
  | 全量标注对照（不开） | 119,600 位置（ds_gen1） | Studio 串行 ≈ 184.4 小时；DGX 串行 ≈ 937 小时 | Studio 22 worker ≈ 8.4 小时；DGX 64 worker ≈ 14.6 小时 |

  注：每局面 guided 续弈按平均剩余 160 步计（采样步 40–240，剩余 60–260，
  均值 160）；greedy 臂与对手走子为裸网毫秒级，可忽略。机器选择由
  Stage A 实测单调用成本定：现行旧数下 Studio 单位吞吐约 1.7× DGX
  （22/5.55 ≈ 3.96 调用/秒 vs 64/28.2 ≈ 2.27 调用/秒），但 DGX 28.2
  与试点 17.8 不一致，重标后以实测为准，不许先入为主定机器。
- **预注册头（跑前必须写死并入仓）**：权重文件＋md5（heads v2
  `06a03d9c…`）、三文件 md5（stage0_teacher.py / cnet2_np.py /
  cnet2_search.py，接线后）、guard 四键、对手（裸网同权重＋md5）、
  场景对局 id 清单（17 局）＋采样种子、续弈种子块、N 公式与 Stage A
  实测 σ̂ 与单调用成本。未预注册不得开跑；改阈值或改对照即本次确认
  作废（§3.4 同款纪律）。
- **验收**：ΔU 点估计＋95% CI（t 区间＋bootstrap 敏感性）＋场景清单
  ＋成本实测（单调用成本、并行度、墙钟）＋guard 触发统计（守卫实际
  动作次数占比，诊断列）。
- **S1-2b（支撑性全场诊断，条件触发、默认关）**：`stage0_guided_match.py`
  200 局整场（对手＝Echo2 固定考试 agent，guided vs greedy，guard-on
  模式——须先给该驱动加 guard-on 开关并改 docstring，当前 docstring
  写的是 guards off）。仅当 S1-2a 通过后才跑，**只作诊断、不作通过/
  不通过判据**（整场 ΔU 混杂开局与局级方差，单位信息成本约为场景闸
  的 2 倍）。成本估计：Studio 约 4.2 小时、DGX 约 7.3 小时（200 局×
  300 步，旧单调用成本口径；对手搜索成本未计，另行实测）。S1-2b 不跑
  不拦蒸馏；跑了只报数。
- **与 §3.4 最终确认的分工**：S1-2a/b 验的是**教师**（蒸馏前）；
  §3.4 最终确认验的是**蒸馏后冻结裸网**（Carago g1 新种子＋未见对手
  各 200 局），在联合动作蒸馏完成后、候选冻结后才启用，一次性。
  两闸的种子块与对手不得重用。

## S1-3 / S1-4 与新顺序的衔接（一句话级）

- **S1-3（多样性机制）**：不变，仍在 gen-1 生产前定；补一条：多样性
  的 A/B 度量（平局率、margin 分布）须在**统一 guard 的生产配置**下测，
  不得用 guard-off 镜像局的平局率当证据（纠正第②点同款错配）。
- **S1-4（标注规模化）**：任务书的两条路（跨机并行／分层抽样）降为
  实现方式；上层计划改按纠正定版走——**按预算稀疏标注（随机覆盖打底
  ＋关键局面加标，未标注局面仍进价值训练）**，标注预算以位置数＋
  纠正后成本算法报价（119,600 位置串行 Studio 184.4 小时，并行除以
  worker 数）；标注产物仍走 F05 生成发布。

## DGX GPU 空等期间可先做的非 GPU 准备工作（不占 GPU、不改仓库）

1. 写场景采样脚本：34 局新对局按 id 核对与 ds_m3s2 不交，产 q 闸
   17 局与 ΔU 闸 17 局两张清单＋采样种子（脚本本身可先写，结果待核查）。
2. 写 Stage A 试点驱动：30 局面分层抽样＋σ̂ 与单调用成本实测输出格式。
3. 守卫抽取：`bankrupt_guard` 抽成有状态类＋与 `make_search_agent`
   的奇偶校验（CPU 活；parity 测试在小批量局面上跑）。
4. U 叶接线实现＋奇偶校验（A1/A2 的代码改动与测试；torch parity
   可在 CPU 上跑，不占 GPU）。
5. F01/F04 终局计分单元测试：手算 3 算例＋20 真实终局局面与 eval_v53
   逐项对数（A2 验收）。
6. 核对 `_extinct_state` 与 `rl2_ppo_rollout._extinct` 的镜像精确性
   （逐项比对规则文本）。
7. 排队说明：Lite-V7 语料 107→126 已于 main `aa8055b0` 落袋，双臂
   复刻训练的语料前置已齐——DGX GPU 若需即刻有活，Lite-V7 训练是
   现成的 GPU 任务（作战室定先后，S1-2 不抢）。

## 未核实项（明文列出，不许当已核实引用）

- 34 局新对局与 ds_m3s2 的对局 id 不交（须采样前逐项核对）。
- DGX 28.2 秒/调用 vs 试点 17.8 秒/调用的口径差异（Stage A 重标）。
- `_extinct_state` 与 `rl2_ppo_rollout._extinct` 镜像精确性（逐项比对）。
- `evaluate_root` 在 CachePolicy 下的数值等价性（换 pol 类后的奇偶校验）。
- ds_m3s2 35,282 样本的对局清单（用以执行 34 局不交核查）是否在仓内
  可直接取用（若无，须 DGX 提供清单或改用对局时间切分：2026-11-12
  20:29 刷新后新增的局——须核实时间戳口径）。

## 草案版本记录

- v0.1（2026-11-13 13:49，设计代理）：初稿。核验基线 main `aa8055b0`；
  文件依据：stage0_teacher.py `d17cf360`、cnet2_model.py `3bb5857f`、
  stage0_train_heads.py `14ba9166`、cnet2_np.py `d95f6384`、
  cnet2_search.py `10a0674e`、stage0_guided_match.py `fc4bb32`、
  stage0_measure_teacher.py `aa4de09`、selfplay_gen.py `95d2e29f`、
  eval_v53.py `aa4c5bb6`；DGX 直读 heads v2 npz `06a03d9c…`。
- 待作战室裁决：(b) q 收益闸阈值建议（+0.01 U / 25% / 锚点 > 0）；
  S1-2b 默认关；S1-2a/S1-2b 与 Lite-V7 训练的 GPU 先后；机器选择
  （Stage A 实测后定）。

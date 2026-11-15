# Echo4 战术搜索线 · 数据与进展汇总（2026-11-16）

- 汇总人：搜索组（本线唯一推进者，负责设计与实现；判据/验收归策略组，排期归作战室，上榜提交归所有者）。
- 用途：搜索线全部相关数据、裁定、实现与证据的一站式汇总，供复核与后续决策。
- 基准时刻：2026-11-16 00:45 CST。文中「估计」均显式标注；实测数字注明口径与出处。

## 1. 注册与裁定链（全部已落仓 main）

| 件 | commit | md5 | 内容 |
|---|---|---|---|
| Codex 设计评审 | 3b954a4c | 8e380849（评审件） | SEARCH_DESIGN_REVIEW_20261115.md＋证据 JSON；源冻结 28841671 |
| 预注册（现行版） | fc523c90 | 70168485 | PREREG_ECHO4_TACTICAL_SEARCH.md；链：初版 2dcb7ca0 → 分工 amendment 07b203c9 → 包尺寸加测 df33f4a4 → MS01–MS04 裁定 fc523c90 |
| EXAM_VOLUME_DEFAULT | 32d8482a | — | 独立入仓；样本单位＝配对 seed；卡线两阶段须附全程错误率控制 |
| 可行性方案 v1.1 | 03a715a5 | d110befc | FEASIBILITY_ECHO4_SEARCH_20261115.md（v1.0 为 94c04d68／01493217） |
| Codex 复核本组方案 | 667bb3f4 | — | REVIEW_MUSE_SEARCH_PLAN_20261115.md＋证据 JSON，提 MS01–MS08 |
| 叶②关闭＋L-vnet33 登记＋实现开工裁定 | 53ca3850 | c57891e1 | 所有者 2026-11-16 00:24 直令「按设计实现」后落字 |
| L-vnet33 身份钉版 | 7f575fd1 | 49eb2efa | 身份＝（检查点 33，state md5 e3df8f02…）钉进注册 |

MS01–MS08 裁定摘要：MS01 S3 降为筛选/诊断、其方差不锁 S4 的 N（改同口径整局 pilot 或具名保守上界）；MS02 S4 固定 N 单次判读、本注册不用卡线；MS03 根名单/总体事前冻结、H=0 真回 A0 记 0、异常/泄漏/失败记无效不填 0、分层加权、先按局聚合，无效上限与估计式由搜索组在 S3 协议附则拟定送验；MS04 叶身份＝（检查点，state md5），晚检查点 critic＝新叶版本另登记。MS05–MS08 为设计侧修订（Sequential Halving 定义、J 与隐藏状态粒子分轴、历史台账拆分、KataGo 引用），已入方案 v1.1。

## 2. 主体与钉版

- 基体包：`code/packs/echo4_int8.py`，4,601,242 B，md5 **819250bfc82632786063dabd46a616bd**；meta：it=20、tensor_count 186；CONF：SEARCH=False／MODE=sweep／ROLL=self／GUARD=True／GUARD_NARROW=True／TEACHER=2／BUDGET=0.7／HARD=0.85。
- 内嵌四源（包内 _BUILD_HASHES 与钉版逐项合）：feat 2a45b08b／np d8c15ad4／fastsim aefe6c83／search 94f7fb64。
- actor 源：ckpt_20.npz，13,972,532 B，md5 43ec2a25b675262dbaad356eabcde6ed（L20 机上在册、自落盘未动）。
- 平台硬限：1.5 核／512 MiB／单文件 ≤10,000,000 B；单步基线 1 s（含通信）、硬顶 3 s；慢步 ≤3/局且超额 ≤5 s/局；init ≤20 s；整场 ≤1500 s。考试钉版集：bigeval 354bcd1b／eval_v53 45792dfb／cnet2_search 94f7fb64。
- 诊断种子段：986200–986499（逐个登记，不入训练池）。

## 3. S0a 实测数据（已关账，策略组验收）

### 3.1 工件盘点（L20 线 gpu-l20-6 直读＋只读加载，2026-11-16 00:18–00:29 CST）

**it=20 完整 trainer state（含 vnet）不存在**：停点只落 ckpt_20.npz（actor），完整 state 未单独快照，state_last/state_previous 两槽轮转已覆盖。现存全部完整 state（/root/carbon/data/rl3/ppou2_m/）：

| 文件 | 轮次 | 字节 | md5 |
|---|---|---|---|
| state_anchor_frozen.pt | it=0（冻结锚） | 54,613,777 | 26631aef5678fa7bf74f55e80a9e2934 |
| state_previous.pt | it=32 | 82,600,176 | 21e1ed9b9e4f8f69e74a99d1eaea628b |
| state_last.pt | it=33（00:09 落地） | 82,600,176 | e3df8f0203ce11b7a858c08f23d87929 |
| （残留目录）aborted state_last.pt | it=1（首跑夭折） | 54,613,777 | 0714cc45…（前 8 位） |

- actor 同源判定「同源」，证据链四环：ckpt_20.npz md5 与备案逐字合且未动；npz 自带 meta 标 it=20；其 186 张量键与 state 内 net 键集逐字同、与包 meta 的 186 对齐；L20 考试包与 DGX 的 Echo4 包同源打包、每跳 md5 验过。
- vnet 构成（现存完整 state 实测，锚与 it=32 逐项相同）：顶层键 net（186 张量）／vnet（186 张量）／opt／vopt／it／kl_coef／training{…}；**vnet 键集与 net 键集逐字相等（186/186）**，含全部头（whead、bhead、U、V、utility.weight/bias）；vnet 共 **3,480,775 参数，fp32 原始 13,923,100 B**（＝参数×4，策略组复核合）。构成与轮次无关（结构推断，已标注）。

### 3.2 基线成本（C1 线 spot-34，certify_pack 钉版口径）

口径：systemd scope CPUQuota=150%＋MemoryMax=512M、BLAS 钉 1；Echo4 原样 SEARCH=False；对手 Master2 真身（2aebfbc1）；20 局、seed0 993000、双席各 10 局；单局内 agent 持久、跨局重构；RC 0、overall PASS、a_errors 0、identity 三方合、cgroup v2 quota 证明在案。工件：机上 /root/carbon/s0a_echo4_cost.json（C1 本地副本已收，归档进仓待补）。

| 项 | 实测 |
|---|---|
| 包尺寸 | 4,601,242 B；距 10,000,000 B 硬线余量 **5,398,758 B** |
| init（包 exec：解压/反量化/Policy 构造） | mean 142.6 ms（142.1–143.1） |
| init_wall（spawn 口径） | max 0.295 s（限 20 s） |
| 首调用 | mean 92.3 ms（91.6–93.3） |
| 稳态逐步（5,980 次调用） | mean 51.2 ms／p50 51.2／p95 51.5／p99 51.6／max 93.4；双席无差 |
| 慢步／超额 | 0/局／0.0 s（限 ≤3 步、≤5 s） |
| 局墙 | mean 22.0 s／max 22.1（限 1500 s） |
| RSS 峰值 | 94.2 MiB（限 512） |

派生包总账（搜索组算，估计）：Echo4 本体 4,601,242 B ＋ vnet int8 载荷约 3.5 MB → 整包约 **8.1–8.3 MB，在硬线内（余量约 1.7 MB）**；fp16 叶则整包约 11.6 MB **出线**；fp32 直出 13.92 MB 出线。新叶部署形态基本钉 int8。

## 4. 叶的状态

- **叶① 启发式叶（现行，已接入）**：cnet2_search.py 中 make_search_agent 内嵌套函数 `leaf(sim, me, left, prev_cash=None)`（行 666–794），公式＝我方−对方的（cash＋CARGO_W·可交付货物＋WORKER_V·工人＋TREE_V·树生命），终局走 F01 口径，含 LEAF_THREAT 残局变体（cfg 给值且 left≤30 触发）。返回**原始分差尺度**（非 U）。仅被 _search（行 1652）与 _mcts（行 1447）调用。
- **叶② it=20 同源 vnet**：**关闭（工件缺失）**，裁定落仓 53ca3850。
- **候选新叶 L-vnet33（已登记）**：来源＝主臂 it=33 完整 state；冻结拷贝 gpu-l20-6:/root/carbon/data/rl3/ppou2_m/state_frozen_it33.pt，82,600,176 B，md5 **e3df8f0203ce11b7a858c08f23d87929**，2026-11-16 00:32:20 CST 落盘（赶在 it=34 轮转前），只读加载读回 state 内 it＝33、kl_coef 0.05、vnet 186 张量在位。身份＝（检查点 33，state md5 e3df8f02…），钉版落仓 7f575fd1。**四项验收（行为匹配／校准／成本／打包尺寸）全过前不进 S3**；其 vnet 与 actor 非同轮（actor 为 it=20），行为匹配是四项之首的原因。

## 5. 实现（code/search_proto/，本汇总同批入仓）

### 5.1 关键设计裁定（搜索组裁量，已报策略组备案）

1. **A0＝post-guard 定型动作**（cnet2_search 行 1004），非裸网输出。
2. 比较器替换现存推进＋判读段（行 1571–1665）：**MIN_DEPTH 接受门（行 1643–1650）、0.45×BUDGET 首层门（行 1604）、MARGIN=20 提交门槛（行 1662）均不带入**——整轮完成才提交、argmax、平局优先 A0。双门复核：叶调用行 1652 严格在两门之后，Codex「深度 2 返回原动作、叶未被调用」结论逐行成立。
3. **守卫一致落在候选准入层**：现存守卫 BankruptcyGuard.apply（行 568，行 989 调用；GUARD_NARROW 收紧行 583–585）只保 A0、从不复核候选；比较器逐候选用 guard.copy() 复核，被改写者进 guard_vetoed。
4. 候选三道准入：根合法性（合法掩码＋去重）→ 守卫一致 → 战术相关性截断 K≤4（含 A0，其余按先验 lg0 顺序；现存 MAX_BR=16 由此收口）。
5. 叶①尺度：原始分差直通（affine 恒等）；argmax 对单调变换不变，比较只在同叶内，不套未注册压缩函数。L-vnet33 的 U 映射固定 (tanh(z)+1)/2。
6. 对手情景 J=1 确定性：sweep 路径无 rng、无种子轴（对手根动作行 1559 根外定型一次；opp_model 只读对手已执行历史，**无任何路径读我方候选动作，窥视面为空**）；scenario_seed 首版只透传登记、未消费。ROBUST 的 hunter 变体是确定性规则改写，不记作 J=2 采样。
7. 嵌套函数提取：leaf/guard/pol 等模块级不可达，real_env 以闭包按名反查（外层 agent→_inner→_search 三级 __closure__）直引原对象、零复刻；逐名断言、缺名即停；**vendor/pack 刷新（md5 变）必须重验提取清单**（漂移风险登记在 real_env.py 头注）。
8. xo 重建坑：Feat.observe 在 step≤self.step 时整表重 init，事后对真 fo 重算 xo 会毁信念状态；解法为基线调用前 feat_copy(fo) 快照、调用后按原式重建（本 CONF 下 INFER_MIRROR/INFER_TREE 关，重建精确）。
9. 外层规则层（行 1975–2048）开关在包 CONF 下**全关**（逐项实测：REC_CUT_ALL/OPEN_PL/TG_*/RG_STEP/PX_MODE/PLANTER_AVOID/SE_STEP/CB_STEP/EH_STEP 均缺失=None）；内层 post-net 只有 GUARD 在活。比较器提交动作不被外层改写；唯一恒活的是异常兜底（_inner 抛异常→返回 {} 空动作）。
10. 比较器合同解释五点：A0 豁免否决（强制兜底）；completed_comparisons 只计完成轮；合同违约（无 A0/多 A0/候选>4）抛 ValueError 不静默降级；abort_reason 词表＝deadline／single_call_overrun；rollout/leaf 异常上抛不吞。

### 5.2 代码清单（md5 为入仓版）

| 文件 | 行数 | md5 | 作用 |
|---|---|---|---|
| comparator.py | 225 | 3218f83a799c04a8b2f82545c6f1b4ae | 比较器核心：入口 monotonic 计时、整轮提交、中断回上轮/首轮回 A0、否决过滤 |
| telemetry.py | 83 | 436b146934799fc958413bc745d1f6df | 独立遥测 StatsCollector（针对旧 stats 死链的显式发布） |
| leaf_adapters.py | 89 | 2bd3925cd574cddc678176e402f16829 | 叶① affine 适配（越界不静默裁剪）；vnet 适配骨架（身份＋tanh 映射，未验收禁用） |
| pack_bridge.py | 109 | 80abf1be62a5ea56e73e9d1c2d8e75e4 | 只读桥接：拉钉版四源验 md5、加载与函数定位 |
| real_env.py | 527 | 7790e71350200d9ed0f91ad6dc15ef70 | 真实环境接线：保留段复刻＋Comparator 替换判读段＋三道准入＋闭包提取 |
| tests_s1.py | — | d9c3ee00ef4b40e466e43f7899eab841 | 桩级机制用例 7 例 |
| smoke_real_s1.py | 156 | 9cad53d6ebfe94e5977f93786ed68d50 | 真实环境冒烟脚本 |
| s1_battery.py | 520 | 6a78101a5a1d4403f1f89a053c1c0453 | 正式 S1 电池：自然根扩量＋精确终局正反例（逐根明细 s1_battery_roots.json，md5 6f3c5b4da9ea43e3170979c8491da987，同批入仓） |
| WIRING_MAP.md | — | 7c17112bfe6fb1dca9fc28cd83bc69e7 | 接线图（cnet2_search 内部结构与行号证据，另存 reports/search_dossier/） |

vendor 四源不入本目录（仓库 code/ 已有钉版原件：cnet2_search.py／cnet2_feat.py／cnet2_np.py／fastsim.py，md5 见 §2）；pack_bridge 按钉版 md5 拉取校验。

### 5.3 验证证据

- **桩级机制（tests_s1.py）：7/7 PASS**——T1 正例必改、T2 反例不改、T3 中断回 A0、T4 守卫一致、T5 平局 A0、T6 候选上限拒收、T7 快步不攒额度。comparator.py 自测 4 例全过。
- **真实环境冒烟（smoke_real_s1.py，本组亲跑复验、两次数字一致，SMOKE: PASS）**：FastCarbon 自对弈、seat0 收根、H=1、K=4、budget=cfg BUDGET 0.7 s −该步基线步耗（下限 0.05 s）：**12 根、完成比较 12/12 根（叶调用 48）、完成轮次 12、中止 0、正例改子 5**（均经守卫一致）；比较器每根 mean 1.5 ms／p50 1.5／p95 2.0／max 2.2。**现象硬要求（完成比较>0 且正例改子>0）在真实环境冒烟级已满足**——旧两套构造在部署形态下从未做到。
- 环境口径提醒：冒烟所在 VM 为共享 CPU，基线步耗 0.42–0.69 s 是测量环境产物，不代表部署步耗（部署口径以 §3.2 的 51.2 ms 为准）；比较器自身耗时同理待 S2 在考位复测。
- **正式 S1 电池（s1_battery.py，墙钟 244 s 同 VM 口径）**——A 自然根：3 局自对弈双席收根、诊断种子 986300 段、H=1/K≤4：**总根 48（双席各 24）、完成比较 48/48（叶调用 192）、中止 0、changed 16（33.3%）、守卫否决 0**、每根 mean 1.30 ms／p95 1.63、深度恒 1；遥测双通道（逐根明细与 StatsCollector 快照）数字一致。B 精确终局例：get_state/set_state 在真实 FastCarbon 造 step=299 根，**双证机器验准——四例×每例 4 候选的 leaf(left=0) 值与逐候选真跑一步到 step 300 的现金差逐个相等（max abs err = 0.0）**；反例 N1 达（A0 踏入交付 +44 唯一最优、changed=False 合期望）；**正例 P1/P2/P3 未达，定性为实质原因非构造不可行**：三种末步几何（交付/捕获交付/避敌基）下 Echo4 基线 A0 均为一步结算最优、比较器无正例可改，且对手根动作会主动拆掉「站我方基地格」类陷阱几何。补测在途：碰撞/逃逸几何正例（注册战术型 1）＋晚局 changed 根续弈验证（本批根 step≤56、预设 step≥220 门槛无一入围，verification 段为空、如实登记）。
- **机制风险项（P3 暴露）**：K≤4 先验截断可丢弃唯一有结算区分度的候选——P3 中被丢的是劣动作（−44）无害，但若最优动作先验垫底，比较器根本评不到它。建议 S1 判读列为机制风险，S3 协议附则考虑给准入第三道加「结算区分度保底」口径。

## 6. 历史结论（MS07 统一口径）

- 旧嫁接构造（KataGo 式价值叶嫁接）：**已有明确负证据**——注册预算 0.38 s、6006 根深度 1/2/≥3 为 2742/3147/117、改子 120；三腿 ΔU −0.1223（归因腿：嫁接叶 vs 常数叶）／−0.4698（直打 Echo3）／−0.1139（参照腿），CI 均在零下，关闭。
- 旧内置部署构造（PREREG_SEARCH Parts 1–4）：**无有效动作改进且计时失败**——Part 1 v1 喂法作废、v2 持久形态 443 根深度全 2 零改子且慢步失败；Part 3 自身 pilot 42 根、叶调用 0/42（双门所致）；Part 2 量纲完成；Part 4 作废。
- 新 Echo4 完整 critic 浅层方案（本线）：S0a 关账、S0b 原型与桩级＋冒烟级机制已过，**收益未测**（S3/S4 未开）。

## 7. 未决与下一步

1. **正式 S1 电池**（已跑，见 §5.3）：自然根 48/48 达标、反例达、正例未达（实质原因）；补测两件在途——碰撞/逃逸几何精确正例、晚局 changed 根续弈验证；齐后整包送策略组 S1 阶段验收。
2. **S0b 成本表**：K={2,4}×H={1,2,3} 完整一轮成本与可完成范围（C_hi 表 v1）；本地先行、部署口径待考位（S2 逐局计时归考场，排期归作战室）。
3. **L-vnet33 四项验收**：行为匹配（与 it=33 actor 的联合行为基线——注意部署 actor 是 it=20，此项判的是 critic 对 it=20 动作分布的适配度口径，须在验收协议中先定义）／校准（U 尺度对真实终局）／成本（单叶调用成本入 C_hi）／打包尺寸（int8 派生整包实测）。
4. **S3 协议附则**（搜索组拟、开测前送策略组验收）：根名单冻结法、无
...[truncated 1107 chars]
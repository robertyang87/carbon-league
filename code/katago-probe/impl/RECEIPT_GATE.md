# RECEIPT_GATE — KataGo 探针②构造＋构造关（第二趟，PREREG 修订一 v1.1＋修订二）

**状态：构造关 PASS（2026-11-14 23:39 CST 落盘，主线直读复核）。** 作战室裁定借 DGX 空窗先跑（23:29 发车、16 worker、wall 574.5 秒）；主线直读 `out_gate/summary.json` 复核，关过。主体按修订二①即刻冻结为 §2 字节（变体 4a3b5f1b）。接线正确性已在本地＋DGX 双路实测证明（§5），含 DGX 上冻结 DEFAULTS 下合成根 6/8 落子改变的接合证据——但那是合成冒烟、**不是构造关判读**；关判读只认 rootset32 登记根集上的正式跑（§8）。

## §1 备齐件状态（每件一行：文件／用途／就绪）

| 文件 | 用途 | 状态 |
|---|---|---|
| `impl/modules/cnet2_search_vleaf.py`（DGX `katago_probe_scratch/probe2/modules/` 同） | 价值叶玩家变体（唯一探针改动件） | ✅ 就绪，md5 4a3b5f1b45fde9b659a383c1c3fba87f，本地与 DGX 逐字合 |
| `impl/modules/cnet2_search.py` | 启发式对照底本（main 现行字节原样） | ✅ 就绪，md5 94f7fb6438bb7577b141e388f068e122 |
| `impl/modules/cnet2_feat.py` / `cnet2_np.py` / `fastsim.py` | 底本依赖三件（main 钉死字节） | ✅ 就绪，md5 2a45b08b…／d8c15ad4…／aefe6c83…，DGX 验合 |
| `impl/gate_driver.py`（DGX 同） | 构造关驱动（三臂同根同种子配对、多进程） | ✅ 就绪，md5 2858ea720d9139d26f0feef9f0560c63，DGX venv py_compile 通过 |
| `impl/run_gate.sh`（DGX 同，已 +x） | DGX 启动脚本（线程四变量钉死=1、16 worker、登记模式） | ✅ 就绪，md5 26ecd4196366b24080b1db34fa006fed |
| `impl/echo3_ref.py` | Echo3 部署包本体（本地证据副本，供 §9 与下趟直测腿） | ✅ 就绪，md5 7b7620499411a088cf4dc17c0e929bd7＝台账登记值 |
| `impl/smoke/tasks.jsonl`（64 根）/`tasks8.jsonl`（8 根） | 合成冒烟根集（SYNTH-SEED777，与登记根集严格分开） | ✅ 就绪，仅诊断用 |
| `impl/smoke/cnet2_search_vleaf_dbg.py` | 变体诊断孪生（仅多 DBG 计数行＋1 空行，diff 已验） | ✅ 就绪，仅接线取证用，**不入部署集** |
| DGX 侧权重/根集（机上原位） | m3s2r1 本体＋s64 rollout 权重＋rootset32 | ✅ 就绪，机上 md5 复验见 §3 |

## §2 变体与对照身份

- 变体 `cnet2_search_vleaf.py` md5 **4a3b5f1b45fde9b659a383c1c3fba87f**：与底本恰差 4 处——①头部来源注释；②`leaf()` 闭包签名加 `fm=None`；③终局 F01 分支之后、启发式主体之前插入 `LEAF_VALUE` 分支；④两个 leaf 调用点（_mcts 第 1470 行、_search 第 1675 行）各传本分支 Feat 句柄 `fs[·][0]`。
- 价值叶（`LEAF_VALUE="value"`）：非终局叶上以 `feat_copy(fm).observe(sim.obs(me))` 在私有拷贝上续接本分支 Feat 历史（绝不推进原 fm），`encode_b` 得 ctx，取 z = ctx @ `value.weightT`[:,0] + `value.bias`[0]，返回 **M_eq = 2000·z**。等价性（代码注释已写明）：û=(tanh z +1)/2 ⇒ 2000·atanh(2û−1) = 2000·z，恒等、无数值发散。
- 常量叶（`LEAF_VALUE="const"`）：非终局叶恒返回 **M_eq ≡ 0.0**（修订一消融口径，非 û:=0）。
- 启发式对照：底本原样（`LEAF_VALUE` 不设），md5 94f7fb64…。
- 除非终局叶评估外全冻结：同 DEFAULTS 预算（§4）、同策略先验、终局 F01 真值计分不动、MARGIN=20 不动。
- **冻结纪律（修订二①）**：构造关一旦通过，价值叶玩家即以本字节（变体 md5 4a3b5f1b…＋底本四件 md5）冻结，过关后不许再动任何参数/映射/代码，开场包用过关那一版字节。本回执 §1/§2 的 md5 即冻结登记。

## §3 权重与根集复验（DGX 机上直读，2026-11-14 23:3x）

- `s26stage/data/weights/cnet2_m3s2r1L.npz` md5 **b6db08c9b076257c54817377bb93130d** ✓（与钉死值合）
- `carbon/data/weights/cnet2_s64.npz` md5 c84eef933d1d71272e89de3742081451 ✓
- 归档 `l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json` md5 c0871c50d0cc1eea7e07f8aed6587838 ✓；`roots/` 32 份回放；登记根集 = 32 局 × 双席 × 步 [200,299) = **6,336 根**。
- 本地副本（冒烟资产）同 md5：m3s2r1 b6db08c9…、s64 c84eef93…。

## §4 search 预算表（DEFAULTS 冻结，部署底本实测）

SEARCH=True、MODE="sweep"、H=5、BUDGET=0.38、CLOCK="wall"、PROJ=1.3、HARD=0.6、MARGIN=20.0、MAX_BR=16、MIN_DEPTH=3、KEY_R=2、MAX_KEY=3、MINC=10.0、CARGO_W=0.9、WORKER_V=30.0、TREE_V=2.0、ENDGAME_STOP=1、TEACHER=1、OPP_TEACHER=1、ROLL="s64"、LEAF_Q=None、GUARD=False、GUARD_NARROW=True、GUARD_CASH=40.0、C_PUCT=1.5、ROOT_S=12、RSEED=0。三臂 conf 仅差 `LEAF_VALUE` 一键（heur 臂连此键亦不设）。
驱动约定（明写）：每（根，臂）一个全新 agent 实例；同根三臂共用种子 `np.random.seed(20261114 + root_index)`；每根以单条记录观测喂全新实例（Feat 历史为首观值）——构造 sanity 口径，rootset32 仅作构造关、不作推断读数，三臂同约定保证配对。

## §5 自检证据（接线三证）

1. **单元一致**：同一 x 下值头预激活 z：`enc_for` 路 −0.42808011 vs 变体叶所用 `encode_b` 路 −0.42807969，|Δ|=4.2e-07；M_eq=−856.16、û=0.2981，有限合理。
2. **分支计数（本地诊断孪生）**：放宽预算（仅诊断、关本身不放宽）下 value 臂叶调用 13/13/5/5 次**全走 value 分支**、const 臂全走 const 分支、终局分支 0 掺入；4/4 根 value≠const 落子。
3. **DGX 目标机冒烟（冻结 DEFAULTS、合成 8 根、workers=4）**：三臂 searched 8/8、errors 0；**changed(value vs const)=6/8**、changed(value vs heur)=6/8、changed(const vs heur)=4/8；mean dt value 0.559s／const 0.296s／heur 0.281s，wall 2.7s。证明 DGX 快核上 rollout 真实接合、价值叶改变多数落子。产物：DGX `probe2/out_smoke8/`。
   - 对照注记：同一 8 根在本地节流 VM 上 changed=0——根因已定位：phase-1 超 0.45×BUDGET 触发 rollout 投影闸、depth=1 撞 C01 MIN_DEPTH 卫哨（depth<3 且非全终局即弃搜索返回默认动作），叶根本未被调用（计数实测 leaf=0）。这是机器速度伪影、非接线问题；关在 DGX 跑即按上行口径接合。

## §6 窗核查证据（DGX 直读）

- 23:25 CST：spark-29db load ≈ 0.00；无 join/annotate 进程；`k0_dist`、`k0_prod_stage` 下无 join 产物、无 σ̂ 报告文件。
- 其后复核（机上时钟 23:28）：C1 回片 62 npz、C2 回片 62 npz（分区 108/92 局）；C2 `progress.json` mtime 23:26（此前 23:24→23:26 仍在推进）；23:00 后 k0 目录无 returns 外新文件；DGX 无 join/annotate 进程（仅 agent_mcp 等常驻）。
- 判读：标注未齐 ⇒ join 未启 ⇒ σ̂ 未出，**窗口未开**。不在本趟抢座。

## §7 启动方式（窗口实测满足后一条命令）

```bash
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'cd /home/arm/junyang/katago_probe_scratch/probe2 && setsid nohup ./run_gate.sh > gate.log 2>&1 & echo started'
```

- 启动后以 `pgrep -f gate_driver.py` 记 PGID 补入本回执；产物 `probe2/out_gate/{results.jsonl, summary.json, examples.json}`。
- 量级估计：19,008 次调用（6,336 根 × 3 臂）、16 worker、DGX 实测单调用 0.28–0.56s ⇒ 约 10–20 分钟（估计）。
- **过关判据（钉死）：changed(value vs const) > 0**；=0 如实报接线未通、停步上报，不许调参凑数。过关 ⇒ §2 字节即刻冻结（修订二①）。

## §8 消融计数全表（构造关正式跑，2026-11-14 23:39 落盘）

| 项 | 值 |
|---|---|
| 总根数 | 6,336（登记） |
| 实搜根数 | 6,006（每臂一致；缺口 330 根原因见 §11.1） |
| 落子改变数 value vs const | **120**（判据 >0，过关） |
| 改变比例 | 120 / 6,006 = **2.0%**（实搜集为分母） |
| 旁参 changed(value vs heur) / changed(const vs heur) | 153 / 70 |
| errors 全臂 | 0 |
| 单根均耗 value / const / heur | 0.4507s / 0.4416s / 0.4366s（value 比 heur +3.2%，计时预读） |
| wall / workers | 574.5s / 16 |
| changed / unchanged 样例存档 | `out_gate/examples.json`（37,463 B）；逐根全记录 `out_gate/results.jsonl`（8,541,337 B） |
| summary md5 复核 | 变体 4a3b5f1b…、底本 94f7fb64…、feat 2a45b08b…、np d8c15ad4…、fastsim aefe6c83…、gate_driver 2858ea72…、权重 b6db08c9…、rweights c84eef93…（与 §1/§2 登记逐字合） |

（§5.3 的 6/8 是合成冒烟诊断数，严禁当作关判读。）

## §9 必答一问（修订二②）：价值叶玩家是否携带 Echo3 的 guard 配置？

**答：部分携带。** GUARD_NARROW、GUARD_CASH 携带且有效值相同；**GUARD 不携带**（Echo3 生产为 True、价值叶玩家为 False）。

证据锚：
- Echo3 部署包本体＝仓库 main `code/packs/echo3_int8.py`（commit 81d1b3e1），本地实测 **md5 7b7620499411a088cf4dc17c0e929bd7、4,576,790 B**，与台账 `reports/live/packs_ledger.md` 登记逐字合；本体权重与本探针同一文件 cnet2_m3s2r1L.npz（b6db08c9…）；打包器 cnet2_search_pack.py（v12 件套、NO_ROLL=1）、CONF 标注「原版 13 键逐字」。
- 包第 27 行生产 CONF 抄录（逐字）：`{"SEARCH": False, "ROLL": "self", "GUARD": True, "GUARD_NARROW": True, "MODE": "sweep", "MIX": False, "TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240, "BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95, "START_DL": None}`。GUARD_CASH **不在 13 键内**。
- 包第 75 行以同一工厂构造：`make_search_agent(CONF, weights=…, rweights=…, oweights=…)`，cfg = 包内嵌 search 源 DEFAULTS ⊕ CONF。包内嵌 search 源（md5 adbe660d…，v12 打包时版）DEFAULTS 中护栏三键为 GUARD=False、GUARD_NARROW=True、GUARD_CASH=40.0——与我变体底本（main 现行 94f7fb64）DEFAULTS 三键**逐键相同**，继承关系成立：同一工厂、同一 DEFAULTS 基线，差异仅在覆盖层。
- Echo3 有效护栏：GUARD=**True**（CONF 覆盖）、GUARD_NARROW=True、GUARD_CASH=40.0（DEFAULTS 回落）。
- 价值叶玩家有效护栏：gate 驱动不注入任何 CONF 覆盖（仅 LEAF_VALUE）⇒ GUARD=**False**、GUARD_NARROW=True、GUARD_CASH=40.0。
- 行为后果：`agent()` 内 `if cfg["GUARD"]: y = bankrupt_guard(...)`——Echo3 出手经破产护栏后处理，我方玩家该步不执行；GUARD_NARROW/GUARD_CASH 只在护栏内部消费，GUARD 关时同为惰性。此差异源于 PREREG 钉死口径「除叶评估外全冻结（DEFAULTS）」，不含 Echo3 部署 CONF 覆盖层；如实登记，是否在开场包对齐由策略组裁（修订二③直测腿正是按 Echo3 生产 CONF 与其包本体对打，见 §10）。

## §10 修订二③登记（下趟开场包范围，本趟不执行）

开场包三腿：①叶对叶配对（种子 985000–985199）；②直测腿——价值叶玩家 vs **Echo3 部署包本体**、按其生产 CONF（§9 抄录）、同块逐种子配对；③Master2 参考腿。包内须含 Echo3 包身份 md5（7b762049…）与 CONF 抄录（§9 已备）。强度场与探针③不在本趟；构造关结果先回主线、过关另行派。

## §11 策略组判后登记（2026-11-14 23:43，关已过、主体已冻结；不改判读）

**策略组判：构造关 PASS——主体冻结 4a3b5f1b。** 三腿开场包待合并后作战室第一窗交 DGX，两腿之间禁任何调参。

1. **实搜缺口 330 根（6,006/6,336）——原因（实测，非推测）**：`gate_driver.py` 的 `tasks_from_replays` 只取 `steps[t][seat]` 的 status 为 None 或 "ACTIVE" 的席步，其余状态的席步在任务生成时即被 `continue` 跳过。`results.jsonl` 逐行复核：330 行三臂 searched 全为 0、无部分臂缺失，分布 26 局、双席各 165、步 200–298 散布（非窗尾集中）——与任务层按席状态过滤一致，非执行失败、非终局集中。消融计数按实搜集 6,006 为正确分母，关的数值不受影响。
2. **实根消融率 2.0%（120/6,006），远低于合成根冒烟 75%（6/8）**——冒烟未作数、未混报，在此记下；2% 是真实杠杆量级：若口径场 ΔU 偏小，与此一致，不作事后惊讶；判读三格与直测腿口径不变。副读在册：value vs heur 153、const vs heur 70；单根均耗 value 0.4507s vs heur 0.4366s（+3.2%，计时预读——天梯部署计时判读仍在部署形态场，不在此数上定）。

## §12 外部 review 处置登记（2026-11-15 04:02，策略组亲核裁定）

报告 `reports/REVIEW_KATAGO_CLAUDE_20261115.md`，策略组逐项亲核（gate 数据复算＋码读）验立。责任划分与处置如下，本组登记入册：

1. **量纲 2000·z 是修订一的注册错误，非本组实现错误**：trainer 标签为 tanh(margin/3000)（build:390），值头是 margin 压缩估计器、不是 U 估计器——正确等价分应是 **3000·z**。本组照注册映射忠实实现（变体码 :691/:700 与注册逐字合）。回执中「MARGIN=20 相对意义不变」一句随注册前提倒而倒，此点记在策略组。
2. **TEACHER=1 是共同漏检**：本回执冻结清单印有 TEACHER=1、§9 只答了 GUARD；策略组 guard 裁定、构造关与修订二亦未点名 TEACHER。Echo3 包 CONF 为 TEACHER=2/SEARCH=False——直测腿因此不隔离价值叶。此漏两边各半，正式记为审查漏检，不作本组单方过失。
3. **深度丢弃与末步**：深度 1–2 被 C01 卫哨丢弃占 98.1%、末步 38 处（297/298），均经策略组亲算验立——构造关「会变」的判据本身不够，属预注册判据设计问题（策略组）；新探针构造关加根上质量检（被改根真实续弈验 U）。
4. **处置**：三腿场照跑完、判读范围按修订三（作战室发）；冻结主体 4a3b5f1b 不动、不返工。新探针预注册由策略组起草（Echo3 包仅翻 SEARCH=True＋sims 定量＋启发式叶基线），出稿后本组实现、策略组验收。

## §13 修订三登记（2026-11-15 04:03，策略组全文补发存档；只改判读范围、不动场与主体）

1. 三腿场照注册跑完；主体冻结 4a3b5f1b 不变、不返工；种子块 985000–985199 不变。
2. **(γ) 收窄**：归因腿落 (γ) 只关闭**本嫁接**（tanh 头＋2000·z 量纲/MARGIN=20＋墙钟预算 0.38＋teacher=1 底）——不关闭 KataGo 路线、不关闭「价值叶」前提本身。
3. **(α) 收窄**：归因腿落 (α) 只作**嫁接级证据**，报告须附混杂清单（teacher=1 底、98.1% 步搜索被丢弃、120 处改变中 38 处在 297/298 末步）；不得外推到 teacher=2 部署底，不得表述为「价值叶成立」。
4. **直测腿读法钉死**：读作「冻结主体对 Echo3 部署包」，附混杂清单（TEACHER 1 对 2、SEARCH 开对关、GUARD 关对开、ROLL s64 对 self、叶评估）——永不读作「价值叶对 Echo3」。「实测超越 Echo3」的宣称仍只挂此腿，但正读数必须带混杂披露。
5. **(β) 的 1,000 种子确认场条款对本主体作废**——不再为本主体加派确认场。
6. Master2 参考腿不变（量级参考、无门效力）。
7. **判读报告常设背景**：构造关深度分布（1/2/≥3＝2742/3147/117）与改变步分布（38/120 在 297/298）须与三腿读数同文交付。

## §14 三腿判读与关账（2026-11-15 10:13，策略组判读，六腿归档 main @4083d571 复算）

- **归因腿**：ΔU **−0.1223**，CI95 [−0.1910, −0.0529] → 落 **(γ)**，**本嫁接关闭**（修订三范围：tanh 头＋2000·z/MARGIN=20＋墙钟 0.38＋teacher=1 底；不关闭 KataGo 路线、不关闭价值叶前提）。
- **直测腿**：ΔU **−0.4698** [−0.5381, −0.3987]，78/400 胜 → **未超越 Echo3**（弱式、强式均不成立；读法＝冻结主体对部署包＋混杂清单）。
- **参考腿**：ΔU −0.1139（量级参考，无门效力）。
- 常设背景已同文交付（深度 2742/3147/117、38/120 末步改变）。**探针②关账**；(β) 确认场对本主体作废、不加派。
- **偏离登记（与本组有关，如实记）**：预注册归因 pairing 为价值叶 vs **启发式叶**（PREREG.md:11/14），实跑归因腿对手为**常量叶**包 36bbe1e4（消融臂）——本回执 §10 开场包只写「叶对叶」未点名对手，交付两包即 value/const。单变量性保持、(γ) 方向保守，判读成立；注册口径 ΔU（价值−启发式）未测、留白。漏检主责在策略组（01:33 口径裁定时两包在册未点名），记为本探针第二次审查漏检（继 TEACHER 后），不作本组单方过失。
- **探针③（985200–985999）：按原注册作废、不开跑（2026-11-15 10:14 策略组裁）**。理由：注册主体即本嫁接，其单步成本已在构造关量过（均 0.4507s/根、p95 1.096s），再跑 800 局只是给不会投产的玩家定价、无新信息。成本账义务不灭、随新探针预注册转移（主体换新探针主体，成本账并入「每代」口径、与教师标注 43–45 秒/根对照）；种子域 985200–985999 释放回缓冲，新预注册需要时另行钉块。
- **下一步**：新探针预注册由策略组起草（Echo3 包仅翻 SEARCH＋sims 定量＋启发式叶基线臂），出稿后本组实现、策略组验收。

## 纪律附注

探针产物一律诊断件、不入任何训练池；未改 main 任何文件（Echo3 包仅经 API 只读拉取作证据）；机器与线程配置：DGX spark-29db（20 核）、venv Python 3.12.3 / numpy 2.5.4、OMP/OPENBLAS/MKL/NUMEXPR 四变量 =1、16 worker、setsid 独立进程组（PGID 待启动时补记）。

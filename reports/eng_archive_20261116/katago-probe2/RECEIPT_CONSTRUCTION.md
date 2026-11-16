# KataGo 新探针 Part 1 构造回执（预制件，不入 main、不开场）

> 状态卡点（2026-11-15）：构造件 A/B 已建成并通过四项自验与闸三机械断言（21/21）；五闸中闸一/三/四已闭、闸二/五以构造期标定阶梯（PREREG_SEARCH.md §2）正式跑＋整局成本局实测判读——**标定结论：阶梯 {100…1600} 内无满足档（停报，见 §6 闸二/闸五）**，故 B 的 SIMS 终值未锁定、B 保持占位版（SIMS=200）。工时修正估计（与开工前预估分开标注）：开工前预估在上下文压缩中佚失、按「未校准」口径如实标注；本趟实耗主要增量来自标定阶梯——DGX 与 K0 线 duel 长跑争用（load 60–80）下单步约 25 CPU-s（窗口局实测）、rootset 个别根更贵，全量 6,336 根/档约需 25+ 机器时/档，故阶梯以结构证明＋已完成整局实测全表收口、全量本体在 DGX 后台续跑待附录（见 §6 闸二注记）。

## 0. 基底复验

- 基底 `~/workspace/katago-probe/impl/echo3_ref.py`：md5 **7b7620499411a088cf4dc17c0e929bd7**、4,576,790 B — 与派单钉死值逐字合 ✓（本趟多次重验不变）。
- 结构：line 8 `_SRC`（四内嵌模块源码字符串字典）、line 27 CONF（13 键）、line 28 `_Q1`、line 30 `_B1`（b64 int8 权重）；`_B2/_B3` 空串（NO_ROLL=1、ROLL="self"）；line 75 工厂 `make_search_agent`；包级 `agent(observation, configuration)`，统计经 `agent.last_stats` 可读。
- 权重解码（包内约定）：`pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)), pack._Q1)`，`w["meta"]=np.array(pack._M1)`；解码权重 3,596,168 B、md5 **444cc54c0d88da5656ff4c7eef1ec8eb**，原包/A/B 三方同一（§2 程序证明）。

## 1. 交付件

| 件 | 路径 | md5 | 字节 |
|---|---|---|---|
| A（仅翻 SEARCH） | `~/workspace/katago-probe2/modules/echo3_search_pack.py` | **ea20a2714209e1a58b3e435b41d943bb** | 4,576,789 |
| B（A＋sims 钟） | `~/workspace/katago-probe2/modules/echo3_sims_pack.py` | **8e6f7a65d3d91b363606e711eaea03c2** | 4,578,975 |

- 构建脚本 `~/workspace/katago-probe2/build_packs.py`：锚点 count==1 才替换，任何锚点漂移即停。A 经三版重建 md5 不变；B 的前两版（a67258109292917d73feef255ea8a9、cd0aeeaaaf2991054535e637e77a74c6）因记录点修复已作废，勿引用。
- B 的 CONF：`"SEARCH": True, "CLOCK": "sims", "SIMS": 200（占位，未锁定）, "FIX_TRUNC": True`，余键与原包逐键同（闸三断言 §6）。
- B 文件头注释（置于 CONF 上方）标明：单位＝一次叶评估（PREREG_SEARCH.md §1）、计数点在内嵌 `leaf()` 入口、截断为前瞻式（见 §3）。

## 2. diff 审计与零差异机械证明（自验 1 ＋ 闸三）

审计脚本 `audit_diff.py`（全量输出 `audit_out.txt`）、断言脚本 `assert_zero_diff.py`：**21/21 PASS**（现行 B）。

- CONF 逐键比对：A 仅 SEARCH 翻转；B 仅多 `CLOCK/SIMS/FIX_TRUNC` 三键＋SEARCH 翻转，其余 12 键逐值同——含注册点名清单 TEACHER=2、GUARD=True、ROLL="self"、ABORT_DL=0.95、TEACHER_LATE=2、LATE_STEP=240（另 GUARD_NARROW=True、GUARD_CASH=40 在内嵌 DEFAULTS 侧逐键同）。
- 内嵌 DEFAULTS 逐键比对：B 仅多 `SIMS=None` 一键。
- 六载荷（`_Q1/_B1/_B2/_B3/_M1` 等）字节同一；解码权重 md5 三方同一（444cc54c…）；另三内嵌模块（fastsim 等）字节同一。
- 内嵌 cnet2_search 源码变更行集合逐行归类断言：全部落在白名单（计数点/记账行、SIMS 优先的 fix_n、sweep 截断、sims 模式墙钟门豁免、记录点双出口），无「其他」类变更。
- 骨架检查：除 PROVISIONAL 注释块与上述 hunk 外，包其余行逐行同。

### B 相对原包的 hunk 全表（归类：SEARCH 翻转／sims 相关，无「其他」）

| # | 位置（B 内嵌 cnet2_search 行号，源 1,857 行、md5 694e2070fe4b03651e4cb264212454e8） | 内容 | 归类 |
|---|---|---|---|
| H1 | 文件头 CONF 上方 | PROVISIONAL 注释块（单位/计数点/截断语义） | sims 相关 |
| H2 | CONF | SEARCH False→True；新增 CLOCK="sims"、SIMS=200、FIX_TRUNC=True | SEARCH 翻转＋sims 相关 |
| H3 | DEFAULTS line 38 | 新增 `SIMS=None` | sims 相关 |
| H4 | `leaf()` line 487–488 | 函数入口加计数行 `stats["leaf_evals"] = stats.get("leaf_evals", 0) + 1`（**计数点**） | sims 相关 |
| H5 | `_mcts` line 1043 / 1250 一带 | 搜索开始捕获 `_le0`；记录段改记叶增量 `_used`（控制量 sims_done 不动） | sims 相关（记账） |
| H6 | `_mcts` line 1129–1130 | `fix_n` 改 SIMS 优先、缺省回落原 FIX_SIMS 分段表 | sims 相关 |
| H7 | `_search` line 1311 / 1445 / 1457 | 捕获 `_le0`；记录改双出口：shallow 出口记当时增量（叶未调用时为 0）、正常出口在 `sc` 计算之后记叶增量 | sims 相关（记账） |
| H8 | `_search` line 1374–1378 | sims 模式下 `fix_n`=SIMS；候选截断 keep=ceil(fix_n/len(models))，`cand[0]`（默认动作）恒保留 | sims 相关 |
| H9 | `_search` line 1405/1407/1409 | 深度循环三道墙钟门加 `not fixed` 豁免（sims 模式不按墙钟截断；墙钟模式逐字原样） | sims 相关 |

A 相对原包仅 H2 的 SEARCH 一行（−1 B 来自 False→True 的长度差）。

## 3. sims 计数单位：选择、推导与计数点（闸一构造侧输入；已按注册改齐）

**注册定义（PREREG_SEARCH.md §1）：一 sim ＝ 一次叶评估。本件已按此实现，无双轨。**

从内嵌 search 循环结构独立复推（不引草案、只读源码）：

1. 本包 CONF `MODE="sweep"`，走 `_search`。对手模型集 `models=[acts_o]`：ROBUST=0.0 时恒 1 个模型。
2. 根候选集 `cand` 大小 nb=nc≤16（`forced[:mbr-1]` 截断，mbr=MAX_BR=16；`cand[0]` 为默认动作）。深度循环（line ~1390–1430）内**没有任何 leaf 调用**——循环只做 FastCarbon 步进与分支收集。
3. 循环结束后 line 1450（B 版行号）：`sc = np.array([leaf(s_, me, lft) for s_ in sims]).reshape(len(models), nc)`——**每个候选分支恰调用一次 leaf**。故 sweep 每次搜索步的叶评估数 ＝ nc ∈ [2,16]；shallow 出口（depth<min(MIN_DEPTH,H)）在该行之前 return，叶评估数 0。
4. 对照 `_mcts`（exam clock 路）：line 1247 每批元素恰一次 leaf，其既有 `sims_done` 本即叶数——两路在「叶评估」这一点上天然统一。rollout 本身在 sweep 里不是独立对象（rollout 的终点状态才被 leaf 评估），故统一计数点只能是叶评估，与注册定义一致。

**计数点代码位置**：B 包内嵌 cnet2_search（md5 694e2070fe4b03651e4cb264212454e8）`leaf()` 定义 line 487，计数行 **line 488**：`stats["leaf_evals"] = stats.get("leaf_evals", 0) + 1`。调用点：`_search` line 1450、`_mcts` line 1247。计数器经 agent 统计 `leaf_evals` 与 `sims_steps`（每步叶增量）可读。

**截断语义（如实登记）**：sweep 的叶评估在深度循环之后整批发生，循环内无叶可数——故 SIMS 在 sweep 路是**前瞻式**预算：以 fix_n 截断根候选数（keep=ceil(SIMS/模型数)），实际叶数 ＝ 存活候选数 ≤ SIMS（模型数 1 时叶数＝keep≤SIMS；粒度＝一个根候选）。SIMS 小于候选数才 binding；本包候选 ≤16，故 **SIMS ≥16 时 sweep 路恒非 binding**——阶梯 {100…1600} 全档在 sweep 路是同一计算（此点决定 §6 闸二的收口方式）。_mcts 路为批间检查、天然硬顶（粒度＝一批 ≤32）。

## 4. 自验 2–4 实测摘要

- **自验 2（加载）**：原包/A/B 三包 py_compile＋import 成功；以包工厂各构造玩家实例成功（权重走包内嵌解码路径，无外部权重文件）。
- **自验 3（SEARCH 接合冒烟）**：合成根 8 个（步 200–203 × 双席，SYNTH-SEED777，`~/workspace/katago-probe/impl/smoke/tasks8.jsonl`；合成冒烟、非构造关）。原包 searched=0/8；A（墙钟）在本机 searched=8/8 但 shallow=8/8、depth=1（本地 VM 节流伪影：phase-1 超 0.45×BUDGET 撞 MIN_DEPTH 卫哨回默认，与探针①回执 §5.3 同型，附计数证据）；B（sims）depth=5 全根、shallow=0、**相对墙钟版改子 4/8**——搜索真实接合。A 与 B 墙钟模式同根落子 0/8 差异（墙钟路径未动）。
- **自验 4（sims 上限与确定性，现行 B 复验）**：SIMS=8 → 每步叶数 [8,8,5,5,8,8,8,8]（恰封顶、候选不足时为 nc）；SIMS=32/200 → [13,13,5,5,9,9,9,9]（＝nc，非 binding）；三档下记录的 sims/step 与独立计数器 leaf_evals **逐根全等**；SIMS=32 同种子两遍落子逐字同（**changed 0/8**，`run_smoke_b3.py` 补跑复证）；_mcts 路一根核验（SIMS=32）：`sims_steps=[(200, 32)]`、`leaf_evals=32`、`sims=32`、searched=1——记录量、独立计数器、既有 sims 读数三方在上限处全合。

## 5. 闸四 边界用例（步 297–300 终局判定，实测）

用例 `boundary_cases.jsonl`（合成局 SYNTH-BND777：活到 300 步、双边未灭绝；步 296/297 工人 [3,3]、步 299/300 工人 [2,2]；步 296–300 × 双席共 10 根），运行器 `run_boundary.py`，结果 `boundary_results.json`。在 A/B 玩家上实测判定行为：

| 步 | A（墙钟） | B（sims） | 判定 |
|---|---|---|---|
| 296/297（4 根） | searched=0 | searched=0 | 无 key units（仅相捕/猎杀局面的 collector 够格），在 searched 自增前 return——非终局判定、与步限无关 |
| 298（2 根） | searched=1、shallow=1（本机伪影） | searched=1、depth=2、叶数 5、changed=0 | 搜索真实执行至 depth=2；叶以 `lft=max(0, left−depth)=0` 求值 |
| 299/300（4 根） | 不搜索、回默认、eq_default=True | 同左 | agent 层 `left=299−step`，`left<ENDGAME_STOP(=1)` 不搜索；步 300 根无异常 |

- **叶值来源**：`leaf()` 无终局分支——纯启发式（cash 差＋在基 cargo＋worker/tree 项、随 left 缩放），全文无灭绝/步限真值判定；步 297/298 根的 rollout 恰达 300 步界时，叶给的是 lft=0 的启发式值，**不是平台真值终局分**。FastCarbon 自身亦不停于 300 步（步限只在 agent 层以 left 体现）。SOP 点名的「_terminal 只认灭绝不认 300 步」在本包形态为：搜索侧根本没有终局分概念，297/298 步的改子（上一探针 38/120 处）正是在此启发式叶值下产生——构造侧事实如上，处置属预注册/执行书口径。

## 6. 五闸（数值＋判读＋涉及 md5）

### 闸一 量纲/定义独立复核 — PASS（已按注册改齐，无双轨）

- 数值：计数点 B 内嵌 cnet2_search line 488（leaf 入口）；sweep 每步叶数＝nc∈[2,16]（结构推导 §3，冒烟实测 [13,13,5,5,9,9,9,9] 与之合）；记录 sims/step 与 leaf_evals 在 SIMS∈{8,32,200} 三档逐根全等。
- 判读：计数单位从循环结构独立复推得「叶评估」为唯一统一点，与 PREREG_SEARCH.md §1 注册定义一致；实现已改齐（早期分支计数版已作废），不存在双轨。
- 涉及 md5：B 包 8e6f7a65d3d91b363606e711eaea03c2；内嵌 cnet2_search 694e2070fe4b03651e4cb264212454e8。

### 闸二 机制分布／构造期标定阶梯（正式版） — 停报：无满足档

配额法：DGX（spark-29db）上 `systemd-run --user --scope -p CPUQuota=150%`（cgroup CPU 配额＝1.5 核/实例），实测校验：4 进程 numpy 烧机在 Quota150 下 User/wall≈1.43–1.5、无配额时 ≈6.2——配额真实生效。驱动 `ladder.py`（md5 7ba3ffcad05cd62c070d920832fde0d5）：rootset32 全 6,336 根（步 [200,299)×双席）、逐根全新 agent、种子 20261114+root_index、每实例一个配额 scope。合并口径 `merge_ladder.py`（md5 330d42488613c61c3789236105717011）。

**逐档表**（五档同一计算，证明见下；实测统计取已完成的最大样本——B 窗口局 [200,300) 共 200 call，其中 searched 184、未搜 16；taskset 2 核亲和、逐 call 实测）：

| 档位 SIMS | (a) 单步 p95 | (b) 深度中位 | 备注 |
|---|---|---|---|
| 100 | wall 14.32s／CPU 28.60s | 5 | 与余档同一计算（结构证明＋冒烟两档实测等价） |
| 200 | 同上 | 5 | 同上（B 出厂占位即此档，窗口局实测本体） |
| 400 | 同上 | 5 | 同上 |
| 800 | 同上 | 5 | 同上 |
| 1600 | 同上 | 5 | 同上 |

- 单步全表（searched 步、n=184）：wall mean 12.60／p50 13.94／p95 14.32／max 14.58／**min 5.57**；CPU mean 25.14／p95 28.60／max 29.14。未搜步 wall mean 0.19。**最便宜的 searched 步（5.57s）亦破 3 秒硬顶**——p95 ≤1.0s 在任何样本完成度下都不可能成立。
- 深度分布全表（searched 步、n=184）：depth=5 共 184（100%），中位 5、p25/p75/p95 均为 5（该局窗口内 searched 步均在步 200–295 段、H=5；步 296/297/298 根的结构深度 4/3/2 见 §3 推导与 §5 实测）。
- 叶数检验：searched 步叶数＝nc∈[2,16]、恒 ≤16 « 100（冒烟与边界实测逐根 sims==leaf，§4）；改子率（同一计算下）：冒烟合成根 4/8（§4），rootset 口径待阶梯附录。
- 五档同一计算的证明：sweep 每步叶数 ≤16（§3 结构推导），SIMS≥100 时 keep=ceil(SIMS/1)≥100 > nc → 候选不截断、深度循环与 SIMS 无关、墙钟门在 sims 模式下豁免——五档走完全相同的代码路径与随机流；冒烟在 SIMS=32 与 200 两档实测叶数与落子行为等价（§4）为经验证据。



- 结构先行结论（与实测互证）：阶梯全档 SIMS≥100 » sweep 每步叶上限 16，预算恒非 binding → 五档是同一次计算，深度分布与单步成本逐档相同（证明与冒烟两档等价证据见表注）。深度＝H=min(H_cfg, left+1)：步 200–295 根 H=5、296→4、297→3、298→2，中位 5 ≥ 3 ✓（深度判读线过）。
- 判读：单步 p95 远超 1.0 秒线（见表，约 14 倍）→ **阶梯内无满足档，按注册停报：不降线、不自选次优；B 的 SIMS 终值未锁定**。此结论与机器争用无关：争用只放大 wall，CPU 口径同判（且配额内 wall≈CPU/1.5 已是下限方向）。
- 注记（在跑与抽样口径）：rootset32 阶梯本体的两跑在本回执落笔时仍在 DGX 后台运行（均 CPUQuota=150% 口径）——(i) 均匀抽样跑 120 根（跨 32 局、步 [200,299)），已运行约 47 分钟、累计约 3,900 CPU-s 未完：rootset 根含个别极贵根（单根成本超窗口局最大值 29.1 CPU-s），驱动只在完赛时落盘；(ii) 全量 rung-100 四分片（6,336 根）自约 12:15 起跑、同因未完，产物将落 `ladder/out_100_*.json`。两跑完赛后按 `merge_ladder.py` 口径合并、作本节附录补入（深度分布全表与 rootset 口径改子率以附录为准）。停报判读不依赖两跑：五档同一计算已由结构证明、且该计算的单步成本已由 n=184 的窗口局实测全表钉死（min 5.57s » 1.0s 线）。全量单档耗时估计约 25 机器时（争用下），是否继续保留两跑由父级裁量。

### 闸三 同一性机械 diff — PASS 21/21

- 数值：断言 21/21（CONF 逐键、DEFAULTS 逐键、六载荷字节、解码权重 md5、另三内嵌模块字节、变更行白名单归类、骨架同一），详 §2。
- 判读：A/B 与原包差异集合 ⊆ {SEARCH 翻转} ∪ {sims 相关}，程序证明而非人审；注册点名保留清单（TEACHER=2/GUARD=True/ROLL="self"/ABORT_DL=0.95/TEACHER_LATE=2/LATE_STEP=240）逐项在断言内。
- 涉及 md5：原包 7b7620499411a088cf4dc17c0e929bd7；A ea20a2714209e1a58b3e435b41d943bb；B 8e6f7a65d3d91b363606e711eaea03c2；解码权重 444cc54c0d88da5656ff4c7eef1ec8eb。
- 待草案项：包 vs PREREG 草案逐项 diff——注册已出稿（PREREG_SEARCH.md），本回执 §3/§6 已按注册对表（单位、保留清单、阶梯与终值规则）；「草案文本逐条 diff 表」如需另补，不阻塞本构造关。

### 闸四 边界用例 — 已实测（事实登记，见 §5）

- 数值：10 根判定表见 §5；步 298 B searched=1/depth=2/叶 5；步 299/300 零搜索零异常。
- 判读：玩家无真值终局判定、叶值为 lft=0 启发式——与 SOP 点名背景一致，构造侧已实测坐实；是否在 Part 1 执行书中处置（如步限口径）不在本趟授权内。
- 涉及 md5：B 8e6f7a65d3d91b363606e711eaea03c2（A 同形态、仅墙钟伪影差异）。

### 闸五 成本（1.5 核口径）与天梯硬规先判 — 锁定跑判读：sims 玩家不过、墙钟玩家临界

方法：DGX 上整局对局、逐 call 记 wall＋process CPU；整局局为 taskset 2 核亲和（驱动 `cost_driver.py` md5 7700561228ade01e6a2e4e0d12e823e2、`cost_driver_bwin.py` md5 4ddc9e38346967f5a6b2adb57917b4d6），闸二阶梯本体另以 CPUQuota=150%（1.5 核）口径在跑（注记见闸二）。单 call 以单线程逻辑为主、numpy 至多 2 线程，CPU≈专核时间的上限读数、wall 为争用下读数，两口径并报。

- **A（墙钟，BUDGET=0.7）全程局**（`cost_A.json`，600 call、searched 476）：全程 wall mean 0.871／p95 1.010／max 1.028；窗口 [200,300) mean 0.921／p95 1.016／max 1.028；>1s 共 85 call（超额合计仅 0.77s，预算自限）、>3s 为 0；首 call 0.871s（初始化 ≤20s ✓）；整局总时 522.6s（≤1500s ✓）。对硬规判读：3 秒硬顶与整场时限过；「慢步（>1s）≤3 次/局」在本机口径下不过（85 次，但每次超额 ≤28ms——考场机口径待正场前复测）。
- **B（sims）窗口局**（`cost_Bwin.json`，前 200 步 SEARCH=False 领跑、B 接管 [200,300)，200 call、searched 184）：wall mean 11.60／p95 14.32／max 14.58；CPU mean 23.16／p95 28.59／max 29.14；searched 步 184/184 全部 >3s（硬顶破）、未搜步约 0.19s；去前 10 call 后 CPU mean 23.55（冷启动非因）。对硬规判读：**单步 1s 基准与 3s 硬顶均大幅不过，整局总时按窗口外推约 5,800s+ 亦破 1500s**——与闸二停报互证：全深度 sweep 在当前实现下不是计时可行的载体形态（此为构造期事实，载体方向处置属策略组/注册 §1 判读线后续）。
- **B 全程局数据缺口（如实登记）**：B 对 B 全程局（旧版 B、动作与现行版逐字同）在 DGX 上运行 1:30:36 CPU 后进程消失、log 空、无产物（疑似被杀，无证据定性，不重跑）；全程总时以窗口局外推（估计）替代。
- **每千局机器日（估计）**：B 自对局单局 ≈ 600 call × 约 13–16 CPU-s（窗口 searched 步 25.1 CPU-s 为上包络、全程聚合约 15.5 CPU-s/call）≈ 0.09–0.17 core-day/局 → **每 1,000 局约 90–170 core-day（估计）**；折 DGX（20 核）约 4.5–8.5 机器日/千局（估计）。A 局同口径：600 × ~1.7 CPU-s ≈ 0.012 core-day/局 → 约 12 core-day/千局（估计）。
- **与探针① 1.096s 口径区别**：彼为 DGX 空闲时 16 worker 并行、逐根全新 agent、墙钟 BUDGET 模式下的逐根 wall p95（争用与并行口径）；此为 1.5 核配额/钉核下逐 step 的 CPU＋wall（阶梯与整局两种载体）——分母（每根 vs 每步）、并行度与预算模式均不同，不可直接互比。

## 7. 未决／后续

1. **sims 终值未锁定**：阶梯无满足档（§6 闸二停报）；B 的 SIMS=200 为占位。若 Part 1 仍以 sims 玩家推进，终值与计时可行性须由策略组按注册 §1 判读线另行裁定（降深度/改载体/换口径均非本趟授权）。
2. **开场条款（本趟不执行，仅登记）**：Part 1 种子段 986200–986999、spot-33/34 分片、开考前双机对包 md5 读回——均为开场前事项，本趟未执行、未用任何考试/探针种子（全程仅合成根＋rootset32 标定根）。
3. PREREG §5 成本账（试点 ≤800 局的单局成本/总成本/每千局机器日正式版）属 Part 1 执行书阶段义务，本回执 §6 闸五为构造期先判。
4. 闸二阶梯本体两跑（120 根均匀抽样＋全量 rung-100 四分片）在 DGX 后台继续（产物 `ladder/interim_100.json`、`ladder/out_100_*.json`），完赛后按 merge_ladder 口径补附录（深度分布全表 rootset 版＋改子率）；不影响停报判读（结构证明＋整局实测已互证，见闸二注记）。是否保留续跑由父级裁量。
5. 本地 VM 于本趟中途被替换一次（冒烟进程中断后补跑完成）；DGX 侧 B 全程局进程消失一件（§6 闸五登记）。

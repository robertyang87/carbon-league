# 设计稿：标注优化第二批（前缀复用＋根级任务队列）

- 日期：2026-11-18。性质：工程组设计稿，主代理已审（2026-11-18），开放点裁定见文末附记。
- 范围（所有者令锁死）：只做两项——① 前缀复用；② 根级任务队列。**跨根拼批不做**（浮点归约顺序翻转风险，等另行裁定；与在验的 leafbatch 分叉是正交的另一层，见 §7）。
- 纪律：策略组验收前一切 VALIDATION-ONLY；生产批继续用钉版对（k0_annotate `7f567a2e`＋stage0_teacher `c6ade5ce`），现行生产链一字不动。
- 审计对象（全部直读，md5 逐件合）：`code/k0_annotate.py` `7f567a2e`（1034 行）、`code/k0_annotate_snapshot.py` `e9c7a0fb`（1114 行）、`code/stage0_teacher.py` `c6ade5ce`、`code/stage1_teacher_gate.py` `e922d9d8`、钉版 `cnet2_np.py` `d8c15ad4`／`fastsim.py` `aefe6c83`／`cnet2_feat.py` `2a45b08b`／`cnet2_search.py` `94f7fb64`（取自 leafbatch bench 包 code_pin 在机件）；裁定 `reports/RULING_POINTA_GEN2_PIN_20261116.md`（`cae8fcbb`，md5 `e0392a83`）＋证据 `reports/eng_verify/compare_pointA_{994400,994401}_result.json`。

---

## 0. 结论先行：item① ＝复用 Point A 本体＋两处补强（非重做）

**判断：复用＋补强。** 依据三条（下文 §1–§3 逐条立证）：

1. 全文 diff 审计：`e9c7a0fb` 相对 `7f567a2e` 的差异**恰为三处、共 90 行 diff**——`_eval_root` 优先取 spec 内 scene（snapshot L336-341）、新增 `_game_root_scenes`（snapshot L377-451）、`_annotate_game` 每局先取全量 scene 再逐根评估（snapshot L454-465）。plan／选根 RNG／teacher_config／code_md5s／分片格式／校验／发布／池调度（仍按局 map）**逐字节未动**。Point A 改的正是所有者 item① 指定的那一层，且只改那一层。
2. 机制与令面逐项合：同局只开一次 gzip、只解析一次（snapshot L405-406）；一遍前向模拟沿途在每个被选步 capture 快照（L443-451 主循环＋L427-440 capture）；整局文件不再逐根重读。前缀重放从「每根 O(step)」降为「每局 O(steps) 一遍」。
3. 等价已按**恰全零**口径裁定在案（裁定 §1：两局各 24/24 根，max|Δq|＝0.0、有符号偏差 0.0、argmax 翻转 0、du Δ 0.0、arms 集合差 0、replay_sha 双局合；证据 JSON 本趟复读合）。本批要补的不是机制，是**证据面**（原验证窗仅 [200,212)、2 局 48 根）与一处**部署形态瑕疵**（见 §4.2）。

**RNG 核心问的定论（先行，§2 立证）：快照不存任何 RNG 状态——因为这条路径上不存在跨根存活的 RNG 状态可存。** 教师侧全部随机流都在 `evaluate_root` 内按根现构造（种子只依赖 `seed_base = seed_teacher + step` 与场景序号 m），前缀路径（reset/step/observe）经钉版代码逐处核查**零随机抽取**。故所有者硬要求「快照连 RNG 状态一起存、belief 采样序列与逐根重放逐位一致」在本构造下是**可证的空满足**：belief 采样序列由 (seed_base, m, 根处 Feat 的未知格集合）唯一决定，快照逐位复现 Feat 即逐位复现采样序列。Point A 的全零比对是这一论证在 48 根上的实测投影；本批验证协议（§6 框 B）把它升级为**流原文哈希级**的直接证伪测试。

---

## 1. 审计一：钉版逐根路径（k0_annotate `7f567a2e`）

### 1.1 每根的代价构成

池任务单元是**局**：`main()` 组 tasks＝每局一条（L1014-1015 区，见 L1018-1021 的 `ProcessPoolExecutor`＋`ex.map(_annotate_game, tasks)`）。`_annotate_game`（L375 起）对局内每根调 `_eval_root(spec, cfg)`（L336），其第一行即 `G.extract_scene(spec)`（L339）。

`extract_scene`（gate L240-279）对**每一根**从零重建现场：

- L244-245：`gzip.open(path)`＋`json.load` —— **整局回放文件逐根重读＋全量解析**；
- L247：`FastCarbon(CFG).reset(seed=1, full_carbon=帧0)`；L249-250：双座 Feat 新建；L258：双 guard 新建；
- L271-275：前缀循环 `for t in range(step)`：每步 `sim.step(记录动作)`＋双座 `Feat.observe`＋`observe_guards()` —— **每根 O(step) 次模拟步**，窗 [200,299] 的根平均重放约 250 步；
- 返回 scene dict：`{full, step, me, state=sim.get_state(), guard_snaps, game, fm, fo, x_me, aux_me, x_opp, aux_opp}`。

`_eval_root` 余下与 gate `_q_scene`（gate L477-495）同路：`_fresh_sim(scene)`（gate L523-526，`set_state(scene["state"])`）＋`T.evaluate_root(..., seed_base=seed_teacher+step, feats=(fm,fo), guard_snaps=_teacher_guard_snaps(scene, belief))`（gate L529-534：sample 模式只传 me 侧 snap）。

冗余量化（每局、窗内 598 根）：文件读/解析 598 次、前缀模拟步合计 ≈ Σstep ≈ 598×250 ≈ 15 万步——而一局本身只有 ≤299 步。**这就是 item① 要杀的全部冗余**；教师评估本体（每根约 596-680 次网络前向）不在本批范围（那是 leafbatch 的层）。

### 1.2 RNG 消费点（annotate 层）

- **选根 RNG**（决定选哪些根、不决定根的值）：`build_plan` L275 `np.random.default_rng(seed)`（seed 默认 12345），固定序消费——pass A 每局一抽（feed 序），pass B 对未中局逐根一抽（feed 序、(step,seat) 序）；选中集 sha256 入身份。两版此段逐字节同（diff 无此段）。
- annotate 层再无其他 RNG；教师层见 §2。

### 1.3 池粒度与落盘形态

- 粒度＝局（`ex.map` over games）：一局 598 根由**单个 worker 串行**算完才落片；尾段剩余局数 ≤ worker 数时其余 worker 全闲，且局长（ft）方差直接变成墙时方差。annot_a 实测一波约 3–3.4 小时（62 worker、50 局单波，墙时＝最长局）即此形态。
- 落盘：每局一分片 `OUT/shards/<game>.npz`（数组＋`_meta` JSON 内嵌、os.replace 原子落），progress.json 记每局 shard_sha256；resume 时逐片全验（`_verify_shard`：逐数组 sha、根键、身份、selection hash）后跳过，验不过点名 die、绝不静默重算。发布 `publish()` 按 gid 序拼接分片、逐字段 sha 封印（此形态 item② 必须原样保住，见 §5.4）。

---

## 2. 审计二：RNG 全图（核心审计问立证）

逐处核查钉版调用链上**全部**随机抽取点，结论：随机流只有三股，全部「按根现构造、根内消费、根终即弃」；前缀路径零抽取。

| # | 随机流 | 构造点 | 种子 | 消费点与序 | 跨根状态 |
|---|--------|--------|------|-----------|---------|
| R1 | 选根流 | annotate `build_plan` L275 | `seed`（12345） | pass A 逐局、pass B 逐根，固定序（§1.2） | 仅 plan 阶段，与评估无关 |
| R2 | belief 碳采样流（每场景一股） | teacher `evaluate_root` L400：`default_rng([seed_base+m, 0xC0])` | `seed_base=seed_teacher+step`、m∈[0,M) | `belief_carbon`（teacher L95-131）：对未知镜像轨道按 rep 升序、**每轨道恒抽 2 次**（L120 注释明示：流位置不依赖其他轨道结果）；抽多少轨道由根处 Feat 的未知格集合决定 | 无：每根每场景新建 |
| R3 | 对手根步采样流（每场景一股） | teacher L408：`default_rng(seed_base+m)` | 同上 | 经 `CachePolicy.act(sample_rng=rng_m)` 在 cnet2_np `Policy.act` 的采样分支消费（`d8c15ad4` L204-211：仅 `sample_rng is not None` 的槽位采样，其余 argmax） | 无：每根每场景新建 |

其余路径逐一验毕为**确定性**：

- 候选生成 `gen_candidates` 与 H 步续弈的 `act()` 均不传 sample_rng → 走 argmax 分支（cnet2_np L211），零抽取；
- `BankruptcyGuard.apply/snapshot/restore`：规则层，无 RNG（钉版 cnet2_search 全文 random 命中仅在与本链无关的 selfplay/opening 函数，且其种子亦为按 (seed,step,me) 现构造）；
- `Feat.observe`：`cnet2_feat.py` 全文零 random 命中；
- `FastCarbon.step`：全文 random 命中仅在 reset/_generate_map；`reset(full_carbon=...)` 在 fastsim L224-226 直接跳过 `_generate_map`——annotate 路径恒给 full_carbon（gate L247／snapshot L409-410），地图生成不可达；
- `sim.obs / full_observation / get_state / set_state`：纯状态读写。

**推论（item① 正确性论证的核心）**：任一根的评估结果是 `(scene, seed_base)` 的纯函数，其中 scene＝（sim 状态、双座 Feat、双 guard 快照、双座 x/aux）。前缀路径零随机 ⇒ 在步 t 处 capture 的 scene 与逐根重放到 t 得到的 scene 逐位同（同一串确定性调用、同一输入字节）。R2 的抽取量由 Feat 未知格集合决定——Feat 在 scene 内 ⇒ 采样序列随 scene 逐位复现。**「快照存 RNG 状态」在此构造下的等价实现就是「快照逐位存 Feat＋sim 状态＋guard 快照」，无额外 RNG 字段可存、亦无必要。** 任何调度（按局／按根／任意 worker 分配）不改变任一根的输入 ⇒ 不改变任一根的输出——这是 item② 「只改调度」的构造依据。

**在案实测投影**：裁定 `cae8fcbb` §1＋compare JSON 两份——窗 [200,212)、994400/994401 两局各 24 根，q/Q/du 全项 Δ 恰 0.0、翻转 0、arms 差 0。局限：样本窄（深窗、48 根、单一 (seed_teacher) 取值），且其依据的 FACT_SHEET_A.md **未入仓**（仅 snapshot docstring 引用，仓内 git tree 无此文件）——本节即其入仓替代，全部论断给钉版行号、可复核。

---

## 3. 审计三：Point A 快照存了什么、还剩什么冗余

### 3.1 快照内容（snapshot `_game_root_scenes` L377-451）

每局一遍：gzip 开一次（L405）、reset 一次（L409-410，seed=1＋帧 0 full_carbon，与 extract_scene L247 同参）、双 Feat/双 guard 随 pass 携带（L411-425 区）、每帧双 observe＋observe_guards（与 extract_scene 同序同组调用）。在每个被选步 `capture(step)`（L427-440）：组 scene dict——字段与 extract_scene 返回**逐项同**（full/state/guard_snaps/fm/fo/x_me/aux_me/x_opp/aux_opp/game/step/me，别名关系亦同：aux_me 即 fm.aux）——并当场 `copy.deepcopy`（L440），每根拿到独立对象图，等价于逐根重建的隔离性。`_annotate_game` 先取全量 scenes（L462）再按 plan 根序逐根以 `spec["scene"]` 注入（L465），`_eval_root` 命中 scene 即跳过 extract_scene、否则回退原路（L336-341）。

**没存的东西**：RNG 状态（§2：不存在）、回放解析结果（steps 数组只活在 pass 内）、sim 本体（只存 get_state() 字典——与钉版 _fresh_sim 的 set_state 口径本来就同）。

### 3.2 还剩的冗余与未治项（本批/后续的边界）

1. **池仍按局 map**（snapshot L1101 与钉版同）——尾段闲置、局长方差未治。→ item② 的靶。
2. **plan 阶段每局还读/解析一次**（build_plan 枚举根＋哈希）：snapshot 后每局总读数＝2（plan 一次＋scene pass 一次），钉版是 1＋598。plan 那次是纯解析（无 sim/observe），毫秒量级；所有者「只读/只解析一次」的靶心是逐根重复，本稿建议接受这 2 次（开放点 O-3）。
3. **scene 全量驻留**：一局最多 598 个 deepcopy scene 驻留 worker 内存至该局算完（Feat est＋sim state 量级，单 scene 约百 KB 级、单局约数十 MB），62 worker 并行下总量可控（128 GiB 机实测口径待验证协议顺带记录峰值 RSS）。
4. 教师评估本体（每根约 600 次前向）未动——leafbatch 的层，本批不碰。

### 3.3 等价口径回顾

Point A 当时按**构造精确**过线：预注册「等价全零唯一过线、无容差后路」（裁定卷首），实测恰全零。本批二次验证沿用同口径并扩样（§6）：验收线写 ≤1e-6 是所有者本批令面，**预期与报告口径仍是恰 0.0**——任何非零 Δ 即按构造被破坏处理、先定位后谈容差。

---

## 4. 设计①：前缀复用（复用＋两处补强）

### 4.1 复用清单（逐字搬入新文件，不改一字）

- `_game_root_scenes` 全函数（snapshot L377-451）；
- `_eval_root` 的 scene 注入分支（snapshot L336-341）；
- `_annotate_game` 的 scenes 取用（snapshot L462-465）——在 item② 下其形态被装配器取代（§5.4），但根行构造 `_root_row` 与数组组装块逐字复用；
- 其余全文件（plan/verify/publish/validate/main 骨架）以 snapshot 版为底本。

### 4.2 补强一：身份文件名瑕疵修正（部署形态）

snapshot 的 `code_md5s()`（L198-202）与 `publish()` producer 块硬编码文件名 `"k0_annotate.py"`：该文件在仓内名为 `k0_annotate_snapshot.py`，只有**部署时改名成 k0_annotate.py** 才自洽（gen-2 当时即如此暂存上座）。新文件按自身文件名登记（fns 列表与 producer 键同步改），杜绝「仓内名≠身份名」。此改只动 code_md5s 字典的键与对应 md5——身份本就因新文件而变（§5.5），config 不受影响。

### 4.3 补强二：证据入仓

本稿 §2 的 RNG 全图随实现件一并入仓（设计文件即证据文件），FACT_SHEET_A 的论断从此有在仓出处；验证协议框 B 的流哈希测试把「零抽取」从静态审计升级为运行时可证伪（§6）。

### 4.4 窗口与快照对应关系（写明防误读）

- 根窗 [lo,hi) 只是 plan 的选根集：scene pass 恒从步 0 走到 `max(被选步)`（snapshot L443-451），窗外步不 capture、窗内每被选 (step,seat) 各一份；[0,299] 全窗时 capture 598 份/局。
- pass A/B 稀疏选根（pi0/pi1<1）时 pass 只走到最大被选步——尾段未选步不模拟，与钉版逐根只算被选根同口径（钉版对未选根同样不付前缀代价之外的任何代价；被选根的前缀内容不因邻居未选而变，§2 推论）。

---

## 5. 设计②：根级任务队列（只改调度）

### 5.1 总形态（三段，单机多进程）

以 snapshot 版为底本新文件（暂名 `k0_annotate_queue.py`，命名待裁 O-1），改动**只许**落在：任务生产/分发、根 worker 入口、结果装配、根级 resume 账本、code_md5s/producer 键名。其余函数逐字。

- **P 段（plan，主进程）**：`build_plan` 逐字。选根 RNG 在此一次性、单进程、固定序消费完毕——调度再乱，选中集与 selected_sha256 不变。
- **S 段（scene 生产者进程 ×p）**：领「局」任务 → `_game_root_scenes(pg)` 逐字跑 → 把该局每根的工件 `(gid, game, step, seat, scene)` 逐根投入根队列（有界队列背压，见 5.2）。生产者不做教师评估。
- **E 段（根评估进程 ×e）**：领根工件 → 组 spec（与 snapshot L465 同构）→ `_eval_root`＋`_root_row` 逐字 → 结果 `(gid, step, seat, row)` 入结果队列。评估进程按 `_init_worker` 逐字从 plan 冻结的 weights_bytes 载教师。
- **A 段（装配，主进程兼任）**：收结果、按局攒行、按 §5.4 装配分片＋记 progress；兼任失败回收（§5.3）。

p＋e＝--workers（默认配比与队列界见 O-4）。进程模型：`multiprocessing` fork 上下文（weights_bytes 与 plan 经 fork 全局继承，不走队列重复序列化）；队列用 `JoinableQueue`（根队列）＋普通 Queue（结果）。

### 5.2 动态领取与背压

- 根队列 `maxsize = 4×e`（根工件含 scene、百 KB 级；界住在途内存 ≈ 4e×scene 大小，62 档约数百 MB 上界，实测量级验证时记录）。
- 生产者整局 scenes 先成 dict 再逐根投递（`_game_root_scenes` 逐字性优先于流式 yield；内存账见 §3.2-3，接受）。
- 终止：全部局投递毕，生产者投 e 个哨兵；评估进程领到哨兵退出；装配端在「分片齐或缺口已回收重投至齐」后收尾走 publish（逐字）。

### 5.3 失败/回收口径

- **根级幂等**：根评估是纯函数（§2 推论），重投任意次结果不变——回收策略可以粗暴简单：宁重投不丢根。
- **worker 进程死亡**：装配端周期核对子进程存活；死者的在途根（已领未回）由装配端在局装配时按缺口检出（期望根集－已回根集），自动重投根队列（scene 需重产：该局由生产者重跑 `_game_root_scenes` 只补缺口根——实现上整局重产、只投缺口根，pass 成本一遍、评估不重付已回根）。
- **整机被回收（spot）**：分片仍是原子记录单位（已落分片 resume 时全验跳过，逐字沿用）；未完局的已回根靠**根级账本**保住：装配端把每局已回根的行追加写 `OUT/.partial/<game>.npz`（原子替换、每攒满 16 根或 5 秒刷一次）＋progress.json 同步记该局 done_roots 计数。resume 时：已验分片跳过；有 .partial 的局载入已回行、只重投缺口根；无 partial 的局整局重投。账本只记行与键，不记任何调度序信息——产物身份不含执行序（§5.5）。
- 账本/partial 是本批**唯一新增的持久化代码**（约占新增代码一半）；若主代理/策略组要压工期，可降级为「只保分片级 resume」（开放点 O-5），功能正确性不受影响、仅回收粒度变粗。

### 5.4 分片装配（与钉版同构的精确含义）

- 装配端按 **plan 根序**（pg["roots"] 的枚举序＝(step 升序, seat 0/1)）排行——到达序只影响等待、不影响行序：`rows` 列表构造、`_root_row` 输出、数组组装块、meta 构造、`np.savez`＋os.replace 全部逐字取自 snapshot `_annotate_game` 后半（抽为共用装配函数，钉版形态与队列形态共用同一份代码文本）。
- 同构口径（按系统实际校验的粒度声明，不多报）：**逐字段数组字节相等（meta 内逐数组 sha256 与参照分片全等）＋meta JSON 相等**；npz 容器字节（zip 条目元数据）不在既有校验面内（`_verify_shard` 验的是逐数组 sha＋progress 记录的本轮文件 sha，自洽于本轮），不声称容器逐字节等。
- 装配毕的分片走原 `_verify_shard` 自验后才记 progress——与钉版同一道闸。

### 5.5 config 不变与身份变化清单

- **config_sha256 不变（构造证明）**：`teacher_config(a)`（snapshot L188-196）只读 K/M/H/teacher_style/belief/train_temp/behavior_temp/seed_teacher/guard_cfg/window/scene_contract——**无任何调度字段**（--workers 本就不在其中，新参数如生产者数/队列界亦严禁入此函数）；config_sha256＝sha256(canonical(teacher_config))（build_plan 内，逐字不动）。scene_contract 串保持 `"prefix-feat-guard-snapshot-v1"`（scene 的内容契约未变：同字段、同别名、同 capture 时点）。**实证义务**写入验证协议：同语料双跑 build_plan，config_sha256／selected_sha256／games_sha256／replay_sha256 四项逐字比对（仅 code_md5s 许差）。
- **身份变化（预期内、逐项列）**：code_md5s 中 annotator 一键换为新文件名＋新 md5（§4.2）；其余六件（stage0_teacher `c6ade5ce`、cnet2_search `94f7fb64`、cnet2_np `d8c15ad4`、cnet2_feat `2a45b08b`、fastsim `aefe6c83`、stage1_teacher_gate `e922d9d8`）**必须逐件不变**——本批验证的配对教师锁钉版 `c6ade5ce`（annot_a 的配对），不叠 Point B 教师 `d446e5b6`（叠栈与否另行裁定，O-6）。weights `06a03d9c` 不变。身份变 ⇒ 新 generation、与钉版产物不混 join（既有身份闸自动执行）。

---

## 6. 二次验证协议（排在 66 同机对照出结果之后执行；本趟只设计）

### 6.1 座位与参照系

- **验证专机（后勤另购）**：**c8a、64 vCPU／128 GiB、M 区**——与 annot_a 生产座 spot-65 同机型同规格。理由：① 参照分片本身是 c8a 上的钉版产出，同档座是同口径比（尤其速度面）的最干净前提；② c9ae 档（67/68）保护期后约 23–33 分钟两度被回收，活不过约 3.5 小时的验证窗，不可用。
- **参照系**：annot_a 的 50 局（selfplay-300002–300051）钉版分片（L20 在案；钉版对 `7f567a2e`＋`c6ade5ce`、config `3cc2e67e`、窗 [0,299]、62 worker、墙时约 3–3.4 小时/波）。
- 候选形态：新文件＋教师 `c6ade5ce`＋同 weights/同 games/同窗/同 seed 全集，VALIDATION-ONLY 独立 out 目录。

### 6.2 框 A：逐根全量对账（正确性主闸）

- 抽样框：**12 局 × 全根窗 [0,299] ≈ 7,176 根**。选局规则（拟，O-2）：在 50 局中取 ft≥299 的满长局、按 feed 序等距抽 12 局，名单在跑前冻结写入验证 manifest。
- 比对项（逐根）：priors／q／Q／Q_std 的 max|Δ|（按步带 [0,100)/[100,200)/[200,299] 分报＋全局）、arms 集合差、argmax 翻转逐根点名、belief/n_unknown 相等、n_act/guard 计数相等；逐局：候选分片逐数组 sha256 与参照分片逐数组比对。
- 闸：所有者令面 **max|Δ| ≤1e-6、翻转＝0**；按 §3.3 的构造口径，**预期恰 0.0**——出现任何非零先按构造破坏定位（查 scene capture／装配序／身份配对），不许直接引容差放行。
- 身份面附带：双跑 plan 四哈希比对（§5.5）随框 A 一并交。

### 6.3 框 B：RNG 序列专项（证伪设计）

目的：把 §2 的静态论证变成运行时可证伪——直接比对随机流原文，而非只看其下游投影。

- **B1 belief 流**：验证专用探针（不进生产文件）在 stage0_teacher 命名空间内给 `np.random.default_rng` 包一层记录器，跑 (3 局 × 步 {200,230,260,290} × 双座 × M=8 场景）根：记录每股流的 (种子参数、抽取次数、全部抽取值的 sha256)。钉版路（逐根 extract_scene）与候选路（快照 scene）各跑一遍，逐股比对——须全等。
- **B2 对手流**：同探针记录 R3 流（seed_base+m 股）的抽取序列哈希，同法比对。
- **B3 阴性对照**：探针故意把 seed_base 偏移 +1 重跑一根——哈希**必须**全变；不变则探针无效、先修探针。此条防「记录器没接对流」的假绿。
- **B4 群体投影**：框 A 的 7,176 根恰零要求本身即 R2/R3 等价的群体级证据（belief 采样任一股错位都会以 ≫1e-6 量级改 Q）；B1/B2 负责把万一的非零定位到具体流。

### 6.4 框 C：速度实测（同机同局）

- 主口径（同座 A/B）：验证专机上对框 A 的 12 局，钉版（`7f567a2e`＋`c6ade5ce`）与候选各跑一遍全窗（62 worker、其余参数逐项同 annot_a 口径），报逐局墙时＋总墙时＋比值；钉版 12 局墙时上界按 annot_a 波速推算约 0.8 小时（估计），全验证窗上界约 3.5 小时（估计，含候选 50 局全跑）。
- 副口径（生产形态）：候选在专机上跑满 50 局的整波墙时，对 annot_a 在 spot-65 的约 3–3.4 小时（同档异座，标注为参考口径）。
- 分项归因：① 的贡献以「同为按局 map、仅换 snapshot 底本」的历史读数（Point A 约 15%，裁定 §3 估算）为参照；② 的贡献＝候选（①＋②）对 ① 单独形态的同座差值——验证时加跑一档「snapshot 底本按局 map」12 局即可三分归因（成本约一局波的 1/4，可选，O-4 附带）。
- 不预先承诺提速数字：结构上限只作定性——50 局/62 worker 单波形态下，钉版墙时＝最长局，队列形态墙时下限≈总根活量/62＋最长单根，尾段回收量即两者之差；实测为准。

### 6.5 时序

66 同机对照（leafbatch）出结果 → 策略组确认本设计 → 建成 → 本协议执行 → 三件套回执（设计说明＝本件定稿、实测提速＝框 C、对账结果＝框 A＋B）交策略组验收。验收前候选不进任何生产批、不进 join。

---

## 7. 风险与不做项

- **不做：跨根拼批。** 本批任何根的教师评估调用与钉版逐字同（同函数、同入参、同 act 次数）；批处理只许出现在 leafbatch 线，本批不引用、不叠加（叠加与否 O-6 留裁）。
- **不做：教师/门/gate 任何文件改动**；六件共享 md5 逐件锁死（§5.5），跑前跑后各验一次。
- 风险 R-1（capture 别名/深拷贝）：Point A 已以 deepcopy 化解且 48 根实测零差；本批框 A 把步带扩到 [0,100)——早步 Feat/guard 状态（lostw 累积、est 初值段）是原样本未覆盖面，正是扩样的理由。
- 风险 R-2（队列 IPC 与内存）：根工件 pickle 过队列，单根百 KB 级；有界队列＋生产者 dict 驻留的内存上界见 §5.2/§3.2-3，验证时实测峰值 RSS 入回执。
- 风险 R-3（装配单点）：装配＝主进程，与钉版 main 同位；partial 账本把整机回收的损失界在「未刷盘的 ≤16 根/局」（§5.3）。
- 风险 R-4（fork 继承的教师缓存）：CachePolicy 在评估进程内各自构建（`_init_worker` 逐字），禁止跨进程共享可变缓存对象——队列只传 scene/row 纯数据。
- 生产隔离：验证全程独立 out、独立目录、VALIDATION-ONLY 标记；候选文件不进生产座（64/65/66）的部署树。

## 8. 开放点（待主代理/策略组裁）

- **O-1 命名与身份键**：新文件暂名 `k0_annotate_queue.py`；code_md5s 键名随文件名（producer 块同步）——下游 join 对 code_md5s 的校验是按值集还是按固定键名，注册时须一并核（若按固定键名，保留旧键名仅换值是备选）。
- **O-2 框 A 选局规则**：拟「ft≥299 满长局、feed 序等距 12 局、跑前冻结名单」；备选分层（长/中/短 ft 各 4 局）——满长局根数齐整（598/局）利于根数对账，分层利于覆盖短局尾段形态。
- **O-3 plan 二次解析**：拟接受每局 2 次读（plan 解析＋scene 解析）；若所有者「只读一次」按字面执行，需把 plan 改为向生产者直传回放字节（plan 需新增字节留存/落盘，改动面与内存面都涨）——建议不做。
- **O-4 配比与归因档**：生产者数默认拟 max(2, workers//8)、根队列界 4×e；框 C 是否加跑「①单独形态」做三分归因（多约 0.2 波成本）。
- **O-5 根级 resume 账本**：拟含（§5.3）；可降级为仅分片级 resume 以压工期（回收时整局重跑，正确性无损）。
- **O-6 与 Point B/leafbatch 的叠栈**：本批只验「①＋② 对钉版教师」；与教师 `d446e5b6`（Point B）、与 leafbatch 教师分叉的叠加组合各自另行对账裁定，本设计不预设。

---

## 附：建成面速览（供建成 ETA 估算）

- 新文件 1 件（约 1,100 行量级，其中约 85% 逐字自 snapshot 底本）；真正新写/改写约 300–380 行：main 分发重构（约 120 行）、根 worker＋装配共用函数（约 80 行）、partial 账本（约 100 行，O-5 可砍）、code_md5s/producer 键名（约 10 行）。
- 验证工具 2 件（约 200 行）：逐根对账器（扩自 shard_compare 口径）＋RNG 流记录探针（框 B）。
- 无新算法、无教师/gate/sim 改动、无 config 变更；教师配对锁 `c6ade5ce`、weights `06a03d9c`。

---

## 附记：主代理开放点裁定（2026-11-18，入仓版）

- **O-1**：新文件定名 `k0_annotate_queue.py`，code_md5s 键名随自身文件名（producer 块同步）。建成时须核 k0_join／k0_join_u 对 code_md5s 的校验方式（值集 vs 固定键名），结果写入建成回执；若为固定键名校验，先报主代理再定键名，不许静默沿用旧键。
- **O-2**：框 A 选局照拟案——ft≥299 满长局、feed 序等距 12 局、跑前冻结名单入验证 manifest。每根计算是 (scene, seed_base) 的纯函数，满长局已覆盖全部步带；短局不产生新步值，不另设分层。
- **O-3**：接受每局 2 次读（plan 解析＋scene 解析），不按字面再压。所有者「只读一次」的靶心是逐根 598 次重复，plan 的单次纯解析为毫秒量级。
- **O-4**：生产者默认 max(2, workers//8)、根队列界 4×e，照拟案。框 C **加跑**「①单独形态（snapshot 底本按局 map）」12 局做三分归因（钉版／①／①＋②），多约 0.2 波成本，接受。
- **O-5**：根级 partial 账本**含**（照 §5.3 拟案，约 100 行，已在建成面估算内）。
- **O-6**：本批配对教师锁钉版 `c6ade5ce`；与 Point B 教师 `d446e5b6`、leafbatch 分叉的叠栈各自另行裁定，本批不预设、不引用。

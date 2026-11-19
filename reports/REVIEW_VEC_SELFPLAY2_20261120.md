# 独立深审报告：vec_selfplay2.py 与 gate_vec2.py（修复版）

- 审查立场：独立审查，只审不修；审查期间未修改任何被审文件，未使用 L20，未落仓。
- 被审对象（本机文件，md5 审前核对一致）：
  - `~/workspace/genexp/vec_selfplay2.py`，md5 `d702d8bc78a8ba00437e2a0ed40d933b`，670 行（下称「候选」，行号记作 V2:nnn）。
  - `~/workspace/genexp/gate_vec2_fixed.py`，md5 `cb4b7c904460122df06cf2df6d462de6`，376 行（下称「闸」，行号记作 G:nnn）。与原版 `gate_vec2.py` 经 `diff` 实测仅第 222 行一词之差（`concatenate`→`stack`），其余逐字相同。
- 对照原件（经 `gh_api.get_file` / git blobs 从 `robertyang87/carbon-league` main 逐字直读）：
  - `tools/katago_m3s2/vec_selfplay.py`，27,313 B，md5 `e59f0700…`（下称「现行」，行号记作 V1:nnn）——等价性的定义方。
  - `tools/katago_m3s2/model.py`（MainNet.forward 输出形态）、`tools/katago_m3s2/bridge.py`、`tools/katago_m3s2/distill_roll.py`（按引用读）、`tools/katago_m3s2/layout.py`（`new_game_state` 的全局 RNG 语义，行号记作 L:nnn）、`tools/katago_m3s2/train.py`（下游消费者，行号记作 T:nnn）。
  - 底盘源 `data/master3s2_source.py`（经 git blobs 取回，2,897,684 B，md5 `57f9a594cce4f2ec9e3573d8d9d26ca1`，与注册的 Master3s2 源身份逐字合；下称「底盘」，行号记作 C:nnn）——仅用于核 `_sample`/`next_state` 的 RNG 消费方式。
- 方法：AST 逐函数提取做归一化文本 diff（把 `client.forward(...)`↔`yield (...)`、`yield from`、函数名差异归一后比对）；numpy 桩实测调度器/切分/闸布尔逻辑（本机无 torch，前向数值不在本次范围，凡依赖数值行为的结论均标注「代码级判断」）。
- 移植逐字性总证据（下文多项引用）：`_onehot_logits`、`sample`、`probs`、`policy_targets_from_probs`、`masks_for`、`write_shard` 六函数 AST 文本逐字相同；`fast_action`/`deep_move`/`play_game` 归一化后仅剩签名换行与 docstring 一句之差，函数体逐字相同。

---

## ① 拼批前向的索引/掩码对齐

调度器本体：V2:512–570 `run_pool`。每轮把 `pending` 中全部未结请求按 tag 分组（V2:547–551，按 `active` 字典插入序遍历），同 tag 内 `np.concatenate` 拼批（V2:554–555），后端返回后以同一请求序、按各请求行数 `n=len(Xk)` 顺序切片回填（V2:557–561）。现行对应物：推理服务器 `_server_main`（V1:137–161）同为按 tag 分组拼接、按请求行数顺序切片回填——拼/拆算法结构同构，差别仅在拼批时机（现行是队列 opportunistic，上限 256 个请求；候选是锁步整轮全拼）。

| 编号 | 检查点 | 结论 | 双方行号 | 证据 |
|---|---|---|---|---|
| 1.1 | 行序构造与按行路由互逆 | **合** | V2:547–561；V1:143–161 | 拼接序＝`reqs` 列表序，切片以同一列表序、同一 `n` 累进 offset，二者严格互逆。numpy 桩实测 T1：7 局×3 请求、每请求行数 1/4/2 交错，后端以行标记回填，每局收到的 `v`/`lw` 与其发送标记逐行全等，零串位。 |
| 1.2 | 不同 tag（cur/m3s2/hof）分组不串位 | **合** | V2:549–556；V1:137–145 | 按 tag 建独立批次、独立后端调用，回复字典按 key 回填，不跨 tag 切片。桩实测 T1 中 cur/m3s2 交错请求同轮共存，后端调用记录为 `[('cur',3),('m3s2',12),('cur',6),…]`，各局标记全对。 |
| 1.3 | batch-cap 切分余数/尾批 | **合** | V2:495–500（`TorchBatchBackend.forward_batch`） | `len<=cap` 单块；否则按 `range(0,len,cap)` 切分、三元组分别 `concatenate` 回拼，顺序保持。桩实测（cap=4，stub `_forward_chunk`）：n=1→[1]、4→[4]、5→[4,1]、8→[4,4]、9→[4,4,1]，回拼后行序与输入逐行全等。 |
| 1.4 | padding 污染 | **合** | V2:445–452、481–500、554–555 | 候选全程**不存在 padding**：拼批是精确行拼接，后端 Numpy 路径逐行精确循环、Torch 路径精确切片，无补行、无掩码需求，故无 padding 污染通道。现行同样无 padding（V1:91–100、143–161）。 |
| 1.5 | 空池边界 | **合** | V2:512–570 | 桩实测 T2：`next_game` 恒 None → 返回 `{}`，无后端调用、无异常。 |
| 1.6 | 单局/池未满 | **合** | V2:540–545（`_fill`）、572 起 main | 桩实测 T3：pool_size=32、仅 1 局 → 正常完成、路由正确。`_fill` 在不足额时以 `next_game() is None` 退出，不虚构局。 |
| 1.7 | 零行请求 | **合** | V2:554–561 | 桩实测 T4：0 行请求与 2 行请求同轮共存，0 行局收到形状 `(0,225,5)/(0,225,3)/(0,)` 的空切片，后续请求不受影响。注：生产路径中零行不可达——fast 恒 1 行（V2:209），deep 滚动段 `live` 非空才 yield（V2:287–296），叶值段有 `if leaf:` 守卫（V2:316）。 |
| 1.8 | 中途补局（refill）的路由 | **合** | V2:562–566 | 每轮 advance 后 `_fill()` 补新局，新局首请求进 `pending`、次轮入批。桩实测 T1（pool=3、7 局）含多轮补局，全部标记对齐。 |
| 1.9 | 单局异常的传播 | **合** | V2:518–528、642–647 | 桩实测 T6：一局抛 `ValueError` → `run_pool` 原样上抛，main 捕获后打印 `SELFPLAY2 FAILED` 并返回 1，不存在「死局之上报完成」。与现行的失败传播纪律（V1:599–609 按 worker 退出码/零分片判失败）等价。 |
| 1.10 | 重复 key 的健壮性 | **不合（健壮性项，生产不可达）** | V2:540–545 | 桩实测 T5：`next_game` 若返回重复 key，后一局在 `active[key]=gen` 处静默覆盖前一局，前一局生成器被孤儿化、永不完成且无报错（2 局输入只出 1 条结果）。生产调用方 `next_game` 以单调递增 gid 为 key（V2:626–633），不可达；仅作调度器单独复用时的契约缺口登记，不构成对当前生成链的判读影响。 |
| 1.11 | 后端返回行数契约 | **合（结构保证，无显式断言）** | V2:557–561；V1:156–161 | 切片逻辑不校验后端返回总行数＝请求总行数；但两个后端（V2:445–452、481–500）的输出均由同一批输入的前向按行构造，行数由构造保证。现行服务器同样无此断言。登记为观察项：若将来替换后端实现，此处无防御。 |

补充（代码级判断）：拼批的尾形状一致性由构造保证——同 tag 全部请求均出自同一 `encode`/`_scalars`（fast 为 `x[None]`，deep 滚动/叶段为 `np.stack`），尾形状恒为特征图形状与 `(5,)`；`np.concatenate` 若遇尾形状不一致会当场抛错（fail-loud），不存在静默错位通道。deep 滚动段对手侧前向复用我方 `SCm`（V2:295）是现行原样（V1:380）的逐字保留，非候选引入。

---

## ② 每局 RNG 流派生与消费顺序

对照总表（消费点｜现行行号｜候选行号｜合/不合）：

| 编号 | 消费点 | 现行 | 候选 | 结论 |
|---|---|---|---|---|
| 2.1 | 种子派生式 `seed + 1000003*gid` | V1:519（`_worker_main`） | V2:629–631（`next_game`） | **合** |
| 2.2 | `% 2**32` 包裹（种子进入 RNG 的唯一点） | V1:442 | V2:360 | **合**（函数体逐字同） |
| 2.3 | per-game `RandomState(seed)` 构造 | V1:443 | V2:361 | **合** |
| 2.4 | 全局 RNG：`layout.new_game_state` 内 `np.random.seed(seed)`＋`uniform`/`choice` | L:158、163–164（双方共用同一 layout，未改） | 同左（V2:362 调用） | **合**，见下方专项论证 |
| 2.5 | 对手抽签：`r = rng.rand()`，`<0.5` mirror／`<0.8` m3s2／否则 hof | V1:446–453 | V2:364–371 | **合**（逐字同，含短路语义：无 hof 时 `r>=0.8` 不消费 `randint`；有 hof 且落入第三支才 `rng.randint(len(hof_tags))`，V1:452／V2:370） |
| 2.6 | deep/fast 抽签：每步每座 `rng.rand() < deep_frac` | V1:466–467 | V2:384–385 | **合**（逐字同，含 `and` 短路：`smoke`/未记录座位/非 cur 座位不消费该 `rand()`；消费顺序为座位 0→1、步序递增，两版一致） |
| 2.7 | fast 采样：`sample(..., rng, ...)` 透传 per-game 流 | V1:294（`fast_action`） | V2:210（`fast_action_g`） | **合**（`sample` 函数逐字同，V1:212／V2:121；底盘 `_sample` 的消费方式见下） |
| 2.8 | deep 内部随机：`srng=RandomState(seed0)`，`seed0=1000003*(step+1)+17*me mod 2^31` | V1:309–310 | V2:225–226 | **合**（逐字同） |
| 2.9 | deep 对手响应：独立 `RandomState(seed0+7)`／`RandomState(seed0+38)` | V1:357–360 | V2:273–276 | **合**（逐字同） |
| 2.10 | deep 滚动段：独立 `RandomState(seed0+7919*ply+31*mi+1/+2)` | V1:382–383 | V2:298–299 | **合**（逐字同） |
| 2.11 | game rng 在 deep 内的（不）消费 | V1:298–427 | V2:214–343 | **合**：AST 实测两版 `deep_move` 体内对裸 `rng` 的属性调用均为 0 次——game 流只作形参透传，deep 的全部随机性来自 2.8–2.10 的 (step,seat) 确定性种子；候选以 `yield` 拆开 deep 的前向段不改变这一点（yield 点之间无任何 rng 消费，恢复后消费序与顺序执行一致）。 |
| 2.12 | 底盘 `_sample` 对 rng 的消费（经候选/现行共同的 `sample` 包装调用） | C:1164–1205 | 同左 | **合**：底盘 `_sample(players, me, lw, lb, rng, step, fixed)` 仅以传入的 rng 做 `rng.choice(3, p=)`（每 base 一次，base 字典序）与 `rng.choice(5, p=)`（每 worker 一次，worker 字典序过滤本座），消费次数由盘面状态决定；两版传入的是同一构造、同一消费历史的 per-game 流，给定相同盘面与相同 logits 则消费序列一致（代码级判断，数值等价归闸）。`_deviations`（C:1206 起）无 rng 参数、纯确定性排序，两版调用点逐字同（V1:354／V2:270）。`next_state`（C:700 起）全文无随机调用，确定性。 |

专项论证 2.4（锁步下全局 RNG 的干扰面）：`new_game_state` 是四个受审/对照文件中唯一的全局 RNG 消费方（L:158 `np.random.seed(seed)` 后 L:163–164 连续消费）。候选中它在 `play_game_g` 协程体内、首个 `yield` 之前被调用（V2:362），协程的首次 advance 在 `_fill` 内同步执行至首个 yield 返回——即一局的建局全局消费在任何挂起发生前已原子完成，不会与另一局的建局交错。补局发生在运行中途、会再次重播全局种子；其安全性取决于「在飞局在挂起后是否再消费全局 RNG」：经底盘源全文检索（md5 `57f9a594…`），底盘内 `np.random` 仅出现在 `RandomState(...)` 构造（C:576 系 ladder `_act` 路径的 per-state 流、vec 路径不调用；C:1114–1115 为自检；C:1278 起为搜索内部确定性种子），**不存在对全局 RNG 的抽取调用**；`next_state`/`encode`/`_sample`/`_deviations` 均不碰全局流。故对 master3s2 底盘，补局重播全局种子不扰动在飞局——**合（代码级判断，已以底盘源全文检索为据）**。fix2/fix3 等其他 `--src` 底盘不在此结论覆盖内（见未覆盖项）。

专项论证（局间干扰总判）：每局持有独立 `RandomState` 对象（V2:361），调度器全文不触碰任何局的 rng（V2:512–570 无 rng 引用）；协程的交错执行只改变局间相对时序，不改变任一局内部的消费顺序（每局的抽取都发生在其自身 advance 的同步段内）。共享的单一 chassis 模块实例（候选 V2:623 `load_module(..., 'sp_src_pool')` 一份 vs 现行每 worker 一份）引入的唯一可变全局是 `_TEMP`：`sample`/`probs` 对 `_TEMP` 的写-用-恢复在单个同步调用内以 try/finally 完成（V2:121–170，与现行逐字同），调用内无 yield，锁步不可能在 `_TEMP` 被改写的窗口内切换到另一局——**合**。其余模块全局（`_POLICY`/`_SID`）在受审路径中只读；ladder 用的 `_ST` 信念状态（C:576 所在路径）vec 路径从不调用。

2.13 同种子重跑确定性：**合（代码级判断）**。候选的拼批装配序由字典插入序与补局序决定，二者均由种子与前向输出确定性导出；无墙钟、无队列竞速（对照现行服务器的 opportunistic 拼批反而带时序不确定性）。须注明：确定性以「同 `--pool-games`、同 `--batch-cap`、同后端数值」为条件——改变池大小/切分会改变批量构成，从而（在 BF16 等批量敏感数值下）可能改变轨迹；此点候选 docstring（V2:41–44）已明示，不构成与现行的口径差，因为现行本身亦不保证跨批量构成的轨迹一致（其批量构成由队列时序决定）。

---

## ③ 落盘 schema 字段级对照

`write_shard` 两版 AST 逐字相同（V1:505–510／V2:424–430）：键序列表字面相同，`np.stack` 按行堆叠，`np.savez_compressed(f, **arrs)` 落盘。记录构造在 `play_game` 体内、归一化后逐字相同（V1:482–501／V2:400–417 区段）。逐字段表（尾形状/dtype 以记录构造＋闸 `SCHEMA` 表 G:80–95 交叉核对）：

| 键（序） | 记录构造（双方同） | dtype | 尾形状 | 结论 |
|---|---|---|---|---|
| x (1) | `rint(clip(x,0,2.55)*100).astype(uint8)` | uint8 | (54,15,15) | **合** |
| sc (2) | `mod._scalars(...)` 原样透传 | float32 | (5,) | **合** |
| tw (3) | 目标矩阵 `.astype(float16)` | float16 | (225,5) | **合** |
| tb (4) | 同上 | float16 | (225,3) | **合** |
| wmask (5) | `masks_for` 构造 | uint8 | (225,) | **合** |
| bmask (6) | 同上 | uint8 | (225,) | **合** |
| carbon (7) | `np.array(st.carbon, dtype=float16)` | float16 | (225,) | **合** |
| margin (8) | `np.float32(final)` 标量 | float32 | () | **合** |
| aux_delta (9) | `np.array([...3 horizons...], dtype=float32)` | float32 | (3,) | **合** |
| win (10) | `np.float32(1/0.5/0)` 标量 | float32 | () | **合** |
| searched (11) | `np.uint8(searched)` 标量 | uint8 | () | **合** |

| 编号 | 检查点 | 结论 | 行号/证据 |
|---|---|---|---|
| 3.1 | 键集合与顺序（11 键） | **合** | V2:91–92 `SHARD_KEYS`、V2:425–426 与 V1:506–507 三处字面一致；闸 G:74–75 第四处一致。标量键经 `np.stack` 成 `(N,)`、尾形状 `()`，闸 G:117 的 `shape[1:] != tail` 检查对标量键成立（本机实测 `(7,)[1:]==()`）。 |
| 3.2 | 伴随文件 `selfplay_config.json` | **合（附加型差异）** | V1:610–614／V2:657–666：候选保留现行全部键（games/seed/deep_frac/ckpt/hof/value_scale/deep{K,D2,tau}），另加 `generator/pool_games/batch_cap` 三键。无键删除、无键改义。 |
| 3.3 | 分片文件名与切分 | **合（消费面无差别，形态有差异，登记）** | 现行 `shard_w{wid}_{n:03d}.npz`（V1:523、527，每 worker 一条缓冲）；候选 `shard_v2_{n:03d}.npz`（V2:639、652，单缓冲、按局完成序攒行）。双方 flush 阈值同为 1024 行（V1:522／V2:637）。行集合按局等价，全局行序不同（现行＝worker 步进序，候选＝完成序）。 |
| 3.4 | 下游 `train.py` 无差别消费 | **合** | `load_dataset`（T:191 起）不硬编码键表：以首个分片的 `z.files` 为键集（T:207–211），按 `STORE_DTYPES`（T:144–157，恰为同一 11 键、dtype 与上表逐项一致，其中 x 以 float16 承接 uint8 原值——对两版分片同等成立）堆叠；`main` 的批次组装与 `augment` 消费的正是该 11 键。目录枚举为 `glob('*.npz')`（T:65 区段），不依赖文件名模式；每 epoch 以 `rng.permutation` 洗牌（T:260、277），故 3.3 的行序差异不进入训练语义。候选分片可被现行消费者无差别消费。 |

---

## ④ 已知发现独立裁断：闸 G:223–224 `report_pass` 恒 False

**① 形态属实——成立。** G:220–222 先以三项浮点最大值构成字典 `d`（仅 `max_dlw/max_dlb/max_dv` 三键）；G:223 `d['hard_pass'] = max(d.values()) <= tol_hard` 求值时 RHS 的 `d.values()` 尚只含三浮点，`hard_pass` 本身计算正确；G:224 `d['report_pass'] = max(d.values()) <= tol_report` 求值时 `d` 已含 `hard_pass` 布尔，`True==1` 参与 `max`，而 `tol_report=0.02`，故凡 `hard_pass=True` 者 `max>=1>0.02`，`report_pass` 必 False；凡 `hard_pass=False` 者三浮点最大值已 `>0.25>0.02`，`report_pass` 亦 False。**即 `report_pass` 在一切输入下恒 False。** 本机以 G:218–224 原逻辑复刻实测三组输入：diffs=(0.01,0.01,0.01) → hard=True、report=False（应为 True，**bug 复现**）；(0.10,0.05,0.03) → report=False 与期望偶合；(0.30,…) → hard=False、report=False 与期望偶合。失真仅发生在「三项最大值 ≤0.02、本应 report_pass=True」一格。

**② 影响面——仅一个观测字段失真，闸判读不受影响。** 失真：`gate_report.json` 与控制台打印 JSON 中 `forward.{cur,m3s2}.report_pass` 两字段（恒 False）。不受影响：`hard_pass`（G:223，计算时字典无布尔）、`forward.pass`（G:226 只取 `hard_pass`）、失败点记录（G:227–231 以三浮点最大值与 `tol_hard` 构成）、分布/ schema 各节、总 `gate_pass` 与结论行/退出码（G:350、main 中 `ok` 只与 `forward.pass` 相与）。三项原始最大值 `max_dlw/max_dlb/max_dv` 均如实记录，report 线（0.02）的人工判读可由原始值直接恢复。

**③ 修法与时机——明确结论：应在 L20 前向重验之前先修。** 理由分两层：(a) 就闸的**有效性**而言非必须——hard 判读、结论与退出码均不依赖该字段，不修先跑得出的 hard 结论仍然有效；(b) 就**证据质量**而言应当先修——前向重验的既定目的之一即取得 report 线（0.02，待策略组钉的观测线）的读数，带着已知缺陷跑，产出的 `gate_report.json` 中该字段整轮作废、只能靠人工读原始三值补判，且修复仅一行（先算 `m = max(三浮点)` 再由 `m` 派生两布尔）、本机即可桩测验证、不触碰任何 hard 口径。建议修法：`m = max(d['max_dlw'], d['max_dlb'], d['max_dv']); d['hard_pass'] = m <= tol_hard; d['report_pass'] = m <= tol_report`。若排期上不修先跑，则该轮报告必须显式声明 `report_pass` 字段作废、只引原始三值——不得让恒 False 的字段进入任何判读材料。

**附带裁断（第 222 行修复本身）：正确且必要。** `rows_v` 的元素是 `v[0]`，即 `(n,)` 数组的 0 维标量；原版 `np.concatenate(rows_v)` 对 0 维数组列表在本机实测直接抛 `ValueError: zero-dimensional arrays cannot be concatenated`——即原版闸在 (a) 段带 `--shard/--ckpt` 运行时会崩溃而非给出错误数值；修复版 `np.stack(rows_v)` 得 `(n,)`，与 `bv` 形状对齐（实测）。同段 `rows_lw/rows_lb` 两行原版即为 `stack`（得 `(n,225,5)/(n,225,3)`；若用 concatenate 将得 `(225n,5)` 的错误形状），修复未触碰、正确。

---

## ⑤ 闸的其他审查项

| 编号 | 检查点 | 结论 | 行号/证据 |
|---|---|---|---|
| 5.1 | (c) schema 机械检查的完备性 | **合** | G:91–125：键序精确比对、逐键 dtype 精确比对、尾形状比对、行数一致性，四项俱全；`SCHEMA` 表（G:80–95）与 §③ 记录构造逐项一致（本审已交叉核对）。 |
| 5.2 | (a) 前向一致性的 tag 覆盖 | **不合（覆盖缺口）** | G:214 只查 `('cur','m3s2')`；hof 系列 tag 的拼批数值从未被 (a) 覆盖。生产生成在配置 hof 时会真实使用 hof tag 前向。 |
| 5.3 | (a) 的对照构造 | **合（附假设登记）** | G:196–231 以候选自己的 `TorchBatchBackend` 做 batch-1 vs 整批对照——测的正是拼批数值残差这一真实风险面；但它不直接对照现行服务器路径。两者数值等价建立在代码同构之上（服务器 V1:141–161 与后端 V2:481–500 的 autocast/`.float()`/切片路径一致、输入均 float32），属代码级判断，数值面归 L20 实测。 |
| 5.4 | (b) aux_delta 的判读地位 | **不合（判线缺失）** | G:165–167 计算了 aux_delta 三个 horizon 的 KS 值，但 `verdicts`（G:172–178）与 `lines`（G:335–339 区段）均无 aux_delta 条目——该分布只报不判，任意大的 aux_delta 分歧不影响 `distributions.pass`。闸 docstring（G:26–30）把 aux_delta 列入 (b) 的比较对象，却无任何线值，与「比较」的表述不符；至少应显式标注为 report-only。 |
| 5.5 | (b) 运行模式的参数透传 | **不合（覆盖缺口）** | `run_generators`（G:236–252）的公共参数不含 `--value-scale` 与 `--threads`，两代生成器恒以默认 `terminal`/4 线程跑 (b)；`--value-scale u` 的桥接路径（bridge 调用点在两版 deep 体内逐字同，V1:407／V2:323）在闸的运行模式下从未被执行——u 口径的等价性目前只有代码级证据，无闸运行证据。 |
| 5.6 | (b) 分布比较的行级加权口径 | **合（口径登记）** | G:133–141、144–190：margin/win/searched 均按行（非按局）汇聚，mirror 局双座记录使其权重为其他局的 2 倍；两代同构造、同加权，对照有效。KS/均值漂移/胜率/searched 比例/每局行数五项判线俱全（G:172–178）。 |
| 5.7 | 闸的纪律形态 | **合** | 全文无重试/修复逻辑；失败点统一 `{check, metric, measured, line}` 汇入 `failure_points`（G:341–349）；结论行与退出码一致（G:350–376）。符合「旁路件、不过即弃置」的定位。 |

---

## 总表

| 大项 | 合 | 不合 | 明细 |
|---|---|---|---|
| ① 拼批索引/掩码对齐（1.1–1.11） | 10 | 1 | 唯一不合为 1.10 重复 key 健壮性（生产不可达）；1.11 为合附观察登记 |
| ② RNG 派生与消费（2.1–2.13） | 13 | 0 | 含 2.4 全局 RNG 专项与局间干扰总判，均以底盘源全文检索为据（代码级判断） |
| ③ 落盘 schema（3.1–3.4 及逐字段 11 项） | 全部 | 0 | `write_shard` 逐字同；config 为附加型超集；文件名/行序差异不进入下游消费语义 |
| ④ `report_pass` 已知发现 | 裁断：属实；影响面仅 `forward.*.report_pass` 两字段；结论＝L20 重验前先修（有效性非必须、证据质量应当） | — | 第 222 行修复本身：正确且必要（原版会崩溃） |
| ⑤ 闸其他项（5.1–5.7） | 4 | 3 | 不合均为覆盖/判线缺口：5.2 hof tag 未测、5.4 aux_delta 只报不判、5.5 (b) 不透传 value-scale |

**总判读（代码级）**：候选在拼批对齐、RNG 等价、落盘 schema 三面与现行逐项相合，未发现任何会使生成语义静默偏移的错位；唯一代码不合（1.10）在生产调用下不可达。闸的 hard 判读链完好，但带三处覆盖缺口与一处已证实的观测字段缺陷（④），其中 ④ 建议在 L20 前向重验前先行修复。

## 未覆盖项清单（需 torch/L20 或本次范围外材料才能验，不默示通过）

1. 前向数值本身：batch-1 与整批（含 batch-cap 切分后）在 BF16/FP16 autocast 下的实际残差分布——闸 (a) 的用途，本机无 torch 未跑；本审对拼批的全部结论为代码级对齐判断。
2. hof tag 的拼批数值（与 5.2 联动）：即使在 L20 跑闸 (a)，现闸也不测 hof；需扩闸或另测。
3. `--value-scale u` 口径下两代的运行等价（与 5.5 联动）：桥接代码逐字同已验，运行证据缺失。
4. fix2/fix3 等其他 `--src` 底盘的全局 RNG 与 `_TEMP` 语义：2.4/局间干扰结论仅对 master3s2 底盘源（md5 `57f9a594…`）成立；候选若以其他底盘为 `--src` 运行，须另核该底盘源。
5. `model.py` MainNet 在批量维上的数值行为（如任何按 batch 统计的算子）：已直读 `forward`（M:112–127 区段）为逐行结构（trunk/heads/critic 均逐行运算、无跨行归约），代码级无跨行耦合；实际内核数值仍归第 1 项。
6. 端到端 200 局分布闸 (b) 的实测结果：需双代同种子实跑，本机未跑（且本审不使用 L20）。
7. 分片文件名模式的下游影响：已核 `train.py` 以 `*.npz` 枚举、不依赖模式；`run_loop.sh` 等其他潜在按名消费者不在本次受审文件集内，未核。
8. 候选在真实池规模（pool=32、deep 滚动批量达数百行）下的内存/批量构成与吞吐：属 L20 bench 范围，本审未测不断言。

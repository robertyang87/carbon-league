# 接线图：comparator.py → vendor/cnet2_search.py（钉版 94f7fb64）

深读对象：`proto/vendor/cnet2_search.py`（2054 行；文件头三行注释声明 code md5（不含头）= `94f7fb64`，整文件 md5 不同是因含头注释，非版本不符）。
目标 API：`proto/comparator.py` 的 `Comparator.decide(state, candidates, rollout, leaf, horizon, scenario_seed, guard_vetoed)`，其中
`RolloutFn = (state, action, horizon, scenario_seed) -> leaf_state`，`LeafFn = (leaf_state) -> float, U∈[0,1]`。
行号均指 vendor 文件（另注文件名者除外）。配套文件：`cnet2_np.py`（下称 NP）、`cnet2_feat.py`（下称 F）、`fastsim.py`。

---

## 1. make_search_agent 构造与返回签名

**构造**：`make_search_agent(conf=None, weights=None, rweights=None, oweights=None)`（行 621）。

- CONF 读取：行 622–623 `cfg = dict(DEFAULTS); cfg.update(conf or {})`——DEFAULTS 在行 38–42，本文件只消费合并后的 `cfg`，包侧 CONF 在构造时传入。**注意 DEFAULTS ≠ Echo4 包 CONF**（DEFAULTS：SEARCH=True、BUDGET=0.38、H=5、MARGIN=20.0、GUARD=False、TEACHER=1；Echo4 包：SEARCH=False、BUDGET=0.7、HARD=0.85、GUARD=True、TEACHER=2）。接线时一切参数只许读 `agent.cfg`（行 2052 挂出），禁止引用 DEFAULTS 或文件头注释（行 4–20）中的数字。
- 权重来源（行 624–640）：`weights` → 主网 b128，`CachePolicy(load_weights(weights))`，缺省路径 `../data/weights/cnet2_b128_q8.npz`（行 625）；`rweights` → rollout 网 `NP.Policy`，缺省在 `cfg["ROLL"]=="s64"` 时取 `cnet2_s64.npz`（行 626–628），为 None 时 `rpol = pol`；`oweights` → 开局网 `opol`（CachePolicy，行 633）；cfg 的 `XNET` → 结构异构混合网 `xpol`（行 629–632）；cfg 的 `ENS` → `epol = NP.Ensemble`（行 636–638）。`load_weights(src)`（行 45–48）：dict 直用，否则 `np.load` NPZ。
- 开关判定点：没有 `SEARCH_MODE` 这个键。模式键是 `cfg["MODE"]`：`_inner` 行 1028 `if cfg["MODE"] == "mcts"` 走 `_mcts`，否则（默认 `"sweep"`）走 `_search`（行 1058）。`cfg["SEARCH"]` 总开关在行 1025：`if not cfg["SEARCH"] or left < cfg["ENDGAME_STOP"] or 开局网期` → 直接返回默认动作。`cfg["MIX"]`/MODE 决定对手建模走 `opp_model`（行 1020–1021），`cfg["OPPID"]` 走 `opp_id`（行 1022–1023）。

**返回**：两层 agent。内层 `_inner = agent`（行 798 定义，行 1667 别名）；外层 `agent`（行 1975）包异常兜底（任何异常 → 返回 `{}` 空动作，行 1977–1981）＋一串默认关的后置规则层（行 1982–2048），`make_search_agent` 最终 `return agent`（行 2054，外层）。外层对象挂 `agent.stats / agent.state / agent.log / agent.cfg`（行 2049–2053）。

- 可调用签名：`agent(obs, config=None) -> dict`。`config` 参数两层都不读（纯占位）。
- 输入 `obs`：平台观测 dict，关键字段 `obs["player"]`（我方座位）、`obs["players"]`（每座 `[cash, {base_id: pos}, {uid: [pos, cargo, kind]}, {tree_id: [pos, age]}]`）、`obs["carbon"]`（225 格、已按可见性过滤）、`obs["step"]`、`obs["trees"]`、`obs["remainingOverageTime"]`。
- 输出动作编码：`{uid: 动作字符串}` 字典，由 `F.Feat.to_actions(y)`（F 行 422–433）从标签向量 `y`（长 `F.OWN`=12：槽 0–9 工人、10–11 基地）转换：工人槽 `y[s]`∈0..4 → `F.MOVES[CANON_MOVE[a] if seat else a]`（如 UP/DOWN/LEFT/RIGHT；`y[s]<=0` 的槽位被省略，即 0=原地/无动作不下发）；基地槽 → `F.BASE_ACTS[a]`（如 RECCOLLECTOR/RECPLANTER）。

## 2. 一步之内的数据流与 A0 定型

`_inner`（行 798）单步主链：

1. 行 799：`t0 = clk()`（`clk` 在行 796 定义：CLOCK="cpu" 用 process_time，否则 `time.time`；另 `tw = time.time()` 墙钟并行）。注意 t0 在内层入口，**不含外层（行 1975）后置规则层的耗时**。
2. 行 828：`x = f.observe(obs)`（`f = st["f"]`，我方座位 Feat 跟踪器；F 行 163 `observe`）。x 是特征 dict（maps/scal/tok_*[22 槽]/hist_*/cash/slots/left）。
3. 行 866：`y, lg0, _ = (epol or pol).act(x, f.aux, teacher=st["T_now"])`——贪心解码得标签 `y` 与先验 `lg0`（开局采样分支在行 837–841、TTA 分支在行 862–864 是另两条互斥路径）。行 867：`st["lg0"] = lg0; st["y0"] = y.copy()`。
   - **先验摘出点**：`lg0` 即每槽 logits，形状 `[OWN=12, 5]`（NP 行 175 `act` 返回 `(labels, logits, value)`，logits 在 NP 行 192 声明为 `(F.OWN, 5)` float32、非法动作置 NEG）。softmax 用法见 `_mcts` 行 1243：`pz = softmax(lg0[s])`。批量路径的等价物是 `decode_b` 的 `lg_out` 参数（见下）。
4. 规则层原地改写 `y`：CAR_CURVE（行 885–901）、PLANTER_CASH（行 905–913）、MACRO（行 922–967）、LATE_PLANT（行 974–987），改写同时登记 `st["base_fix"]`。
5. 行 989–997：`if cfg["GUARD"]: y = bankrupt_guard(x, f.aux, y, lg0)`，守卫后与 base_fix 对账（被守卫推翻的 base_fix 条目删除）。
6. MASTER_NO_PLANTER 改写（行 1000–1006）。
7. 行 1004：`acts = f.to_actions(y)`——**A0 在此处定型：A0 = 经规则层＋守卫后的 `acts`（不是行 867 的裸网输出 `st["y0"]`）**。
8. 行 1013–1019：构造对手视角观测 `obs_o`（换座位、碳图 = 对手可见格 × 我方 `f.est` 信念），行 1019 `xo = fo.observe(obs_o)`；行 1025 过 SEARCH 门后分发 `_mcts`（行 1030）/`_search`（行 1058），两者入参均为 `(obs, x, y, acts, xo, left, t0, tw)`。

**Prefix/解码调用链**：批量自回归解码在模块级 `decode_b(pol, X, auxs, tok, ctx, forced=None, logp=None, noise=None, temp=None, lg_out=None)`（行 238–354）：`forced` 为 `[B, OWN]` int（−1=自由），行 326–328 处 `a = np.where(fz >= 0, fz, a)` 强制覆盖 argmax；每槽 logits `lg` 形状 `[n, 5]`（行 ~318），传 `lg_out` 可逐槽摘出。`encode_b(pol, X, teacher)`（行 159）= `trunk_b` + `ctx_b`（行 165，teacher/style 只在此进入 ctx）。`act_b(pol, xs, auxs, teachers, forced=None)`（行 363–368）= stack_x → encode_b → decode_b，是 rollout 步的批量双座解码入口。`_search` 根候选解码在行 1546–1547 直接调 `decode_b`（tok/ctx 来自行 1510 `pol.enc_for(x, st["T_now"])`，CachePolicy 行 86–90 的原帧编码守卫）。

## 3. 模拟器 FastCarbon 与对手情景

- 实例构造/池：`pool = []` 是 `make_search_agent` 闭包变量（行 643）；`_search` 行 1574–1575 `while len(pool) < nb: pool.append(FastCarbon())` 按需扩容、跨步复用（`_mcts` 行 1362–1363 同）。`FastCarbon.__init__(config=None)` 在 fastsim 行 135。
- 分支初始化（行 1564–1590）：根状态 dict `full = {"step", "players": obs["players"], "trees", "full_carbon": [float(v) for v in f.est], "carbon": obs["carbon"], "remainingOverageTime": 60}`（行 1564–1565，我方信念图作全图）；每分支 `s_.load_observation(full)`（行 1580；fastsim 行 286，经 `reset(full_carbon=...)` 重建全部状态），行 1581 把 `_tid_counter` +500 错开分支间新树 id。
- 一步推进：`s_.step(a, ao_)` 或 `s_.step(ao_, a)`（按我方座位定序，行 1583–1586；fastsim 行 504 `step(actions_a, actions_b)`，入参为 `{uid: 动作字符串}`）。rollout 步在行 1610–1640：双座 Feat 各自 `observe(s_.obs(me))` / `observe(s_.obs(1-me))`（fastsim 行 404 `obs(player_index)`，按该座可见性置零碳图）→ 行 1614 `lab = act_b(rpol, xs, auxs, tch)` 批量解码 → `to_actions` → `s_.step(...)`（行 1634–1637）。
- 状态克隆/恢复：**没有 deepcopy，也没有 FastCarbon.copy()**。现存做法是「池复用 + `load_observation(full)` 从根观测全量重建」（等价于每分支一份新状态，但计数器是下界恢复，见 fastsim 行 286–300 的注释）。Feat 跟踪器则用 `feat_copy(f)` / `feat_copy(fo)`（行 372–383、调用点行 1590）：浅拷贝全部字段＋对会被 observe 原地改写的字段（est/last/last_step/hist/own_uid/inf）做私有副本。fastsim 另有精确快照对 `get_state()`（行 424）/`set_state()`（行 458，含 id 计数器原样恢复），但 `_search`/`_mcts` 都没用它——接线若要「同一根状态多次 rollout」，优先用 get_state/set_state 而不是反复 load_observation（后者每调一次都重跑 reset＋重建，成本更高且计数器语义是下界）。
- 随机种子：`FastCarbon.step` 本身确定性（fastsim 内 `np.random` 只出现在 reset/地图生成，行 203–286；分支走 full_carbon 直供，不触发）。**`_search` 路径全程无 rng**：对手情景是确定性单点（J=1，见下）。`_mcts` 才有种子：行 1286 `rng = np.random.default_rng(cfg["RSEED"] * 1000003 + step * 7919 + me)`，用于风格抽签与 rollout 噪声（common random numbers 跨候选共享）。开局采样另有 `st["orng"]`（行 831，seed = OPEN_SEED*7+me）。**结论：comparator 的 `scenario_seed` 参数在 sweep 路径上没有现存对应物，接线时必须新造并显式登记，不能声称与现存行为等价。**
- 对手情景生成点（sweep）：行 1553–1562——`st["oenc"]`（agent 主链行 1019–1023 由 opp_id/opp_model 或缺省路径备好的对手座编码 `(Xo, toko, preo)`）上 `yo = decode_b(pol, Xo, [fo.aux], toko, ctx_b(pol, preo, [st["ot"]]))[0]`，只解一次，`acts_o = fo.to_actions(yo)` 对所有分支共用。`cfg["ROBUST"]` 非零时行 1567–1570 追加第二个 model：`hunter_acts(obs, me, acts_o, cfg)`（行 486–509）的规则改写版，最终叶值按 `(1-w)*sc[0] + w*sc.min(0)` 混合（行 1654–1657）——这是确定性变体，不是采样。
- `opp_model(obs, xo)`（行 1150–1217）：签名只有 `(obs, xo)`。它读的是：上一 步对手编码 `st["oprev"]`（Xp/tokp/prep/auxp/uidp/wp）、对手**已执行的**上一步动作标签（`_labels_prev`，行 1131–1148，从 `fo.hist` 取实际发生过的动作）、`obs["players"]` 里对手新招募单位的种类；产出风格后验权重 `st["ow"]`、当前风格 `st["ot"]`、对手编码 `st["oenc"]`。**窥视面结论：opp_model 与根对手解码（行 1559）都不读我方当前候选动作（cand/ys/forced 在它们运行之后才生成，或与之无数据通路）；`hunter_acts` 读的是 obs 中我方 collector 的位置/载货——根状态的公开量，不是候选动作。唯一需在接线时守住的纪律：新造的 scenario 生成不得引入对 cand 的依赖（现存代码天然满足）。**

## 4. 嵌套 leaf(sim, me, left, prev_cash=None)（行 666–794）

- 定义：`make_search_agent` 闭包内嵌函数，行 666 `def leaf(sim, me, left, prev_cash=None):`，行 794 `return m[me] - m[1 - me]`。
- 参数：`sim` = 已推进到叶深度的 FastCarbon 实例；`me` = 我方座位（0/1）；`left` = 叶处的剩余步数（调用方传 `lft = max(0, left - depth)`，行 1651）；`prev_cash` = 「使该 sim 进入终局的那一步之前」的双座 cash 列表（F01 存活方计分语义），非终局时不参与计算，缺省 None 时终局分支用 `sim.cash`。
- 返回尺度：**原始分差尺度，不是 U∈[0,1]**。终局分支（行 667–677）：灭绝方记 `pen = sim.step_count - episodeSteps(300) - 1`，存活方记其带入终局步的 cash，返回两者之差。非终局：`m[p] = cash[p] + Σ工人 WORKER_V*min(1, left/100) + Σ可交付载货 CARGO_W*cargo（仅当到基地距离 d <= left）+ 树值（TREE_V*剩余寿命，或 TREE_C 期望收入变体）+ 可选 PD_POT/PD_W 开局项`，返回 `m[me] - m[1-me]`。量级与 `MARGIN=20.0` 同尺度（分差十至数百）。**与 comparator 的 LeafFn 合同（U∈[0,1]）直接冲突，接线必须加注册口径的尺度适配层（如 (tanh(z/2000)+1)/2 一类），原始 leaf 不得直喂 comparator。**
- LEAF_THREAT 变体：触发判定在行 681 `lt = cfg.get("LEAF_THREAT") is not None and left <= cfg.get("LT_LEFT", 30)`——需 CONF 显式给 LEAF_THREAT 且叶处剩余步 ≤ LT_LEFT（默认 30）。生效段行 706–727：对每个载货工人，若存在敌方 collector 载货 ≤ 我方且环面距离 ≤ 1（road 路径，`LT_ROAD` 默认开），或敌方 collector 贴我方回家基地口且该基地距工人 ≤ 5 步（mouth 路径，需 `LT_BASE` 开），则该工人载货的计价因子 `f_` 由 1.0 改为 `cfg["LEAF_THREAT"]`（单一折扣、不连乘）。另有 WT_LEFT 晚期工人计价变体（行 685–705）与之独立。
- 闭包依赖：leaf 读 `cfg` 多键（WORKER_V/CARGO_W/TREE_V/LEAF_THREAT/LT_*/WT_*/WC_*/TREE_C/TREE_SAT/TSURV/TC_ONCE/PD_*）与 `st` 的根快照（`st["root_wids"]`、`st["root_tree_owner"]`、`st["root_tids"]`，由 `_inner` 行 803–808 每步在根处写入），并依赖同闭包的 `_terminal/_extinct`（行 649–664）。**它不是纯函数：脱离闭包单独调用会读到上一根的 st 快照。**
- 被调用点：`_search` 行 1652（sweep 路径唯一调用，`sc = np.array([leaf(s_, me, lft, term_cash[i]) ...]).reshape(len(models), nc)`）；`_mcts` 行 1447。agent 主链与守卫都不调它。

## 5. 守卫 bankrupt_guard 与 GUARD/GUARD_NARROW

- 实现：类 `BankruptcyGuard`（行 512–618，注释声明系从原闭包逐字抽出、行为不变）；`make_search_agent` 行 641 建唯一实例 `guard = BankruptcyGuard(cfg)`（每局每座一份，`st["guard_base"]` 是其 `guard_base` 集合的别名，行 642）。闭包薄封装 `bankrupt_guard(x, aux, y, lg0)`（行 1069–1074）转发 `guard.apply(x, aux, y, lg0, pol, st["f"].own_uid, st["T_now"], stats)`。
- 判定点：调用在 `_inner` 行 989 `if cfg["GUARD"]:`（try 包裹，异常只记 stats 不中断）。判定逻辑在 `apply`（行 568–618）：`risky = 工人数 nw <= cfg["LOWW"]`（默认 2，行 582）；`GUARD_NARROW` 开（行 583–585）时再收紧为 `risky and lostw > 0 and cash < cfg["GUARD_CASH"]`（默认 40；lostw = 本局已损失工人数，由 apply 行 579–581 按 prevw 差集累计）。另有一条独立于 risky 的招募规则（行 605–609）：无 collector 且 cash ≥ 30、有空槽、left > 10 时强制一个基地招 collector。
- 否决/改写方式：不是逐动作打回，而是「构造 forced 向量 + 整步重解码」。apply 行 589–604：对每个槽位算危险掩码（collector 看 `aux["en_min"][dest] <= cargo`，planter 看 `aux["thr"][dest] > 0`），若网络所选动作危险且非全危险，则按 `lg0` 排序取最优安全动作写入 `fz[s]`；行 611–612 若无任何改动（first is None）原样返回 y；否则行 613 `fz[:first] = y[:first]`（首个改动槽之前的槽位强制为原计划），行 614–617 用 `pol.enc_for(x, teacher)` 取原帧编码、行 617 `decode_b(pol, X1, [aux], tok, ctx, fz[None])` 整步重解码——改动槽之后的槽位由解码器自回归地跟着变。状态性警告：apply 会原地更新 lostw/prevw/guard_base，**不可对同一 guard 实例重复试探多个候选**；类提供 `snapshot/restore`（行 550–561）与 `copy()`（行 563–566）正是为分支隔离准备的。
- 比较器接入的守卫一致层应做在哪：现存链路只保证 A0 过守卫（行 989 在 `_search` 之前）；`_search` 的候选靠前缀强制（fz[:s]=y[:s]）＋ base_fix 强制（行 1523–1525）＋ guard_base/base_fix 基地槽禁变（行 1517）部分继承守卫，但**被偏离的那个槽位本身从未经守卫复核，胜出候选在行 1662–1665 直接返回、不再过守卫**。因此接线时守卫一致必须落在**候选准入层**：对每个候选的标签向量用 `guard.copy()` 跑一遍 `apply(...)`，凡被改写的候选索引即 comparator 的 `guard_vetoed` 集合（A0 按 comparator 合同豁免）；不要等提交后再补守卫——那会改变比较器已提交的动作、破坏「提交即最终」语义（且外层行 1975 之后还有规则层，见 §7 冲突 6）。

## 6. MIN_DEPTH 接受门与首层预算门（复核 Codex 结论）

`_search` rollout 循环（行 1600–1640）内的预算门（每轮迭代开头判定，`now = clk()`）：

- 行 1602：投影门——`if last_dt is not None and now - t0 + cfg["PROJ"] * last_dt > cfg["BUDGET"]: break`（用上一轮耗时投影下一轮）。
- 行 1604：**首层预算门——`if last_dt is None and now - t0 > 0.45 * cfg["BUDGET"]: break`**（尚无一轮 rollout 完成时，入口已耗超 45% 预算则不开第一层）。
- 行 1606–1607：墙钟 HARD 门（`time.time() - tw + PROJ * last_wall > cfg["HARD"]`）。

MIN_DEPTH 接受门在行 1643–1650：

```
if depth < min(cfg["MIN_DEPTH"], H) and not all(term):
    stats["shallow"] += 1
    return acts
```

`depth` 初值 1（根步已推进），每完成一轮 rollout 迭代 `depth = hh + 1`（行 1639）。叶调用在行 1652，**严格位于该门之后**。

**Codex 结论复核：成立。** 典型路径：BUDGET 被根解码＋首轮投影吃紧时，循环在 depth=1 或完成一轮后 depth=2 处 break；`min(MIN_DEPTH=3, H=5)=3`，depth<3 且非全终局 → 行 1650 直接返回原动作 `acts`，行 1652 的 leaf 从未被调用——「深度 2 返回原动作、叶未被调用」属实，且存在两道独立的门（首层 0.45 门可在 depth=1 就截断；MIN_DEPTH 门在叶求值前截断）。唯一例外是行 1643 的 `not all(term)`：全部分支已达精确终局时绕过该门、按 F01 真实终局分判读（C01 注释，行 1644–1649）。`_mcts` 的对应物：首轮门行 1351（`rounds == 0 and now - t0 > 0.4 * BUDGET`）与行 1455–1457（`rounds == 0 → return acts`）。

## 7. 接线方案

### 7.1 比较器包在哪一层

包在 `_search` 内部、替换其「固定深度 rollout + MIN_DEPTH 门 + leaf + argmax」段，保留其候选生成与分支构造：

- 保留：行 1503–1551（key_units 触发、forced 前缀重解码、去重得 `cand/cmeta`，`cand[0]=acts` 即 A0）、行 1553–1562（对手根动作 `acts_o`）、行 1564–1565（`full` 根状态）。
- 替换：行 1571–1665（分支批量推进、深度循环、行 1643 MIN_DEPTH 门、行 1652 leaf、行 1658–1665 argmax+MARGIN 提交）。新段对每个 horizon 档构造 rollout/leaf 闭包（见 7.2/7.3）并调用 `Comparator.decide(state=full, candidates=[Candidate(i, a, is_a0=(i==0)) for i, a in enumerate(cand)], rollout=..., leaf=..., horizon=H档, scenario_seed=..., guard_vetoed=...)`，用 `Decision.action` 替代原返回值，`Decision` 的遥测字段喂 stats（替代 depth_sum/shallow 口径须另登记）。
- 不选「包在 `_inner` 行 1058 外层」的方案：那样要重抄候选生成（行 1513–1551 与 st/f/fo/enc_for 强耦合），且 A0 的 post-guard 形态、base_fix、oenc 都在闭包内，外层重建必然漂移。
- 注意 K 上限：现存 `cand` 可达 MAX_BR=16（行 1546 截断 `forced[:mbr-1]`），comparator 合同 K≤4——接线段须在调用 decide 前把候选截到 4（A0＋至多 3 个偏离，截断顺序按 key_units 的 stake 序，即 meta 的生成序），这一步是新增逻辑，须在 S1 机制回执中明示。

### 7.2 rollout 可调用对象怎么构造

在替换段内定义闭包（捕获 pool、f、fo、rpol、st、cfg、full、acts_o、me）：

```
def rollout(state, action, horizon, scenario_seed):
    s_ = 取一台池内 FastCarbon（pool 不足则 FastCarbon() 扩容，同行 1574–1575）
    s_.load_observation(state)          # state 即 full（行 1564）
    s_._tid_counter = [c + 500 for c in s_._tid_counter]   # 同行 1581
    fm, fop = feat_copy(f), feat_copy(fo)                  # 每 call 必须新拷贝（行 1590 的做法）
    pc = list(s_.cash)
    s_.step(action, acts_o) if me == 0 else s_.step(acts_o, action)   # 根步，同行 1583–1586
    for hh in range(1, horizon):        # 后续步：复刻行 1610–1640 的双座 observe→act_b→step
        xs = [fm.observe(s_.obs(me)), fop.observe(s_.obs(1 - me))]
        lab = act_b(rpol, xs, [fm.aux, fop.aux], [st["T_now"], st["ot"]])
        s_.step(fm.to_actions(lab[0]), fop.to_actions(lab[1]))（按 me 定序）
        若 _terminal(s_): 记录 term_cash 并冻结后续步（同行 1631–1638 的 R02 处理）
    return (s_, fm, fop, pc)            # leaf_state（不透明）
```

要点：① scenario_seed 在现存 sweep 无对应物——J=1 首版可令 rollout 忽略它（acts_o 已在根外定型、双座 rollout 贪心确定性），但必须在接线代码与回执中明示「seed 暂未消费」；若后续要 J≥2 或风格采样，种子应注入对手风格抽签（参照 `_mcts` 行 1286 的 default_rng 派生式），不得注入 FastCarbon（其 step 无随机源）。② comparator 是逐候选串行调用 rollout 的，池实例可复用同一台（每次 load_observation 全量重建）；但 Feat 拷贝必须每 call 新做——feat_copy 漏字段正是文件头注释提到的 2026-11-10 INFER bug 的成因（行 372–375）。③ horizon 的语义对齐：现存 `depth` 含根步（H=5 → 根步＋至多 4 步 rollout），comparator 的 horizon 应按同一口径登记（建议 horizon = 总深度，rollout 内部步数 = horizon−1），避免与注册 H 定义错一格。

### 7.3 leaf 怎么提取/复刻

- 首选（零漂移）：**不提取、不复刻，直接在闭包内引用**。leaf 是 `make_search_agent` 的嵌套函数（行 666），与 cfg/st/_terminal 同闭包；7.1 的接线段本就写在 `_search` 内（或同闭包的新增函数），可直接构造 `def leaf_fn(leaf_state): return adapt(leaf(leaf_state.sim, me, max(0, left - horizon), leaf_state.pc))`，其中 `adapt` 是注册口径的尺度适配（原始分差 → U∈[0,1]）。leaf 读 st 根快照（root_wids 等）的时序也天然正确（由 `_inner` 行 803–808 在本步根处刚写好）。
- 次选（comparator 必须在闭包外运行时）：不要抄源码。从已构造的 agent 对象反查闭包单元（`_inner` 的 `__closure__` 中名为 leaf 的函数对象）取出原函数引用——零代码复制，但依赖 CPython 闭包布局，vendor 一刷新（md5 钉版变化）必须重验。再次选才考虑在桥接层（pack_bridge 侧）按行 666–794 复刻，并接受漂移风险：leaf 引用 20+ 个 cfg 键与 3 个 st 根快照字段、两个终局语义（F01/R02），复刻版与钉版之间没有任何自动对账机制，vendor 刷新后复刻版会静默过期——这正是历史上「叶量纲从未校准」类事故的温床，复刻路线必须配逐字 diff 闸（每次 vendor 刷新对复刻段与钉版行 666–794 做文本比对，不合即停）。

### 7.4 候选三道准入的代码对应检查点

| 准入 | 代码检查点 | 说明 |
|---|---|---|
| ① 根合法性 | `key_units` 触发后 `_search` 行 1511 `uid2slot` 存在性；行 1517 基地槽条件（s≥10 须 BASE_KEY 开且不在 base_fix/guard_base）；行 1519 `m == y[s]` 跳过同动作；`decode_b` 内合法掩码（行 ~316 `lg = np.where(legal, lg, NEG)`，forced 也只能落在解码器步进的槽位上）；行 1550–1554 `seen` 去重（与 A0 标签全同的候选丢弃） | 候选在生成端即保证「是该槽的一个合法替代标签、且与 A0 不同」 |
| ② 守卫一致 | A0 已过守卫：行 989；候选继承：行 1521 `fz[:s] = y[:s]`（前缀 = post-guard 的 y）、行 1523–1525 base_fix 槽全候选强制、行 1517 guard_base 槽禁变。**缺口**：偏离槽本身未过守卫（见 §5）→ 接线新增：逐候选 `guard.copy().apply(x, f.aux, cand_labels, lg0, pol, f.own_uid, st["T_now"])`，被改写者进 `guard_vetoed` | apply 有状态，必须用 copy()（行 563），不得用主 guard 实例试探 |
| ③ 战术相关性 | `key_units(obs, cfg)`（行 386–484）：KEY_R（默认 2）环面距离内、载货 ≥ MINC（默认 10）的 collector 捕获/被捕获情境（HUNT 猎杀向、THREAT_FIRST/SPLIT 变体、KEY_LATE_HOME 末盘返航变体），按 stake 降序、MAX_KEY 截断；行 1546 MAX_BR 截断 | 注册覆盖三类战术中，碰撞/逃逸是默认触发面；返航交付与近终局变现依赖 KEY_LATE_HOME/KEY_LATE_CARGO 变体（默认关）——接线时覆盖口径以实际 CONF 为准，不得按 key_units 的全部变体声称覆盖 |

---

## 附：与注册合同冲突的现存行为（接线前必须知晓）

1. **leaf 尺度冲突**：嵌套 leaf 返回原始分差（§4），comparator.LeafFn 合同要求 U∈[0,1]；且现存提交判据带 `MARGIN=20` 的分差门槛（行 1662），comparator 是纯 argmax——尺度适配与「是否保留等价 MARGIN」必须按注册口径先定，不能默认直连。
2. **MIN_DEPTH＋首层门丢弃已算结果**：行 1643/1604 使浅深度时叶根本不被调用、整轮作废回 A0，与注册「完成一轮同深度完整比较即提交、中断只回退上一完整轮」相冲突——这正是 7.1 要替换掉的段，不得在新接线里保留任何等价卫哨。
3. **候选数超合同**：现存 MAX_BR=16（行 1546），注册 K≤4；且现存候选是「每 key 单位 × 每个替代动作」的笛卡尔展开，接线截断规则（按 stake 序取前 3）是新增口径，须登记。
4. **对手情景无种子、无 J 轴**：sweep 是 J=1 确定性单情景（行 1559），`scenario_seed` 无现存对应物；ROBUST 的 hunter 变体（行 1567–1570）是确定性规则改写＋min 混合，不是情景采样，不得在回执中记作 J=2。
5. **守卫只保 A0、不保候选**：偏离槽未过守卫、胜出候选不再过守卫（§5/§7.4②）——注册「守卫一致」在现存代码里是缺口，接线层补齐前不得声称已满足。
6. **外层规则层可改写搜索结果**：外层 agent（行 1975–2048）在 `_inner` 返回后还有 REC_CUT/OPEN_PL/tree_gate/recruit_gate/planter_explore/planter_avoid/safe_entry/camp_bust/endgame_home 等后置层（Echo4 CONF 下应为全关），且异常兜底返回 `{}` 空动作。接线验收时必须逐项核这些开关在目标 CONF 下为关，否则 comparator 提交的动作不是最终下发动作。
7. **触发面窄于注册覆盖**：默认口径下 key_units 只覆盖 collector 战术情境；注册三类战术中的返航交付/近终局变现要靠默认关的变体（KEY_LATE_HOME 等），覆盖表不得照 key_units 的变体清单全额声称（见 §7.4③）。
8. **遥测可见性**：`agent.last_stats` 只在 `_inner` 内赋值（行 1026/1062/1066）且是 `dict(stats)` 快照；外层不更新。接线遥测应直接读 `agent.stats`（行 2049，活引用）或新遥测通道，不要重蹈历史上「统计字典未发布」的死链。

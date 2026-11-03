"""Carbon League 提交策略（无敌v2：修掉「带货运原地待机 / 基地震荡」死锁，保留 v1 的致胜行为）。

规则依据（来自 zerosum_env 的 helpers.py / carbon.json）：
  * 坐标：观测位置是扁平索引 index = row * size + col；UP/DOWN 改 row，RIGHT/LEFT 改 col，
    边界环绕（15x15 环面）。距离用环面最短曼哈顿距离。
  * 采集：采集员「停留」才会采集（拿走该格 30% 碳）；回到自家转化中心「停留」才把
    在途碳结算成现金（smeltCost = 0）。
  * 种树：种树员在无树、无基地的格子上「停留」即种树，花费 plantCost(20) + 5*1.235^n；
    树每步吸收四连通邻格碳并换成现金；树 50 步后枯死并留下 30 碳。
  * 碰撞（关键）：同格若存在采集员，「碳最少」的那个获胜并吞掉同格其他单位的碳；
    最少碳并列则同格全灭；同格没有采集员则全灭。=> 携带量越大越危险。
  * 招募：新兵生成在转化中心所在格，若该格已有单位，会和自家单位撞死。

v3 相对 v2 的改动，全部来自「无敌」线上对局（/api/summary 的逐步事件流）暴露的问题：
  1. 硬性携带上限（22 + 2*离基地距离，≤45）→ 轻装高频交付，既滚现金流也避免被更轻的
     敌方采集员吃掉（v2 曾带着 105 碳被撞死，一局死 30+ 单位）。
  2. 招募目标抬到 7 采集员 + 3 种树员，并优先把采集员铺到 5 个（v2 只有 6 人，对手 10 人）。
  3. 基地格上若有我方单位在停留交付，本步不在该基地招募（避免新兵撞死）。
  4. 种树改为按边际收益决策：(0.8*邻格碳 - 本格碳 - 种树成本) > 0 才种；
     种树员闲置时去抢敌方树，或停到自家树上（不再占基地）。
  5. 空载单位绝不停留在自家基地（占位会挡住交付并被新兵撞死）；采集员不再踏进树格。

每步决策的优先级阶梯（高 → 低）：
  P0 发育：能招就招，采集员优先铺满（经济引擎），与下面的动作在同一回合并行执行；
  P1 自保：先保住已投入的资产——采集员贴脸遇到更轻的敌采集员必被吃，优先脱身/回基地交付；
           种树员碰撞必输，遇敌先躲；
  P2 原有策略：每步解析局势 + 预测敌方下一步落点，对每个候选动作算「预期收益」，
           只执行正收益动作；若全部 ≤ 0，走对策阶梯：保收益回程 → 降权重探索 → 原地待机。

状态仅保存在本进程内，step == 0 时重置。

无敌v1 线上 9 局复盘（对手均为 claude-opus-v18 系衍生策略）暴露的真实输因：
  * 输掉的 5 局里 4 局的第 60 步现金只有 8~113，而对手有 120~427 —— 开局经济引擎熄火。
  * f5aaeccc 那局逐帧证据：第 27~75 步现金死死冻结在 20.8，采集员 C49 携带 14 碳原地不动
    49 步、C51 携带 25 碳原地不动 58 步，空载的 C160 在基地↔邻格间来回震荡；全队 60 步
    零交付 → 招不起人 → 直接输掉比赛。

无敌v2 的改动（只针对上面这条根因，其余致胜逻辑原样保留）：
  1. ★反卡死看门狗：连续 STALL_LIMIT 步原地不动且没有采集收益 → 强制脱离（带货回城、空载换格）。
     这是本次最关键的修复——v1 里那两个采集员是「策略自己选择原地不动」的，不是被环境卡住的。
  2. 空载单位禁止进入自家基地格 → 断掉「让位→回头」的 2 步震荡，基地不再被自己人堵死。
  3. 终局 ENDGAME 步内强制回城结算，不让在途碳烂在路上。
  4. 「对策 C 原地待机」在带货时取消：收益全负时必须先往家的方向走。

实测否掉的方案（记录在此，避免以后重复踩坑；两组指标：死锁局面推演 90 步交回的碳 / 无对手沙盘 300 步现金）：
  * 「在途碳持有成本 hold cost」（claude-opus-v18 系的做法）：hold=0 → 1218 / 13580，
    hold=0.02 → 1073，hold=0.05 → 689，hold=0.2 → 384。它让采集员频繁空跑回城、采集时长下降，
    在这套贪心收益评估里是净亏，不采用。
  * 「禁止折返上一格」：BACK=3 → 沙盘 10295（比 v1 的 13977 低 26%）；不开启 → 沙盘 13580（只低 2.8%），
    而两者在死锁推演里都是 1090。收益不值，不采用。
  * 结论：v1 的死锁不靠「改收益函数」解决，而是靠看门狗直接兜底。
"""

SIZE = 15
N = SIZE * SIZE

PRIOR = 12.0            # 未观测格子的先验碳量
STEP_COST = 1.0         # 距离在位置估值里的折算代价
RETURN_W = 0.35         # 归程距离权重
EXPLORE_W = 0.45        # 对策阶梯里探索用的低距离权重
COLLECTOR_ABS = 0.1     # 敌方树对周边我方采集员在途碳的吸收率
PLANTER_KILL = 15.0     # 击杀敌方种树员的附加收益
DANGER_UNIT = 25.0      # 单位损失的估值
CARRY_BASE = 30.0       # 携带上限基数
CARRY_PER_DIST = 5.0    # 携带上限随离基地距离增长（回程越贵，越该多带）
CARRY_MAX = 75.0        # 携带上限硬顶（线上胜者单次交付约 100~130，故不宜过小）
TREE_ABS_EFF = 0.8      # 树能吸收到的邻格碳比例（估收益用）
COLLECTOR_TARGET = 8    # 发育优先：先把采集员（经济引擎）铺满
PLANTER_TARGET = 2
TREE_TARGET = 8         # 上限；实际由「边际收益 > 0」决定是否种
EPISODE_DEFAULT = 300

ENDGAME = 14            # 最后 N 步强制回城结算
STALL_LIMIT = 4         # 连续 N 步原地不动且无收益 → 强制脱离

STATE = {"est": None, "seen": None}

DIR_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _rci(p):
    return p // SIZE, p % SIZE


def _idx(r, c):
    return (r % SIZE) * SIZE + (c % SIZE)


def _dist(a, b):
    ar, ac = _rci(a)
    br, bc = _rci(b)
    dr = abs(ar - br)
    dr = min(dr, SIZE - dr)
    dc = abs(ac - bc)
    dc = min(dc, SIZE - dc)
    return dr + dc


def _step(p, d):
    r, c = _rci(p)
    if d == 0:
        r -= 1
    elif d == 1:
        c += 1
    elif d == 2:
        r += 1
    else:
        c -= 1
    return _idx(r, c)


def _go(cur, tgt, blocked):
    """贪心走一步靠近 tgt，返回方向下标；无可行方向返回 None。"""
    best_d = _dist(cur, tgt)
    best = None
    for d in range(4):
        nxt = _step(cur, d)
        if nxt in blocked:
            continue
        nd = _dist(nxt, tgt)
        if nd < best_d:
            best_d = nd
            best = d
    return best


def _neighbors(p):
    r, c = _rci(p)
    return (_idx(r - 1, c), _idx(r + 1, c), _idx(r, c - 1), _idx(r, c + 1))


def _cfg(conf, key, default):
    if conf is None:
        return default
    try:
        v = conf.get(key)
        if v is not None:
            return v
    except Exception:
        pass
    try:
        return conf[key]
    except Exception:
        return default


def agent(observation, configuration):
    try:
        return _agent(observation, configuration)
    except Exception:
        return {}


def _agent(observation, configuration):
    step = int(observation["step"])
    me = int(observation["player"])
    players = observation["players"]
    obs_carbon = observation["carbon"]
    my = players[me]
    opp = players[1 - me]

    cash = float(my[0])
    my_base_items = list(my[1].items())
    my_base_set = {int(v) for v in my[1].values()}
    en_base = {int(v) for v in opp[1].values()}
    base_all = my_base_set | en_base

    my_workers = my[2]
    opp_workers = opp[2]
    my_trees = my[3]
    trees = observation["trees"]

    # ---------------- 参数 ----------------
    conf = configuration
    worker_limit = int(_cfg(conf, "workerLimit", 10))
    rec_c_cost = float(_cfg(conf, "recCollectorCost", 30))
    rec_p_cost = float(_cfg(conf, "recPlanterCost", 30))
    plant_cost_base = float(_cfg(conf, "plantCost", 20))
    plant_ratio = float(_cfg(conf, "plantCostInflationRatio", 5))
    plant_inf_base = float(_cfg(conf, "plantCostInflationBase", 1.235))
    protective = int(_cfg(conf, "playerTreeProtectiveNumber", 0))
    seize_cost = float(_cfg(conf, "seizeCost", 20))
    lifespan = int(_cfg(conf, "treeLifespan", 50))
    episode = int(_cfg(conf, "episodeSteps", EPISODE_DEFAULT))

    remaining = episode - 1 - step

    # ---------------- 全图碳量估计 ----------------
    st = STATE
    if step == 0 or st["est"] is None:
        st["est"] = [PRIOR] * N
        st["seen"] = [False] * N
        st["hist"] = {}
        for b in base_all:
            st["est"][b] = 0.0
            st["seen"][b] = True
    est = st["est"]
    seen = st["seen"]

    tree_owner = {}
    for _tid, t in my_trees.items():
        tree_owner[int(t[0])] = me
    for _tid, t in opp[3].items():
        tree_owner[int(t[0])] = 1 - me
    tree_cells = set(tree_owner.keys())
    enemy_tree_near = set()
    for tc, own in tree_owner.items():
        if own != me:
            r, c = _rci(tc)
            for dr, dc in NB8:
                enemy_tree_near.add(_idx(r + dr, c + dc))

    # 可见格：采集员只看得见自己所在格，种树员看得见四连通邻格
    visible = set()
    for w in my_workers.values():
        p = int(w[0])
        if w[2] == "COLLECTOR":
            visible.add(p)
        else:
            visible.update(_neighbors(p))

    for p in visible:
        v = float(obs_carbon[p])
        est[p] = v
        seen[p] = True
        r, c = _rci(p)
        for mr, mc in ((r, SIZE - 1 - c), (SIZE - 1 - r, c), (SIZE - 1 - r, SIZE - 1 - c)):
            m = _idx(mr, mc)
            if not seen[m]:
                est[m] = v * 0.85

    # 未观测格按复利增长近似（真实规则：未被踩到且非树邻格的格子每步 *1.03，上限 100）
    for i in range(N):
        if i in visible:
            continue
        e = est[i]
        if e > 0.0:
            e *= 1.03
            est[i] = e if e < 100.0 else 100.0

    for tc in tree_cells:
        est[tc] = 0.0
        seen[tc] = True

    # ---------------- 敌方下一步落点预测 ----------------
    def _best_cell(src, weight):
        bc = -1
        bs = -1e9
        for i in range(N):
            if i in base_all or i in tree_cells:
                continue
            e = est[i]
            if e <= 1.0:
                continue
            s = e - weight * _dist(src, i)
            if s > bs:
                bs = s
                bc = i
        return bc

    enemy_col_next = {}
    enemy_planter_next = set()
    for w in opp_workers.values():
        p = int(w[0])
        is_col = (w[2] == "COLLECTOR")
        if is_col and float(w[1]) > 0 and p in en_base:
            continue  # 正在交碳
        bc = _best_cell(p, STEP_COST)
        q = p
        if bc >= 0:
            d = _go(p, bc, set())
            if d is not None:
                q = _step(p, d)
        if is_col:
            enemy_col_next.setdefault(q, []).append(float(w[1]))
        else:
            enemy_planter_next.add(q)

    # 敌方「下一步可能占据」的格子汇总：命中者含当前位置（可能原地不动）+ 预测落点。
    # 碰撞规则只看同格集合里的最小碳，所以这里同时维护最小碳与总碳：
    #   poss_min[q] > 我方携带量  => 我方是唯一最小者 => 主动撞上去可吞掉它们全部在途碳。
    poss_min = {}
    poss_sum = {}
    for epos, ecs in enemy_col_next.items():
        poss_min[epos] = min(ecs)
        poss_sum[epos] = sum(ecs)
    planter_possible = set(enemy_planter_next)
    for w in opp_workers.values():
        q = int(w[0])
        if w[2] == "COLLECTOR":
            c = float(w[1])
            if q in poss_min:
                if c < poss_min[q]:
                    poss_min[q] = c
                poss_sum[q] += c
            else:
                poss_min[q] = c
                poss_sum[q] = c
        else:
            planter_possible.add(q)

    # ---------------- 兵力统计与种树成本 ----------------
    n_col = 0
    n_pla = 0
    for w in my_workers.values():
        if w[2] == "COLLECTOR":
            n_col += 1
        else:
            n_pla += 1

    alive_trees = 0
    for t in trees.values():
        if int(t[1]) < lifespan:
            alive_trees += 1
    if len(my_trees) <= protective:
        plant_cost = plant_cost_base
    else:
        plant_cost = plant_cost_base + plant_ratio * (plant_inf_base ** alive_trees)

    # ---------------- 招募（跳过「本步有单位停留交付」的基地，避免新兵撞死） ----------------
    base_busy = set()
    for w in my_workers.values():
        if float(w[1]) > 0 and int(w[0]) in my_base_set:
            base_busy.add(int(w[0]))

    actions = {}
    # ===== P0 发育（最高优先级）：能招就招，采集员优先把经济引擎铺满 =====
    # 线上胜者对比：CarbonV14d / CarbonV12 都是 10 采集员 0 种树员打上 9k~10k；
    # 我方旧版第 50 步只有 6 人 → 雪球滚不起来。所以先堆采集员，再补种树员。
    if step < max(0, episode - 40):
        budget = cash
        got_c = 0
        got_p = 0
        for bid, bpos in my_base_items:
            if int(bpos) in base_busy:
                continue
            if len(my_workers) + got_c + got_p >= worker_limit:
                break
            if n_col + got_c < COLLECTOR_TARGET and budget - rec_c_cost >= 0:
                actions[bid] = "RECCOLLECTOR"
                got_c += 1
                budget -= rec_c_cost
            elif n_pla + got_p < PLANTER_TARGET and budget - rec_p_cost >= plant_cost:
                actions[bid] = "RECPLANTER"
                got_p += 1
                budget -= rec_p_cost
            else:
                break

    # ---------------- 种树潜力图（减掉种树成本，不赚不种） ----------------
    plant_pot = [0.0] * N
    if cash >= plant_cost and len(my_trees) < TREE_TARGET and remaining > 12:
        scale = min(1.0, remaining / 70.0)
        for i in range(N):
            if i in base_all or i in tree_cells:
                continue
            r, c = _rci(i)
            s = (est[_idx(r - 1, c)] + est[_idx(r + 1, c)]
                 + est[_idx(r, c - 1)] + est[_idx(r, c + 1)])
            plant_pot[i] = (TREE_ABS_EFF * s - est[i] - plant_cost) * scale

    # ---------------- 预期收益评估 ----------------
    def pos_value(q, cargo):
        """站在 q 的长期期望收益（移动代价已含在距离折算里，不再重复扣）。"""
        if remaining <= 0:
            return 0.0
        hq = min(my_base_set, key=lambda b: _dist(q, b))
        cc = 0.0 if q in my_base_set else cargo
        best = -1e9
        for i in range(N):
            if i in base_all or i in tree_cells:
                continue
            e = est[i]
            if e <= 1.0:
                continue
            dq = _dist(q, i)
            if dq <= 0:
                v = e - RETURN_W * _dist(i, hq)
            else:
                di = _dist(i, hq)
                if dq + di > remaining:
                    continue
                v = e - STEP_COST * dq - RETURN_W * di
            if v > best:
                best = v
        if cc > 0 and _dist(q, hq) <= remaining:
            dv = cc - STEP_COST * _dist(q, hq)
            if dv > best:
                best = dv
        return best if best > -1e8 else 0.0

    def eval_collector(p, cargo, q):
        prof = 0.0
        if q in my_base_set:
            prof += cargo                                   # 本步落点即基地 → 立刻结算
        elif q in en_base:
            prof -= cargo + DANGER_UNIT                     # 对方基地会吞掉在途碳
        prof += pos_value(q, cargo)
        ec = poss_min.get(q)
        if ec is not None:
            if ec > cargo:
                prof += poss_sum[q]                         # 我方碳最少 → 吃掉同格对手
            else:
                prof -= cargo + DANGER_UNIT                 # 输或平局 → 丢货丢人
        elif q in planter_possible:
            prof += PLANTER_KILL                            # 撞上对方种树员 → 我方必胜
        if cargo > 0:
            if q in enemy_tree_near:
                prof -= COLLECTOR_ABS * cargo               # 敌方树会吸走在途碳
            # 威胁按距离分级：贴脸最危险、隔一格次之、>=2 格不计（否则所有候选同等扣分 = 不会逃）
            mind = 9
            for epos, mc in poss_min.items():
                if mc < cargo:
                    dd = _dist(epos, q)
                    if dd < mind:
                        mind = dd
            if mind <= 1:
                prof -= (2 - mind) * (0.6 * cargo + 10.0)
        return prof

    def eval_planter(p, q):
        prof = 0.0
        best = -1e9
        for i in range(N):
            if plant_pot[i] > 0.0:
                v = plant_pot[i] - STEP_COST * _dist(q, i)
                if v > best:
                    best = v
        if best > 0.0:
            prof += best
        if q == p and q in tree_cells and tree_owner.get(q, me) != me and cash >= seize_cost:
            r, c = _rci(q)
            s = (est[_idx(r - 1, c)] + est[_idx(r + 1, c)]
                 + est[_idx(r, c - 1)] + est[_idx(r, c + 1)])
            prof += max(0.0, 0.6 * TREE_ABS_EFF * s * min(1.0, remaining / 70.0) - seize_cost)
        if q in poss_min or q in planter_possible or q in en_base:
            prof -= DANGER_UNIT                             # 种树员不该撞上对方单位（它必输）
        return prof

    # ---------------- 单位调度 ----------------
    occ = set()
    for w in my_workers.values():
        occ.add(int(w[0]))

    order = sorted(my_workers.items(), key=lambda kv: -float(kv[1][1]))

    for wid, w in order:
        p = int(w[0])
        cargo = float(w[1])
        is_col = (w[2] == "COLLECTOR")
        occ.discard(p)
        if is_col:
            # 只回避：我方单位格、对方基地、树格，以及「撞了会输或会平局」的敌方采集员格。
            # 自带量更少的敌方采集员格、以及敌方种树员格一律不回避（踩上去就吃掉/击杀）。
            blocked = occ | en_base | tree_cells
            if cargo <= 0:
                blocked |= my_base_set          # ★ 空载单位禁止进入自家基地，断掉「让位→回头」震荡
            for q, mc in poss_min.items():
                if mc <= cargo:
                    blocked.add(q)
        else:
            # 种树员碰撞必输，回避所有可能落点有敌方单位的格子
            blocked = occ | en_base | set(poss_min) | planter_possible

        home = min(my_base_set, key=lambda b: _dist(p, b))
        dh = _dist(p, home)

        free_nb = [q for q in _neighbors(p) if q not in blocked]

        # ★ 反卡死看门狗：连续 STALL_LIMIT 步原地不动且没有采集收益 → 强制脱离。
        # 线上证据（f5aaeccc）：C49 携带 14 碳在 49 号格卡 49 步、C51 携带 25 碳卡 58 步，
        # 全队 60 步零交付、现金冻结 20.8 → 招不起人 → 输。
        hist = st.setdefault("hist", {})
        h = hist.setdefault(wid, [])
        same = 0
        for e in reversed(h):
            if e[0] == p:
                same += 1
            else:
                break
        cargo_start = h[len(h) - same][1] if same > 0 else cargo
        h.append((p, cargo))
        if len(h) > 12:
            del h[0:len(h) - 12]
        if same >= STALL_LIMIT and cargo <= cargo_start + 0.5:
            if cargo > 0:
                d = _go(p, home, blocked)          # 带货卡死 → 直接回城结算
                if d is not None and _step(p, d) != p:
                    actions[wid] = DIR_NAMES[d]
                    occ.add(_step(p, d))
                    continue
            kick = None
            best_v = -1e18
            for q in free_nb:
                v = est[q] - 0.5 * _dist(q, home)
                if v > best_v:
                    best_v = v
                    kick = q
            if kick is not None:
                d = None
                for dd in range(4):
                    if _step(p, dd) == kick:
                        d = dd
                        break
                if d is not None:
                    actions[wid] = DIR_NAMES[d]
                    occ.add(kick)
                    continue

        if p in my_base_set and cargo <= 0:
            # ★ 死锁根因修复：自家基地格绝不能被自己人占住。
            # 线上证据：14.95 那局第 32 步在两个基地各招募了 1 个种树员，之后 268 步
            # 一次交付都没有（现金精确冻结在 14.95），在途碳却涨到 1120 —— 就是基地格
            # 被自家种树员占死、所有采集员都落不了地的结果。这里强制让位。
            if free_nb:
                best_q = free_nb[0]
                best_v = -1e18
                for q in free_nb:
                    v = eval_collector(p, cargo, q) if is_col else eval_planter(p, q)
                    if v > best_v:
                        best_v = v
                        best_q = q
                d = None
                for dd in range(4):
                    if _step(p, dd) == best_q:
                        d = dd
                        break
                if d is None:
                    d = 0
                actions[wid] = DIR_NAMES[d]
                occ.add(_step(p, d))
            else:
                occ.add(p)
            continue

        cands = [p] + free_nb

        # 采集员停在自家基地且有货 → 立即交付（最安全的动作，优先于自保与其它策略）
        if is_col and p in my_base_set and cargo > 0:
            occ.add(p)
            continue

        # ===== P1 自保（次高优先级）：先保住已投入的资产（单位 + 在途碳）=====
        # 采集员：贴脸存在「比我更轻」的敌方采集员时必被吃 → 优先拉开距离，能回基地就交付
        # 种树员：任何敌方单位撞上来它都必死（碰撞里种树员必输）→ 同样脱身
        if is_col:
            threats = [e for e, mc in poss_min.items() if mc < cargo]
        else:
            threats = list(poss_min) + list(planter_possible)
        if threats:
            md0 = 9
            for e in threats:
                dd = _dist(e, p)
                if dd < md0:
                    md0 = dd
            if md0 <= 1:
                best_q = None
                best_key = None
                for q in cands:
                    md = 9
                    for e in threats:
                        dd = _dist(e, q)
                        if dd < md:
                            md = dd
                    # 排序键：① 采集员能落到自家基地＝直接把在途碳清零（最彻底的自保）
                    #          ② 落点离所有更轻的威胁 ≥2 格（安全） ③ 越靠近家越好（尽快交付）
                    base_bonus = 1 if (is_col and q in my_base_set) else 0
                    key = (base_bonus, 1 if md >= 2 else 0, -_dist(q, home))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_q = q
                if best_q is not None and best_q != p:
                    d = None
                    for dd in range(4):
                        if _step(p, dd) == best_q:
                            d = dd
                            break
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(best_q)
                        continue

        # ===== P2 原有策略：预期收益评估（与 P0 招募在同一回合并行执行）=====
        if is_col:
            cap = min(CARRY_MAX, CARRY_BASE + CARRY_PER_DIST * dh)
            # ★ 交付优先级提高：v1 输掉的局全部是「开局现金熄火」。
            #   满载 / 时间不够往返 / 终局 → 立刻回城结算。
            if p != home and (cargo >= cap or remaining <= dh + 1 + ENDGAME):
                d = _go(p, home, blocked)
                if d is not None:
                    actions[wid] = DIR_NAMES[d]
                    occ.add(_step(p, d))
                else:
                    occ.add(p)
                continue

            # 猎杀：轻装（几乎必然在碰撞中获胜）时主动贴近携带量更大的敌方采集员。
            # 依据：线上胜者单笔交付可达 300~490，那是撞死对手缴获的在途碳，是最大单笔收入；
            # 我方 294 那局为碰撞损失约 1000 碳，同一机制反过来用就是纯收益。
            if remaining > 30 and cargo <= 10.0:
                tgt = -1
                best_v = 0.0
                for epos, mc in poss_min.items():
                    if mc < 30.0 or mc <= cargo:
                        continue
                    dd = _dist(p, epos)
                    if dd <= 3:
                        v = poss_sum[epos] - 2.0 * dd
                        if v > best_v:
                            best_v = v
                            tgt = epos
                if tgt >= 0:
                    d = _go(p, tgt, blocked)
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(_step(p, d))
                        continue

            best_q = None
            best_prof = -1e18
            for q in cands:
                pr = eval_collector(p, cargo, q)
                if pr > best_prof:
                    best_prof = pr
                    best_q = q
            if best_prof <= 0.0:
                # 负收益 → 对策阶梯（★ 带货时取消「原地待机」，那等于把钱锁死在路上）
                if cargo > 0 and p != home:
                    dest = home                            # 对策 A：先保住已有收益
                else:
                    dest = _best_cell(p, EXPLORE_W)         # 对策 B：降低距离权重去探索
                if dest is not None and dest >= 0:
                    d = _go(p, dest, blocked)
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(_step(p, d))
                        continue
                if cargo > 0 and p != home:
                    d = _go(p, home, blocked)              # 被堵也要往家的方向挪，绝不带货发呆
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(_step(p, d))
                        continue
                occ.add(p)
                continue
            if best_q == p:
                occ.add(p)
                continue
            d = None
            for dd in range(4):
                if _step(p, dd) == best_q:
                    d = dd
                    break
            if d is None:
                occ.add(p)
                continue
            actions[wid] = DIR_NAMES[d]
            occ.add(best_q)

        else:  # 种树员
            best_q = None
            best_prof = 0.0
            for q in cands:
                pr = eval_planter(p, q)
                if pr > best_prof:
                    best_prof = pr
                    best_q = q
            if best_q is None:
                # 无利可图 → 去抢敌方树，或停到自家树上（不占基地、不会被误种树）
                dest = -1
                bd = 1e9
                if cash >= seize_cost:
                    for tc, own in tree_owner.items():
                        if own != me:
                            dd = _dist(p, tc)
                            if dd < bd:
                                bd = dd
                                dest = tc
                if dest < 0:
                    bd = 1e9
                    for tc, own in tree_owner.items():
                        if own == me:
                            dd = _dist(p, tc)
                            if dd < bd:
                                bd = dd
                                dest = tc
                if dest >= 0 and dest != p:
                    d = _go(p, dest, blocked)
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(_step(p, d))
                        continue
                occ.add(p)
                continue
            if best_q == p:
                occ.add(p)
                continue
            d = None
            for dd in range(4):
                if _step(p, dd) == best_q:
                    d = dd
                    break
            if d is None:
                occ.add(p)
                continue
            actions[wid] = DIR_NAMES[d]
            occ.add(best_q)

    return actions

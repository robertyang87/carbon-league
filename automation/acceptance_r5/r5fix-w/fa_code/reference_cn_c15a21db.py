"""Carbon League 提交策略（v2：每步做「预期收益」评估后再行动）。

规则依据（来自 zerosum_env 的 helpers.py / carbon.json，本文件未借鉴他人策略源码）：
  * 坐标：观测位置是扁平索引 index = row * size + col；UP/DOWN 改 row，RIGHT/LEFT 改 col，
    边界环绕（15x15 环面）。距离用环面最短曼哈顿距离。
  * 采集：采集员「停留」才会采集（拿走该格 30% 碳）；回到自家转化中心「停留」才把
    在途碳结算成现金（smeltCost = 0）。
  * 种树：种树员在无树、无基地的格子上「停留」即种树，花费 plantCost(20) + 5*1.235^n；
    树每步吸收四连通邻格碳（3.75%）并换成现金；树 50 步后枯死并留下 30 碳。
  * 抢树：种树员在敌方树上「停留」，花 seizeCost(20) 即可夺取该树。
  * 碰撞（关键、反直觉）：同一格上若存在采集员，则「碳最少」的那个采集员获胜，
    吞掉同格其他单位携带的碳，其余单位全部消失；若最少碳并列，则同格单位全灭；
    若同格没有采集员，则全灭。所以碳少的采集员撞上碳多的采集员＝稳赚。

决策方式（对应用户要求的三条）：
  1. 每步先解析局势（可见碳、各单位位置与携带量）并预测敌方下一步落点；
     然后对每个己方单位的每个候选动作计算「预期收益」，只执行收益为正的最佳动作。
  2. 若所有候选动作收益都 ≤ 0，则进入对策阶梯：先保收益（回基地交付在途碳），
     再按降低移动权重的方式重新评估探索（换取信息），仍不行则原地待机（不亏）。
  3. 每一步都重复以上流程，直到对局结束。

状态仅保存在本进程内，step == 0 时重置。
"""

SIZE = 15
N = SIZE * SIZE

PRIOR = 12.0          # 未观测格子的先验碳量
STEP_COST = 1.0       # 距离在位置估值里的折算代价（时间成本）
RETURN_W = 0.35       # 归程距离权重
EXPLORE_W = 0.45      # 对策阶梯里探索用的低距离权重
COLLECTOR_ABS = 0.1   # 敌方树对周边我方采集员在途碳的吸收率
PLANTER_KILL = 15.0   # 击杀敌方种树员的附加收益（对方需再花 30 招募）
DANGER_UNIT = 25.0    # 单位损失的估值
COLLECTOR_TARGET = 6
PLANTER_TARGET = 2
TREE_TARGET = 4
EPISODE_DEFAULT = 300

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
        for b in base_all:
            st["est"][b] = 0.0
            st["seen"][b] = True
    est = st["est"]
    seen = st["seen"]

    tree_owner = {}
    for tid, t in my_trees.items():
        tree_owner[int(t[0])] = me
    for tid, t in opp[3].items():
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

    # 未观测格按复利增长近似（真实规则：未被踩到且无树/非树邻格的格子 step *1.03，上限 100）
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
            continue  # 正在交碳，不必预测
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

    # ---------------- 招募 ----------------
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

    actions = {}
    budget = cash
    queue = []
    recruit_until = max(0, episode - 40)
    if step < recruit_until:
        while len(queue) < len(my_base_items):
            if len(my_workers) + len(queue) >= worker_limit:
                break
            col_n = n_col + queue.count("RECCOLLECTOR")
            pla_n = n_pla + queue.count("RECPLANTER")
            if col_n < 2:
                if budget - rec_c_cost < 0:
                    break
                queue.append("RECCOLLECTOR")
                budget -= rec_c_cost
            elif pla_n < PLANTER_TARGET and budget - rec_p_cost >= plant_cost:
                queue.append("RECPLANTER")
                budget -= rec_p_cost
            elif col_n < COLLECTOR_TARGET and budget - rec_c_cost >= 0:
                queue.append("RECCOLLECTOR")
                budget -= rec_c_cost
            else:
                break
    for i, (bid, _bpos) in enumerate(my_base_items):
        if i < len(queue):
            actions[bid] = queue[i]

    # ---------------- 种树潜力图（只算一次，供所有种树员复用） ----------------
    plant_pot = [0.0] * N
    want_tree = (cash >= plant_cost and len(my_trees) < TREE_TARGET and remaining > 10)
    if want_tree:
        scale = min(1.0, remaining / 70.0)
        for i in range(N):
            if i in base_all or i in tree_cells:
                continue
            r, c = _rci(i)
            s = (est[_idx(r - 1, c)] + est[_idx(r + 1, c)]
                 + est[_idx(r, c - 1)] + est[_idx(r, c + 1)])
            plant_pot[i] = (s - 0.6 * est[i]) * scale

    # ---------------- 预期收益评估 ----------------
    def pos_value(q, cargo):
        """站在 q 的长期期望收益。

        位置估值本身就是一张「距离+归程」折算后的距离变换图，移动的代价已经包含在
        里面，因此比较候选落点时不再重复扣一步移动费。
        q 是己方基地时先交付清零，故在 q 处不再计在途碳（交付收益在调用方单独计）。
        """
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
        ec = enemy_col_next.get(q)
        if ec:
            allc = list(ec) + [cargo]
            m = min(allc)
            if allc.count(m) == 1 and m == cargo:
                prof += sum(ec)                             # 我方碳最少 → 吃掉对手
            else:
                prof -= cargo + DANGER_UNIT                 # 输或平局 → 丢货丢人
        elif q in enemy_planter_next:
            prof += PLANTER_KILL                            # 单独撞上对方种树员 → 击杀
        if cargo > 0:
            if q in enemy_tree_near:
                prof -= COLLECTOR_ABS * cargo               # 敌方树会吸走在途碳
            for epos, ecs in enemy_col_next.items():
                if _dist(epos, q) <= 1:
                    for ecar in ecs:
                        if ecar < cargo:
                            prof -= 0.4 * cargo             # 附近有更轻的敌方采集员 → 我可能是猎物
                            break
        return prof

    def eval_planter(p, q):
        prof = 0.0
        # 未来最优种树落点的价值（已含走到该点的距离折算）
        best = -1e9
        for i in range(N):
            if plant_pot[i] > 0.0:
                v = plant_pot[i] - STEP_COST * _dist(q, i)
                if v > best:
                    best = v
        if best > 0.0:
            prof += best
        if q == p and q in tree_cells and tree_owner.get(q, me) != me and cash >= seize_cost:
            # 站在敌方树上停留 → 抢树
            r, c = _rci(q)
            s = (est[_idx(r - 1, c)] + est[_idx(r + 1, c)]
                 + est[_idx(r, c - 1)] + est[_idx(r, c + 1)])
            prof += max(0.0, 0.6 * s * min(1.0, remaining / 70.0) - seize_cost)
        if q in enemy_col_next or q in enemy_planter_next or q in en_base:
            prof -= DANGER_UNIT                             # 种树员不该撞上对方单位
        return prof

    # ---------------- 单位调度 ----------------
    occ = set()
    for w in my_workers.values():
        occ.add(int(w[0]))
    for w in opp_workers.values():
        occ.add(int(w[0]))

    order = sorted(my_workers.items(), key=lambda kv: -float(kv[1][1]))

    for wid, w in order:
        p = int(w[0])
        cargo = float(w[1])
        occ.discard(p)
        blocked = occ | en_base

        cands = [p]
        for q in _neighbors(p):
            if q in blocked:
                continue
            cands.append(q)

        if w[2] == "COLLECTOR":
            # 第一步：只挑正收益的动作
            best_q = None
            best_prof = 0.0
            for q in cands:
                pr = eval_collector(p, cargo, q)
                if pr > best_prof:
                    best_prof = pr
                    best_q = q
            if best_q is None:
                # 第二步：负收益 → 找对策
                home = min(my_base_set, key=lambda b: _dist(p, b))
                if cargo > 0 and p != home:
                    dest = home                                    # 对策 A：先保住已有收益
                else:
                    dest = _best_cell(p, EXPLORE_W)                # 对策 B：降低距离权重去探索，换信息
                if dest is not None and dest >= 0:
                    d = _go(p, dest, blocked)
                    if d is not None:
                        actions[wid] = DIR_NAMES[d]
                        occ.add(_step(p, d))
                        continue
                occ.add(p)
                continue                                           # 对策 C：原地待机，不亏
            if best_q == p:
                occ.add(p)
                continue                                           # 停留（采集/交付）
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
                home = min(my_base_set, key=lambda b: _dist(p, b))
                if p != home:
                    d = _go(p, home, blocked)
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

# -*- coding: utf-8 -*-
"""
捕碳大作战 strong_agent v2 —— 信念地图 + 满编经济 + 树/对抗战术

设计依据 (仅环境规则与对战公开数据, 不读对手代码):
1. 视野: 环境只把"己方捕碳员所在格 + 种树员四邻域"的碳值给 obs.carbon,
   但双方工人/树/基地/现金位置全程公开 => 维护信念图:
   - 未见过格: 用"地图四象限对称(初始同值) + 21.6 均值先验"估计初值, 每回合按
     3% 再生 / 树4邻域 3.75% 吸收 / 敌方捕碳员停留 30% 采集 演化;
     自己传感器每回合用真实值覆写。
   - 双方树位置公开 => 树抑制/吸收区精确; 敌方捕碳员停留格可精确扣减。
2. 经济: 早期买捕碳员滚现金 -> 中期满编(约 6-7 采 + 3 种)。
   种树员: a) 抢对方树(固定20, 保留树龄); b) 信念图上 4 邻域碳和最大的空地种
   新树, 滚动补种对冲 50 回合寿命; 市场价过高则只抢不种。
3. 对抗: 空载捕碳员当保镖守高价值树(敌种树员来抢=撞死); 猎杀对方满载捕碳员
   (碰撞规则: 载碳最少者存活并缴获死者碳); 满载员绕开敌方树 8 邻域(抽税10%/回合)
   与敌方低载捕碳员。
4. 终局: 剩余回合 <= 回家距离+2 时强制回存变现; 树吃到终局; 剩<45步不再种新树。
5. 防淘汰: 没工人没树时优先招人; 招募保留现金下限。

性能: 信念演化/打分为 225 格 numpy 向量化; 路径为 225 节点 Dijkstra, 单回合 << 1s。

索引约定: idx = (size-1-y)*size + x; 内部用 (r,c)=(idx//size, idx%size)(r 自上而下),
方向: UP=r-1, DOWN=r+1, LEFT=c-1, RIGHT=c+1, 环面 wrap。
"""

import heapq
import time

import numpy as np

# ---------------- 可调参数 ----------------
P = {
    "prior": 21.6,           # 初始格碳先验 (2500 / 约116个非零格)
    "lam_move": 5.0,         # 捕碳目标评分: 每步距离的代价
    "mine_stay_min": 8.0,    # 当前格信念值低于此值换格
    "mine_margin_min": 1.0,  # 边际(0.3*v - 0.1*敌树数*载碳)低于此值换格
    "dep_base": 18.0,        # 回存阈值 T(d)=clamp(dep_base+dep_k*d, dep_min, dep_max)
    "dep_k": 5.0,
    "dep_min": 28.0,
    "dep_max": 90.0,
    "hunt_carry": 70.0,      # 敌载碳超过此值才值得猎杀
    "hunt_reach": 6,         # 猎杀者到拦截点的最大环面距离
    "guard_max_age": 16,     # 树龄<=此值 且 上回合吸收>=guard_abs 才护卫
    "guard_abs": 2.0,
    "guard_threat_d": 4,     # 敌种树员距树<=此值算威胁
    "claim_pen": 45.0,       # 己方其他捕碳员已认领格的惩罚
    "mob_risk_loaded": 4.0,  # 敌低载捕碳员可能到的格(对我满载)的路径附加成本
    "mob_risk_planter": 9.0, # 敌捕碳员可能到的格(对我种树员)的附加成本
    "mob_risk_pl_vs_pl": 6.0,# 敌种树员可能到的格(对我种树员)的附加成本
    "enemy_tree_taxk": 0.08, # 敌树8邻域对我载碳(10%/回合)的路径成本系数
    "reserve_cost": 3.0,     # 我方已占用格的路径附加成本
    "lam_plant": 5.0,        # 种树选址: 每步距离的代价(近者优先)
    "plant_margin": 6.0,     # 种树ROI余量: plant_rate*sum4 >= 市场价+margin
    "plant_rate": 0.85,      # 树一生约能吸收邻域碳的比例
    "own_tree_cap": 10,      # 自有树软上限
    "tree_near_pen": 10.0,   # 选址与已有树相邻的惩罚
    "en_pl_near_pen": 25.0,  # 种树点附近有敌方种树员的惩罚(会被反抢)
    "lam_seize": 4.0,        # 抢树选址: 每步距离的代价
    "seize_min": 30.0,       # 抢树最低评分
    "plant_stick": 60.0,     # 种树任务黏性: 上回合目标格加分(防打分抖动)
    "recruit_floor": 22.0,   # 招募后至少保留的现金(防淘汰)
}

_CTX = {}      # player_id -> 信念状态(跨回合)
_STATS = {"max_turn_s": 0.0}
_DEBUG = False


# ---------------- 观测解析 ----------------
def _g(o, k, d=None):
    try:
        v = o[k]
        if v is not None:
            return v
    except Exception:
        pass
    try:
        return getattr(o, k, d)
    except Exception:
        return d


def _items(d):
    try:
        return list(d.items())
    except Exception:
        return []


class _W(object):
    __slots__ = ("step", "size", "n", "cash", "my_base", "en_base", "my_coll",
                 "my_pl", "en_coll", "en_pl", "my_trees", "en_trees",
                 "tree_abs", "carbon", "bases")


def _parse(obs, size):
    n = size * size
    players = _g(obs, "players", [])
    me = int(_g(obs, "player", 0) or 0)
    w = _W()
    w.size = size
    w.n = n
    w.step = int(_g(obs, "step", 0) or 0)
    try:
        my = players[me]
        en = players[1 - me]
        w.cash = float(my[0])
    except Exception:
        my, en = [80.0, {}, {}, {}], [80.0, {}, {}, {}]
        w.cash = 80.0
    w.bases = [(str(bid), int(bidx)) for bid, bidx in _items(my[1])]
    w.my_base = [bidx for _, bidx in w.bases]
    w.en_base = [int(v) for _, v in _items(en[1])]
    w.my_coll, w.my_pl = [], []
    for k, v in _items(my[2]):
        wid, idx, car, occ = str(k), int(v[0]), float(v[1]), str(v[2])
        if occ == "COLLECTOR":
            w.my_coll.append((wid, idx, car))
        else:
            w.my_pl.append((wid, idx))
    w.en_coll, w.en_pl = [], []
    for k, v in _items(en[2]):
        wid, idx, car, occ = str(k), int(v[0]), float(v[1]), str(v[2])
        if occ == "COLLECTOR":
            w.en_coll.append((wid, idx, car))
        else:
            w.en_pl.append((wid, idx))
    w.my_trees = {str(k): (int(v[0]), int(v[1])) for k, v in _items(my[3])}
    w.en_trees = {str(k): (int(v[0]), int(v[1])) for k, v in _items(en[3])}
    w.tree_abs = {}
    for k, v in _items(_g(obs, "trees", {}) or {}):
        try:
            w.tree_abs[str(k)] = float(v[1])
        except Exception:
            pass
    carb = _g(obs, "carbon", [])
    w.carbon = [float(x) for x in carb] if carb and len(carb) == n else None
    return w


def _cfg(conf):
    return {
        "size": int(_g(conf, "size", 15) or 15),
        "regen": float(_g(conf, "regenRate", 0.03)),
        "abs_cell": float(_g(conf, "cellAbsorptionRate", 0.0375)),
        "abs_coll": float(_g(conf, "collectorAbsorptionRate", 0.1)),
        "collect": float(_g(conf, "initialCollectRate", 0.3)),
        "tree_life": int(_g(conf, "treeLifespan", 50)),
        "plant_cost": float(_g(conf, "plantCost", 20)),
        "infl_ratio": float(_g(conf, "plantCostInflationRatio", 5)),
        "infl_base": float(_g(conf, "plantCostInflationBase", 1.235)),
        "protective": int(_g(conf, "playerTreeProtectiveNumber", 0)),
        "seize_cost": float(_g(conf, "seizeCost", 20)),
        "wlimit": int(_g(conf, "workerLimit", 10)),
        "rec_coll": float(_g(conf, "recCollectorCost", 30)),
        "rec_pl": float(_g(conf, "recPlanterCost", 30)),
        "steps": int(_g(conf, "episodeSteps", 300)),
        "wither": float(_g(conf, "co2FrmWithered", 30)),
    }


# ---------------- 信念地图 ----------------
class _Ctx(object):
    __slots__ = ("size", "n", "t", "ev", "v0", "v0w", "pure", "seen", "cur",
                 "nb", "nb_act", "row", "col", "prev_trees", "prev_ecoll",
                 "regen_f", "mirrors", "ptarget")

    def __init__(self, size):
        self.size = size
        self.n = n = size * size
        self.t = 0
        self.ev = np.full(n, P["prior"])
        self.v0 = np.full(n, P["prior"])
        self.v0w = np.ones(n)
        self.pure = np.ones(n, bool)
        self.seen = np.full(n, -1)
        self.cur = np.full(n, P["prior"])
        idx = np.arange(n)
        r, c = idx // size, idx % size
        self.row, self.col = r, c
        self.nb = [((r - 1) % size) * size + c, ((r + 1) % size) * size + c,
                   r * size + (c - 1) % size, r * size + (c + 1) % size]
        self.nb_act = ["UP", "DOWN", "LEFT", "RIGHT"]
        mir = np.empty((n, 3), dtype=int)
        mir[:, 0] = r * size + (size - 1 - c)
        mir[:, 1] = (size - 1 - r) * size + c
        mir[:, 2] = (size - 1 - r) * size + (size - 1 - c)
        self.mirrors = mir
        self.prev_trees = set()
        self.prev_ecoll = {}
        self.regen_f = 1.0
        self.ptarget = {}      # planter 种树任务黏性: wid -> 目标格


def _belief_update(ctx, w, cfg):
    """演化上一回合事件, 写入本回合传感器, 算 ctx.cur(决策期望值)."""
    n, size, t = ctx.n, ctx.size, w.step
    regf = (1.0 + cfg["regen"]) ** t
    ctx.regen_f = regf
    all_tree = set(w.my_trees.values()) | set(w.en_trees.values())
    tree_pos = {p for p, _ in all_tree}

    cover = np.zeros(n)                       # 树对 4 邻域覆盖数
    for p in tree_pos:
        for k in range(4):
            cover[int(ctx.nb[k][p])] += 1.0
    stay = np.zeros(n, bool)                  # 敌捕碳员停留格
    en_coll_now = {wid: idx for wid, idx, _ in w.en_coll}
    for wid, idx in en_coll_now.items():
        if ctx.prev_ecoll.get(wid) == idx:
            stay[idx] = True
    dead = ctx.prev_trees - tree_pos          # 本回合枯死的树

    mat = ctx.pure & ((cover > 0) | stay)
    for p in dead:
        mat[p] = True
    if mat.any():
        ctx.ev[mat] = np.minimum(ctx.v0[mat] * regf, 100.0)
        ctx.pure[mat] = False

    live_idx = np.nonzero(~ctx.pure)[0]
    regv = 1.0 + cfg["regen"]
    drain = 1.0 - cfg["abs_cell"]
    mult = np.where(cover > 0, drain ** cover, regv)
    if live_idx.size:
        ctx.ev[live_idx] = np.minimum(ctx.ev[live_idx] * mult[live_idx], 100.0)
    if stay.any():
        ctx.ev[stay] *= (1.0 - cfg["collect"])
    for p in dead:
        ctx.ev[p] = cfg["wither"]
        ctx.pure[p] = False
    for bp in w.my_base:
        ctx.ev[bp] = 0.0
    for bp in w.en_base:
        ctx.ev[bp] = 0.0

    # 传感器: 己捕碳员所在格 + 种树员四邻域
    if w.carbon is not None:
        sensors = []
        for _, idx, _ in w.my_coll:
            sensors.append(idx)
        for _, idx in w.my_pl:
            for k in range(4):
                sensors.append(int(ctx.nb[k][idx]))
        base_set = set(w.my_base) | set(w.en_base)
        for idx in sensors:
            val = w.carbon[idx]
            first = bool(ctx.pure[idx])
            ctx.ev[idx] = val
            ctx.pure[idx] = False
            ctx.seen[idx] = t
            if first:
                v0c = 0.0 if val <= 0.5 else min(val / regf, 50.0)
                for m in ctx.mirrors[idx]:
                    if ctx.pure[m] and m not in tree_pos and m not in base_set:
                        wgt = ctx.v0w[m] + 1.0
                        ctx.v0[m] = (ctx.v0[m] * ctx.v0w[m] + v0c) / wgt
                        ctx.v0w[m] = wgt

    ctx.cur = np.where(ctx.pure, np.minimum(ctx.v0 * regf, 100.0), ctx.ev)
    ctx.prev_trees = tree_pos
    ctx.prev_ecoll = en_coll_now


# ---------------- 路径: 环面 Dijkstra ----------------
def _route(ctx, start, target, cost):
    """从 start 向 target 走一步; 返回方向字符串或 None(已在/无路)."""
    if start == target:
        return None
    n = ctx.n
    INF = 1e15
    dist = [INF] * n
    pred = [-1] * n
    dist[start] = 0.0
    h = [(0.0, start)]
    nb = ctx.nb
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u] + 1e-12:
            continue
        if u == target:
            break
        for k in range(4):
            v = int(nb[k][u])
            nd = d + 1.0 + cost[v]
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(h, (nd, v))
    if dist[target] >= INF / 2:
        return None
    cur = target
    guard = 0
    while pred[cur] != start and guard <= n:
        cur = pred[cur]
        guard += 1
        if cur < 0:
            return None
    if pred[cur] != start:
        return None
    for k in range(4):
        if int(nb[k][start]) == cur:
            return ctx.nb_act[k]
    return None


def _torus_dist(ctx, a, b):
    size = ctx.size
    dr = abs(a // size - b // size)
    dc = abs(a % size - b % size)
    return min(dr, size - dr) + min(dc, size - dc)


def _dist_field(ctx, idx0):
    size = ctx.size
    r0, c0 = idx0 // size, idx0 % size
    dr = np.abs(ctx.row - r0)
    dc = np.abs(ctx.col - c0)
    return np.minimum(dr, size - dr) + np.minimum(dc, size - dc)


def _greedy_path(ctx, a, b):
    """a→b 贪心路径(不含起点), 用于预测敌方回基路线."""
    size = ctx.size
    path = []
    cur = a
    for _ in range(2 * size + 2):
        if cur == b:
            break
        best, bestd = None, 1e9
        r, c = cur // size, cur % size
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nn = ((r + dr) % size) * size + (c + dc) % size
            d = _torus_dist(ctx, nn, b)
            if d < bestd:
                bestd, best = d, nn
        cur = best
        path.append(cur)
    return path


def _fallback_step(ctx, idx, tgt, w, reserved, my_cells_now, planter):
    """Dijkstra 无路时的贪心一步: 硬避开敌基/敌树/我方占位."""
    size = ctx.size
    en_tree = {p for p, _ in w.en_trees.values()}
    en_base = set(w.en_base)
    bestact, bestd = None, 1e9
    r, c = idx // size, idx % size
    for dr, dc, act in ((-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")):
        nn = ((r + dr) % size) * size + (c + dc) % size
        if nn in reserved or nn in my_cells_now or nn in en_base:
            continue
        if planter and nn in en_tree and nn != tgt:
            continue
        d = _torus_dist(ctx, nn, tgt)
        if d < bestd:
            bestd, bestact = d, act
    return bestact


def _dir_between(ctx, a, b):
    """相邻格 a->b 的方向名."""
    size = ctx.size
    ra, ca, rb, cb = a // size, a % size, b // size, b % size
    if (ra - rb) % size == 1:
        return "UP"
    if (rb - ra) % size == 1:
        return "DOWN"
    if (ca - cb) % size == 1:
        return "LEFT"
    if (cb - ca) % size == 1:
        return "RIGHT"
    return None


def _park_step_out(ctx, idx, w, reserved, my_cells_now):
    """种树员站在基地格时的挪出一步: (action, next_idx) 或 None."""
    size = ctx.size
    en_tree = {p for p, _ in w.en_trees.values()}
    en_base = set(w.en_base)
    r, c = idx // size, idx % size
    for dr, dc, act in ((1, 0, "DOWN"), (-1, 0, "UP"), (0, 1, "RIGHT"), (0, -1, "LEFT")):
        nn = ((r + dr) % size) * size + (c + dc) % size
        if nn in reserved or nn in my_cells_now or nn in en_base or nn in en_tree:
            continue
        return (act, nn)
    return None


# ---------------- 编制与招募 ----------------
def _ladder(step, wlimit):
    """(目标捕碳员数, 目标种树员数)"""
    if step < 2:
        pl, cl = 0, 2
    elif step < 8:
        pl, cl = 1, 3
    elif step < 26:
        pl, cl = 2, 4
    elif step < 70:
        pl, cl = 2, 5
    elif step < 140:
        pl, cl = 3, 6
    else:
        pl, cl = 3, 7
    return min(cl, max(0, wlimit - pl)), pl


def _recruit(cfg, w, actions, reserved, my_cells_now):
    """招募: 按目标编制补员; 招募基地须无己方工人落点(避免同格碰撞)."""
    n_w = len(w.my_coll) + len(w.my_pl)
    if n_w >= cfg["wlimit"]:
        return
    cash = w.cash
    cl_t, pl_t = _ladder(w.step, cfg["wlimit"])
    free = []
    for bid, bidx in w.bases:
        if bidx in my_cells_now or bidx in reserved:
            continue
        free.append((bid, bidx))
    if not free:
        return
    floor = 0.0 if w.step <= 1 else 12.0
    if _DEBUG and w.step % 10 == 0:
        print("[recruit] t=%d cash=%.1f pl_t=%d cl_t=%d npl=%d ncl=%d free=%d n_w=%d" % (
            w.step, cash, pl_t, cl_t, len(w.my_pl), len(w.my_coll), len(free), n_w))
    # 救命招: 没工人没树 => 有钱就招, 避免被淘汰
    if n_w == 0 and not w.my_trees:
        bid, _ = free[0]
        if cash >= cfg["rec_coll"]:
            actions[bid] = "RECCOLLECTOR"
        elif cfg["rec_pl"] <= cash < cfg["rec_coll"]:
            actions[bid] = "RECPLANTER"
        return
    # 1) 种树员: 现金只需 30+15 (=首树价), 树经济尽早启动
    if len(w.my_pl) < pl_t and free and cash >= cfg["rec_pl"] + 15.0:
        bid, _ = free.pop(0)
        actions[bid] = "RECPLANTER"
        if _DEBUG:
            print("[recruit:P] t=%d gate_cash=%.1f bases_free_after=%d" % (w.step, cash, len(free)))
        cash -= cfg["rec_pl"]
    # 2) 捕碳员 (两个基地均可, 互不占用同一基地格)
    while free and cash >= cfg["rec_coll"] + floor:
        if len(w.my_coll) >= cl_t:
            break
        bid, _ = free.pop(0)
        if dict(actions).get(bid):
            continue
        actions[bid] = "RECCOLLECTOR"
        if _DEBUG:
            print("[recruit:C] t=%d gate_cash=%.1f" % (w.step, cash))
        cash -= cfg["rec_coll"]


# ---------------- 主决策 ----------------
def _agent_impl(obs, conf):
    me = int(_g(obs, "player", 0) or 0)
    cfg = _cfg(conf)
    size = cfg["size"]
    w = _parse(obs, size)
    n = w.n
    step = w.step

    if step == 0 or me not in _CTX or _CTX[me].size != size:
        _CTX[me] = _Ctx(size)
    ctx = _CTX[me]
    ctx.t = step
    _belief_update(ctx, w, cfg)

    actions = {}
    reserved = set()          # 本回合我方预计落点
    my_cells_now = set()
    for _, idx, _ in w.my_coll:
        my_cells_now.add(idx)
    for _, idx in w.my_pl:
        my_cells_now.add(idx)

    tree_pos = {p for p, _ in (set(w.my_trees.values()) | set(w.en_trees.values()))}
    base_all = set(w.my_base) | set(w.en_base)
    remaining = cfg["steps"] - step

    # 敌方威胁图
    en_reachc = np.full(n, np.inf)
    for _, eidx, ecar in w.en_coll:
        if ecar < en_reachc[eidx]:
            en_reachc[eidx] = ecar
        for k in range(4):
            v = int(ctx.nb[k][eidx])
            if ecar < en_reachc[v]:
                en_reachc[v] = ecar
    en_tax = np.zeros(n)                  # 敌树 8 邻域覆盖数(对载碳抽税)
    for p, _ in w.en_trees.values():
        rr, cc = p // size, p % size
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                en_tax[((rr + dr) % size) * size + (cc + dc) % size] += 1.0
    en_pl_cells = np.zeros(n)
    en_pl_far = np.zeros(n)      # 敌 planter 活动圈(dist<=4)密度
    for _, pidx in w.en_pl:
        en_pl_cells[pidx] += 1.0
        dfield = _dist_field(ctx, pidx)
        en_pl_far = en_pl_far + (dfield <= 4).astype(float)
        for k in range(4):
            en_pl_cells[int(ctx.nb[k][pidx])] += 1.0

    # ================= 捕碳员角色 =================
    roles = {}      # wid -> (kind, target)
    assigned = set()

    # 1) 守树保镖: 幼树(吸收可观时) + 任一敌种树员在可达范围内 => 派空载 collector
    #    驻守树格本体(敌 planter 要抢必须踩上来 => 撞死), 核心防"种完几帧被偷"
    guarded_trees = 0
    my_trees_sorted = sorted(
        w.my_trees.items(),
        key=lambda kv: (kv[1][0], -w.tree_abs.get(kv[0], 0.0)))
    for tid, (tidx, age) in my_trees_sorted:
        if guarded_trees >= 3:
            break
        absr = w.tree_abs.get(tid, 0.0)
        if age > P["guard_max_age"] or absr < P["guard_abs"]:
            # 幼树还没吸收记录的, 出生头几帧按"必然保护"处理
            if not (age <= 6):
                continue
        # 已有我方 worker 停在树格上则天然安全
        if tidx in my_cells_now:
            continue
        threatened = any(_torus_dist(ctx, tidx, pidx) <= P["guard_threat_d"]
                         for _, pidx in w.en_pl)
        if not threatened:
            continue
        best = None
        for wid, idx, car in w.my_coll:
            if wid in assigned or car > 4.0:
                continue
            d = _torus_dist(ctx, tidx, idx)
            if best is None or d < best[0]:
                best = (d, wid)
        if best is not None:
            roles[best[1]] = ("guard", tidx)
            assigned.add(best[1])
            guarded_trees += 1

    # 2) 猎杀敌种树员: 撞死对方 planter(空载 collector 撞 planter 必胜),
    #    断其树引擎; 同时也保护了我方新树不被立刻偷走
    for _, pidx_enemy in w.en_pl:
        best = None
        for wid, idx, car in w.my_coll:
            if wid in assigned or car > 2.0:
                continue
            d = _torus_dist(ctx, idx, pidx_enemy)
            if d <= 5 and (best is None or d < best[0]):
                best = (d, wid)
        if best is not None:
            roles[best[1]] = ("huntpl", pidx_enemy)
            assigned.add(best[1])

    # 2.5) 猎杀: 拦截敌方满载捕碳员(空载最小载碳者胜并缴获)
    for _, eidx, ecar in sorted(w.en_coll, key=lambda x: -x[2]):
        if ecar < P["hunt_carry"]:
            continue
        ebase = min(w.en_base, key=lambda b: _torus_dist(ctx, eidx, b))
        path = _greedy_path(ctx, eidx, ebase)
        best = None
        for wid, idx, car in w.my_coll:
            if wid in assigned or car >= ecar - 5.0:
                continue
            for si, pc in enumerate(path):
                ed = si + 1
                hd = _torus_dist(ctx, pc, idx)
                if hd <= ed and hd <= P["hunt_reach"] and pc not in w.en_base:
                    key = (ed - hd, hd)
                    if best is None or key < best[0]:
                        best = (key, wid, pc)
        if best is not None:
            roles[best[1]] = ("hunt", best[2])
            assigned.add(best[1])

    # 3) 回存: 满载 / 终局 / 撤离威胁
    for wid, idx, car in w.my_coll:
        if wid in assigned:
            continue
        d_base = min(_torus_dist(ctx, idx, b) for b in w.my_base)
        thr = max(P["dep_min"], min(P["dep_max"], P["dep_base"] + P["dep_k"] * d_base))
        forced = car > 0 and remaining <= d_base + 2
        flee = car >= 30.0 and np.isfinite(en_reachc[idx]) and en_reachc[idx] < car
        if car >= thr or forced or flee:
            tgt = min(w.my_base, key=lambda b: _torus_dist(ctx, idx, b))
            roles[wid] = ("deposit", tgt)
            assigned.add(wid)

    # 4) 采碳: 原地过阈值停留, 否则 belief 全图择优
    claims = {}
    for wid, idx, car in sorted(w.my_coll, key=lambda x: -x[2]):
        if wid in assigned:
            continue
        ok_here = (idx not in tree_pos) and (idx not in base_all)
        curv = float(ctx.cur[idx])
        margin = cfg["collect"] * curv - cfg["abs_coll"] * en_tax[idx] * car
        if ok_here and curv >= P["mine_stay_min"] and margin >= P["mine_margin_min"]:
            roles[wid] = ("mine", idx)
            claims[idx] = wid
            continue
        df = _dist_field(ctx, idx)
        score = ctx.cur - P["lam_move"] * df
        banned = np.zeros(n, bool)
        for p in tree_pos:
            banned[p] = True
        for b in base_all:
            banned[b] = True
        for c in claims:
            score[c] -= P["claim_pen"]
        for cidx in my_cells_now:
            if cidx != idx:
                score[cidx] -= 15.0
        if car > 0:
            score[np.isfinite(en_reachc) & (en_reachc <= car)] -= P["mob_risk_loaded"] * 10.0
        else:
            score[en_reachc == 0.0] -= 1.5
        score -= 0.6 * en_tax
        score[banned] = -1e9
        t = int(np.argmax(score))
        roles[wid] = ("mine", t)
        claims[t] = wid

    # ================= 捕碳员移动(回存/守/猎优先占位) =================
    def _collect_cost(idx, car):
        cost = np.zeros(n)
        for b in w.en_base:
            cost[b] += 1e6
        if car > 0:
            cost += P["enemy_tree_taxk"] * car * en_tax
            cost[np.isfinite(en_reachc) & (en_reachc <= car)] += P["mob_risk_loaded"]
        for rc in reserved:
            cost[rc] += P["reserve_cost"]
        for c in my_cells_now:
            if c != idx:
                cost[c] += P["reserve_cost"]
        return cost

    order = sorted(
        w.my_coll,
        key=lambda x: 0 if roles.get(x[0], ("mine",))[0] in ("deposit", "guard", "hunt") else 1)
    for wid, idx, car in order:
        if wid not in roles:
            reserved.add(idx)
            continue
        kind, tgt = roles[wid]
        if idx == tgt:
            reserved.add(idx)
            continue  # None 动作 => 停留(采/守/等)
        if kind == "deposit" and (tgt in reserved or tgt in my_cells_now):
            # 基地落点已被占用(同伴/idle工人/刚出生的种树员) => 在旁边等一拍,
            # 避免同格碰撞损员(0载者胜, 满载回存员可能被撞死)
            alt = None
            for k in range(4):
                v = int(ctx.nb[k][tgt])
                if v not in reserved and v not in my_cells_now:
                    d = _torus_dist(ctx, v, idx)
                    if alt is None or d < alt[0]:
                        alt = (d, v)
            if alt is not None and alt[1] != idx and _torus_dist(ctx, idx, alt[1]) == 1:
                act = _dir_between(ctx, idx, alt[1])
                if act:
                    actions[wid] = act
                    reserved.add(alt[1])
                    continue
        act = _route(ctx, idx, tgt, _collect_cost(idx, car))
        if act is None:
            act = _fallback_step(ctx, idx, tgt, w, reserved, my_cells_now, planter=False)
        if act is None:
            reserved.add(idx)
        else:
            k = ctx.nb_act.index(act)
            reserved.add(int(ctx.nb[k][idx]))
            actions[wid] = act

    # ================= 种树员 =================
    price = cfg["plant_cost"] if len(w.my_trees) <= cfg["protective"] \
        else cfg["plant_cost"] + cfg["infl_ratio"] * (cfg["infl_base"] ** len(tree_pos))
    planter_roles = {}
    taken = set()
    if w.my_pl:
        # 抢树候选: 当前吸收*剩余寿命 - 距离代价 - 敌种树员护卫惩罚
        seize_cand = []
        for tid, (tidx, age) in w.en_trees.items():
            rem = max(0, cfg["tree_life"] - age)
            absr = w.tree_abs.get(tid, 0.0)
            s0 = absr * rem
            if any(_torus_dist(ctx, tidx, pidx) <= 1 for _, pidx in w.en_pl):
                s0 -= 40.0
            if s0 >= P["seize_min"] and w.cash >= cfg["seize_cost"]:
                seize_cand.append((tidx, s0))
        # 种树候选(向量化): 4 邻域碳和最大
        sum4 = ctx.cur[ctx.nb[0]] + ctx.cur[ctx.nb[1]] + \
            ctx.cur[ctx.nb[2]] + ctx.cur[ctx.nb[3]]
        block = np.zeros(n, bool)
        for p in tree_pos:
            block[p] = True
        for b in base_all:
            block[b] = True
        near_tree = np.zeros(n)
        for p in tree_pos:
            rr, cc = p // size, p % size
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near_tree[((rr + dr) % size) * size + (cc + dc) % size] += 1.0
        # 剩<=14步: 新树仍能吸收若干回合, 不必保守
        roi_pool_ok = bool(remaining >= 14 and len(w.my_trees) < P["own_tree_cap"])
        seize_pos = {t for t, _ in seize_cand}
        for pi, (wid, idx) in enumerate(sorted(w.my_pl, key=lambda x: x[0])):
            if idx in seize_pos and idx not in taken:
                planter_roles[wid] = ("seize", idx)   # 已站在可抢的敌树上
                taken.add(idx)
                continue
            best_seize = None
            for tidx, s0 in seize_cand:
                if tidx in taken:
                    continue
                s = s0 - P["lam_seize"] * _torus_dist(ctx, tidx, idx)
                if best_seize is None or s > best_seize[0]:
                    best_seize = (s, tidx)
            best_plant = None
            if roi_pool_ok and w.cash >= price + 5.0:
                df = _dist_field(ctx, idx)
                # 早期近距选址: 尽快把第一波树种下去抢开局窗口
                lam_pl = 9.0 if step < 40 else P["lam_plant"]
                s = sum4 - lam_pl * df - P["tree_near_pen"] * near_tree
                s[en_pl_cells > 0] -= P["en_pl_near_pen"]
                # 避开敌方 planter 活动圈(4步内): 这种地方种的树几帧内就被偷
                s = s - 30.0 * en_pl_far
                s[block] = -1e9
                for tc in taken:
                    s[tc] = -1e9
                for cidx in my_cells_now:
                    if cidx != idx:
                        s[cidx] -= 20.0
                # 任务黏性: 上回合的选址目标加分, 防止打分抖动导致来回跑
                stick = ctx.ptarget.get(wid)
                if stick is not None and 0 <= stick < n and not block[stick]:
                    s[stick] += P["plant_stick"]
                t = int(np.argmax(s))
                if _DEBUG and w.step % 10 == 0:
                    print("[plant] t=%d planter@%d best=%d sum4=%.0f price=%.1f cash=%.0f dist=%d" % (
                        w.step, idx, t, float(sum4[t]), price, w.cash, int(df[t])))
                if s[t] > -1e8 and P["plant_rate"] * float(sum4[t]) >= price + P["plant_margin"]:
                    best_plant = (float(s[t]), t)
                    ctx.ptarget[wid] = t
            if best_seize is not None and (best_plant is None or
                                           best_seize[0] >= best_plant[0] - 30.0):
                planter_roles[wid] = ("seize", best_seize[1])
                taken.add(best_seize[1])
            elif best_plant is not None:
                planter_roles[wid] = ("plant", best_plant[1])
                taken.add(best_plant[1])
            else:
                # 无任务: 停到自己树格/基格(这两类格子停留无副作用)
                if w.my_trees:
                    tgt = min((p for p, _ in w.my_trees.values()),
                              key=lambda p: _torus_dist(ctx, p, idx))
                else:
                    # 早期无树: 种树员分散停到不同基地附近, 避免互相/撞回存员
                    bi = pi % max(1, len(w.my_base))
                    tgt = w.my_base[bi]
                planter_roles[wid] = ("park", tgt)

    en_tree_pos = {p for p, _ in w.en_trees.values()}
    my_tree_pos = {p for p, _ in w.my_trees.values()}
    if _DEBUG and w.step % 10 == 0 and w.my_pl:
        kinds = {}
        for r in planter_roles.values():
            kinds[r[0]] = kinds.get(r[0], 0) + 1
        print("[planter] t=%d kinds=%s T=%d cash=%.0f" % (
            w.step, kinds, len(w.my_trees), w.cash))
    for wid, idx in list(w.my_pl):
        role = planter_roles.get(wid)
        if role is None:
            role = ("park", min(w.my_base, key=lambda b: _torus_dist(ctx, b, idx)))
        kind, tgt = role
        if idx == tgt:
            if kind == "plant" and w.cash < price:
                continue        # 钱不够: 先原地等(站在空格领不到钱也没副作用,
                                # 不会误种 - env 只在 cash>=价 才种)
            if kind == "park" and not w.my_trees:
                # 早期无树停在基格 => 立刻挪到安全邻格(震荡), 绝不占基地格:
                # 1) 防碍回存 collector 落点撞击  2) 防碍新一轮招募
                mm = _park_step_out(ctx, idx, w, reserved, my_cells_now)
                if mm is not None:
                    act, nxt = mm
                    actions[wid] = act
                    reserved.add(nxt)
                else:
                    reserved.add(idx)
                continue
            reserved.add(idx)
            continue
        cost = np.zeros(n)
        for b in w.en_base:
            cost[b] += 1e6
        cost[en_reachc < np.inf] += P["mob_risk_planter"]
        cost[en_pl_cells > 0] += P["mob_risk_pl_vs_pl"]
        for p in en_tree_pos:
            cost[p] += 1e5           # 误入敌树 = 花钱误抢
        for p in my_tree_pos:
            if p != tgt:
                cost[p] += 10.0
        if kind == "seize":
            cost[tgt] = 0.0          # 目标敌树格允许进入
        for rc in reserved:
            cost[rc] += P["reserve_cost"]
        for cidx in my_cells_now:
            if cidx != idx:
                cost[cidx] += P["reserve_cost"]
        act = _route(ctx, idx, tgt, cost)
        if act is None:
            act = _fallback_step(ctx, idx, tgt, w, reserved, my_cells_now, planter=True)
        if act is None:
            reserved.add(idx)
        else:
            k = ctx.nb_act.index(act)
            reserved.add(int(ctx.nb[k][idx]))
            actions[wid] = act

    # ================= 招募 =================
    _recruit(cfg, w, actions, reserved, my_cells_now)
    return actions


# ============ 入口(必须是文件中最后一个 callable) ============
def agent(observation, configuration):
    t0 = time.perf_counter()
    try:
        acts = _agent_impl(observation, configuration)
    except Exception:
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        acts = {}
    _STATS["max_turn_s"] = max(_STATS["max_turn_s"], time.perf_counter() - t0)
    return acts

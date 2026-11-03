# Carbon League 入门策略 v1：轮换采矿 + 回程交付 + 招募 + 种树 + 避碰
#
# 核心思路（对应原版规则，全部参数读 configuration，不硬编码）：
#   1. 起始资金有限，每回合最多招 1 名工人（同回合同中心招 2 个会出生在同格、
#      货量相同判和 → 双双死亡，见 helpers.next() 的 resolve_collision）。
#   2. 捕碳员：脚下有碳就停留采集（initialCollectRate/回合）；采到阈值以下
#      或货满就走向最近的己方转化中心；站上中心自动结算（cash += cargo - smeltCost）。
#   3. 格子再生是纯乘法（×regenRate），被采到 0 的格子永远枯死，
#      所以采到阈值就换格轮换，不做绝户采。
#   4. 碰撞规则：同格捕碳员中"携带量最少且唯一"者获胜并缴获全部货物，
#      其余全灭（含种树员）。因此：货多要躲；我货少而敌货多 → 上去击杀缴获；
#      踩敌种树员白赚击杀（场上唯一捕碳员必胜）。
#   5. 敌方树的 8 邻域（切比雪夫距离 1）每回合吸走 10% 货物，带货绕行。
#   6. 种树员：停在无树无中心的空格自动种树（扣当前树价），树的 4 邻域每回合
#      吸 3.75% 格子碳变成树主的现金；树价随全场活树数指数上涨，超过回本上限停种。
#      种树会清空所在格，选点避开任何树的 8 邻域。
#   7. 终局：剩余步数不够跑一趟回程就提前回家；临近结束停止招募/种树。
#
# 观测说明：carbon 只有捕碳员所在格、种树员及其 4 邻域为真实值，其余是 0
# （未知≠零）。工人/中心/树双方全可见，且每个格子至多 1 名工人（回合末碰撞
# 已结算）。用跨步内存记录见过的碳量。
# 索引约定：idx = 行*15 + 列；UP=-size、DOWN=+size、RIGHT=+1、LEFT=-1，环形绕回。

_MEM = {}
_UNK = 10.0          # 未知格先验估值（地图均值 ~11）
_DELIVER_CARGO = 25.0 # 载货达到该值回中心
_CELL_FLOOR = 8.0     # 格子采到该值以下换地方（留底按比例再生）
_HUNT_MARGIN = 5.0    # 敌货比我多出该值才追杀（留安全余量防同归）


def _reset_mem():
    _MEM.clear()
    _MEM["carbon"] = {}       # idx -> 上次看到的真实碳量
    _MEM["seen_step"] = {}    # idx -> 上次看到的回合
    _MEM["center_busy"] = {}  # 中心 idx -> 新兵占位解除的回合


def agent(observation, configuration):
    step = observation["step"]
    if step == 0:
        _reset_mem()

    size = configuration["size"]
    me = observation["player"]
    foe = 1 - me
    cash, centers, workers, my_trees = observation["players"][me]
    _fc, foe_centers, foe_workers, foe_trees = observation["players"][foe]
    carbon = observation["carbon"]
    steps_left = configuration["episodeSteps"] - step

    cfg = configuration
    rec_c_cost = cfg["recCollectorCost"]
    rec_p_cost = cfg["recPlanterCost"]
    plant_cost = cfg["plantCost"]
    plant_ratio = cfg["plantCostInflationRatio"]
    plant_base = cfg["plantCostInflationBase"]
    worker_limit = cfg["workerLimit"]
    tree_lifespan = cfg["treeLifespan"]
    max_cell = cfg["maxCellCarbon"]
    regen = cfg["regenRate"]

    # ---------- 几何工具（索引空间，环形） ----------
    DIRS = [("UP", 0, -1), ("RIGHT", 1, 0), ("DOWN", 0, 1), ("LEFT", -1, 0)]
    DMAP = {d[0]: (d[1], d[2]) for d in DIRS}

    def rc(i):
        return i % size, i // size

    def tdist(a, b):
        ax, ay = rc(a)
        bx, by = rc(b)
        dx = abs(ax - bx)
        dy = abs(ay - by)
        return min(dx, size - dx) + min(dy, size - dy)

    def cheb(a, b):
        ax, ay = rc(a)
        bx, by = rc(b)
        dx = abs(ax - bx)
        dy = abs(ay - by)
        return max(min(dx, size - dx), min(dy, size - dy))

    def tstep(i, d):
        dx, dy = d
        x, y = rc(i)
        return ((x + dx) % size) + ((y + dy) % size) * size

    # ---------- 可见性与内存 ----------
    my_center_set = set(centers.values())
    foe_center_set = set(foe_centers.values())
    visible = set()
    for _wid, (pos, _c, _t) in workers.items():
        visible.add(pos)
        if _t == "PLANTER":
            for _n, dx, dy in DIRS:
                visible.add(tstep(pos, (dx, dy)))
    for i in visible:
        _MEM["carbon"][i] = carbon[i]
        _MEM["seen_step"][i] = step

    def est(i):
        if i in _MEM["carbon"]:
            v = _MEM["carbon"][i]
            if v <= 0:
                return 0.0  # 乘法再生：见过的 0 永远是 0
            dt = step - _MEM["seen_step"].get(i, step)
            return min(v * (1.0 + regen) ** dt, max_cell)
        return _UNK

    foe_tree_pos = [info[0] for info in foe_trees.values()]
    my_tree_pos = [info[0] for info in my_trees.values()]

    def foe_tree_drain(i):
        # 敌树 8 邻域（切比雪夫 1）每回合吸 10% 货
        return any(cheb(i, t) <= 1 for t in foe_tree_pos)

    # 全场活树数与当前种树价（引擎同款公式）
    alive_trees = sum(1 for _p in (my_trees, foe_trees)
                      for info in _p.values() if info[1] < tree_lifespan)
    plant_price = plant_cost + plant_ratio * (plant_base ** alive_trees)

    # ---------- 占用与预约（防自撞 / 防扎堆） ----------
    all_workers = dict(workers)
    all_workers.update(foe_workers)
    occupied = {pos for pos, _c, _t in all_workers.values()}
    claimed = set()
    actions = {}

    # ---------- 招募（每回合最多 1 个，防同格双生互撞） ----------
    n_workers = len(workers)
    n_planters = sum(1 for _p, _c, t in workers.values() if t == "PLANTER")
    n_collectors = n_workers - n_planters
    target_collectors = 5 if step < 120 else 6
    # 不在新兵占位期、也没有己方工人正站在上面的中心才招（防出生互撞）
    free_centers = [c for c in centers.values()
                    if step >= _MEM["center_busy"].get(c, -1)
                    and c not in occupied]

    if steps_left > 30 and free_centers and n_workers < worker_limit:
        # 动作字典以中心 ID 为键；_MEM 以中心格子位置记账
        pos2id = {v: k for k, v in centers.items()}
        if n_collectors < target_collectors and cash >= rec_c_cost:
            cid = min(free_centers)
            actions[pos2id[cid]] = "RECCOLLECTOR"
            _MEM["center_busy"][cid] = step + 2  # 新兵占位 2 回合
        elif n_planters < 1 and cash >= rec_c_cost + rec_p_cost:
            cid = min(free_centers)
            actions[pos2id[cid]] = "RECPLANTER"
            _MEM["center_busy"][cid] = step + 2

    # ---------- 捕碳员 ----------
    for wid in sorted(workers):
        pos, cargo, wtype = workers[wid]
        if wtype != "COLLECTOR":
            continue
        others = occupied - {pos}
        at_center = pos in my_center_set

        home_d = min((tdist(pos, c) for c in my_center_set), default=999)
        go_home = bool(my_center_set) and (
            cargo >= _DELIVER_CARGO or (cargo > 0 and steps_left <= home_d + 2))

        if go_home:
            if at_center:
                if pos not in claimed:
                    claimed.add(pos)  # 站中心自动结算，无需动作
                continue
            tgt = min(my_center_set,
                      key=lambda c: (step < _MEM["center_busy"].get(c, -1),
                                     tdist(pos, c), c))
            best_name, best_d = None, tdist(pos, tgt)
            for name, dx, dy in DIRS:
                nb = tstep(pos, (dx, dy))
                if nb in others or nb in claimed or nb in foe_center_set:
                    continue
                d = tdist(nb, tgt)
                if d < best_d:
                    best_name, best_d = name, d
            if best_name:
                actions[wid] = best_name
                claimed.add(tstep(pos, DMAP[best_name]))
            continue

        # 采矿：脚下有碳（且不在敌树吸货区）就停留采集
        here = est(pos)  # 捕碳员所在格必然可见，est 即真实值
        if (here >= _CELL_FLOOR and not foe_tree_drain(pos)
                and not at_center and pos not in claimed):
            claimed.add(pos)
            continue

        # 换格：四邻打分（真实可见值优先，含追杀判定）
        best_name, best_score = None, None
        for name, dx, dy in DIRS:
            nb = tstep(pos, (dx, dy))
            if nb in claimed or nb in foe_center_set:
                continue
            sc = carbon[nb] if nb in visible else est(nb)
            # 敌人占据该格：只有"击杀收益"情形才允许进入
            foe_here = next((v for v in foe_workers.values() if v[0] == nb), None)
            if foe_here is not None:
                fcargo, ftype = foe_here[1], foe_here[2]
                if ftype == "COLLECTOR" and fcargo > cargo + _HUNT_MARGIN:
                    sc += 30.0 + 0.5 * fcargo  # 击杀并缴获
                elif ftype == "PLANTER":
                    sc += 15.0  # 场上唯一捕碳员必胜，白赚击杀
                else:
                    continue  # 打不过，绕开
            elif nb in others:
                continue  # 己方工人占的格不进（防自撞）
            if foe_tree_drain(nb):
                sc -= 50.0 if cargo > 0 else 8.0
            if best_score is None or sc > best_score + 1e-9:
                best_name, best_score = name, sc
        if best_name is not None and best_score > max(here, 0.0) - 2.0:
            actions[wid] = best_name
            claimed.add(tstep(pos, DMAP[best_name]))
        elif pos not in claimed:
            claimed.add(pos)  # 原地待着（有碳就采，没碳休整）

    # ---------- 种树员 ----------
    for wid in sorted(workers):
        pos, cargo, wtype = workers[wid]
        if wtype != "PLANTER":
            continue
        others = occupied - {pos}

        def plantable(i):
            if i in others or i in claimed:
                return False
            if i in my_center_set or i in foe_center_set:
                return False
            for t in my_tree_pos + foe_tree_pos:
                if cheb(i, t) <= 1:
                    return False  # 避开任何树的 8 邻域，防止互抢碳
            return True

        def spot_score(i):
            if not plantable(i):
                return None
            # 树吸 4 邻域碳；种下时所在格被清 0，故只计邻域
            return sum(est(tstep(i, (dx, dy))) for _n, dx, dy in DIRS) \
                - 1.5 * tdist(pos, i)

        can_plant = (steps_left >= 35 and plant_price <= 90.0
                     and cash >= plant_price + rec_c_cost)
        here_ok = plantable(pos)

        if steps_left < 25:
            # 终局停种：停到自家树的格子上（树下无采集/种树判定，绝对安全）
            if my_tree_pos:
                tgt = min(my_tree_pos, key=lambda t: (tdist(pos, t), t))
                if pos == tgt or pos in claimed:
                    continue
                for name, dx, dy in DIRS:
                    nb = tstep(pos, (dx, dy))
                    if nb not in others and nb not in claimed \
                            and nb not in foe_center_set \
                            and tdist(nb, tgt) < tdist(pos, tgt):
                        actions[wid] = name
                        claimed.add(nb)
                        break
            continue

        if can_plant and here_ok:
            claimed.add(pos)  # 停下 → 引擎自动种树（钱够才生效）
            continue

        # 找最佳种树点走过去
        best_i, best_s = None, None
        x, y = rc(pos)
        for yy in range(y - 5, y + 6):
            for xx in range(x - 5, x + 6):
                i = (xx % size) + (yy % size) * size
                s = spot_score(i)
                if s is not None and (best_s is None or s > best_s + 1e-9):
                    best_i, best_s = i, s
        if best_i is None:
            if pos not in claimed:
                claimed.add(pos)
            continue
        best_name, best_d = None, tdist(pos, best_i)
        for name, dx, dy in DIRS:
            nb = tstep(pos, (dx, dy))
            if nb in others or nb in claimed or nb in foe_center_set:
                continue
            d = tdist(nb, best_i)
            if d < best_d:
                best_name, best_d = name, d
        if best_name:
            actions[wid] = best_name
            claimed.add(tstep(pos, DMAP[best_name]))
        elif pos not in claimed:
            claimed.add(pos)

    # ---------- 合法性兜底 ----------
    safe = {}
    for wid, act in actions.items():
        if wid in centers and act in ("RECCOLLECTOR", "RECPLANTER"):
            safe[wid] = act
        elif wid in workers and act in ("UP", "RIGHT", "DOWN", "LEFT"):
            safe[wid] = act
    return safe

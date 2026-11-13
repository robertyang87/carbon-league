# submission.py
# Carbon League 入门策略：单文件、标准库、快速决策。
# 思路：招募采集员；有负重回最近基地；无负重去最近可见碳；简单避让己方单位。

from zerosum_env.envs.carbon.helpers import Board, WorkerAction, RecrtCenterAction


def _cfg(configuration, names, default):
    """从 configuration 中稳健读取配置。"""
    if not isinstance(configuration, dict):
        return default
    for n in names:
        if n in configuration:
            return configuration[n]
    lower = {str(k).lower(): v for k, v in configuration.items()}
    for n in names:
        ln = n.lower()
        if ln in lower:
            return lower[ln]
    for k, v in lower.items():
        for n in names:
            if n.lower() in k:
                return v
    return default


def _attr(obj, names, default=None):
    """兼容 dict / 对象属性的读取。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return default


def _pos(obj):
    """读取单位坐标，返回 (x, y) 或 None。"""
    p = _attr(obj, ["position", "pos", "location"])
    if p is not None:
        if isinstance(p, dict):
            x = p.get("x", p.get(0, p.get("0")))
            y = p.get("y", p.get(1, p.get("1")))
            if x is not None and y is not None:
                return (int(x), int(y))
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return (int(p[0]), int(p[1]))
        x = _attr(p, ["x", "0"], None)
        y = _attr(p, ["y", "1"], None)
        if x is not None and y is not None:
            return (int(x), int(y))
    x = _attr(obj, ["x"], None)
    y = _attr(obj, ["y"], None)
    if x is not None and y is not None:
        return (int(x), int(y))
    return None


def _amount(obj):
    return int(_attr(obj, ["carbon", "amount", "value", "count"], 0) or 0)


def _carried(worker):
    return int(_attr(worker, ["carbon", "carried", "load"], 0) or 0)


def _money(board):
    p = _attr(board, "current_player", None)
    return int(_attr(p, ["carbon", "money", "funds", "balance"], 0) or 0)


def _units(board, names):
    p = _attr(board, "current_player", None)
    for name in names:
        arr = _attr(p, [name], None)
        if arr is not None:
            try:
                return list(arr)
            except Exception:
                return []
    return []


def _size(configuration):
    return int(_cfg(configuration, ["boardSize", "board_size", "size"], 15) or 15)


def _dist(a, b, size):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, size - dx)
    dy = min(dy, size - dy)
    return dx + dy


def _delta(a, b, size):
    dx = (b[0] - a[0]) % size
    if dx > size // 2:
        dx -= size
    dy = (b[1] - a[1]) % size
    if dy > size // 2:
        dy -= size
    return dx, dy


def _next_pos(pos, action, size):
    x, y = pos
    if action == WorkerAction.UP:
        return (x % size, (y + 1) % size)
    if action == WorkerAction.RIGHT:
        return ((x + 1) % size, y % size)
    if action == WorkerAction.DOWN:
        return (x % size, (y - 1) % size)
    if action == WorkerAction.LEFT:
        return ((x - 1) % size, y % size)
    return pos


def _dir_candidates(src, dst, size):
    dx, dy = _delta(src, dst, size)
    cand = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            cand.append(WorkerAction.RIGHT)
        elif dx < 0:
            cand.append(WorkerAction.LEFT)
        if dy > 0:
            cand.append(WorkerAction.UP)
        elif dy < 0:
            cand.append(WorkerAction.DOWN)
    else:
        if dy > 0:
            cand.append(WorkerAction.UP)
        elif dy < 0:
            cand.append(WorkerAction.DOWN)
        if dx > 0:
            cand.append(WorkerAction.RIGHT)
        elif dx < 0:
            cand.append(WorkerAction.LEFT)
    return cand


def _choose_move(pos, dst, size, occupied):
    for a in _dir_candidates(pos, dst, size):
        np = _next_pos(pos, a, size)
        if np not in occupied:
            return a, np
    return None, pos


def _action_str(a):
    if a is None:
        return None
    if isinstance(a, str):
        return a
    v = getattr(a, "value", None)
    if isinstance(v, str):
        return v
    n = getattr(a, "name", None)
    if isinstance(n, str):
        return n
    return str(a)


def _visible_carbon(observation, board):
    """返回可见碳列表 [(x, y, amount), ...]。"""
    c = _attr(board, ["carbon"], None)
    if c is None and isinstance(observation, dict):
        c = observation.get("carbon", None)
    if c is None:
        return []

    out = []

    # numpy 数组兼容
    try:
        import numpy as np
        if isinstance(c, np.ndarray):
            if c.ndim == 2:
                for x in range(c.shape[0]):
                    for y in range(c.shape[1]):
                        v = int(c[x, y])
                        if v > 0:
                            out.append((x, y, v))
                return out
    except Exception:
        pass

    if isinstance(c, dict):
        for k, v in c.items():
            if isinstance(k, (list, tuple)) and len(k) >= 2:
                try:
                    if float(v) > 0:
                        out.append((int(k[0]), int(k[1]), int(v)))
                except Exception:
                    pass
        return out

    if isinstance(c, (list, tuple)):
        # 二维地图
        if len(c) > 0 and all(isinstance(row, (list, tuple)) for row in c):
            for x, row in enumerate(c):
                for y, v in enumerate(row):
                    try:
                        if float(v) > 0:
                            out.append((x, y, int(v)))
                    except Exception:
                        pass
            return out

        # 对象列表或三元组列表
        for item in c:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    if float(item[2]) > 0:
                        out.append((int(item[0]), int(item[1]), int(item[2])))
                except Exception:
                    pass
            else:
                p = _pos(item)
                amt = _amount(item)
                if p is not None and amt > 0:
                    out.append((p[0], p[1], amt))
    return out


def agent(observation, configuration):
    try:
        board = Board(observation, configuration)
    except Exception:
        # 解析失败时返回空动作，保证不抛异常。
        return {}

    size = _size(configuration)
    step = int(_attr(board, ["step"], observation.get("step", 0) if isinstance(observation, dict) else 0) or 0)
    episode = int(_cfg(configuration, ["episodeSteps", "episode_steps", "steps"], 300) or 300)

    workers = _units(board, ["workers"])
    centers = _units(board, ["recrtCenters", "recrtcenters", "recrt_centers"])
    center_positions = [p for p in (_pos(c) for c in centers) if p is not None]

    visible = _visible_carbon(observation, board)

    occupied = set()
    for w in workers:
        p = _pos(w)
        if p is not None:
            occupied.add(p)

    actions = {}

    # 招募：留出终局时间，控制单位数量。
    money = _money(board)
    cost = int(_cfg(
        configuration,
        ["recollectorCost", "recCollectorCost", "collectorCost",
         "recollector_cost", "recCollector_cost", "RECCOLLECTOR_COST"],
        50
    ) or 50)
    max_workers = int(_cfg(configuration, ["maxWorkers", "max_workers", "workerLimit"], 10) or 10)

    if step < max(0, episode - 40) and len(workers) < max_workers and money >= cost:
        for c in centers:
            cid = _attr(c, ["id", "unit_id", "index"], None)
            if cid is not None:
                actions[cid] = _action_str(RecrtCenterAction.RECCOLLECTOR)
                break

    # 工人动作
    for w in workers:
        wid = _attr(w, ["id", "unit_id", "index"], None)
        pos = _pos(w)
        if wid is None or pos is None:
            continue

        carried = _carried(w)
        target = None

        if carried > 0 and center_positions:
            # 负重：回最近基地交付。
            target = min(center_positions, key=lambda p: _dist(pos, p, size))
        elif visible:
            # 空载：去最近可见碳。
            target = min(
                ((x, y) for x, y, v in visible),
                key=lambda p: _dist(pos, p, size)
            )
        else:
            # 没有可见碳时，向地图中心附近做低频探索，避免原地不动。
            cx, cy = size // 2, size // 2
            if _dist(pos, (cx, cy), size) > 2:
                target = (cx, cy)

        if target is not None and target != pos:
            a, np = _choose_move(pos, target, size, occupied)
            if a is not None:
                actions[wid] = _action_str(a)
                occupied.discard(pos)
                occupied.add(np)

    return actions
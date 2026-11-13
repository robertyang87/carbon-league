"""Parity harness: FastCarbon vs the original zerosum_env carbon Board.

Compares, step by step:
  * map generation (populate_board replica vs a real env reset),
  * full 300-step games driven by an identical deterministic "chaos" policy
    (recruiting, collecting, planting, seizing, hunting collisions, deposits),
  * crafted micro-scenarios for collisions / deposits / planting costs /
    seizing / withering / tree absorption,
  * per-player observation visibility (obs()) against the interpreter's
    filtering rule,
  * drop-in compatibility: example.py and reference_juzi.py run unmodified
    on FastCarbon observations,
  * speed: original env.run vs FastCarbon self-play.

Writes fastsim_report.md next to this file.  Exit code 0 iff all parity
checks pass.
"""

import copy
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from zerosum_env import make  # noqa: E402
from zerosum_env.envs.carbon.helpers import Board  # noqa: E402
from zerosum_env.envs.carbon import idgen  # noqa: E402

from fastsim import FastCarbon  # noqa: E402

RESULTS = []          # (section, name, ok, detail)
MISMATCHES = []       # human-readable first-mismatch descriptions


def record(section, name, ok, detail=""):
    RESULTS.append((section, name, ok, detail))
    if not ok:
        MISMATCHES.append(f"[{section}] {name}: {detail}")


def load_config():
    spec = json.load(open(os.path.join(
        HERE, "env_src", "carbon_challenge-master",
        "zerosum_env", "envs", "carbon", "carbon.json")))
    cfg = {}
    for k, v in spec["configuration"].items():
        cfg[k] = v.get("default") if isinstance(v, dict) else v
    return cfg


CFG = load_config()
SIZE = 15


def idx(x, y):
    return (SIZE - 1 - y) * SIZE + x


def xy(pos):
    return pos % SIZE, SIZE - 1 - pos // SIZE


# --------------------------------------------------------------------------
# state extraction / comparison
# --------------------------------------------------------------------------
def state_from_obs(o):
    workers, trees = {}, {}
    for p in range(2):
        for wid, (pos, cargo, occ) in o["players"][p][2].items():
            workers[wid] = (p, pos, cargo, occ)
        for tid, (pos, age) in o["players"][p][3].items():
            trees[tid] = (p, pos, age)
    return {
        "cash": [o["players"][0][0], o["players"][1][0]],
        "carbon": list(o["carbon"]),
        "workers": workers,
        "trees": trees,
        "step": o["step"],
        "trees_dict": {k: list(v) for k, v in o["trees"].items()},
    }


def sim_state(sim):
    s = sim.get_state()
    return {
        "cash": s["cash"],
        "carbon": s["carbon"],
        "workers": {k: (v["player"], v["pos"], v["carbon"], v["type"])
                    for k, v in s["workers"].items()},
        "trees": {k: (v["player"], v["pos"], v["age"])
                  for k, v in s["trees"].items()},
        "step": s["step"],
        "trees_dict": s["trees_dict"],
    }


def diff_states(ref, sim):
    """Return list of mismatch strings (empty == identical)."""
    out = []
    if ref["step"] != sim["step"]:
        out.append(f"step ref={ref['step']} sim={sim['step']}")
    if ref["cash"] != sim["cash"]:
        out.append(f"cash ref={ref['cash']} sim={sim['cash']}")
    if ref["carbon"] != sim["carbon"]:
        d = [(i, ref["carbon"][i], sim["carbon"][i]) for i in range(SIZE * SIZE)
             if ref["carbon"][i] != sim["carbon"][i]]
        out.append(f"carbon {len(d)} cells differ, first: {d[:5]}")
    for name in ("workers", "trees"):
        if ref[name] != sim[name]:
            for k in sorted(set(ref[name]) | set(sim[name])):
                if ref[name].get(k) != sim[name].get(k):
                    out.append(f"{name}[{k}] ref={ref[name].get(k)} sim={sim[name].get(k)}")
                    if len(out) > 12:
                        break
    if ref["trees_dict"] != sim["trees_dict"]:
        for k in sorted(set(ref["trees_dict"]) | set(sim["trees_dict"])):
            if ref["trees_dict"].get(k) != sim["trees_dict"].get(k):
                out.append(f"trees_dict[{k}] ref={ref['trees_dict'].get(k)} "
                           f"sim={sim['trees_dict'].get(k)}")
                if len(out) > 12:
                    break
    return out


class RefChain:
    """Drives the original Board directly from a full observation."""

    def __init__(self, full_obs, sim_counters=None):
        self.obs = copy.deepcopy(full_obs)
        idgen.reset()
        if sim_counters is not None:
            wc, tc = sim_counters
            for p in range(2):
                for _ in range(wc[p]):
                    idgen.new_worker_id(p)
                for _ in range(tc[p]):
                    idgen.new_tree_id(p)

    def step(self, a, b):
        board = Board(copy.deepcopy(self.obs), CFG, [a, b])
        self.obs = board.next().observation
        return self.obs


# --------------------------------------------------------------------------
# 1. map generation parity
# --------------------------------------------------------------------------
def check_map_generation():
    seed = 42
    np.random.seed(seed)
    env = make("carbon", configuration={"episodeSteps": 300, "randomSeed": 999})
    states = env.reset(2)
    env_map = list(states[0].observation.full_carbon)
    # make()+reset() populates twice; FastCarbon.reset(seed) is generation #1.
    # Burn one generation, then the second must equal the env map.
    sim = FastCarbon(CFG)
    sim.reset(seed=seed)                       # generation #1 from seed
    gen1 = list(sim.carbon)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        sim2 = FastCarbon(CFG)
        sim2.reset()                           # generation #1 (ambient stream)
        sim2.reset()                           # generation #2 == env map
    finally:
        np.random.set_state(state)
    record("map", "generation#2 == env.reset() full_carbon",
           list(sim2.carbon) == env_map,
           f"equal={list(sim2.carbon) == env_map}, sums {sum(sim2.carbon)} vs {sum(env_map)}")
    record("map", "reset(seed) deterministic & 4-fold symmetric",
           gen1 == list(FastCarbon(CFG).reset(seed=seed).carbon)
           and all(gen1[idx(x, y)] == gen1[idx(SIZE - 1 - x, y)] for x in range(8) for y in range(SIZE)),
           f"gen1 sum={sum(gen1)} nonzero={sum(1 for v in gen1 if v)}")


# --------------------------------------------------------------------------
# 2. chaos policy for full games
# --------------------------------------------------------------------------
def _torus_dist(a, b):
    ax, ay = xy(a)
    bx, by = xy(b)
    return min(abs(ax - bx), SIZE - abs(ax - bx)) + min(abs(ay - by), SIZE - abs(ay - by))


def _step_toward(pos, target):
    if pos == target:
        return None
    best, best_d = None, None
    px, py = xy(pos)
    for name, (dx, dy) in (("UP", (0, 1)), ("RIGHT", (1, 0)),
                           ("DOWN", (0, -1)), ("LEFT", (-1, 0))):
        npos = idx((px + dx) % SIZE, (py + dy) % SIZE)
        d = _torus_dist(npos, target)
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


ANCHORS = [(6, 10), (4, 7), (7, 4), (10, 7), (12, 10), (10, 12), (7, 12), (12, 7)]


def chaos_actions(state, full_obs, step):
    """Deterministic policy of the (identical) state: exercises recruiting,
    collecting+deposit, planting, seizing, collision hunting."""
    rng = random.Random(987654 + step)
    acts = [{}, {}]
    for p in range(2):
        a = acts[p]
        cash = state["cash"][p]
        workers = {k: v for k, v in state["workers"].items() if v[0] == p}
        collectors = [k for k, v in workers.items() if v[3] == "COLLECTOR"]
        planters = [k for k, v in workers.items() if v[3] == "PLANTER"]
        bases = list(full_obs["players"][p][1].items())  # [(bid, pos)]
        # recruiting on even steps, composition-balancing, budget-tracked
        if step % 2 == 0 and cash >= 30:
            budget = cash
            n = len(workers)
            n_coll, n_plan = len(collectors), len(planters)
            for bid, _ in bases:
                if n >= 10 or budget < 30:
                    break
                a[bid] = "RECCOLLECTOR" if n_coll <= n_plan else "RECPLANTER"
                if a[bid] == "RECCOLLECTOR":
                    n_coll += 1
                else:
                    n_plan += 1
                budget -= 30
                n += 1
        enemy_workers = [v for k, v in state["workers"].items() if v[0] != p]
        enemy_trees = [v for k, v in state["trees"].items() if v[0] != p]
        base_positions = [pos for _, pos in bases]
        for i, wid in enumerate(collectors):
            _, pos, cargo, _ = workers[wid]
            act = None
            if cargo >= 50 or step >= 288:
                tgt = min(base_positions, key=lambda b: _torus_dist(pos, b))
                act = _step_toward(pos, tgt)
            elif i == 0 and enemy_workers:
                tgt = min((v[1] for v in enemy_workers),
                          key=lambda e: _torus_dist(pos, e))
                act = _step_toward(pos, tgt)
            elif state["carbon"][pos] >= 20:
                act = None  # stay and collect
            else:
                best = max(range(SIZE * SIZE),
                           key=lambda c: state["carbon"][c] / (1 + _torus_dist(pos, c)))
                act = _step_toward(pos, best) if state["carbon"][best] > 0 else \
                    rng.choice(["UP", "RIGHT", "DOWN", "LEFT"])
            if rng.random() < 0.08:
                act = rng.choice(["UP", "RIGHT", "DOWN", "LEFT", None])
            if act:
                a[wid] = act
        for i, wid in enumerate(planters):
            _, pos, _, _ = workers[wid]
            act = None
            if i == 0 and enemy_trees:
                tgt = min((v[1] for v in enemy_trees),
                          key=lambda e: _torus_dist(pos, e))
                act = None if pos == tgt else _step_toward(pos, tgt)
            else:
                anchor = idx(*ANCHORS[(i + 3 * p) % len(ANCHORS)])
                act = None if pos == anchor else _step_toward(pos, anchor)
            if rng.random() < 0.08:
                act = rng.choice(["UP", "RIGHT", "DOWN", "LEFT", None])
            if act:
                a[wid] = act
    return acts


def check_full_game(seed, steps=300):
    sim = FastCarbon(CFG).reset(seed=seed)
    ref = RefChain(sim.full_observation())
    stats = {"plants": 0, "max_workers": 0, "max_trees": 0}
    prev_trees = 0
    ok = True
    for step in range(steps):
        rs = state_from_obs(ref.obs)
        ss = sim_state(sim)
        d = diff_states(rs, ss)
        if d:
            record("game", f"seed={seed} state@step{step}", False, "; ".join(d[:6]))
            ok = False
            break
        stats["max_workers"] = max(stats["max_workers"], len(rs["workers"]))
        stats["max_trees"] = max(stats["max_trees"], len(rs["trees"]))
        if len(rs["trees"]) > prev_trees:
            stats["plants"] += len(rs["trees"]) - prev_trees
        prev_trees = len(rs["trees"])
        a = chaos_actions(rs, ref.obs, step)
        ref.step(a[0], a[1])
        sim.step(a[0], a[1])
        if step % 25 == 0:
            # observation visibility parity for both players (post-step state)
            rs = state_from_obs(ref.obs)
            for p in range(2):
                o = sim.obs(p)
                exp = [0] * (SIZE * SIZE)
                for wid, (wp, pos, cargo, occ) in rs["workers"].items():
                    if wp != p:
                        continue
                    if occ == "COLLECTOR":
                        exp[pos] = rs["carbon"][pos]
                    else:
                        x, y = xy(pos)
                        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                            exp[idx((x + dx) % SIZE, (y + dy) % SIZE)] = \
                                rs["carbon"][idx((x + dx) % SIZE, (y + dy) % SIZE)]
                if o["carbon"] != exp or o["players"] != ref.obs["players"] \
                        or o["step"] != ref.obs["step"] or o["trees"] != ref.obs["trees"]:
                    record("game", f"seed={seed} obs({p})@step{step}", False,
                           "obs mismatch vs interpreter visibility rule")
                    ok = False
    if ok:
        record("game", f"seed={seed} obs() visibility parity (every 25 steps)", True,
               "carbon visibility, players, trees, step all match interpreter rule")
        rs, ss = state_from_obs(ref.obs), sim_state(sim)
        d = diff_states(rs, ss)
        if d:
            record("game", f"seed={seed} final state", False, "; ".join(d[:6]))
        else:
            record("game", f"seed={seed} 300 steps full parity", True,
                   f"final cash={rs['cash']}, plants>={stats['plants']}, "
                   f"max_workers={stats['max_workers']}, max_trees={stats['max_trees']}")
    return ok


# --------------------------------------------------------------------------
# 3. crafted scenarios
# --------------------------------------------------------------------------
BASES = {0: {"player-0-recrtCenter-0": idx(4, 10), "player-0-recrtCenter-1": idx(10, 4)},
         1: {"player-1-recrtCenter-0": idx(10, 10), "player-1-recrtCenter-1": idx(4, 4)}}


def crafted_obs(players, carbon_cells=None):
    """players: list of (cash, {wid:(pos,cargo,occ)}, {tid:(pos,age)})"""
    carbon = [0] * (SIZE * SIZE)
    for pos, v in (carbon_cells or {}).items():
        carbon[pos] = v
    obs = {"carbon": carbon, "players": [], "player": 0, "step": 0,
           "trees": {}, "remainingOverageTime": 60}
    for p in range(2):
        cash, workers, trees = players[p]
        obs["players"].append([
            cash, dict(BASES[p]),
            {wid: [pos, cargo, occ] for wid, (pos, cargo, occ) in workers.items()},
            {tid: [pos, age] for tid, (pos, age) in trees.items()},
        ])
        for tid in trees:
            obs["trees"][tid] = [None, 0]
    return obs


def run_scenario(name, obs, action_steps, asserts):
    sim = FastCarbon(CFG).load_observation(copy.deepcopy(obs))
    ref = RefChain(obs, sim_counters=(sim.worker_counters, sim.tree_counters))
    ok = True
    for i, (a, b) in enumerate(action_steps):
        ref.step(a, b)
        sim.step(a, b)
        d = diff_states(state_from_obs(ref.obs), sim_state(sim))
        if d:
            record("scenario", f"{name} parity@step{i}", False, "; ".join(d[:6]))
            ok = False
            break
    if ok:
        record("scenario", f"{name} parity", True, f"{len(action_steps)} step(s)")
        st = sim_state(sim)
        for label, fn in asserts:
            try:
                good, detail = fn(st)
            except Exception as e:  # noqa: BLE001
                good, detail = False, f"assert error: {e}"
            record("scenario", f"{name}: {label}", bool(good), str(detail))
    return ok


def W(p, n):
    return f"player-{p}-worker-{n}"


def T(p, n):
    return f"player-{p}-tree-{n}"


def check_scenarios():
    C, P = "COLLECTOR", "PLANTER"
    # S1a collision: least-cargo collector wins and captures
    run_scenario("S1a collision capture",
                 crafted_obs([(100, {W(0, 0): (idx(7, 7), 50, C)}, {}),
                              (100, {W(1, 0): (idx(7, 6), 10, C)}, {})]),
                 [({}, {W(1, 0): "UP"})],
                 [("winner is P1 collector with 60 cargo",
                   lambda s: (list(s["workers"].values()) == [(1, idx(7, 7), 60, C)],
                              s["workers"]))])
    # S1b collision tie: both die, cargo drops to cell
    run_scenario("S1b collision tie",
                 crafted_obs([(100, {W(0, 0): (idx(7, 7), 20, C)}, {}),
                              (100, {W(1, 0): (idx(7, 6), 20, C)}, {})]),
                 [({}, {W(1, 0): "UP"})],
                 [("both dead, cell carbon 40",
                   lambda s: (not s["workers"] and s["carbon"][idx(7, 7)] == 40,
                              (s["workers"], s["carbon"][idx(7, 7)])))])
    # S1c collector kills planter and captures its (crafted) cargo
    run_scenario("S1c collision vs planter",
                 crafted_obs([(100, {W(0, 0): (idx(7, 7), 5, P)}, {}),
                              (100, {W(1, 0): (idx(7, 6), 30, C)}, {})]),
                 [({}, {W(1, 0): "UP"})],
                 [("planter dead, collector cargo 35",
                   lambda s: (list(s["workers"].values()) == [(1, idx(7, 7), 35, C)],
                              s["workers"]))])
    # S1d two planters collide on an enemy tree cell: tree owner gets cargo
    run_scenario("S1d planter pileup on tree",
                 crafted_obs([(100, {}, {T(0, 0): (idx(7, 7), 10)}),
                              (100, {W(1, 0): (idx(7, 6), 5, P), W(1, 1): (idx(6, 7), 5, P)}, {})]),
                 [({}, {W(1, 0): "UP", W(1, 1): "RIGHT"})],
                 [("both planters dead, P0 cash 110",
                   lambda s: (not s["workers"] and s["cash"][0] == 110, (s["workers"], s["cash"])))])
    # S2a own-base deposit
    run_scenario("S2a own deposit",
                 crafted_obs([(100, {W(0, 0): (idx(4, 9), 80, C)}, {}), (100, {}, {})]),
                 [({W(0, 0): "UP"}, {})],
                 [("cash 180, cargo 0, worker alive on base",
                   lambda s: (s["cash"][0] == 180 and
                              list(s["workers"].values()) == [(0, idx(4, 10), 0, C)],
                              (s["cash"], s["workers"])))])
    # S2b enemy-base deposit hands cash to base owner
    run_scenario("S2b enemy deposit",
                 crafted_obs([(100, {}, {}), (100, {W(1, 0): (idx(5, 10), 50, C)}, {})]),
                 [({}, {W(1, 0): "LEFT"})],
                 [("P0 cash 150, P1 collector alive cargo 0",
                   lambda s: (s["cash"][0] == 150 and
                              list(s["workers"].values()) == [(1, idx(4, 10), 0, C)],
                              (s["cash"], s["workers"])))])
    # S2c empty enemy collector stepping on a base is deleted
    run_scenario("S2c enemy empty on base dies",
                 crafted_obs([(100, {}, {}), (100, {W(1, 0): (idx(5, 10), 0, C)}, {})]),
                 [({}, {W(1, 0): "LEFT"})],
                 [("worker deleted, cash unchanged",
                   lambda s: (not s["workers"] and s["cash"] == [100, 100],
                              (s["workers"], s["cash"])))])
    # S3 planting: flat cost for first tree, newborn does not absorb,
    # absorption starts next step
    nb = {idx(7, 8): 100, idx(8, 7): 100, idx(7, 6): 100, idx(6, 7): 100, idx(7, 7): 55}
    run_scenario("S3 plant + absorb timing",
                 crafted_obs([(1000, {W(0, 0): (idx(7, 7), 0, P)}, {}), (100, {}, {})],
                             carbon_cells=nb),
                 [({}, {}), ({W(0, 0): "RIGHT"}, {})],
                 [("cash 995 after absorb step, neighbours 96.25",
                   lambda s: (s["cash"][0] == 995 and s["carbon"][idx(7, 8)] == 96.25
                              and len(s["trees"]) == 1, (s["cash"], s["carbon"][idx(7, 8)])))])
    # S3b second tree pays market price 20 + 5*1.235^1
    run_scenario("S3b plant cost inflation",
                 crafted_obs([(1000, {W(0, 0): (idx(7, 7), 0, P)}, {T(0, 0): (idx(2, 2), 5)}),
                              (100, {}, {})]),
                 [({}, {})],
                 [("cash == round(1000-26.175,2), 2 trees",
                   lambda s: (s["cash"][0] == round(1000 - (20 + 5 * 1.235 ** 1), 2)
                              and len(s["trees"]) == 2, (s["cash"], len(s["trees"]))))])
    # S4 seize an enemy tree (age preserved, then absorbs for new owner)
    run_scenario("S4 seize",
                 crafted_obs([(100, {W(0, 0): (idx(7, 7), 0, P)}, {}),
                              (100, {}, {T(1, 0): (idx(7, 7), 20)})],
                             carbon_cells={idx(7, 8): 100, idx(8, 7): 100,
                                           idx(7, 6): 100, idx(6, 7): 100}),
                 [({}, {})],
                 [("tree flipped to P0 age 21, cash 100-20+15=95",
                   lambda s: (list(s["trees"].values()) == [(0, idx(7, 7), 21)]
                              and s["cash"][0] == 95, (s["trees"], s["cash"])))])
    # S5 withering: age-50 tree dies leaving 30 carbon, collector harvests it
    run_scenario("S5 wither",
                 crafted_obs([(100, {W(0, 0): (idx(7, 7), 0, C)}, {T(0, 0): (idx(7, 7), 50)}),
                              (100, {}, {})]),
                 [({}, {})],
                 [("tree gone, collector cargo 9, cell carbon 21",
                   lambda s: (not s["trees"] and
                              list(s["workers"].values()) == [(0, idx(7, 7), 9, C)]
                              and s["carbon"][idx(7, 7)] == 21,
                              (s["trees"], s["workers"], s["carbon"][idx(7, 7)])))])
    # S6 absorption: quad blocked by staying collectors, 8-neighbour siphon
    run_scenario("S6 absorption & siphon",
                 crafted_obs([(500, {W(0, 0): (idx(7, 8), 40, C)}, {T(0, 0): (idx(7, 7), 10)}),
                              (500, {W(1, 0): (idx(7, 6), 200, C), W(1, 1): (idx(8, 8), 50, C)}, {})],
                             carbon_cells={idx(7, 8): 100, idx(8, 7): 100,
                                           idx(7, 6): 100, idx(6, 7): 100}),
                 [({}, {})],
                 [("P0 cash 535.5 (23+5 siphon, 3.75+3.75 cells)",
                   lambda s: (s["cash"][0] == 535.5, s["cash"])),
                  ("siphoned collector 230 collected -> 207 after 10%",
                   lambda s: (s["workers"][W(1, 0)][2] == 207, s["workers"])),
                  ("blocked cell carbon 70 (collected, no tree absorption)",
                   lambda s: (s["carbon"][idx(7, 6)] == 70, s["carbon"][idx(7, 6)]))])
    # S7 worker limit = 10 enforced across both bases in one step
    many = {W(0, i): (idx(2 + i, 2), 0, C) for i in range(9)}
    run_scenario("S7 worker limit",
                 crafted_obs([(1000, many, {}), (100, {}, {})]),
                 [({"player-0-recrtCenter-0": "RECCOLLECTOR",
                    "player-0-recrtCenter-1": "RECPLANTER"}, {})],
                 [("exactly 10 workers after double recruit attempt",
                   lambda s: (len(s["workers"]) == 10, len(s["workers"])))])


# --------------------------------------------------------------------------
# 4. drop-in agent compatibility
# --------------------------------------------------------------------------
def check_agent_compat():
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.agent

    try:
        example_agent = load("example_agent", os.path.join(HERE, "example.py"))
        sim = FastCarbon(CFG).reset(seed=3)
        for step in range(30):
            a = example_agent(sim.obs(0), CFG)
            b = example_agent(sim.obs(1), CFG)
            assert isinstance(a, dict) and isinstance(b, dict)
            sim.step(a, b)
        record("agents", "example.py runs 30 steps on FastCarbon", True,
               f"cash={sim.cash}")
    except Exception as e:  # noqa: BLE001
        record("agents", "example.py runs 30 steps on FastCarbon", False, repr(e))
    try:
        juzi_agent = load("juzi_agent", os.path.join(HERE, "reference_juzi.py"))
        sim = FastCarbon(CFG).reset(seed=5)
        t0 = time.perf_counter()
        for step in range(10):
            a = juzi_agent(sim.obs(0), CFG)
            b = juzi_agent(sim.obs(1), CFG)
            sim.step(a, b)
        record("agents", "reference_juzi.py runs 10 steps on FastCarbon", True,
               f"cash={sim.cash}, {time.perf_counter() - t0:.1f}s for 10 steps (agent-bound)")
    except Exception as e:  # noqa: BLE001
        record("agents", "reference_juzi.py runs 10 steps on FastCarbon", False, repr(e))


# --------------------------------------------------------------------------
# 5. benchmarks
# --------------------------------------------------------------------------
def bench():
    def fast_random(obs, config):
        me = obs["players"][obs["player"]]
        cash, centers, workers = me[0], me[1], me[2]
        actions = {}
        for wid in workers:
            actions[wid] = random.choice(["UP", "RIGHT", "DOWN", "LEFT"])
        for cid in centers:
            if cash >= 30:
                actions[cid] = random.choice(["RECCOLLECTOR", "RECPLANTER"])
                cash -= 30
        return actions

    random.seed(2026)
    t0 = time.perf_counter()
    env = make("carbon", configuration={"episodeSteps": 300, "randomSeed": 3})
    env.run([fast_random, fast_random])
    orig_t = time.perf_counter() - t0
    j = env.toJSON()
    orig_rewards = [s["reward"] for s in j["steps"][-1]]

    times = []
    rewards = None
    for _ in range(5):
        random.seed(2026)
        sim = FastCarbon(CFG).reset(seed=3)
        t0 = time.perf_counter()
        for step in range(300):
            a = fast_random(sim.obs(0), CFG)
            b = fast_random(sim.obs(1), CFG)
            sim.step(a, b)
        times.append(time.perf_counter() - t0)
        rewards = list(sim.cash)
    fast_t = sorted(times)[len(times) // 2]

    # pure transition cost: no agents at all, empty actions, pre-built board
    sim = FastCarbon(CFG).reset(seed=3)
    t0 = time.perf_counter()
    for _ in range(300):
        sim.step({}, {})
    empty_fast = time.perf_counter() - t0

    # full-length comparison: Board chain vs FastCarbon, same chaos policy,
    # always 300 steps (env.run above stops early on elimination, so its
    # wall time is NOT for 300 steps -- see report note).
    def chaos_game(sim_obj, ref_obj):
        for step in range(300):
            st = state_from_obs(ref_obj.obs) if ref_obj else sim_state(sim_obj)
            fo = ref_obj.obs if ref_obj else sim_obj.full_observation()
            a = chaos_actions(st, fo, step)
            if ref_obj:
                ref_obj.step(a[0], a[1])
            else:
                sim_obj.step(a[0], a[1])

    sim_c = FastCarbon(CFG).reset(seed=123)
    ref_c = RefChain(sim_c.full_observation())
    t0 = time.perf_counter()
    chaos_game(None, ref_c)
    board_chain_t = time.perf_counter() - t0
    fast_chaos_times = []
    for _ in range(3):
        sim_c = FastCarbon(CFG).reset(seed=123)
        t0 = time.perf_counter()
        chaos_game(sim_c, None)
        fast_chaos_times.append(time.perf_counter() - t0)
    fast_chaos_t = sorted(fast_chaos_times)[len(fast_chaos_times) // 2]
    return (orig_t, orig_rewards, fast_t, rewards, empty_fast, times,
            board_chain_t, fast_chaos_t)


# --------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    check_map_generation()
    print("map generation checks done", flush=True)
    check_full_game(seed=7)
    print("full game seed=7 done", flush=True)
    check_full_game(seed=123)
    print("full game seed=123 done", flush=True)
    check_scenarios()
    print("scenarios done", flush=True)
    check_agent_compat()
    print("agent compat done", flush=True)
    (orig_t, orig_r, fast_t, fast_r, empty_fast, all_fast,
     board_chain_t, fast_chaos_t) = bench()
    print("bench done", flush=True)

    n_ok = sum(1 for r in RESULTS if r[2])
    n_all = len(RESULTS)
    speedup = orig_t / fast_t if fast_t else float("inf")
    lines = []
    lines.append("# FastCarbon 对拍验证报告\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（本地）  ")
    lines.append(f"总检查项：{n_all}，通过：{n_ok}，失败：{n_all - n_ok}  ")
    lines.append(f"对拍总耗时：{time.perf_counter() - t_start:.1f}s\n")
    lines.append("## 结果明细\n")
    cur = None
    for section, name, ok, detail in RESULTS:
        if section != cur:
            lines.append(f"\n### {section}\n")
            cur = section
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    lines.append("\n## 测速（300 步一局，随机招募/移动 agent 自对弈）\n")
    chaos_speedup = board_chain_t / fast_chaos_t if fast_chaos_t else float("inf")
    lines.append(f"- 原版 zerosum_env `env.run`（随机 agent）：{orig_t:.2f}s，最终 reward={orig_r}。"
                 "注意：该局因一方被淘汰（compute_next_board 的 DONE 判定）提前结束，"
                 "实际步数 < 300，此数字仅作量级参考。")
    lines.append(f"- FastCarbon（同 agent 逻辑，obs+step，满 300 步）：中位数 {fast_t:.3f}s"
                 f"（5 次：{', '.join(f'{t:.3f}' for t in all_fast)}），最终 cash={fast_r}")
    lines.append(f"- **同策略满 300 步对比（chaos 策略，含招募/种树/吸收的真实负载）："
                 f"原版 Board 链 {board_chain_t:.2f}s vs FastCarbon {fast_chaos_t:.3f}s "
                 f"→ 加速比 {chaos_speedup:.0f}×**")
    lines.append(f"- FastCarbon 纯转移（空动作 300 步）：{empty_fast:.3f}s"
                 f"（{empty_fast / 300 * 1000:.2f} ms/步）")
    lines.append("- 注：env.run 与 FastCarbon 的 agent 随机数流不完全一致，故最终分数"
                 "不要求相同；语义一致性由上面的逐步对拍保证。\n")
    lines.append("## 语义对齐范围与已知边界\n")
    lines.append("- 已逐条对齐 Board.next()：招募成本（按全局工人数计价）与 "
                 "worker_limit=10、环绕移动、碰撞消解（最少携带碳采集员胜出/并列全灭/"
                 "胜者缴获/败者碳落树·基地·格子）、基地存碳（含敌方基地规则）、"
                 "停留采集 30%、种植成本（首棵 20，之后 20+5·1.235^存活树总数）、"
                 "抢树 20、树龄 50 枯萎留 30 碳、树吸收（4 邻 3.75% 带 quad 条件、"
                 "8 邻+树格敌方采集员 10%）、未受影响格再生 3%、cash 2 位/格碳 3 位"
                 "四舍五入、新生树当步不吸收。")
    lines.append("- 复刻的原版怪癖：抢树时 _delete_tree 会清掉抢树者所在格的 worker "
                 "标记（影响当步吸收/再生判定）；工人死亡时其所种树的 trees_dict "
                 "creator 字段置 None；quad flag 字典的键（任意树的 4 邻格）当步一律"
                 "不再生，即使 flag 集合为空。")
    lines.append("- 地图生成：populate_board 使用全局 np.random 流（原版从不自行播种），"
                 "FastCarbon.reset(seed) 等价于 np.random.seed(seed) 后第一次 "
                 "populate；env 的 make()+reset() 会 populate 两次，本报告已按此校验"
                 "（第二次生成与 env full_carbon 完全一致）。")
    lines.append("- 未覆盖（属 env core 而非 Board.next 的职责，未做也不影响对拍）："
                 " agent 超时与 remainingOverageTime 扣减（本实现原样透传）、"
                 "compute_next_board 的淘汰/DONE 判定、无效动作的 INVALID 状态处理"
                 "（本实现按 Board 行为将未知动作字符串视为无动作）。")
    lines.append("- obs() 的 trees 字典条目顺序与 Board.observation 在个别步骤可能"
                 "不同（本实现每步按玩家顺序重建，与 Board 构造顺序一致；吸收阶段的"
                 "重插入顺序也已复刻），字典内容逐项一致——对拍按字典内容比较。")
    if MISMATCHES:
        lines.append("\n## 不一致点\n")
        for m in MISMATCHES:
            lines.append(f"- {m}")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "fastsim_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n==== {n_ok}/{n_all} checks passed ====")
    for m in MISMATCHES:
        print("FAIL:", m)
    print(f"bench: orig {orig_t:.2f}s vs fast {fast_t:.3f}s -> {speedup:.0f}x; "
          f"empty-step 300: {empty_fast:.3f}s")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())

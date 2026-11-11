"""rl2_ppo_rollout -- CPU rollout / evaluation workers for rl2_ppo.

A worker task plays several FastCarbon games IN LOCKSTEP so the learner's
forward pass is batched over all their units (one NumPy call per step);
opponents are called per game. Game seeding follows eval_v53.play_game:
random.seed(rng_seed(seed, side)) -> sim.reset(seed) -> load opponent.
(With >1 lockstep game the global `random` stream is shared between the
games' opponents, so training games are not bit-reproducible vs play_game;
evaluation tasks therefore run one game per task.)

Opponent spec strings:
    "exp/opp_1cz_v5.py"      a bot file (module agent(), eval_v53 loader rule)
    "SNAP:<adapter>:<npz>"   frozen snapshot of a learner (greedy or sampled at
                             SNAP_TEMP), played through that adapter's actor
"""

import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import eval_v53  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import rl2_ppo_adapters as AD  # noqa: E402

CFG = eval_v53.CFG
_W_CACHE = {}      # (adapter, key) -> actor   (worker-local)


def _actor(adapter_name, wkey, weights=None):
    k = (adapter_name, wkey)
    a = _W_CACHE.get(k)
    if a is None:
        if weights is None:
            weights = AD.load_npz(wkey)
        a = AD.get_adapter(adapter_name).make_actor(weights)
        if len(_W_CACHE) > 8:
            _W_CACHE.clear()
        _W_CACHE[k] = a
    return a


def load_bot(path):
    """eval_v53.load_agent_fn, but each load gets a PRIVATE nn_agent module:
    files like cand_c1z_v1.py `import nn_agent` and nn_agent keeps per-player
    module state (_stall), which concurrent lockstep games would share."""
    saved = sys.modules.pop("nn_agent", None)
    try:
        return eval_v53.load_agent_fn(os.path.join(HERE, path))
    finally:
        if saved is not None:
            sys.modules["nn_agent"] = saved
        else:
            sys.modules.pop("nn_agent", None)


class _SnapAgent:
    def __init__(self, adapter_name, path, side, temp, seed):
        self.actor = _actor(adapter_name, path)
        self.st = self.actor.new_game(side)
        self.temp = temp
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, cfg):
        greedy = self.temp <= 0
        return self.actor.act([self.st], [obs], self.rng, max(self.temp, 1e-3), greedy)[0][0]


def make_opponent(spec, side, seed):
    if spec.startswith("SNAP:"):
        _, ad, path = spec.split(":", 2)
        return _SnapAgent(ad, path, side, float(os.environ.get("RL2_SNAP_TEMP", "0")), seed * 13 + 5)
    return load_bot(spec)


def _own_events(pre, post, me, my_bases):
    """(delivered cargo, cargo lost by death) for player `me` in one step."""
    dep = lost = 0.0
    pw, qw = pre["workers"], post["workers"]
    for wid, w in pw.items():
        if w["player"] != me:
            continue
        q = qw.get(wid)
        if q is None:
            lost += w["carbon"]
        elif w["carbon"] > 0 and q["carbon"] == 0 and q["pos"] in my_bases:
            dep += w["carbon"]
    return dep, lost


def _worth_margin(sim, post, side, left):
    """(our worth - their worth): cash + cargo on board + remaining value of standing trees; at the end cash only.
    Potential-based shaping for --reward worth: the per-step differences sum to the final margin exactly, but carbon
    is credited when it is picked up / a tree is planted, not tens of steps later when it reaches a base."""
    v = [float(post["cash"][0]), float(post["cash"][1])]
    if left > 0:
        for w in post["workers"].values():
            v[w["player"]] += float(w["carbon"])
        car = sim.carbon
        for t_ in post["trees"].values():
            r, c = divmod(int(t_["pos"]), 15)
            nb = car[((r - 1) % 15) * 15 + c] + car[((r + 1) % 15) * 15 + c] + car[r * 15 + (c - 1) % 15] + car[r * 15 + (c + 1) % 15]
            v[t_["player"]] += nb * (1.0 - 0.9625 ** max(0, min(50 - int(t_["age"]), left, 30))) * 0.8
    return v[side] - v[1 - side]


def play_batch(task):
    """task keys: adapter, wkey, weights (or None -> load wkey), games [(opp, seed, side)],
    temp, greedy, record (bool), rng_seed.
    Returns list of per-game dicts (traj recs if record) + timing."""
    t0 = time.time()
    actor = _actor(task["adapter"], task["wkey"], task.get("weights"))
    games = []
    for opp, seed, side in task["games"]:
        random.seed(eval_v53.rng_seed(seed, side))
        sim = FastCarbon(CFG).reset(seed=seed)
        g = {"opp": opp, "seed": seed, "side": side, "sim": sim,
             "op": make_opponent(opp, 1 - side, seed), "st": actor.new_game(side),
             "recs": [], "R": [], "W": [], "dep": [], "lost": [], "prev": 0.0, "prevw": 0.0,
             "my_bases": None}
        games.append(g)
    rng = np.random.default_rng(task.get("rng_seed", 0))
    temp, greedy, record = task.get("temp", 1.0), task.get("greedy", False), task.get("record", True)
    t_pol = t_opp = t_sim = 0.0
    ndec = 0
    for step in range(1, 301):
        obs_me = [g["sim"].obs(g["side"]) for g in games]
        ta = time.time()
        outs = actor.act([g["st"] for g in games], obs_me, rng, temp, greedy)
        t_pol += time.time() - ta
        ndec += len(games)
        for g, o, (acts, rec) in zip(games, obs_me, outs):
            sim, side = g["sim"], g["side"]
            tb = time.time()
            a_op = g["op"](sim.obs(1 - side), CFG)
            tc = time.time()
            t_opp += tc - tb
            pre = sim.get_state() if record else None
            sim.step(*((acts, a_op) if side == 0 else (a_op, acts)))
            post = sim.get_state()
            t_sim += time.time() - tc
            d = post["cash"][side] - post["cash"][1 - side]
            if record:
                if g["my_bases"] is None:
                    g["my_bases"] = set(o["players"][side][1].values())
                dep, lost = _own_events(pre, post, side, set(o["players"][side][1].values()))
                g["recs"].append(rec)
                g["R"].append(d - g["prev"])
                wm = _worth_margin(sim, post, side, 300 - step)
                g["W"].append(wm - g["prevw"]); g["prevw"] = wm
                g["dep"].append(dep); g["lost"].append(lost)
            g["prev"] = d
    out = []
    for g in games:
        st = g["sim"].get_state()
        side = g["side"]
        r = {"opp": g["opp"], "seed": g["seed"], "side": side,
             "my": float(st["cash"][side]), "op": float(st["cash"][1 - side]),
             "margin": float(st["cash"][side] - st["cash"][1 - side])}
        if record:
            r.update(recs=g["recs"], dmargin=np.array(g["R"], np.float32), dworth=np.array(g["W"], np.float32),
                     dep=np.array(g["dep"], np.float32), lost=np.array(g["lost"], np.float32))
        out.append(r)
    return {"games": out, "wall": time.time() - t0, "t_pol": t_pol, "t_opp": t_opp,
            "t_sim": t_sim, "ndec": ndec, "pid": os.getpid()}

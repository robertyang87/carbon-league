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
    "SNAP:cnet2:<npz>#T5"    any cnet2 policy net as a cheap fixed opponent (style row 5): no rule layer, no
                             search, trunk on the inference server -- costs what a snapshot costs
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


_REMOTE = [None]   # rl2_ppo_server.Client of this worker (None: NumPy trunk in the worker)


def _init_worker(ids, req_q, resp_qs):
    """Pool initializer when the trainer runs an inference server."""
    import queue
    import rl2_ppo_server as SV
    try:
        wid = ids.get(timeout=5)
    except queue.Empty:                 # a replacement worker: no free response queue, fall back to the NumPy trunk
        return
    if isinstance(req_q, (list, tuple)):          # several server processes: this worker talks to one of them
        req_q = req_q[wid % len(req_q)]
    _REMOTE[0] = SV.Client(wid, req_q, resp_qs[wid])


def _actor(adapter_name, wkey, weights=None, role="snap"):
    # F06: the role is part of the cache identity. The old key was only
    # (adapter, wkey), so an actor first built as a task's primary -- on
    # which play_batch set temp_kind=True -- was handed back for a SNAP of
    # the same weights, and the snapshot silently played at the learner's
    # per-kind temperatures instead of the temperature it was handed
    # (review measurement: cached_snapshot_is_primary=True; temps 0.5/1.0
    # produced identical logprobs). temp_kind is now fixed by role at
    # construction: only the "learner" actor may be governed by
    # CNET2_TEMP_KIND; snapshot actors keep temp_kind=False.
    k = (adapter_name, wkey, role)
    a = _W_CACHE.get(k)
    if a is None:
        teacher = None
        if isinstance(wkey, str) and "#T" in wkey:      # "path.npz#T5": play this net with style row 5 (fixed opponents)
            wkey, t_ = wkey.rsplit("#T", 1); teacher = int(t_)
        if weights is None:
            weights = AD.load_npz(wkey)
        a = AD.get_adapter(adapter_name).make_actor(weights)
        a.temp_kind = role == "learner"
        if teacher is not None:
            a.teacher = teacher; a.sched = False
        if _REMOTE[0] is not None and hasattr(a, "remote") and isinstance(wkey, str) and os.path.exists(wkey):
            a.remote = _REMOTE[0]; a.key = os.path.abspath(wkey)
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
        self.actor = _actor(adapter_name, path, role="snap")
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


def _extinct(post, p):
    """Original-env elimination (zerosum_env/envs/carbon/carbon.py): player p is finished when it
    has no workers, no trees, and no way to recruit -- no recruitment center left, or cash below
    the cheaper of the two recruit costs. The game ends for both sides at that moment."""
    if any(w["player"] == p for w in post["workers"].values()):
        return False
    if any(t["player"] == p for t in post["trees"].values()):
        return False
    if not post["bases"].get(p):
        return True
    return post["cash"][p] < min(CFG.get("recCollectorCost", 30), CFG.get("recPlanterCost", 30))


def play_batch(task):
    """task keys: adapter, wkey, weights (or None -> load wkey), games [(opp, seed, side)],
    temp, greedy, record (bool), rng_seed.
    Opponent "SELF": the same weights play the other side with the same temperature, and that side is recorded too
    (returned as a second game dict with opp "SELF", the other side, mirror=True).
    All positions that use the same model in one step go through ONE actor.act call (one server request).
    Returns list of per-game dicts (traj recs if record) + timing."""
    t0 = time.time()
    actor = _actor(task["adapter"], task["wkey"], task.get("weights"), role="learner")
    # R04/F06: the task's primary actor is the learner(-side) actor -- the only one
    # CNET2_TEMP_KIND may govern. It is built with role="learner" (temp_kind=True);
    # snapshot actors built below (and _SnapAgent's) are separate cached objects
    # with temp_kind=False and play at the temperature they are handed, so the
    # learner's per-kind temperatures no longer leak into them.
    snap_temp = float(os.environ.get("RL2_SNAP_TEMP", "0"))
    games = []
    for opp, seed, side in task["games"]:
        random.seed(eval_v53.rng_seed(seed, side))
        sim = FastCarbon(CFG).reset(seed=seed)
        g = {"opp": opp, "seed": seed, "side": side, "sim": sim, "st": actor.new_game(side),
             "recs": [], "R": [], "W": [], "dep": [], "lost": [], "prev": 0.0, "prevw": 0.0,
             "prevcash": [float(sim.cash[0]), float(sim.cash[1])], "term_margin": 0.0,
             "kind": "bot", "op": None, "done": False, "term": False, "nsteps": 0}
        if opp == "SELF":
            g.update(kind="self", st2=actor.new_game(1 - side), recs2=[], dep2=[], lost2=[])
        elif opp.startswith("SNAP:"):
            _, ad_, path_ = opp.split(":", 2)
            sa = _actor(ad_, path_, role="snap")
            g.update(kind="snap", sactor=sa, st2=sa.new_game(1 - side))
        else:
            g["op"] = make_opponent(opp, 1 - side, seed)
        games.append(g)
    rng = np.random.default_rng(task.get("rng_seed", 0))
    rng_s = np.random.default_rng(task.get("rng_seed", 0) * 13 + 5)
    temp, greedy, record = task.get("temp", 1.0), task.get("greedy", False), task.get("record", True)
    t_pol = t_opp = t_sim = 0.0
    ndec = 0
    # R01: the original env plays 299 transitions (its final observation is step=299 and agents
    # are last called at step=298); the old range(1, 301) played one action step that does not
    # exist in the real game.
    for step in range(1, 300):
        # Review §5 (efficiency): infer ONLY for unfinished games. The old loop prepared and
        # decoded every game at every step and threw the finished ones' outputs away in the
        # step loop below -- with one of two SELF games ending at step 1 it recorded 600 valid
        # decisions but decoded 1196 states. A finished game's record is final (R02), and once
        # no game is left the batch is over. (RNG draws therefore shrink once a sibling game
        # ends; multi-game tasks were never bit-reproducible anyway -- module docstring.)
        live = [g for g in games if not g["done"]]
        if not live:
            break
        lself = [g for g in live if g["kind"] == "self"]
        lsnaps = {}
        for g in live:
            if g["kind"] == "snap":
                lsnaps.setdefault(id(g["sactor"]), []).append(g)
        ta = time.time()
        obs_me = [g["sim"].obs(g["side"]) for g in live]
        obs_s = [g["sim"].obs(1 - g["side"]) for g in lself]
        st_all = [g["st"] for g in live] + [g["st2"] for g in lself]
        two = hasattr(actor, "prepare")
        # all trunk requests of this step go out before any answer is awaited (one server batch, not one per model)
        prep = actor.prepare(st_all, obs_me + obs_s) if two else None
        sprep = [(gs, gs[0]["sactor"].prepare([g["st2"] for g in gs], [g["sim"].obs(1 - g["side"]) for g in gs]))
                 for gs in lsnaps.values()] if two else []
        outs = actor.finish(st_all, prep, rng, temp, greedy) if two else actor.act(st_all, obs_me + obs_s, rng, temp, greedy)
        for g, o in zip(lself, outs[len(live):]):
            g["a_op"], g["rec2"] = o
        ndec += len(outs)
        tb = time.time()
        t_pol += tb - ta
        if two:
            for gs, sp in sprep:
                so = gs[0]["sactor"].finish([g["st2"] for g in gs], sp, rng_s, max(snap_temp, 1e-3), snap_temp <= 0)
                for g, o in zip(gs, so):
                    g["a_op"] = o[0]
        else:
            for gs in lsnaps.values():
                so = gs[0]["sactor"].act([g["st2"] for g in gs], [g["sim"].obs(1 - g["side"]) for g in gs], rng_s,
                                         max(snap_temp, 1e-3), snap_temp <= 0)
                for g, o in zip(gs, so):
                    g["a_op"] = o[0]
        t_opp += time.time() - tb
        for g, o, (acts, rec) in zip(live, obs_me, outs):
            sim, side = g["sim"], g["side"]
            tb = time.time()
            a_op = g["op"](sim.obs(1 - side), CFG) if g["kind"] == "bot" else g["a_op"]
            tc = time.time()
            t_opp += tc - tb
            pre = sim.get_state() if record else None
            sim.step(*((acts, a_op) if side == 0 else (a_op, acts)))
            post = sim.get_state()
            g["nsteps"] += 1
            # R02: extinction ends the game here (original env marks the side DONE and stops):
            # no further steps are played or recorded, so the trainer's GAE sees a true terminal
            # boundary (its last-step bootstrap is already 0) instead of a game that plays on.
            ext_now = [_extinct(post, 0), _extinct(post, 1)]
            term_now = any(ext_now)
            if term_now:
                g["done"] = True; g["term"] = True
                # F01: original terminal scoring (carbon.py compute_next_board), NOT the
                # post-step cash difference: an extinct side scores (step - episodeSteps - 1);
                # a survivor keeps the reward it carried INTO this step -- its cash in the
                # pre-step state (g["prevcash"]) -- because the original marks it DONE
                # without a reward update. The dense/worth increments below and the final
                # margin all land on this value; the SELF mirror negates it as before.
                pen = float(step - eval_v53.EPISODE_STEPS - 1)
                rew = [pen if ext_now[p] else g["prevcash"][p] for p in (0, 1)]
                g["term_margin"] = rew[side] - rew[1 - side]
            t_sim += time.time() - tc
            d = post["cash"][side] - post["cash"][1 - side]
            seen = g.setdefault("tseen", set())          # trees planted per side (new tree ids), for the train log
            for tid_, t_ in post["trees"].items():
                if tid_ not in seen:
                    seen.add(tid_)
                    k_ = "tp_my" if t_["player"] == side else "tp_op"
                    g[k_] = g.get(k_, 0) + 1
            if record:
                dep, lost = _own_events(pre, post, side, set(o["players"][side][1].values()))
                g["recs"].append(rec)
                # F01: at an extinction finish the last increment lands on the original
                # terminal margin, not on the post-step cash difference
                g["R"].append((g["term_margin"] if term_now else d) - g["prev"])
                # at the terminal step (extinction or the step-299 horizon) worth is cash only;
                # at an extinction finish that means the original terminal rewards above
                wm = g["term_margin"] if term_now else _worth_margin(sim, post, side, 299 - step)
                g["W"].append(wm - g["prevw"]); g["prevw"] = wm
                g["dep"].append(dep); g["lost"].append(lost)
                if g["kind"] == "self":
                    dep2, lost2 = _own_events(pre, post, 1 - side, set(o["players"][1 - side][1].values()))
                    g["recs2"].append(g["rec2"]); g["dep2"].append(dep2); g["lost2"].append(lost2)
            g["prev"] = d
            g["prevcash"] = [float(post["cash"][0]), float(post["cash"][1])]
    if record and hasattr(actor, "pack") and os.environ.get("RL2_PACK", "1") == "1":
        for g in games:
            g["recs"] = actor.pack(g["recs"])
            if g["kind"] == "self":
                g["recs2"] = actor.pack(g["recs2"])
    out = []
    for g in games:
        st = g["sim"].get_state()
        side = g["side"]
        r = {"opp": g["opp"], "seed": g["seed"], "side": side,
             "my": float(st["cash"][side]), "op": float(st["cash"][1 - side]),
             "margin": float(g["term_margin"]) if g["term"] else float(st["cash"][side] - st["cash"][1 - side]),
             "term": bool(g["term"]), "steps": g["nsteps"],
             "planted": g.get("tp_my", 0), "planted_op": g.get("tp_op", 0)}
        if record:
            r.update(recs=g["recs"], dmargin=np.array(g["R"], np.float32), dworth=np.array(g["W"], np.float32),
                     dep=np.array(g["dep"], np.float32), lost=np.array(g["lost"], np.float32))
        out.append(r)
        if record and g["kind"] == "self":          # the other side of a self-play game: every per-step margin negated
            out.append({"opp": "SELF", "seed": g["seed"], "side": 1 - side, "mirror": True,
                        "my": r["op"], "op": r["my"], "margin": -r["margin"],
                        "term": r["term"], "steps": r["steps"], "recs": g["recs2"],
                        "planted": r["planted_op"], "planted_op": r["planted"],
                        "dmargin": -r["dmargin"], "dworth": -r["dworth"],
                        "dep": np.array(g["dep2"], np.float32), "lost": np.array(g["lost2"], np.float32)})
    return {"games": out, "wall": time.time() - t0, "t_pol": t_pol, "t_opp": t_opp,
            "t_sim": t_sim, "ndec": ndec, "pid": os.getpid()}

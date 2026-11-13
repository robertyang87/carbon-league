"""train_rl2 -- self-play PPO, stage 2, for the Carbon League NN line.

Stage-2 changes vs train_rl.py (stage 1), per rl_stage1_report S6:
  * belief channels: featurize2 (bc_features + four-fold-symmetric
    belief carbon map, belief.py) -- worker 156, base 28 dims; PPO
    warm-starts from the stage-2 BC (rl2_init_weights.npz), whose val
    top-1 was A/B-checked against the no-belief control (train_bc2).
  * decorrelated action decoding everywhere (shadow v2 protocol):
    sequential sampling with destination reservation (own base cells
    exempt); the exact sampling mask is stored per transition and
    re-applied in the PPO update, so behaviour logp is exact.
  * hard-rule shields INSIDE the training loop (same code as the
    deployed policy, rl2_common): (a) danger mask for loaded
    collectors, (b) last-30-steps return scheduling, (c) opening
    cash-chain protection. Shield-determined transitions are excluded
    from the policy batch (they are not policy choices) but stay in
    the GAE chain.
  * checkpoint selection on a FIXED probe set every iteration:
    vs v0.5 seeds 4100/4101 both sides (4), vs shadow v2 same (4),
    vs base_v53 seed 4100 both sides (2), plus vs example (2) as
    acceptance-only (self-collision deaths + endgame stranding
    reported, not scored). Probe score = mean margin over the 10
    selection games. (Probe originally carried 2 juzi games; juzi was
    retired as a sparring opponent by user directive 2026-11-03
    09:46 and replaced by base_v53 -- juzi was never trained against
    either way.)
  * opponent pool per 96-game iteration: shadow v2 38 / self-snapshot
    24 / policy_v0_5 24 / example 10. Training seeds 30000+.

Run: venv/bin/python train_rl2.py        (full run)
     RL2_SMOKE=1 venv/bin/python train_rl2.py
Outputs: rl2_ckpt/iter{n}.npz, rl2_train_log.jsonl, rl2_weights.npz
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src",
                                "carbon_challenge-master"))

import numpy as np  # noqa: E402

PROBE_SEEDS = (4100, 4101)


# ----------------------------------------------------------------------
# Torch nets (belief dims, warm start from stage-2 BC)
# ----------------------------------------------------------------------
def build_nets2():
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, bc, prefix):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.policy = nn.Linear(hidden, out_dim)
            self.value = nn.Linear(hidden, 1)
            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor(bc[prefix + "W1"]))
                self.fc1.bias.copy_(torch.tensor(bc[prefix + "b1"]))
                self.fc2.weight.copy_(torch.tensor(bc[prefix + "W2"]))
                self.fc2.bias.copy_(torch.tensor(bc[prefix + "b2"]))
                self.policy.weight.copy_(torch.tensor(bc[prefix + "W3"]))
                self.policy.bias.copy_(torch.tensor(bc[prefix + "b3"]))
                self.value.weight.mul_(0.01)
                self.value.bias.zero_()
            self.register_buffer(
                "mean", torch.tensor(bc[prefix + "mean"]))
            self.register_buffer(
                "std", torch.tensor(bc[prefix + "std"]))

        def forward(self, x):
            import torch
            h = (x - self.mean) / self.std
            h = torch.relu(self.fc1(h))
            h = torch.relu(self.fc2(h))
            return self.policy(h), self.value(h).squeeze(-1)

    raw = np.load(os.path.join(HERE, "rl2_init_weights.npz"))
    bc = {}
    for k in raw.files:
        # train_bc2 exports worker_*/base_*; the net convention is w_*/b_*
        nk = k.replace("worker_", "w_", 1).replace("base_", "b_", 1)
        bc[nk] = raw[k].astype(np.float32)
    from belief import BASE_DIM2, WORKER_DIM2
    wnet = Head(WORKER_DIM2, 192, 5, bc, "w_")
    bnet = Head(BASE_DIM2, 96, 3, bc, "b_")
    return wnet, bnet


def export_np_weights(wnet, bnet):
    out = {}
    for prefix, net in (("w_", wnet), ("b_", bnet)):
        out[prefix + "mean"] = net.mean.numpy().astype(np.float32)
        out[prefix + "std"] = net.std.numpy().astype(np.float32)
        out[prefix + "W1"] = net.fc1.weight.detach().numpy()
        out[prefix + "b1"] = net.fc1.bias.detach().numpy()
        out[prefix + "W2"] = net.fc2.weight.detach().numpy()
        out[prefix + "b2"] = net.fc2.bias.detach().numpy()
        out[prefix + "W3"] = net.policy.weight.detach().numpy()
        out[prefix + "b3"] = net.policy.bias.detach().numpy()
    return out


# ----------------------------------------------------------------------
# Rollout worker
# ----------------------------------------------------------------------
_W = {}


def _worker_init():
    import torch
    torch.set_num_threads(1)
    import rl_common
    _W["cfg"] = rl_common.CFG
    _W["shadow2_weights"] = rl_common.load_weights_npz(
        os.path.join(HERE, "shadow_v2_weights.npz"))


def _plant_price(pre_trees, player):
    import rl_common
    cfg = rl_common.CFG
    mine = sum(1 for (o, a) in pre_trees.values() if o == player)
    if mine <= cfg["playerTreeProtectiveNumber"]:
        return float(cfg["plantCost"])
    alive = sum(1 for (o, a) in pre_trees.values()
                if a < cfg["treeLifespan"])
    return (cfg["plantCost"] + cfg["plantCostInflationRatio"]
            * (cfg["plantCostInflationBase"] ** alive))


def _rollout_game2(wnet, bnet, job, snap_weights):
    """One game. Returns (traj, result, stats).

    traj: {unit_key: {"steps": [(feat, act, logp, val, rew), ...],
                      "masks": [bool (5,) or None, ...],
                      "excluded": [bool, ...]}}  (learner side only)
    """
    import torch
    import rl2_common
    import rl_common
    from belief import BeliefTracker, featurize2
    from fastsim import FastCarbon

    cfg = _W["cfg"]
    seed, lp, opp = job["seed"], job["side"], job["opp"]
    sim = FastCarbon(cfg).reset(seed=seed)
    seq = rl_common.SeqState()
    tracker = BeliefTracker()
    shield = rl2_common.OpeningShield()
    rng = np.random.default_rng(seed * 31 + 7)
    stats = {"fallbacks": 0, "endgame_overrides": 0,
             "opening_forced": 0, "opening_throttled": 0}

    opp_actor = None
    opp_fn = None
    if opp == "shadow2":
        opp_actor = rl2_common.NumpyActorD(
            _W["shadow2_weights"], seed=9000 + seed, stay_penalty=0.4)
    elif opp == "self":
        opp_actor = rl2_common.NumpyActor2(snap_weights,
                                           seed=9000 + seed)
    elif opp == "v05":
        opp_fn = rl_common.load_agent_fn(
            os.path.join(HERE, "policy_v0_5.py"), "policy")
    elif opp == "example":
        opp_fn = rl_common.load_agent_fn(
            os.path.join(HERE, "example.py"), "opp")
    else:
        raise ValueError(opp)

    own_base_pos = {pos for _, pos in sim.bases[lp]}
    traj = {}
    MOVE = rl_common._MOVE
    REC = rl_common._REC

    def push(key, feat, a, logp, val, r, mask, excluded):
        e = traj.setdefault(key, {"steps": [], "masks": [],
                                  "excluded": []})
        e["steps"].append((feat, a, logp, val, r))
        e["masks"].append(mask)
        e["excluded"].append(bool(excluded))

    for step in range(300):
        canon = rl_common.canonical_from_sim(sim, lp)
        (p, carbon, cash, bases, workers, wmask, trees, tmask, stp,
         my_wids, my_bids) = canon
        belief = tracker.update(carbon, workers, wmask, trees, tmask,
                                p, stp)
        n_now = int(np.sum(wmask[p]))
        sq = seq.seq_for(n_now, stp)
        Wf, Bf = featurize2(carbon, cash, bases, workers, wmask,
                            trees, tmask, p, stp, belief, seq=sq)
        pre_w = {wid: (w.player, w.pos, w.carbon, w.is_collector)
                 for wid, w in sim.workers.items()}
        pre_t = {tid: (t.player, t.age) for tid, t in sim.trees.items()}
        pre_cash = list(sim.cash)

        actions = {}
        acted_keys = []
        with torch.no_grad():
            if my_wids:
                x = torch.tensor(Wf[:len(my_wids)],
                                 dtype=torch.float32)
                logits, vals = wnet(x)
                lg = logits.numpy()
                lg = lg - lg.max(axis=1, keepdims=True)
                probs = np.exp(lg)
                probs /= probs.sum(axis=1, keepdims=True)
                own_bases = {int(b) for b in bases[p] if int(b) >= 0}
                allowed = [rl2_common.danger_allowed(canon, j)
                           for j in range(len(my_wids))]
                positions = [int(workers[p, j, 0])
                             for j in range(len(my_wids))]
                acts, masks, fb = rl2_common.sequential_sample(
                    probs, positions, own_bases, allowed, rng)
                stats["fallbacks"] += fb
                for j, wid in enumerate(my_wids):
                    a = int(acts[j])
                    pr = np.where(masks[j], probs[j], 0.0)
                    tot = pr.sum()
                    pa = (pr[a] / tot) if tot > 1e-12 else 1e-12
                    logp = float(np.log(max(pa, 1e-12)))
                    excl = False
                    ea = rl2_common.endgame_action(canon, j)
                    if ea is not None:
                        if ea != a:
                            stats["endgame_overrides"] += 1
                        a = ea
                        excl = True
                    push(("w", wid), Wf[j].astype(np.float32), a, logp,
                         float(vals[j]), 0.0, masks[j], excl)
                    acted_keys.append(("w", wid))
                    if a in MOVE:
                        actions[wid] = MOVE[a]
            n_rec = 0
            if my_bids:
                dirs = shield.directives(canon)
                x = torch.tensor(Bf[:len(my_bids)],
                                 dtype=torch.float32)
                logits, vals = bnet(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts_t = dist.sample()
                logp_t = dist.log_prob(acts_t)
                for i, bid in enumerate(my_bids):
                    a = int(acts_t[i])
                    logp = float(logp_t[i])
                    excl = False
                    d = dirs[i] if i < len(dirs) else ("free",)
                    if d[0] == "force":
                        a = d[1]
                        excl = True
                        stats["opening_forced"] += 1
                    elif d[0] == "none":
                        a = 0
                        excl = True
                        stats["opening_throttled"] += 1
                    push(("b", bid), Bf[i].astype(np.float32), a, logp,
                         float(vals[i]), 0.0, None, excl)
                    acted_keys.append(("b", bid))
                    if a in REC:
                        actions[bid] = REC[a]
                        n_rec += 1
        seq.update(n_now, stp, n_rec)

        if opp_actor is not None:
            _r = opp_actor.act_canonical(
                rl_common.canonical_from_sim(sim, 1 - lp))
            # rl_common.NumpyActor returns a bare dict; NumpyActorD
            # returns (actions, info).
            oacts = _r[0] if isinstance(_r, tuple) else _r
        else:
            oacts = opp_fn(sim.obs(1 - lp), cfg)
        both = [None, None]
        both[lp] = actions
        both[1 - lp] = oacts
        sim.step(both[0], both[1])

        # ---- rewards (stage-1 attribution, verbatim) ----
        post_w = {wid: (w.player, w.pos, w.carbon, w.is_collector)
                  for wid, w in sim.workers.items()}
        post_t = {tid: t.player for tid, t in sim.trees.items()}
        dcash = sim.cash[lp] - pre_cash[lp]
        tree_credit = {}
        for tid, (creator, absorb) in sim.trees_dict.items():
            if post_t.get(tid) == lp and creator is not None \
                    and absorb:
                tree_credit[creator] = tree_credit.get(creator, 0.0) \
                    + float(absorb)
        planted = {}
        seized = set()
        for tid, owner in post_t.items():
            if owner != lp:
                continue
            if tid not in pre_t:
                creator = sim.trees_dict.get(tid, [None, 0])[0]
                if creator is not None:
                    planted[creator] = _plant_price(pre_t, lp)
            elif pre_t[tid][0] != lp:
                creator = sim.trees_dict.get(tid, [None, 0])[0]
                if creator is not None:
                    seized.add(creator)
        # rewards attach to exactly the transitions added this step
        # (stage-1 semantics).
        for key in acted_keys:
            e = traj[key]
            kind, uid = key
            f, a, logp, val, _r = e["steps"][-1]
            r = 0.0
            if kind == "b":
                r = dcash / 100.0
            else:
                pre = pre_w.get(uid)
                post = post_w.get(uid)
                if post is None:
                    r = -3.0
                elif pre is not None and pre[3]:
                    pre_cargo, post_cargo, post_pos = pre[2], post[2], \
                        post[1]
                    if pre_cargo > 0 and post_cargo == 0 \
                            and post_pos in own_base_pos:
                        r = pre_cargo / 100.0
                    else:
                        dd = post_cargo - pre_cargo
                        if dd > 0:
                            r = min(dd, 50.0) / 100.0
                        elif dd < 0:
                            r = dd / 100.0
                elif pre is not None:
                    inc = tree_credit.get(uid, 0.0)
                    r = inc / 100.0
                    if uid in planted:
                        r -= planted[uid] / 100.0
                    if uid in seized:
                        r -= 0.2
                    if inc == 0 and uid not in planted \
                            and uid not in seized:
                        r -= 0.01
            e["steps"][-1] = (f, a, logp, val, r)

    final = list(sim.cash)
    margin = final[lp] - final[1 - lp]
    term = float(np.clip(margin / 1000.0, -3.0, 3.0))
    for key, e in traj.items():
        if e["steps"]:
            f, a, logp, val, r = e["steps"][-1]
            e["steps"][-1] = (f, a, logp, val, r + term)
    result = {"seed": seed, "side": lp, "opp": opp,
              "my_cash": round(final[lp], 2),
              "opp_cash": round(final[1 - lp], 2),
              "margin": round(margin, 2)}
    return traj, result, stats


def _rollout_batch(task):
    state, snap_weights, jobs = task
    wnet, bnet = build_nets2()
    import torch
    wnet.load_state_dict(state[0])
    bnet.load_state_dict(state[1])
    wnet.eval()
    bnet.eval()
    out = []
    for job in jobs:
        out.append(_rollout_game2(wnet, bnet, job, snap_weights))
    return out


# ----------------------------------------------------------------------
# Probe (fixed set, pool-executed)
# ----------------------------------------------------------------------
def _probe_game(task):
    weights, opp_kind, seed, side = task
    import rl2_common
    import rl_common
    from fastsim import FastCarbon
    cfg = rl_common.CFG
    sim = FastCarbon(cfg).reset(seed=seed)
    me = rl2_common.NumpyActor2(weights, seed=4242 + seed)
    opp_actor = None
    opp_fn = None
    if opp_kind == "shadow2":
        opp_actor = rl2_common.NumpyActorD(
            rl_common.load_weights_npz(
                os.path.join(HERE, "shadow_v2_weights.npz")),
            seed=777 + seed, stay_penalty=0.4)
    elif opp_kind == "v05":
        opp_fn = rl_common.load_agent_fn(
            os.path.join(HERE, "policy_v0_5.py"), "policy")
    elif opp_kind == "v53":
        opp_fn = rl_common.load_agent_fn(
            os.path.join(HERE, "base_v53.py"), "opp")
    elif opp_kind == "example":
        opp_fn = rl_common.load_agent_fn(
            os.path.join(HERE, "example.py"), "opp")
    else:
        raise ValueError(opp_kind)
    DELTA = {None: (0, 0), "UP": (0, 1), "RIGHT": (1, 0),
             "DOWN": (0, -1), "LEFT": (-1, 0)}
    from bc_features import _idx, _xs, _ys
    suicides = 0
    for _ in range(300):
        pre = {wid: (w.player, w.pos) for wid, w in
               sim.workers.items()}
        acts = [None, None]
        for p in range(2):
            if p == side:
                acts[p] = me.act_canonical(
                    rl_common.canonical_from_sim(sim, p))[0]
            elif opp_actor is not None:
                acts[p] = opp_actor.act_canonical(
                    rl_common.canonical_from_sim(sim, p))
            else:
                acts[p] = opp_fn(sim.obs(p), cfg)
        # landing groups for self-collision attribution
        groups = {}
        for wid, (pl, pos) in pre.items():
            a = (acts[pl] or {}).get(wid)
            dx, dy = DELTA.get(a, (0, 0))
            land = int(_idx(int(_xs[pos]) + dx, int(_ys[pos]) + dy))
            groups.setdefault(land, []).append((wid, pl))
        sim.step(acts[0], acts[1])
        alive = set(sim.workers)
        for land, members in groups.items():
            sides = {pl for _w, pl in members}
            if sides == {side} and len(members) > 1:
                suicides += sum(1 for w, _pl in members
                                if w not in alive)
        # NOTE: landing-group attribution mirrors eval_shadow_v2's
        # instrument (group composition at the landing cell).
    st = sim.get_state()
    stranded = float(sum(w.carbon for w in sim.workers.values()
                         if w.player == side))
    return {"opp": opp_kind, "seed": seed, "side": side,
            "margin": round(st["cash"][side] - st["cash"][1 - side], 2),
            "my_cash": round(st["cash"][side], 2),
            "opp_cash": round(st["cash"][1 - side], 2),
            "suicides": suicides, "stranded": round(stranded, 1)}


def probe_jobs(weights):
    jobs = []
    for opp_kind in ("v05", "shadow2"):
        for seed in PROBE_SEEDS:
            for side in (0, 1):
                jobs.append((weights, opp_kind, seed, side))
    for side in (0, 1):
        jobs.append((weights, "v53", PROBE_SEEDS[0], side))
        jobs.append((weights, "example", PROBE_SEEDS[0], side))
    return jobs


# ----------------------------------------------------------------------
# PPO update (worker head is mask-aware)
# ----------------------------------------------------------------------
def _gae(steps, gamma=0.995, lam=0.95):
    advs = np.zeros(len(steps), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(steps))):
        v = steps[t][3]
        v_next = steps[t + 1][3] if t + 1 < len(steps) else 0.0
        delta = steps[t][4] + gamma * v_next - v
        last = delta + gamma * lam * last
        advs[t] = last
    rets = advs + np.array([s[3] for s in steps], dtype=np.float32)
    return advs, rets


def _ppo_update_masked(net, opt, X, A, LOGP, MASK, ADV, RET, epochs,
                       mb, clip=0.2, ent_coef=0.01, vf_coef=0.5):
    import torch
    n = len(X)
    if n == 0:
        return {}
    xt = torch.tensor(X)
    at = torch.tensor(A, dtype=torch.long)
    lpt = torch.tensor(LOGP)
    mt = torch.tensor(MASK)
    advt = torch.tensor(ADV)
    rett = torch.tensor(RET)
    tot = {"pi": 0.0, "vf": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, mb):
            idx = perm[k:k + mb]
            logits, vals = net(xt[idx])
            logits = logits.masked_fill(~mt[idx], -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(at[idx])
            ratio = torch.exp(logp - lpt[idx])
            adv = advt[idx]
            pi = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1 - clip, 1 + clip)
                            * adv).mean()
            vf = ((vals - rett[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi + vf_coef * vf - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            tot["pi"] += float(pi)
            tot["vf"] += float(vf)
            tot["ent"] += float(ent)
            tot["n"] += 1
    return {k: (v / tot["n"] if k != "n" else v)
            for k, v in tot.items() if k != "n" or True}


def _ppo_update_plain(net, opt, X, A, LOGP, ADV, RET, epochs, mb,
                      **kw):
    ones = np.ones((len(X), X.shape[1] and 3), dtype=bool) \
        if len(X) else np.zeros((0, 3), dtype=bool)
    # base head has 3 actions; mask is all-True.
    return _ppo_update_masked(net, opt, X, A, LOGP, ones, ADV, RET,
                              epochs, mb, **kw)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import torch
    torch.set_num_threads(2)

    smoke = bool(os.environ.get("RL2_SMOKE"))
    n_iter = 1 if smoke else int(os.environ.get("RL2_ITERS", "10"))
    games_per_iter = 8 if smoke else int(os.environ.get("RL2_GAMES",
                                                        "96"))
    epochs = 1 if smoke else 4

    wnet, bnet = build_nets2()
    opt_w = torch.optim.Adam(wnet.parameters(), lr=5e-4)
    opt_b = torch.optim.Adam(bnet.parameters(), lr=5e-4)

    os.makedirs(os.path.join(HERE, "rl2_ckpt"), exist_ok=True)
    logf = open(os.path.join(HERE, "rl2_train_log.jsonl"), "a")
    snap_weights = export_np_weights(wnet, bnet)  # iter0 self = BC init
    best_score, best_iter = None, None

    mix = [("shadow2", 0.40), ("self", 0.25), ("v05", 0.25),
           ("example", 0.10)]

    # rollout parallelism capped at 2 processes (user compute
    # scheduling directive 2026-11-03 09:44, all lines share 4 cores)
    with ProcessPoolExecutor(max_workers=2,
                             initializer=_worker_init) as ex:
        for it in range(n_iter):
            t0 = time.time()
            jobs = []
            counts = {}
            for name, frac in mix:
                counts[name] = int(round(games_per_iter * frac))
            counts["shadow2"] += games_per_iter - sum(counts.values())
            idx = 0
            for name, cnt in counts.items():
                for _ in range(cnt):
                    jobs.append({"seed": 30000 + it * 10000 + idx * 7,
                                 "side": idx % 2, "opp": name})
                    idx += 1
            state = (wnet.state_dict(), bnet.state_dict())
            chunks = [jobs[i::4] for i in range(4)]
            tasks = [(state, snap_weights, c) for c in chunks]
            results = []
            for batch in ex.map(_rollout_batch, tasks):
                results.extend(batch)
            roll_s = time.time() - t0

            # ---- assemble PPO batch (excluded transitions dropped
            # from the loss but kept in GAE chains) ----
            Xw, Aw, Lw, Mw, Vw_adv, Vw_ret = [], [], [], [], [], []
            Xb, Ab, Lb, Vb_adv, Vb_ret = [], [], [], [], []
            margins_by_opp = {}
            agg_stats = {"fallbacks": 0, "endgame_overrides": 0,
                         "opening_forced": 0, "opening_throttled": 0}
            for traj, res, stats in results:
                margins_by_opp.setdefault(res["opp"], []).append(
                    res["margin"])
                for k in agg_stats:
                    agg_stats[k] += stats[k]
                for key, e in traj.items():
                    advs, rets = _gae(e["steps"])
                    for i, (feat, a, logp, val, r) in \
                            enumerate(e["steps"]):
                        if e["excluded"][i]:
                            continue
                        if key[0] == "w":
                            Xw.append(feat)
                            Aw.append(a)
                            Lw.append(logp)
                            Mw.append(e["masks"][i])
                            Vw_adv.append(advs[i])
                            Vw_ret.append(rets[i])
                        else:
                            Xb.append(feat)
                            Ab.append(a)
                            Lb.append(logp)
                            Vb_adv.append(advs[i])
                            Vb_ret.append(rets[i])

            def prep(X, A, L, ADV, RET):
                ADV = np.array(ADV, dtype=np.float32)
                ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
                return (np.array(X, dtype=np.float32),
                        np.array(A), np.array(L, dtype=np.float32),
                        ADV, np.array(RET, dtype=np.float32))

            batch_w = prep(Xw, Aw, Lw, Vw_adv, Vw_ret)
            batch_b = prep(Xb, Ab, Lb, Vb_adv, Vb_ret)
            Mw_arr = np.array(Mw, dtype=bool)

            t1 = time.time()
            info_w = _ppo_update_masked(wnet, opt_w, batch_w[0],
                                        batch_w[1], batch_w[2], Mw_arr,
                                        batch_w[3], batch_w[4], epochs,
                                        8192)
            info_b = _ppo_update_plain(bnet, opt_b, *batch_b, epochs,
                                       8192)
            upd_s = time.time() - t1

            # ---- snapshot + checkpoint + probe ----
            np_w = export_np_weights(wnet, bnet)
            snap_weights = np_w
            ckpt = os.path.join(HERE, "rl2_ckpt", f"iter{it}.npz")
            np.savez(ckpt, **np_w)
            t2 = time.time()
            if smoke:
                probe = []
                score = float("nan")
            else:
                probe = list(ex.map(_probe_game, probe_jobs(np_w)))
                sel = [g for g in probe if g["opp"] != "example"]
                score = float(np.mean([g["margin"] for g in sel]))
            probe_s = time.time() - t2
            pb = {}
            for g in probe:
                pb.setdefault(g["opp"], []).append(g)
            rec = {
                "iter": it, "rollout_s": round(roll_s, 1),
                "update_s": round(upd_s, 1), "probe_s":
                round(probe_s, 1),
                "games": len(results),
                "worker_samples": len(Xw), "base_samples": len(Xb),
                "train_margin": {k: round(float(np.mean(v)), 1)
                                 for k, v in margins_by_opp.items()},
                "probe_score": None if smoke else round(score, 1),
                "probe_margin": {k: round(float(np.mean(
                    [g["margin"] for g in v])), 1)
                    for k, v in pb.items()},
                "probe_example_suicides": [g["suicides"] for g in
                                           pb.get("example", [])],
                "probe_example_stranded": [g["stranded"] for g in
                                           pb.get("example", [])],
                "shield_stats": agg_stats,
                "w_ent": round(info_w.get("ent", 0.0), 3),
                "w_vf": round(info_w.get("vf", 0.0), 3),
            }
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if not smoke and (best_score is None or score > best_score):
                best_score, best_iter = score, it
                np.savez(os.path.join(HERE, "rl2_weights.npz"), **np_w)
    logf.close()
    print(f"done. best_iter={best_iter} best_score={best_score}")


if __name__ == "__main__":
    main()

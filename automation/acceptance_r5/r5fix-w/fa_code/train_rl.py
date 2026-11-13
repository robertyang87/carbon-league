"""train_rl -- self-play PPO, stage 1, for the Carbon League NN line.

Algorithm verdict (throughput first, measured by bench_rl.py):
  actor-vs-actor 0.78 s/game single-core (~6800 games/h on 4 procs),
  actor-vs-v0.5 2.38 s/game (~2200 games/h). Even after torch-forward
  and reward-bookkeeping overhead, a 96-game iteration costs ~1-2 min,
  so PPO's sample appetite is affordable -- NO downgrade to ES. This
  is PPO (clipped surrogate + GAE), warm-started from the BC weights.

Design (mirrors the official baseline's proven structure, see
shoulders.md S1):
  * Parameter-shared per-unit actors: worker MLP 124->192->192->{5
    logits, value}, base MLP 25->96->96->{3 logits, value}; feature
    normalization frozen at the BC mean/std so the warm start is exact
    for the policy heads; value heads are fresh.
  * Per-unit shaped rewards (baseline attribution, scaled /100):
      collector: deposit +cargo/100 (detected: alive, cargo -> 0 on an
        own base cell); cargo gain +min(d,50)/100; cargo loss d/100
        (enemy-tree absorption etc.); death -3.0
      planter:   + own trees' absorption/100 credited via trees_dict
        creator id each step; plant step -market_price/100; seize step
        -0.2; zero-income idle step -0.01
      base:      team dcash/100 (all income and spend flows through it)
      terminal:  clip(final margin/1000, -3, 3) added to every unit
        trajectory's last transition.
  * gamma 0.995, GAE lambda 0.95, clip 0.2, epochs 4, lr 5e-4,
    entropy 0.01, minibatch 8192 unit-samples.
  * Opponent pool per 96-game iteration: shadow 38 / self-snapshot 24
    / policy_v0_5 24 / example 10. juzi is NEVER in training (0.31
    s/step); it is eval-only.
  * Learner acts on the fast canonical path (bit-identical to the obs
    path -- test_rl_common.py); per-iteration eval uses NumpyActor
    with the shadow protocol (workers sampled T=1, bases argmax) on
    held-out seeds 3000/3001 vs shadow and v0.5.

Run: venv/bin/python train_rl.py            (full run)
     RL_SMOKE=1 venv/bin/python train_rl.py (1 iter x 8 games)
Outputs: rl_ckpt/iter{n}.npz, rl_train_log.jsonl, rl_weights.npz
(best checkpoint by eval-vs-v0.5 margin, tie-break vs-shadow).
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

import numpy as np  # noqa: E402

CFG = None  # set in main / worker init from rl_common


# ----------------------------------------------------------------------
# Torch nets
# ----------------------------------------------------------------------
def build_nets():
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

    bc = {k: v.astype(np.float32) for k, v in
          np.load(os.path.join(HERE, "shadow_weights.npz")).items()}
    from bc_features import WORKER_DIM, BASE_DIM
    wnet = Head(WORKER_DIM, 192, 5, bc, "w_")
    bnet = Head(BASE_DIM, 96, 3, bc, "b_")
    return wnet, bnet


def export_np_weights(wnet, bnet):
    """Torch nets -> numpy weights dict in shadow_weights.npz format."""
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
# Rollout worker (separate process)
# ----------------------------------------------------------------------
_W = {}


def _worker_init():
    import torch
    torch.set_num_threads(1)
    import rl_common
    _W["cfg"] = rl_common.CFG
    _W["bc_actor_weights"] = rl_common.load_weights_npz(
        os.path.join(HERE, "shadow_weights.npz"))
    _W["v05"] = None
    _W["example"] = None


def _plant_price(pre_trees, player):
    """Market plant price from pre-step sim trees {tid: (owner, age)}."""
    import rl_common
    cfg = rl_common.CFG
    mine = sum(1 for (o, a) in pre_trees.values() if o == player)
    if mine <= cfg["playerTreeProtectiveNumber"]:
        return float(cfg["plantCost"])
    alive = sum(1 for (o, a) in pre_trees.values()
                if a < cfg["treeLifespan"])
    return (cfg["plantCost"] + cfg["plantCostInflationRatio"]
            * (cfg["plantCostInflationBase"] ** alive))


def _rollout_game(wnet, bnet, job, snap_weights):
    """Play one game; return (trajectories, result).

    trajectories: {unit_key: [(feat, act, logp, val, rew), ...]} with
    unit_key ('w', wid) / ('b', bid) -- learner side only.
    """
    import torch
    from bc_features import featurize
    import rl_common
    from fastsim import FastCarbon

    cfg = _W["cfg"]
    seed, lp, opp = job["seed"], job["side"], job["opp"]
    sim = FastCarbon(cfg).reset(seed=seed)
    seq = rl_common.SeqState()

    # --- opponent setup ---
    opp_actor = None
    opp_fn = None
    if opp in ("shadow", "self"):
        w = _W["bc_actor_weights"] if opp == "shadow" else snap_weights
        opp_actor = rl_common.NumpyActor(w, seed=9000 + seed)
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

    for step in range(300):
        canon = rl_common.canonical_from_sim(sim, lp)
        (p, carbon, cash, bases, workers, wmask, trees, tmask, stp,
         my_wids, my_bids) = canon
        n_now = int(np.sum(wmask[p]))
        sq = seq.seq_for(n_now, stp)
        Wf, Bf = featurize(carbon, cash, bases, workers, wmask, trees,
                           tmask, p, stp, seq=sq)
        # pre-step snapshot for rewards
        pre_w = {wid: (w.player, w.pos, w.carbon, w.is_collector)
                 for wid, w in sim.workers.items()}
        pre_t = {tid: (t.player, t.age) for tid, t in sim.trees.items()}
        pre_cash = list(sim.cash)

        actions = {}
        step_units = []  # (unit_key, feat, act, logp, val)
        with torch.no_grad():
            if my_wids:
                x = torch.tensor(Wf[:len(my_wids)], dtype=torch.float32)
                logits, vals = wnet(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts_t = dist.sample()
                logp = dist.log_prob(acts_t)
                for j, wid in enumerate(my_wids):
                    a = int(acts_t[j])
                    step_units.append((("w", wid),
                                       Wf[j].astype(np.float32), a,
                                       float(logp[j]), float(vals[j])))
                    if a in MOVE:
                        actions[wid] = MOVE[a]
            if my_bids:
                x = torch.tensor(Bf[:len(my_bids)], dtype=torch.float32)
                logits, vals = bnet(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts_t = dist.sample()
                logp = dist.log_prob(acts_t)
                n_rec = 0
                for i, bid in enumerate(my_bids):
                    a = int(acts_t[i])
                    step_units.append((("b", bid),
                                       Bf[i].astype(np.float32), a,
                                       float(logp[i]), float(vals[i])))
                    if a in REC:
                        actions[bid] = REC[a]
                        n_rec += 1
            else:
                n_rec = 0
        seq.update(n_now, stp, n_rec)

        # opponent action
        if opp_actor is not None:
            oacts = opp_actor.act_canonical(
                rl_common.canonical_from_sim(sim, 1 - lp))
        else:
            oacts = opp_fn(sim.obs(1 - lp), cfg)
        both = [None, None]
        both[lp] = actions
        both[1 - lp] = oacts
        sim.step(both[0], both[1])

        # ---- rewards from post-step state ----
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
        for key, feat, a, logp, val in step_units:
            kind, uid = key
            r = 0.0
            if kind == "b":
                r = dcash / 100.0
            else:
                pre = pre_w.get(uid)
                post = post_w.get(uid)
                if post is None:
                    r = -3.0  # death
                elif pre is not None and pre[3]:  # collector
                    pre_cargo, post_cargo, post_pos = pre[2], post[2], \
                        post[1]
                    if pre_cargo > 0 and post_cargo == 0 \
                            and post_pos in own_base_pos:
                        r = pre_cargo / 100.0  # deposit credit
                    else:
                        d = post_cargo - pre_cargo
                        if d > 0:
                            r = min(d, 50.0) / 100.0
                        elif d < 0:
                            r = d / 100.0
                elif pre is not None:  # planter
                    inc = tree_credit.get(uid, 0.0)
                    r = inc / 100.0
                    if uid in planted:
                        r -= planted[uid] / 100.0
                    if uid in seized:
                        r -= 0.2
                    if inc == 0 and uid not in planted \
                            and uid not in seized:
                        r -= 0.01
            traj.setdefault(key, []).append((feat, a, logp, val, r))

    final = list(sim.cash)
    margin = final[lp] - final[1 - lp]
    term = float(np.clip(margin / 1000.0, -3.0, 3.0))
    for key, lst in traj.items():
        if lst:
            f, a, logp, val, r = lst[-1]
            lst[-1] = (f, a, logp, val, r + term)
    result = {"seed": seed, "side": lp, "opp": opp,
              "my_cash": round(final[lp], 2),
              "opp_cash": round(final[1 - lp], 2),
              "margin": round(margin, 2)}
    return traj, result


def _rollout_batch(task):
    state, snap_weights, jobs = task
    wnet, bnet = build_nets()
    import torch
    wnet.load_state_dict(state[0])
    bnet.load_state_dict(state[1])
    wnet.eval()
    bnet.eval()
    out = []
    for job in jobs:
        out.append(_rollout_game(wnet, bnet, job, snap_weights))
    return out


# ----------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------
def _gae(traj, gamma=0.995, lam=0.95):
    """traj: list of (feat, act, logp, val, rew) -> (adv, ret) arrays."""
    advs = np.zeros(len(traj), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(traj))):
        v = traj[t][3]
        v_next = traj[t + 1][3] if t + 1 < len(traj) else 0.0
        delta = traj[t][4] + gamma * v_next - v
        last = delta + gamma * lam * last
        advs[t] = last
    rets = advs + np.array([t[3] for t in traj], dtype=np.float32)
    return advs, rets


def _ppo_update(net, opt, X, A, LOGP, ADV, RET, epochs, mb, clip=0.2,
                ent_coef=0.01, vf_coef=0.5):
    import torch
    n = len(X)
    if n == 0:
        return {}
    xt = torch.tensor(X)
    at = torch.tensor(A, dtype=torch.long)
    lpt = torch.tensor(LOGP)
    advt = torch.tensor(ADV)
    rett = torch.tensor(RET)
    tot = {"pi": 0.0, "vf": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, mb):
            idx = perm[k:k + mb]
            logits, vals = net(xt[idx])
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


# ----------------------------------------------------------------------
# Eval during training (main process, NumpyActor protocol)
# ----------------------------------------------------------------------
def _eval_weights(np_weights, opp_kind, seeds=(3000, 3001)):
    import rl_common
    from fastsim import FastCarbon
    cfg = rl_common.CFG
    margins = []
    for seed in seeds:
        for side in (0, 1):
            sim = FastCarbon(cfg).reset(seed=seed)
            me = rl_common.NumpyActor(np_weights, seed=4242 + seed)
            if opp_kind == "shadow":
                opp = rl_common.NumpyActor(
                    _W_BC, seed=777 + seed)
                opp_fn = None
            else:
                opp = None
                opp_fn = rl_common.load_agent_fn(
                    os.path.join(HERE, "policy_v0_5.py"), "policy")
            for _ in range(300):
                acts = [None, None]
                for p in range(2):
                    if p == side:
                        acts[p] = me.act_canonical(
                            rl_common.canonical_from_sim(sim, p))
                    elif opp is not None:
                        acts[p] = opp.act_canonical(
                            rl_common.canonical_from_sim(sim, p))
                    else:
                        acts[p] = opp_fn(sim.obs(p), cfg)
                sim.step(acts[0], acts[1])
            st = sim.get_state()
            margins.append(st["cash"][side] - st["cash"][1 - side])
    return float(np.mean(margins)), margins


_W_BC = None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import torch
    torch.set_num_threads(2)
    import rl_common
    global _W_BC
    _W_BC = rl_common.load_weights_npz(
        os.path.join(HERE, "shadow_weights.npz"))

    smoke = bool(os.environ.get("RL_SMOKE"))
    n_iter = 1 if smoke else int(os.environ.get("RL_ITERS", "12"))
    games_per_iter = 8 if smoke else int(os.environ.get("RL_GAMES", "96"))
    epochs = 1 if smoke else 4

    wnet, bnet = build_nets()
    opt_w = torch.optim.Adam(wnet.parameters(), lr=5e-4)
    opt_b = torch.optim.Adam(bnet.parameters(), lr=5e-4)

    os.makedirs(os.path.join(HERE, "rl_ckpt"), exist_ok=True)
    logf = open(os.path.join(HERE, "rl_train_log.jsonl"), "a")
    snap_weights = _W_BC  # iteration 0 self-opponent = BC shadow weights
    best_score, best_iter = None, None

    # opponent mix template (fractions of games_per_iter)
    mix = [("shadow", 0.40), ("self", 0.25), ("v05", 0.25),
           ("example", 0.10)]

    with ProcessPoolExecutor(max_workers=4,
                             initializer=_worker_init) as ex:
        for it in range(n_iter):
            t0 = time.time()
            jobs = []
            counts = {}
            for name, frac in mix:
                counts[name] = int(round(games_per_iter * frac))
            counts["shadow"] += games_per_iter - sum(counts.values())
            idx = 0
            for name, cnt in counts.items():
                for _ in range(cnt):
                    jobs.append({"seed": 20000 + it * 10000 + idx * 7,
                                 "side": idx % 2, "opp": name})
                    idx += 1
            state = (wnet.state_dict(), bnet.state_dict())
            # chunk jobs over 4 workers
            chunks = [jobs[i::4] for i in range(4)]
            tasks = [(state, snap_weights, c) for c in chunks]
            results = []
            for batch in ex.map(_rollout_batch, tasks):
                results.extend(batch)
            roll_s = time.time() - t0

            # ---- assemble PPO batch ----
            Xw, Aw, Lw, Vw_adv, Vw_ret = [], [], [], [], []
            Xb, Ab, Lb, Vb_adv, Vb_ret = [], [], [], [], []
            margins_by_opp = {}
            for traj, res in results:
                margins_by_opp.setdefault(res["opp"], []).append(
                    res["margin"])
                for key, lst in traj.items():
                    advs, rets = _gae(lst)
                    for i, (feat, a, logp, val, r) in enumerate(lst):
                        if key[0] == "w":
                            Xw.append(feat)
                            Aw.append(a)
                            Lw.append(logp)
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

            t1 = time.time()
            info_w = _ppo_update(wnet, opt_w, *batch_w, epochs, 8192)
            info_b = _ppo_update(bnet, opt_b, *batch_b, epochs, 8192)
            upd_s = time.time() - t1

            # ---- snapshot + checkpoint + eval ----
            np_w = export_np_weights(wnet, bnet)
            snap_weights = np_w
            ckpt = os.path.join(HERE, "rl_ckpt", f"iter{it}.npz")
            np.savez(ckpt, **np_w)
            t2 = time.time()
            if smoke:
                ev_shadow = ev_v05 = (float("nan"), [])
            else:
                ev_shadow = _eval_weights(np_w, "shadow")
                ev_v05 = _eval_weights(np_w, "v05", seeds=(3000,))
            ev_s = time.time() - t2
            score = ev_v05[0] + 0.5 * ev_shadow[0]
            rec = {
                "iter": it, "rollout_s": round(roll_s, 1),
                "update_s": round(upd_s, 1), "eval_s": round(ev_s, 1),
                "games": len(results),
                "worker_samples": len(Xw), "base_samples": len(Xb),
                "train_margin": {k: round(float(np.mean(v)), 1)
                                 for k, v in margins_by_opp.items()},
                "eval_vs_shadow": round(ev_shadow[0], 1),
                "eval_vs_v05": round(ev_v05[0], 1),
                "w_ent": round(info_w.get("ent", 0.0), 3),
                "w_vf": round(info_w.get("vf", 0.0), 3),
                "b_ent": round(info_b.get("ent", 0.0), 3),
            }
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if not smoke and (best_score is None or score > best_score):
                best_score, best_iter = score, it
                np.savez(os.path.join(HERE, "rl_weights.npz"), **np_w)
    logf.close()
    print(f"done. best_iter={best_iter} best_score={best_score}")


if __name__ == "__main__":
    main()

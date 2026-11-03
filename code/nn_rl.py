"""nn_rl -- PPO fine-tuning of the BC policy by self-play in FastCarbon.

Rollouts (worker processes, NumPy policy, 1 thread each): our side samples
from softmax(logits / T) with the same own-collision guard as nn_agent;
recruit actions are sampled and kept only if legal. Opponent = a pool
file (rule bots) or a frozen snapshot of ourselves.
Reward (team, zero-sum): r_t = d(cash_me - cash_opp)/10000 each step,
plus +0.3/-0.3 at the end for win/loss. Team advantage by GAE over
V_t = mean unit value; every unit sample at step t gets A_t, value
target = team return G_t. PPO clip 0.2, entropy bonus, small LR.

Usage: python3 nn_rl.py INIT.npz OUTDIR [minutes]
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nnfeat as F  # noqa: E402

GAMMA, LAM = 0.99, 0.95
GAMES_PER_ITER = int(os.environ.get("RL_GAMES", "24"))
WORKERS = int(os.environ.get("RL_WORKERS", "4"))
POOL = [("nocamp.py", 0.5), ("exp/spar_tf2.py", 0.15), ("exp/spar_tf3.py", 0.15), ("SELF", 0.2)]
TEMP = float(os.environ.get("RL_TEMP", "0.5"))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def rollout(task):
    sd, opp, seed, side, snap_sd = task
    import eval_v53
    from fastsim import FastCarbon
    import nn_agent as A
    random.seed(eval_v53.rng_seed(seed, side))
    rng = np.random.default_rng(seed * 7 + side)
    pol = A.Policy(sd)
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    if opp == "SELF":
        opp_fn = A.make_agent(snap_sd)
    else:
        opp_fn = eval_v53.load_agent_fn(os.path.join(HERE, opp))
    tr = F.Tracker(side)
    cap = 10
    X, Sc, K, Ac, LP, T_ = [], [], [], [], [], []
    rew = []
    prev = None
    for step in range(1, 301):
        o_me = sim.obs(side)
        o_op = sim.obs(1 - side)
        tr.update(o_me)
        G, scal, info = tr.global_planes(o_me)
        mine = o_me["players"][side]
        units, xs, ss = [], [], []
        for wid, w in mine[2].items():
            k = 0 if w[2] == "COLLECTOR" else 1
            x, s = F.unit_inputs(G, scal, w[0], k, w[1])
            units.append((wid, k, w[0])); xs.append(x); ss.append(s)
        for bid, b in mine[1].items():
            x, s = F.unit_inputs(G, scal, b, 2, 0.0)
            units.append((bid, 2, b)); xs.append(x); ss.append(s)
        acts = {}
        t_idx = step - 1
        if units:
            xa = np.stack(xs).astype(np.float32); sa = np.stack(ss).astype(np.float32)
            lg = pol.forward(xa, sa)
            claimed = set()
            order = []
            for i, (uid, k, pos) in enumerate(units):
                if k == 2:
                    continue
                pr = _softmax(lg[i, 0:5] / TEMP)
                a = int(rng.choice(5, p=pr))
                order.append((-pr[a], i, a, pr))
            order.sort(key=lambda t: t[0])
            for _, i, a, pr in order:
                uid, k, pos = units[i]
                cand = [a] + [int(b) for b in np.argsort(-pr) if b != a]
                for b in cand:
                    tgt = F.nb(pos, F.MOVES[b])
                    if tgt not in claimed:
                        claimed.add(tgt)
                        if b != 0:
                            acts[uid] = F.MOVES[b]
                        X.append(xs[i]); Sc.append(ss[i]); K.append(k); Ac.append(b)
                        LP.append(float(np.log(max(pr[b], 1e-8)))); T_.append(t_idx)
                        break
            # recruitment: same rule gate as nn_agent (not trained by PPO)
            cash = info["cash"]; nw = len(mine[2])
            n_pl = sum(1 for w in mine[2].values() if w[2] != "COLLECTOR")
            for i, (uid, k, pos) in enumerate(units):
                if k != 2:
                    continue
                a = A.recruit_choice(lg[i, 5:8], cash, nw, cap, n_pl, info["tree_price"],
                                     o_me["step"], pos in claimed)
                if a != 0:
                    acts[uid] = F.BASE_ACTS[a]; cash -= 30; nw += 1; claimed.add(pos)
                    n_pl += (a == 2)
        a_op = opp_fn(o_op, eval_v53.CFG)
        sim.step(*((acts, a_op) if side == 0 else (a_op, acts)))
        stt = sim.get_state()
        d = stt["cash"][side] - stt["cash"][1 - side]
        rew.append((d - (prev if prev is not None else 0.0)) / 10000.0)
        prev = d
    final = prev
    rew[-1] += 0.3 if final > 0 else (-0.3 if final < 0 else 0.0)
    X = np.clip(np.stack(X) * 127.0, 0, 255).astype(np.uint8) if X else np.zeros((0, F.C_PLANES, 15, 15), np.uint8)
    return {"X": X, "Sc": np.array(Sc, np.float32), "K": np.array(K, np.int64), "A": np.array(Ac, np.int64),
            "LP": np.array(LP, np.float32), "T": np.array(T_, np.int64), "R": np.array(rew, np.float32),
            "margin": float(final), "opp": opp, "seed": seed, "side": side}


def main(init, outdir, minutes=60):
    import torch
    import torch.nn.functional as Fn
    import nn_train_bc as TB
    torch.set_num_threads(4)
    os.makedirs(outdir, exist_ok=True)
    net = TB.Net()
    w = np.load(init)
    net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    opt = torch.optim.Adam(net.parameters(), lr=float(os.environ.get("RL_LR", "1e-4")))
    log = open(os.path.join(outdir, "rl_log.jsonl"), "a")
    snaps = [{k: v.detach().numpy().copy() for k, v in net.state_dict().items()}]
    deadline = time.time() + 60 * minutes
    it = 0
    seed = int(os.environ.get("RL_SEED0", "500000"))
    while time.time() < deadline:
        it += 1
        sd = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        names, probs = zip(*POOL)
        tasks = []
        for g in range(GAMES_PER_ITER):
            opp = random.choices(names, probs)[0]
            tasks.append((sd, opp, seed, g % 2, random.choice(snaps)))
            seed += 1
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(rollout, tasks))
        t_roll = time.time() - t0
        # value estimates per sample, team baseline per step
        Xs, Ss, Ks, As, LPs, ADV, RET = [], [], [], [], [], [], []
        net.eval()
        with torch.no_grad():
            for r in res:
                if len(r["A"]) == 0:
                    continue
                xb = torch.from_numpy(r["X"]).float() / 127.0
                _, v = net(xb, torch.from_numpy(r["Sc"]))
                v = v.numpy()
                Tn = len(r["R"])
                Vt = np.zeros(Tn + 1, np.float32)
                cnt = np.zeros(Tn, np.float32)
                np.add.at(Vt, r["T"], v); np.add.at(cnt, r["T"], 1)
                Vt[:Tn] = np.where(cnt > 0, Vt[:Tn] / np.maximum(cnt, 1), 0.0)
                adv = np.zeros(Tn, np.float32); last = 0.0
                for t in range(Tn - 1, -1, -1):
                    delta = r["R"][t] + GAMMA * Vt[t + 1] - Vt[t]
                    last = delta + GAMMA * LAM * last
                    adv[t] = last
                ret = adv + Vt[:Tn]
                Xs.append(r["X"]); Ss.append(r["Sc"]); Ks.append(r["K"]); As.append(r["A"]); LPs.append(r["LP"])
                ADV.append(adv[r["T"]]); RET.append(ret[r["T"]])
        X = torch.from_numpy(np.concatenate(Xs)); S = torch.from_numpy(np.concatenate(Ss))
        K = torch.from_numpy(np.concatenate(Ks)); A = torch.from_numpy(np.concatenate(As))
        LP0 = torch.from_numpy(np.concatenate(LPs)); ADVt = torch.from_numpy(np.concatenate(ADV))
        RETt = torch.from_numpy(np.concatenate(RET))
        ADVt = (ADVt - ADVt.mean()) / (ADVt.std() + 1e-6)
        net.train()
        n = len(A); bs = 1024
        t1 = time.time()
        stats = []
        for ep in range(2):
            perm = torch.randperm(n)
            for b in range(0, n, bs):
                ii = perm[b:b + bs]
                logits, v = net(X[ii].float() / 127.0, S[ii])
                ml = TB.masked_logits(logits, K[ii])
                # temperature-consistent log-probs over the kind's action subset
                lp_all = Fn.log_softmax(ml / TEMP, dim=1)
                lp = lp_all.gather(1, A[ii][:, None]).squeeze(1)
                ratio = torch.exp(lp - LP0[ii])
                a_ = ADVt[ii]
                pl = -torch.min(ratio * a_, torch.clamp(ratio, 0.8, 1.2) * a_).mean()
                vl = Fn.mse_loss(v, torch.clamp(RETt[ii], -1, 1))
                p_all = lp_all.exp()
                ent = -(p_all * torch.where(p_all > 0, lp_all, torch.zeros_like(lp_all))).sum(1).mean()
                loss = pl + 0.5 * vl - 0.01 * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((pl.item(), vl.item(), ent.item(), (ratio - 1).abs().mean().item()))
        st = np.mean(stats, axis=0)
        by = {}
        for r in res:
            by.setdefault(r["opp"], []).append(r["margin"])
        rec = {"it": it, "games": len(res), "samples": n, "roll_s": round(t_roll), "upd_s": round(time.time() - t1),
               "pl": round(st[0], 4), "vl": round(st[1], 4), "ent": round(st[2], 3), "dratio": round(st[3], 4),
               "margin": {k: [round(float(np.mean(v))), len(v)] for k, v in by.items()},
               "winrate": round(float(np.mean([r["margin"] > 0 for r in res])), 3)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if it % 5 == 0:
            snaps.append({k: v.detach().numpy().copy() for k, v in net.state_dict().items()})
            snaps = snaps[-6:]
            TB.export(net, os.path.join(outdir, f"rl_it{it}.npz"), {"it": it})
        TB.export(net, os.path.join(outdir, "rl_last.npz"), {"it": it})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 60)

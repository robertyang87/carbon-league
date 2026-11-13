"""cnet2_search_recruit_rank -- paired ranking test for recruit search (mfhr go/no-go).

Positions: ladder replays in data/v10z (1C-Zero-v10's losses: the WINNER's side; v10's wins over our v12/v13: OUR side)
at steps ~150/200/250 where a base recruit is affordable.  For each position the first base slot is branched
{NONE, RECCOLLECTOR, RECPLANTER} (rest of the joint re-decoded by cnet2 b128 style 1 with that slot forced).
  value head  : roll each branch 10 steps greedily (b128 style 1 both sides), cnet2 value from our seat
  material    : same 10-step positions, cash + 0.9 cargo + 30/worker + tree life (ours - theirs)
  policy head : the root base-slot probability of the branch's action
  truth       : mean final margin over playouts to the end vs TRAIN-pool opponent nets (v12=b128 s1, b128v3 s1,
                b128 style 0, b128 style 3) x 2 sampling seeds (both sides sample at temperature T), common random
                numbers across sibling branches.
Output: pairwise ordering agreement of each predictor with the truth, with counts.
  python3 cnet2_search_recruit_rank.py [T]
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402
import cnet2_search as S  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import eval_v53  # noqa: E402

W = os.path.join(HERE, "..", "data", "weights")
TEMP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
GAMES = {  # id: side to evaluate (0/1 index into steps' player order)
}


def pick_games():
    M = json.load(open(os.path.join(HERE, "..", "data", "v10z_matches.json")))
    out = []
    for m in M:
        a, b = m["name_a"], m["name_b"]
        oa = m["order"]                      # order[seat] -> 'a'/'b'
        seat_of = {oa[0]: 0, oa[1]: 1}
        if m["winner"] == "b" and a == "1C-Zero-v10":
            out.append((m["id"], seat_of["b"], "v10_loss:" + b))
        elif m["winner"] == "a" and a == "1C-Zero-v10" and b.startswith("claude-opus55-cloud-"):
            out.append((m["id"], seat_of["b"], "our_loss:" + b))
    return out


class NetPlayer:
    """cnet2 net from one seat with its own Feat; greedy or sampled."""

    def __init__(self, pol, teacher, seat):
        self.pol, self.t, self.f = pol, teacher, F.Feat(seat)

    def __call__(self, obs, rng=None, temp=1.0, forced=None):
        x = self.f.observe(obs)
        if forced is None:
            y, lg, _ = self.pol.act(x, self.f.aux, teacher=self.t, sample_rng=rng, temp=temp)
        else:
            fz = forced
            hook = (lambda s, lg, pf, a: int(fz[s]) if fz[s] >= 0 else None)
            y, lg, _ = self.pol.act(x, self.f.aux, teacher=self.t, sample_rng=rng, temp=temp, hook=hook)
        self.last = (x, lg, y)
        return self.f.to_actions(y)


def copy_player(p):
    q = NetPlayer.__new__(NetPlayer)
    q.pol, q.t = p.pol, p.t
    q.f = S.feat_copy(p.f)
    q.f.aux = p.f.aux
    return q


def mk_sim(full):
    sim = FastCarbon(eval_v53.CFG).load_observation(full)
    sim._tid_counter = [c + 500 for c in sim._tid_counter]     # seized trees keep foreign ids
    return sim


def material(sim, me, left):
    m = [float(sim.cash[0]), float(sim.cash[1])]
    for w in sim.workers.values():
        m[w.player] += 0.9 * w.carbon + 30.0 * min(1.0, left / 100.0)
    for t in sim.trees.values():
        m[t.player] += 2.0 * max(0, min(50 - t.age, left))
    return m[me] - m[1 - me]


def main():
    b128 = NP.Policy(S.load_weights(os.path.join(W, "cnet2_b128_q8.npz")))
    v3 = NP.Policy(S.load_weights(os.path.join(W, "cnet2_b128v3_q8.npz")))
    OPPS = [("v12", b128, 1), ("b128v3", v3, 1), ("t0", b128, 0), ("t3", b128, 3)]
    rows = []
    prev = os.environ.get("RESUME_LOG")
    done = {}
    if prev and os.path.exists(prev):
        for l in open(prev):
            if l.startswith('{"game"'):
                r = json.loads(l); done[(r["game"], r["step"] // 50)] = r
    for gid, seat, tag in pick_games():
        d = json.load(gzip.open(os.path.join(HERE, "..", "data", "v10z", gid + "-1.json.gz")))
        steps = d["steps"]
        for t_target in (150, 200, 250):
            key = (gid, t_target // 50)
            if key in done:
                rows.append(done[key]); print(json.dumps(done[key]), flush=True); continue
            # warm Feats on the replay up to the decision step (both seats, all candidate opponents)
            me_p = NetPlayer(b128, 1, seat)
            opp_ps = {nm: NetPlayer(pol, tt, 1 - seat) for nm, pol, tt in OPPS}
            opp_g = NetPlayer(b128, 1, 1 - seat)
            chosen = None
            for t in range(len(steps) - 1):
                o0 = steps[t][0]["observation"]
                if o0["step"] > t_target + 15:
                    break
                ob = {"step": o0["step"], "players": o0["players"], "player": seat, "carbon": steps[t][seat]["observation"]["carbon"]}
                oo = {"step": o0["step"], "players": o0["players"], "player": 1 - seat, "carbon": steps[t][1 - seat]["observation"]["carbon"]}
                cash = o0["players"][seat][0]; nw = len(o0["players"][seat][2]); nb = len(o0["players"][seat][1])
                if o0["step"] >= t_target and cash >= 30 and nw < 10 and nb > 0:
                    chosen = (t, o0, ob, oo)
                    break
                me_p(ob); opp_g(oo)
                for p in opp_ps.values():
                    p(oo)
            if chosen is None:
                continue
            t, o0, ob, oo = chosen
            full = {"step": o0["step"], "players": o0["players"], "trees": o0.get("trees") or {},
                    "full_carbon": o0["full_carbon"], "carbon": o0["carbon"], "remainingOverageTime": 60}
            # root: policy-head probabilities of the first base slot, branch forcings
            root = copy_player(me_p)
            x = root.f.observe(ob)
            y0, lg0, _ = b128.act(x, root.f.aux, teacher=1)
            bslot = [s for s in (10, 11) if x["tok_mask"][s]][0]
            z = lg0[bslot].astype(np.float64); z = z[:3]; pz = np.exp(z - z.max()); pz /= pz.sum()
            res = {"game": gid, "tag": tag, "step": o0["step"], "policy": [round(float(v), 3) for v in pz], "branches": []}
            for a_b in (0, 1, 2):
                fz = np.full(F.OWN, -1, np.int64); fz[:bslot] = y0[:bslot]; fz[bslot] = a_b
                # (1) 10-step greedy rollout -> value head + material
                sim = mk_sim(full)
                pm, pg = copy_player(me_p), copy_player(opp_g)
                am = pm(ob, forced=fz); ao = pg(oo)
                sim.step(*((am, ao) if seat == 0 else (ao, am)))
                for h in range(9):
                    if sim.step_count >= 299:
                        break
                    am = pm(sim.obs(seat)); ao = pg(sim.obs(1 - seat))
                    sim.step(*((am, ao) if seat == 0 else (ao, am)))
                xx = pm.f.observe(sim.obs(seat))
                _, _, val = b128.act(xx, pm.f.aux, teacher=1)
                left = 299 - sim.step_count
                br = {"a": a_b, "value10": round(float(val), 4), "mat10": round(material(sim, seat, left), 1), "truth": []}
                # (2) playouts to the end: train-pool opponents x 2 sampling seeds, both sides sample at TEMP
                for nm, pol, tt in OPPS:
                    for rs in (0, 1):
                        rng_m = np.random.default_rng(1000 + rs); rng_o = np.random.default_rng(2000 + rs)
                        sim = mk_sim(full)
                        pm, po = copy_player(me_p), copy_player(opp_ps[nm])
                        am = pm(ob, rng=rng_m, temp=TEMP, forced=fz); ao = po(oo, rng=rng_o, temp=TEMP)
                        sim.step(*((am, ao) if seat == 0 else (ao, am)))
                        while sim.step_count < 300:
                            am = pm(sim.obs(seat), rng=rng_m, temp=TEMP); ao = po(sim.obs(1 - seat), rng=rng_o, temp=TEMP)
                            sim.step(*((am, ao) if seat == 0 else (ao, am)))
                        br["truth"].append(round(sim.cash[seat] - sim.cash[1 - seat]))
                br["truth_mean"] = round(float(np.mean(br["truth"])))
                res["branches"].append(br)
            rows.append(res)
            print(json.dumps(res), flush=True)
    # pairwise agreement
    agg = {k: [0, 0] for k in ("value", "material", "policy")}
    for r in rows:
        B = r["branches"]
        for i in range(3):
            for j in range(i + 1, 3):
                dt = B[i]["truth_mean"] - B[j]["truth_mean"]
                if dt == 0:
                    continue
                for k, a_, b_ in (("value", B[i]["value10"], B[j]["value10"]), ("material", B[i]["mat10"], B[j]["mat10"]),
                                  ("policy", r["policy"][B[i]["a"]], r["policy"][B[j]["a"]])):
                    agg[k][1] += 1
                    agg[k][0] += int(np.sign(a_ - b_) == np.sign(dt))
    print(json.dumps({"agreement": {k: {"agree": v[0], "pairs": v[1], "frac": round(v[0] / max(v[1], 1), 3)} for k, v in agg.items()},
                      "positions": len(rows), "temp": TEMP}))


if __name__ == "__main__":
    main()

"""dagger_gen -- narrow DAgger data (plan A): planter labels from a labeler net on OUR OWN game states.

  python3 dagger_gen.py OUT_DIR OURS.py LABELER.npz LABELER_SLOT SEED0 N [WORKERS]

Plays OURS (a pure-net wrapper) against a training pool (never an exam opponent). Every step records our side in
the cnet2_build sample format; the label of every own PLANTER slot is the labeler's greedy move, teacher-forced on
our actual joint action (earlier slots = what we really played). Collector and base slots keep our own move but get
unit weight 0 (uw), so training on these shards supervises the planter head only; the teacher id is LABELER_SLOT.
One shard per game: OUT_DIR/shards/dg_<seed>.npz (fields of cnet2_build.FIELDS + uw); games.jsonl lists them.
"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("BC_W", "64")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

# training pool only: our own nets (no stand-ins, no v13, no WB/1C sources); Master-line-like styles weighted up
UW_CHANGED = float(os.environ.get("DG_UW_CHANGED", "3.0"))
POOL = ["exp/cnet2_v6t5g.py", "exp/cnet2_v6t5g.py", "exp/cnet2_v5t5g.py", "exp/cnet2_v7s57_240.py",
        "exp/cnet2_v9s57_240.py", "exp/cnet2_style_t0.py", "exp/submission_cnet2_v15.py"]


if os.environ.get("DG_POOL"):              # override, comma separated (still: training-pool nets only, never exams)
    POOL = os.environ["DG_POOL"].split(",")


def play(task):
    import numpy as np
    import eval_v53
    import cnet2_feat as F
    import cnet2_np as NP
    from cnet2_agent import load_weights
    from cnet2_build import FIELDS
    from fastsim import FastCarbon
    out, ours, lab_w, lab_slot, seed = task
    opp = POOL[seed % len(POOL)]; side = (seed // len(POOL)) % 2
    random.seed(eval_v53.rng_seed(seed, side))
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    fa = eval_v53.load_agent_fn(os.path.join(HERE, ours)); fb = eval_v53.load_agent_fn(os.path.join(HERE, opp))
    lab = NP.Policy(load_weights(lab_w))
    feat = F.Feat(side)
    rows = {k: [] for k in FIELDS}; rows["uw"] = []
    nchg = npl = 0
    for step in range(1, 301):
        o_me = sim.obs(side)
        x = feat.observe(o_me)
        a_me = fa(o_me, eval_v53.CFG)
        a_op = fb(sim.obs(1 - side), eval_v53.CFG)
        y = feat.labels(a_me)
        _, lg, _ = lab.act(x, feat.aux, teacher=lab_slot, forced=y)
        ym = y.copy(); uw = np.zeros(F.OWN, np.float32)
        for s in range(10):
            if x["tok_mask"][s] and int(x["tok_kind"][s]) == 1 and y[s] >= 0:
                t = int(lg[s].argmax()); npl += 1; nchg += t != int(y[s])
                uw[s] = UW_CHANGED if t != int(y[s]) else 1.0      # gradient goes where we disagree with the labeler
                ym[s] = t
        pf = F.Prefix(x, feat.aux)
        uf = np.zeros((F.OWN, F.UF), np.float32); mf = np.zeros((F.OWN, 5, F.MF), np.float32); legal = np.zeros((F.OWN, 5), bool)
        for s in range(F.OWN):
            if not x["tok_mask"][s]:
                continue
            uf[s], mf[s], legal[s] = pf.feats(s)
            if not legal[s, ym[s]]:
                ym[s] = 0
            pf.advance(s, int(y[s]) if y[s] >= 0 else int(ym[s]))       # prefix = what we ACTUALLY played (the labeler was teacher-forced on it too)
        order = F.CANON if side else F.GRID
        for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move"):
            rows[k].append(x[k])
        rows["uf"].append(uf); rows["mf"].append(mf); rows["legal"].append(legal); rows["y"].append(ym)
        rows["carbon"].append(F.q8(np.asarray(sim.carbon, np.float64)[order] / 100.0))
        rows["teacher"].append(lab_slot); rows["gid"].append(0); rows["step"].append(step - 1); rows["uw"].append(uw)
        sim.step(*((a_me, a_op) if side == 0 else (a_op, a_me)))
    st = sim.get_state(); mg = float(st["cash"][side] - st["cash"][1 - side])
    rows["value"] = [np.tanh(mg / 3000.0)] * len(rows["y"])
    arr = {k: np.asarray(v).astype(FIELDS[k][0]) for k, v in rows.items() if k in FIELDS}
    arr["uw"] = np.asarray(rows["uw"], np.float16)
    fp = os.path.join(out, "shards", f"dg_{seed}.npz"); np.savez(fp, **arr)
    return {"seed": seed, "opp": opp, "side": side, "margin": mg, "n": len(rows["y"]), "planter": npl, "changed": nchg, "shard": fp}


def main():
    out, ours, lab_w, lab_slot, s0, n = sys.argv[1:7]
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    log = open(os.path.join(out, "games.jsonl"), "a")
    tasks = [(out, ours, lab_w, int(lab_slot), s) for s in range(int(s0), int(s0) + int(n))]
    P = C = 0
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(play, tasks):
            log.write(json.dumps(r) + "\n"); log.flush(); P += r["planter"]; C += r["changed"]
    print(json.dumps({"games": len(tasks), "planter_decisions": P, "relabelled": C, "rate": round(C / max(1, P), 3)}))


if __name__ == "__main__":
    main()

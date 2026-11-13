"""exit_gen -- expert-iteration data generator (round 1): mfxn (clock-free cnet2_search, 60 sims, narrowed
bankruptcy guard) plays vs a diverse opponent pool; for the search side every step is recorded in the
cnet2_build sample format (features + the PLAYED, search-modified joint action under teacher forcing) plus
  searched (bool), changed (bool: search or guard replaced the net's greedy action), q [5] (root mean leaf value
  per move of the searched unit, nan = not evaluated), pi [OWN,5] (softmax(q/30) at pi_slot, zeros elsewhere),
  pi_slot (-1 if none), y_net (the net's own greedy labels).
value = tanh(final margin / 3000); teacher = EXIT_TEACHER (7, an unused style slot).
One shard per game: data/exit1/shards/<seed>_<opp>_s<side>.npz; one line per game in data/exit1/games.jsonl.
In self-play (opp 'mfxn') both seats are recorded.
  python3 exit_gen.py SEED0 N [WORKERS]      (opponent = pool[seed % len(pool)], seat = seed // len(pool) % 2)
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "data", "exit1")
EXIT_TEACHER = 7
Q_TEMP = 30.0      # material points
# TRAIN pool only (held-out test opponents v13, b160, opp_1cz_v5 are never used here); b128v3 weighted x3
POOL = ["exp/submission_cnet2_v12.py", "exp/cnet2_b128v3_t1_packed.py", "exp/cnet2_style_t0.py",
        "exp/cnet2_b128v3_t1_packed.py", "exp/cnet2_style_t2.py", "exp/cnet2_style_t3.py",
        "exp/cnet2_b128v3_t1_packed.py", "exp/cnet2_style_t4.py", "self"]
TCFG = os.environ.get("EXIT_CFG", "mfxn")        # search teacher: mfxn | mfh
MFH_EXTRA = {'SPLIT': True, 'PREY_MIN': 50.0, 'MAX_KEY': 2, 'MAX_BR': 9, 'ROOT_S': 7}
MFXN = {'MODE': 'mcts', 'CLOCK': 'cpu', 'BUDGET': 99.0, 'HARD': 99.0, 'START_DL': None, 'ABORT_DL': None, 'ROOT_S': 12,
        'MINC': 20.0, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True,
        'GUARD': True, 'GUARD_NARROW': True}
OPEN_T = float(os.environ.get("EXIT_OPEN_T", "0.7")); OPEN_STEPS = 50
EXTRA = {"sampled": (np.bool_, ()), "searched": (np.bool_, ()), "changed": (np.bool_, ()), "pi": (np.float16, (12, 5)), "pi_slot": (np.int8, ()),
         "y_net": (np.int8, (12,)), "q": (np.float16, (5,))}


def opp_name(path):
    return "self" if path == "self" else os.path.basename(path).replace(".py", "").replace("submission_", "")


NETOPP = {"exp/submission_cnet2_v12.py": ("cnet2_b128_q8.npz", 1), "exp/cnet2_b128v3_t1_packed.py": ("cnet2_b128v3_q8.npz", 1),
          "exp/cnet2_style_t0.py": ("cnet2_b128_q8.npz", 0), "exp/cnet2_style_t2.py": ("cnet2_b128_q8.npz", 2),
          "exp/cnet2_style_t3.py": ("cnet2_b128_q8.npz", 3), "exp/cnet2_style_t4.py": ("cnet2_b128_q8.npz", 4)}


class NetOpp:
    """cnet2 opponent (same weights/teacher as the pool file) that samples at OPEN_T in the opening, greedy after."""

    def __init__(self, path, seed):
        import cnet2_np as NP, cnet2_feat as F, cnet2_search as S
        wf, self.t = NETOPP[path]
        self.pol = NP.Policy(S.load_weights(os.path.join(HERE, "..", "data", "weights", wf)))
        self.F = F; self.f = None; self.rng = np.random.default_rng(seed * 7 + 3)

    def __call__(self, obs, cfg=None):
        if self.f is None:
            self.f = self.F.Feat(obs["player"])
        x = self.f.observe(obs)
        if OPEN_T and obs["step"] < OPEN_STEPS:
            y, _, _ = self.pol.act(x, self.f.aux, teacher=self.t, sample_rng=self.rng, temp=OPEN_T)
        else:
            y, _, _ = self.pol.act(x, self.f.aux, teacher=self.t)
        return self.f.to_actions(y)


def make_mfxn(seed=0):
    import cnet2_search as S
    W = os.path.join(HERE, "..", "data", "weights")
    conf = dict(MFXN); conf.update(OPEN_T=OPEN_T, OPEN_STEPS=OPEN_STEPS, OPEN_SEED=seed)
    if TCFG == "mfh":
        conf.update(MFH_EXTRA)
    return S.make_search_agent(conf, weights=os.path.join(W, "cnet2_b128_q8.npz"),
                               rweights=os.path.join(W, "cnet2_s64_q8.npz"))


def record(ag, side, sim, rows):
    import cnet2_feat as F
    st = ag.state
    f = st["f"]; x = st["x"]; aux = f.aux
    y_net = np.asarray(st["y_net"], np.int64)
    act = st["acts_out"]
    y = f.labels(act)
    pf = F.Prefix(x, aux)
    uf = np.zeros((F.OWN, F.UF), np.float32); mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    for s in range(F.OWN):
        if not x["tok_mask"][s]:
            continue
        uf[s], mf[s], legal[s] = pf.feats(s)
        if not legal[s, y[s]]:
            y[s] = 0
        pf.advance(s, int(y[s]))
    order = F.CANON if side else F.GRID
    full = np.asarray(sim.carbon, np.float64)
    rec = st.get("rec")
    pi = np.zeros((F.OWN, 5), np.float32); ps = -1; q = np.full(5, np.nan, np.float32)
    if rec is not None and np.isfinite(rec["q"]).any():
        # soft target from root values (visits are flat at the per-move cap): softmax((Q - max) / Q_TEMP)
        ps = rec["slot"]; q = rec["q"].astype(np.float32)
        z = np.where(np.isfinite(q), (q - np.nanmax(q)) / Q_TEMP, -np.inf)
        e = np.exp(z); pi[ps] = e / e.sum()
    r = rows
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move"):
        r[k].append(x[k])
    r["uf"].append(uf); r["mf"].append(mf); r["legal"].append(legal); r["y"].append(y)
    r["carbon"].append(F.q8(full[order] / 100.0))
    r["teacher"].append(EXIT_TEACHER)
    r["sampled"].append(bool(st.get("sampled"))); r["searched"].append(rec is not None); r["changed"].append(bool((y != y_net)[x["tok_mask"][:F.OWN]].any()))
    r["pi"].append(pi); r["pi_slot"].append(ps); r["y_net"].append(y_net); r["q"].append(q)


def play(task):
    seed, opp = task
    import eval_v53
    import cnet2_feat as F
    from cnet2_build import FIELDS
    from fastsim import FastCarbon
    t0 = time.time()
    npool = len(POOL)
    side = (seed // npool) % 2
    try:
        A = make_mfxn(seed)
        agents = [None, None]; rec_sides = [side]
        agents[side] = A
        if opp == "self":
            B = make_mfxn(seed + 99991); agents[1 - side] = B; rec_sides = [0, 1]
        else:
            ob = NetOpp(opp, seed) if opp in NETOPP else eval_v53.load_agent_fn(opp); agents[1 - side] = ob
        sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
        rows = {p: {k: [] for k in list(FIELDS) + list(EXTRA)} for p in rec_sides}
        for t in range(300):
            o = [sim.obs(0), sim.obs(1)]
            acts = [None, None]
            for p in (0, 1):
                a = agents[p](o[p], eval_v53.CFG)
                if p in rec_sides:
                    st = agents[p].state
                    st["acts_out"] = a
                    # net's own greedy choice (before guard/search): stored by the agent as lg0 argmax path
                    st["y_net"] = st.get("y0", np.full(F.OWN, -1))
                    record(agents[p], p, sim, rows[p])
                acts[p] = a
            sim.step(acts[0], acts[1])
        cash = [float(sim.cash[0]), float(sim.cash[1])]
        out = []
        for p in rec_sides:
            mg = cash[p] - cash[1 - p]
            R = rows[p]
            n = len(R["y"])
            R["value"] = [np.tanh(mg / 3000.0)] * n
            R["gid"] = [0] * n
            R.pop("step", None); R["step"] = list(range(n))
            arr = {k: np.asarray(v).astype((FIELDS.get(k) or EXTRA[k])[0]) for k, v in R.items()}
            fn = os.path.join(OUT, "shards_" + TCFG, "%d_%s_s%d.npz" % (seed, opp_name(opp), p))
            np.savez(fn, **arr)
            stt = agents[p].stats
            out.append({"tcfg": TCFG, "open_t": OPEN_T, "seed": seed, "opp": opp_name(opp), "side": p, "margin": round(mg), "cash": [round(cash[p]), round(cash[1 - p])],
                        "n": n, "searched": int(arr["searched"].sum()), "changed": int(arr["changed"].sum()),
                        "sims": stt.get("sims", 0), "lost_ours": len(stt.get("lost_ours", [])), "lost_ours_cargo": round(sum(stt.get("lost_ours", []))),
                        "lost_theirs": len(stt.get("lost_theirs", [])), "lost_theirs_cargo": round(sum(stt.get("lost_theirs", []))),
                        "guard": stt.get("guard_moves", 0) + stt.get("guard_recruit", 0), "errors": stt["errors"],
                        "shard": os.path.basename(fn), "s": round(time.time() - t0)})
        return out
    except Exception as e:
        import traceback
        return [{"seed": seed, "opp": opp_name(opp), "error": repr(e)[:300], "tb": traceback.format_exc()[-600:]}]


def main(s0, n, workers):
    os.makedirs(os.path.join(OUT, "shards_" + TCFG), exist_ok=True)
    tasks = [(s, POOL[s % len(POOL)]) for s in range(s0, s0 + n)]
    done = {l.split('"shard": "')[1].split('"')[0] for l in open(os.path.join(OUT, "games.jsonl"))} \
        if os.path.exists(os.path.join(OUT, "games.jsonl")) else set()
    tasks = [t for t in tasks if "%d_%s_s%d.npz" % (t[0], opp_name(t[1]), (t[0] // len(POOL)) % 2) not in done]
    lf = open(os.path.join(OUT, "games.jsonl"), "a")
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(play, tasks):
            for r in res:
                lf.write(json.dumps(r) + "\n"); lf.flush()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 4)

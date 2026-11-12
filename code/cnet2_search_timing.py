"""cnet2_search_timing -- platform-style load (exec, no __file__) of a packed search submission, play N games
vs an opponent, record per-step wall ms of our agent and sims per searched step.
python3 cnet2_search_timing.py PACKED.py OPP.py SEED0 N"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, eval_v53
from fastsim import FastCarbon
pk, opp, s0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
W, SIMS, res = [], [], []
for seed in range(s0, s0 + n):
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    exec(compile(open(pk).read(), "<submission>", "exec"), g)
    ob = eval_v53.load_agent_fn(opp); side = seed % 2
    sim = FastCarbon(eval_v53.CFG).reset(seed=seed); prev = (0, 0)
    for t in range(299):     # R01: a game is 299 transfers (last agent call at obs step 298), not 300
        o = [sim.obs(0), sim.obs(1)]
        ts = time.time(); a = g["agent"](o[side], eval_v53.CFG); W.append((time.time() - ts) * 1000)
        st = g["_AG"].stats; cur = (st.get("sims", 0), st["searched"])
        if cur[1] > prev[1]:
            SIMS.append(cur[0] - prev[0])
        prev = cur
        b = ob(o[1 - side], eval_v53.CFG)
        sim.step(*((a, b) if side == 0 else (b, a)))
    st = g["_AG"].stats
    # C03: unified error accounting, the SAME rule as eval_v53's a_errors: "errors" plus every
    # numeric *_errors category (tg_errors, pa_errors, px_errors, cb_errors, ...). Checking only
    # errors/root_errors let rule-layer failures pass this gate while bigeval failed the same pack.
    err_cat = {k: (v.item() if hasattr(v, "item") else v) for k, v in st.items()
               if (k == "errors" or k.endswith("_errors")) and isinstance(v, (int, float))}
    res.append({"seed": seed, "margin": round(sim.cash[side] - sim.cash[1 - side]), "errors": st.get("errors", 0),
                "root_errors": st.get("root_errors", 0), "aborted": st.get("aborted", 0),
                "a_errors": sum(err_cat.values()), "err_by_cat": err_cat})
    print(json.dumps(res[-1]), flush=True)
W = np.array(W); S = np.array(SIMS)
# R11: the general wall-time summary must print even when nothing was searched (SEARCH=False
# pure-net packs): an empty SIMS used to crash the percentile summary (IndexError) and lose it.
wall = {"mean": round(W.mean(), 1), "p50": round(np.percentile(W, 50)), "p90": round(np.percentile(W, 90)),
        "p99": round(np.percentile(W, 99)), "max": round(W.max())} if len(W) else \
       {"mean": None, "p50": None, "p90": None, "p99": None, "max": None}
sims = {"mean": round(S.mean(), 1), "p10": float(np.percentile(S, 10)), "p50": float(np.percentile(S, 50)),
        "p90": float(np.percentile(S, 90)), "searched": len(S)} if len(S) else \
       {"mean": None, "p10": None, "p50": None, "p90": None, "searched": 0}
tot_errors = sum(r["errors"] for r in res)
tot_root_errors = sum(r["root_errors"] for r in res)
err_tot = {}
for r in res:
    for k, v in r["err_by_cat"].items():
        err_tot[k] = err_tot.get(k, 0) + v
tot_a_errors = sum(err_tot.values())
print(json.dumps({"steps": len(W), "wall_ms": wall, "sims_per_searched_step": sims,
                  "errors": tot_errors, "root_errors": tot_root_errors,
                  "a_errors": tot_a_errors, "error_categories": err_tot}))
# F09/R11 + C03: this tool doubles as the acceptance gate for packed submissions.
# A pack whose agent fails at the root of every step (root_errors == steps)
# previously printed its statistics and exited 0; any error total must exit
# nonzero so exit-code callers cannot read it as a pass. The gate now uses the
# unified total (a_errors: errors + all numeric *_errors categories, as in
# eval_v53), so a rule-layer failure such as tg_errors > 0 fails here exactly
# as it fails bigeval. Statistics above are printed either way.
if tot_a_errors:
    print(json.dumps({"FAIL": True, "errors": tot_errors, "root_errors": tot_root_errors,
                      "a_errors": tot_a_errors, "error_categories": err_tot}), file=sys.stderr, flush=True)
    sys.exit(1)

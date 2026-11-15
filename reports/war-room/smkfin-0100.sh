#!/bin/bash
# Final-version smoke (baseline 0f704a9d): self-syncs the DGX repo (fetch + ff-only), then runs
# play_batch standard batch + constructed-extinction batch + R12 (TTA=2) re-run against the
# final code. Artifacts: /home/arm/junyang/carbon/data/smoke_pb_final/
# Marker: SMKFIN0100A
MARK=SMKFIN0100A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 1800 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "SMKFIN0100A DGX-BEGIN $(date '+%F %T %Z')"
hostname
W=/home/arm/junyang/carbon
R=/home/arm/junyang/carbon-repo
PY=$W/.venv/bin/python
ART=$W/data/smoke_pb_final
mkdir -p $ART
echo '--- SYNC ---'
ps aux | grep -E 'git (fetch|pull)' | grep -v grep || echo "no git residue"
cd $R
timeout 60 git fetch origin 2>&1 | tail -2
git merge --ff-only 0f704a9da8da05e68976bfd1442467b52106bf5c 2>&1 | tail -2
echo "HEAD $(git rev-parse HEAD)"
md5sum code/rl2_ppo_rollout.py code/eval_v53.py code/cnet2_search.py code/cnet2_np.py
echo '--- TRIO ---'
ps aux | grep -E 'cnet2_train|bigeval|rl2_ppo' | grep -v grep || echo "no train/eval processes"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader
cat > $ART/smoke_final.py <<'PYEOF'
import os, sys, json, time, copy
import numpy as np
sys.path.insert(0, os.getcwd())
import eval_v53
import rl2_ppo_rollout as RO
from fastsim import FastCarbon as RealFC

W = "/home/arm/junyang/carbon/data/weights/cnet2_b128m2t3.npz"
BOT = "/home/arm/junyang/carbon/code/exp/opp_1cz_v5.py"
ART = "/home/arm/junyang/carbon/data/smoke_pb_final"
EPS = eval_v53.EPISODE_STEPS
CFG = eval_v53.CFG
rep = {"eps_steps": EPS, "weights": W, "bot": BOT, "batches": {}, "errors": []}

def check_game(g, tag):
    dw = np.asarray(g["dworth"], dtype=np.float64); dm = np.asarray(g["dmargin"], dtype=np.float64)
    nrec = len(g["recs"]) if g.get("recs") is not None else -1
    row = {"tag": tag, "opp": str(g["opp"])[:24], "seed": int(g["seed"]), "side": int(g["side"]),
           "mirror": bool(g.get("mirror")), "steps": int(g["steps"]), "term": bool(g["term"]),
           "margin": round(float(g["margin"]), 4), "nrec": nrec,
           "len_ok": bool(nrec == g["steps"] == len(dw) == len(dm)),
           "dw_err": round(abs(float(dw.sum()) - float(g["margin"])), 6),
           "dm_err": round(abs(float(dm.sum()) - float(g["margin"])), 6),
           "horizon_ok": bool(g["term"] or g["steps"] == 299)}
    print("GAME", json.dumps(row), flush=True)
    return row

def run_batch(games, tag, rng_seed):
    t = {"adapter": "cnet2", "wkey": os.path.abspath(W), "games": games,
         "temp": 1.0, "greedy": False, "record": True, "rng_seed": rng_seed}
    out = RO.play_batch(t)
    rows = [check_game(g, tag) for g in out["games"]]
    prim = [r for r in rows if not r["mirror"]]
    exp_ndec = sum(r["steps"] * (2 if r["opp"] == "SELF" else 1) for r in prim)
    b = {"n_records": len(rows), "n_term": sum(1 for r in rows if r["term"]),
         "all_len_ok": all(r["len_ok"] for r in rows), "all_horizon_ok": all(r["horizon_ok"] for r in rows),
         "max_dw_err": max((r["dw_err"] for r in rows), default=None),
         "max_dm_err": max((r["dm_err"] for r in rows), default=None),
         "ndec_actual": out["ndec"], "ndec_expected": exp_ndec,
         "ndec_ok": bool(out["ndec"] == exp_ndec), "wall": round(out["wall"], 1)}
    rep["batches"][tag] = b
    print("BATCH", tag, json.dumps(b), flush=True)
    return out, rows

# ---------- standard batch (unpatched) ----------
try:
    run_batch([("SELF", 991000 + i, i % 2) for i in range(6)] +
              [(BOT, 990500 + i, i % 2) for i in range(6)], "std", 22)
except Exception as e:
    import traceback; traceback.print_exc(); rep["errors"].append("std:" + repr(e))

# ---------- constructed extinction ----------
def side_counts(st, p):
    return {"workers": sum(1 for v in st["workers"].values() if v.get("player") == p),
            "trees": sum(1 for v in st["trees"].values() if v.get("player") == p),
            "base": bool(st["bases"].get(p)), "cash": float(st["cash"][p])}

def extinct_by_rule(st, p):
    c = side_counts(st, p)
    if c["workers"] > 0 or c["trees"] > 0:
        return False
    if not c["base"]:
        return True
    return c["cash"] < min(CFG.get("recCollectorCost", 30), CFG.get("recPlanterCost", 30))

def try_real_arrange(real, p, log):
    """Remove side p's workers/trees and zero its cash in the sim's real state.
    Route 1: get_state() dicts are live. Route 2: internal dict attrs with matching keys."""
    st = real.get_state()
    for wid in [k for k, v in list(st["workers"].items()) if v.get("player") == p]:
        st["workers"].pop(wid, None)
    for tid in [k for k, v in list(st["trees"].items()) if v.get("player") == p]:
        st["trees"].pop(tid, None)
    try:
        st["cash"][p] = 0.0
    except Exception:
        pass
    if extinct_by_rule(real.get_state(), p):
        log["route"] = "live_state_dicts"
        return True
    for attr in dir(real):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(real, attr)
        except Exception:
            continue
        if isinstance(v, dict) and st["workers"] is not None and len(v) and all(isinstance(x, dict) and "player" in x for x in list(v.values())[:3]):
            keys = set(v.keys())
            if keys == set(st["workers"].keys()) or keys == set(st["trees"].keys()):
                for k in [k for k, x in list(v.items()) if x.get("player") == p]:
                    v.pop(k, None)
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            try:
                v[p] = 0.0
            except Exception:
                pass
    if extinct_by_rule(real.get_state(), p):
        log["route"] = "internal_attrs"
        return True
    return False

class ProxySim:
    def __init__(self, real, plan, log):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_n", 0)
        object.__setattr__(self, "_intercepted", False)
        st = real.get_state()
        log["cash_hist"] = [[float(st["cash"][0]), float(st["cash"][1])]]
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)
    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real"), name, value)
    def step(self, a0, a1):
        real = object.__getattribute__(self, "_real")
        r = real.step(a0, a1)
        n = object.__getattribute__(self, "_n") + 1
        object.__setattr__(self, "_n", n)
        plan = object.__getattribute__(self, "_plan")
        log = object.__getattribute__(self, "_log")
        if plan and n == plan["at_step"]:
            ok = try_real_arrange(real, plan["side"], log)
            log["real_arrange_ok"] = bool(ok)
        st = real.get_state()
        log["cash_hist"].append([float(st["cash"][0]), float(st["cash"][1])])
        return r
    def get_state(self):
        real = object.__getattribute__(self, "_real")
        st = real.get_state()
        plan = object.__getattribute__(self, "_plan")
        log = object.__getattribute__(self, "_log")
        if plan and not object.__getattribute__(self, "_intercepted") \
           and object.__getattribute__(self, "_n") >= plan["at_step"]:
            if extinct_by_rule(st, plan["side"]):
                log.setdefault("arranged_post", {"counts": side_counts(st, plan["side"]),
                                                 "extinct_by_rule": True, "route": log.get("route", "real")})
                object.__setattr__(self, "_intercepted", True)
                return st
            # route 3: intercept -- hand play_batch an arranged copy for this one read
            st2 = copy.deepcopy(st)
            for wid in [k for k, v in list(st2["workers"].items()) if v.get("player") == plan["side"]]:
                st2["workers"].pop(wid, None)
            for tid in [k for k, v in list(st2["trees"].items()) if v.get("player") == plan["side"]]:
                st2["trees"].pop(tid, None)
            st2["cash"][plan["side"]] = 0.0
            log["route"] = "get_state_intercept"
            log["arranged_post"] = {"counts": side_counts(st2, plan["side"]),
                                    "extinct_by_rule": extinct_by_rule(st2, plan["side"])}
            object.__setattr__(self, "_intercepted", True)
            return st2
        return st

class FCWrap:
    def __init__(self, plan, log):
        self._plan = plan; self._log = log
    def reset(self, seed=None):
        return ProxySim(RealFC(CFG).reset(seed=seed), self._plan, self._log)

plans = [{"side": 1, "at_step": 1},   # C1 SELF seed 992001 side 0: opponent side extinct at step 1
         {"side": 0, "at_step": 1},   # C2 bot  seed 992002 side 0: learner side extinct at step 1
         None, None]                  # S1/S2 SELF siblings run full 299
logs = [{}, {}, {}, {}]
plan_iter = iter(plans); log_iter = iter(logs)
RO.FastCarbon = lambda cfg: FCWrap(next(plan_iter), next(log_iter))
ext_rows = []
try:
    out_ext, ext_rows = run_batch([("SELF", 992001, 0), (BOT, 992002, 0),
                                   ("SELF", 992003, 0), ("SELF", 992004, 1)], "ext", 77)
finally:
    RO.FastCarbon = RealFC

pen = 1 - EPS - 1
ext_checks = {}
for name, idx, seed, surv_side in (("C1", 0, 992001, 0), ("C2", 1, 992002, 1)):
    prim = next((r for r in ext_rows if r["seed"] == seed and not r["mirror"]), None)
    lg = logs[idx]
    prevcash = lg.get("cash_hist", [[None, None]])[0]
    exp_margin = (prevcash[surv_side] - pen) if surv_side == (prim["side"] if prim else surv_side) else None
    # expected margin from the learner side's perspective:
    if prim is not None:
        side = prim["side"]
        rew = [pen if s == (1 - surv_side) and False else None for s in (0, 1)]
        rew = [None, None]
        rew[1 - surv_side] = pen            # extinct side
        rew[surv_side] = prevcash[surv_side]  # survivor keeps pre-step cash
        exp_margin = rew[side] - rew[1 - side]
    ext_checks[name] = {"found": prim is not None,
                        "term": prim["term"] if prim else None,
                        "steps": prim["steps"] if prim else None,
                        "len_ok": prim["len_ok"] if prim else None,
                        "margin": prim["margin"] if prim else None,
                        "expected_margin": exp_margin,
                        "margin_err": (abs(prim["margin"] - exp_margin) if prim and exp_margin is not None else None),
                        "dw_err": prim["dw_err"] if prim else None,
                        "route": lg.get("route"), "real_arrange_ok": lg.get("real_arrange_ok"),
                        "arranged_post": lg.get("arranged_post"),
                        "prevcash_reset": prevcash}
    if name == "C1":
        mir = next((r for r in ext_rows if r["seed"] == seed and r["mirror"]), None)
        ext_checks[name]["mirror_margin"] = mir["margin"] if mir else None
        ext_checks[name]["mirror_dw_err"] = mir["dw_err"] if mir else None
    print("EXTCHECK", name, json.dumps(ext_checks[name]), flush=True)
rep["ext_checks"] = ext_checks
json.dump(rep, open(os.path.join(ART, "smoke_final_summary.json"), "w"), indent=1)
print("SMOKE-FINAL errors:", rep["errors"], flush=True)
PYEOF
echo '--- SMOKE RUN ---'
cd $R/code && timeout 1300 $PY $ART/smoke_final.py 2>&1 | tail -40
echo '--- R12 RUN ---'
cat > $ART/r12_final.py <<'PYEOF'
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.getcwd())
out = {}
try:
    import cnet2_np as NP, cnet2_search as S, cnet2_feat as F
    from fastsim import FastCarbon
    import eval_v53
    P = "/home/arm/junyang/carbon/data/weights/cnet2_b128m2t3.npz"
    out["weights"] = P
    out["md5"] = hashlib.md5(open(P, "rb").read()).hexdigest()
    w = S.load_weights(P)
    out["np_format"] = bool("stem.weight" in w and "gru.weight_ih" in w and "teach.weight" in w)
    pol = S.CachePolicy(w)
    sim = FastCarbon(eval_v53.CFG).reset(seed=991001)
    f = F.Feat(0)
    x = f.observe(sim.obs(0))
    tok0, ctx0 = pol.encode(x, 1)
    y, lg = NP.act_tta(pol, x, f.aux, teacher=1, k=2)
    tokS, ctxS = pol.last_enc
    out["sig_mismatch_after_tta"] = bool(S.CachePolicy._sig(x, 1) != pol.last_sig)
    out["diff_stale_view_vs_raw"] = round(max(float(np.abs(tokS - tok0).max()), float(np.abs(ctxS - ctx0).max())), 6)
    tok1, ctx1 = pol.enc_for(x, 1)
    out["diff_consumed_encfor_vs_raw"] = round(max(float(np.abs(tok1 - tok0).max()), float(np.abs(ctx1 - ctx0).max())), 6)
except Exception as e:
    import traceback; traceback.print_exc()
    out["error"] = repr(e)
json.dump(out, open(os.path.join(sys.argv[1], "r12_final_summary.json"), "w"), indent=1)
print("R12-FINAL", json.dumps(out), flush=True)
PYEOF
cd $R/code && timeout 240 $PY $ART/r12_final.py $ART 2>&1 | tail -12
echo "SMKFIN0100A DGX-END art=$ART"
DGXEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# Probe SMOKE: play_batch end-to-end smoke (new rollout code from repo) + R12 TTA=2 enc_for retest.
# Runs on DGX via L20 ssh. Smoke only: no deployment, no training touched. Marker: PBSMOKE2345A
MARK=PBSMOKE2345A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 1500 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSMOKE2345A DGX-BEGIN $(date '+%F %T %Z')"
hostname
W=/home/arm/junyang/carbon
R=/home/arm/junyang/carbon-repo
PY=$W/.venv/bin/python
ART=$W/data/smoke_pb_2345
mkdir -p $ART
echo '--- PRE: repo vs workspace py diff (names) ---'
diff -rq $R/code $W/code 2>/dev/null | grep -E '\.py' | head -40
echo '--- PRE: weights candidates ---'
find $W/data $W/code -name '*.npz' 2>/dev/null | head -40
BW=$(find $W -name 'cnet2_b128m2t3.npz' 2>/dev/null | head -1)
echo "B128M2T3=$BW"
cat > $ART/smoke_pb.py <<'PYEOF'
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.getcwd())
import rl2_ppo_rollout as RO

W, ART = sys.argv[1], sys.argv[2]
res = {"weights": W, "games": [], "errors": []}

def check(g, tag):
    dw = np.asarray(g["dworth"], dtype=np.float64); dm = np.asarray(g["dmargin"], dtype=np.float64)
    nrec = len(g["recs"]) if g.get("recs") is not None else -1
    row = {"tag": tag, "opp": str(g["opp"])[:24], "seed": int(g["seed"]), "side": int(g["side"]),
           "steps": int(g["steps"]), "term": bool(g["term"]), "margin": round(float(g["margin"]), 3),
           "nrec": nrec, "len_ok": bool(nrec == g["steps"] == len(dw) == len(dm)),
           "dw_err": round(abs(float(dw.sum()) - float(g["margin"])), 5),
           "dm_err": round(abs(float(dm.sum()) - float(g["margin"])), 5),
           "horizon_ok": bool(g["term"] or g["steps"] == 299)}
    res["games"].append(row)
    print("GAME", json.dumps(row), flush=True)
    return row["term"]

def batch(games, tag, rng):
    t = {"adapter": "cnet2", "wkey": os.path.abspath(W), "games": games,
         "temp": 1.0, "greedy": False, "record": True, "rng_seed": rng}
    t0 = time.time()
    out = RO.play_batch(t)
    print(f"BATCH {tag} ngames={len(out['games'])} wall={time.time()-t0:.1f}s", flush=True)
    return any(check(g, tag) for g in out["games"])

term_found = False
try:
    term_found |= batch([("exp/opp_1cz_v5.py", 990500 + i, i % 2) for i in range(6)], "botA", 11)
    term_found |= batch([("SELF", 991000 + i, i % 2) for i in range(6)], "selfB", 22)
    extra = 0
    while not term_found and extra < 24:
        gs = [("SELF", 992000 + extra + i, (extra + i) % 2) if i % 2 == 0
              else ("exp/opp_1cz_v5.py", 993000 + extra + i, (extra + i) % 2) for i in range(8)]
        term_found |= batch(gs, f"hunt{extra}", 33 + extra)
        extra += 8
except Exception as e:
    import traceback; traceback.print_exc()
    res["errors"].append(repr(e))
gs = res["games"]
res["summary"] = {"n_games": len(gs), "n_term": sum(1 for g in gs if g["term"]),
                  "all_len_ok": all(g["len_ok"] for g in gs),
                  "max_dw_err": max((g["dw_err"] for g in gs), default=None),
                  "max_dm_err": max((g["dm_err"] for g in gs), default=None),
                  "all_horizon_ok": all(g["horizon_ok"] for g in gs)}
json.dump(res, open(os.path.join(ART, "smoke_summary.json"), "w"), indent=1)
print("SMOKE-SUMMARY", json.dumps(res["summary"]), flush=True)
PYEOF
cat > $ART/r12.py <<'PYEOF'
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.getcwd())
out = {}
try:
    import cnet2_np as NP, cnet2_search as S, cnet2_feat as F
    from fastsim import FastCarbon
    import eval_v53
    cands = [l.strip() for l in open(sys.argv[1]) if l.strip().endswith(".npz")]
    chosen, pol = None, None
    for p in cands:
        try:
            w = S.load_weights(p)
            if "stem.weight" not in w or "gru.weight_ih" not in w:
                continue
            cand = S.CachePolicy(w)
            chosen, pol = p, cand
            break
        except Exception:
            continue
    out["chosen"] = chosen
    if chosen is None:
        out["error"] = "no NP-format weights candidate"
    else:
        out["md5"] = hashlib.md5(open(chosen, "rb").read()).hexdigest()
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
json.dump(out, open(os.path.join(sys.argv[2], "r12_summary.json"), "w"), indent=1)
print("R12-SUMMARY", json.dumps(out), flush=True)
PYEOF
find $W/data $W/code -name '*.npz' 2>/dev/null | head -40 > $ART/cands.txt
echo '--- SMOKE RUN ---'
cd $R/code && timeout 1200 $PY $ART/smoke_pb.py "$BW" $ART 2>&1 | tail -60
echo '--- R12 RUN ---'
cd $R/code && timeout 240 $PY $ART/r12.py $ART/cands.txt $ART 2>&1 | tail -30
echo "PBSMOKE2345A DGX-END art=$ART"
DGXEOF
echo "$MARK L20-END rc=$?"

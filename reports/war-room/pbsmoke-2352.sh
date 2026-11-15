#!/bin/bash
# Probe SMOKE v2: play_batch smoke + R12 retest. v1 failed on setup: repo exp/ lacks the bench
# default bots (they live in the workspace exp/) and the R12 candidate list was flooded by
# dataset shards. v2: SELF games first, bot via absolute workspace path, weights from data/weights.
# Marker: PBSMOKE2352B
MARK=PBSMOKE2352B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 1500 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSMOKE2352B DGX-BEGIN $(date '+%F %T %Z')"
hostname
W=/home/arm/junyang/carbon
R=/home/arm/junyang/carbon-repo
PY=$W/.venv/bin/python
ART=$W/data/smoke_pb_2345
mkdir -p $ART
echo '--- PRE: workspace exp bots ---'
ls $W/code/exp/ | grep -E '^(opp_|cand_)' | head -20
echo '--- PRE: data/weights npz ---'
ls -la $W/data/weights/*.npz 2>/dev/null | head -30
BOT=$(ls $W/code/exp/opp_1cz_v5.py 2>/dev/null || ls $W/code/exp/cand_c1z_v1.py 2>/dev/null || ls $W/code/exp/opp_*.py 2>/dev/null | head -1)
echo "BOT=$BOT"
BW=$W/data/weights/cnet2_b128m2t3.npz
cat > $ART/smoke_pb2.py <<'PYEOF'
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.getcwd())
import rl2_ppo_rollout as RO

W, ART, BOT = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"weights": W, "bot": BOT, "games": [], "errors": []}

def check(g, tag):
    dw = np.asarray(g["dworth"], dtype=np.float64); dm = np.asarray(g["dmargin"], dtype=np.float64)
    nrec = len(g["recs"]) if g.get("recs") is not None else -1
    row = {"tag": tag, "opp": str(g["opp"])[:28], "seed": int(g["seed"]), "side": int(g["side"]),
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
    term_found |= batch([("SELF", 991000 + i, i % 2) for i in range(6)], "selfB", 22)
except Exception as e:
    import traceback; traceback.print_exc(); res["errors"].append("selfB:" + repr(e))
if BOT:
    try:
        term_found |= batch([(BOT, 990500 + i, i % 2) for i in range(6)], "botA", 11)
    except Exception as e:
        import traceback; traceback.print_exc(); res["errors"].append("botA:" + repr(e))
extra = 0
while not term_found and extra < 24:
    try:
        gs = [("SELF", 992000 + extra + i, (extra + i) % 2) if i % 2 == 0
              else (BOT, 993000 + extra + i, (extra + i) % 2) for i in range(8)] if BOT else \
             [("SELF", 992000 + extra + i, (extra + i) % 2) for i in range(8)]
        term_found |= batch(gs, f"hunt{extra}", 33 + extra)
    except Exception as e:
        import traceback; traceback.print_exc(); res["errors"].append(f"hunt{extra}:" + repr(e)); break
    extra += 8
gs = res["games"]
res["summary"] = {"n_games": len(gs), "n_term": sum(1 for g in gs if g["term"]),
                  "all_len_ok": all(g["len_ok"] for g in gs) if gs else None,
                  "max_dw_err": max((g["dw_err"] for g in gs), default=None),
                  "max_dm_err": max((g["dm_err"] for g in gs), default=None),
                  "all_horizon_ok": all(g["horizon_ok"] for g in gs) if gs else None,
                  "errors": res["errors"]}
json.dump(res, open(os.path.join(ART, "smoke_summary2.json"), "w"), indent=1)
print("SMOKE-SUMMARY", json.dumps(res["summary"]), flush=True)
PYEOF
cat > $ART/r12b.py <<'PYEOF'
import os, sys, json, hashlib, glob
import numpy as np
sys.path.insert(0, os.getcwd())
out = {"tried": []}
try:
    import cnet2_np as NP, cnet2_search as S, cnet2_feat as F
    from fastsim import FastCarbon
    import eval_v53
    W = "/home/arm/junyang/carbon"
    cands = sorted(glob.glob(W + "/data/weights/*.npz")) + \
            sorted(glob.glob(W + "/code/*.npz")) + \
            sorted(glob.glob(os.getcwd() + "/*.npz"))
    chosen, pol = None, None
    for p in cands:
        try:
            w = S.load_weights(p)
            keys = sorted(w.keys())
            ok = ("stem.weight" in w and "gru.weight_ih" in w and "teach.weight" in w)
            out["tried"].append({"path": p, "np_format": bool(ok), "nkeys": len(keys),
                                 "sample": keys[:6]})
            if ok and chosen is None:
                chosen, pol = p, S.CachePolicy(w)
        except Exception as e:
            out["tried"].append({"path": p, "error": repr(e)[:100]})
    out["chosen"] = chosen
    if chosen is None:
        out["error"] = "no NP-format weights in data/weights or code dirs"
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
json.dump(out, open(os.path.join(sys.argv[1], "r12_summary2.json"), "w"), indent=1)
print("R12-SUMMARY", json.dumps(out)[:3000], flush=True)
PYEOF
echo '--- SMOKE RUN v2 ---'
cd $R/code && timeout 1200 $PY $ART/smoke_pb2.py "$BW" $ART "$BOT" 2>&1 | tail -70
echo '--- R12 RUN v2 ---'
cd $R/code && timeout 240 $PY $ART/r12b.py $ART 2>&1 | tail -30
echo "PBSMOKE2352B DGX-END art=$ART"
DGXEOF
echo "$MARK L20-END rc=$?"

#!/bin/bash
# R12 retest pinned to the review's b128 weights (cnet2_b128m2t3.npz), same construction
# as r12b.py in the v2 smoke: TTA=2, then compare stale cached view vs raw encoding and
# the consumed enc_for encoding vs raw. Marker: PBSUM0013B
MARK=PBSUM0013B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSUM0013B DGX-BEGIN $(date '+%F %T %Z')"
hostname
cd /home/arm/junyang/carbon-repo/code && timeout 200 /home/arm/junyang/carbon/.venv/bin/python - <<'PYEOF'
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
print("R12-B128", json.dumps(out), flush=True)
PYEOF
echo "PBSUM0013B DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"

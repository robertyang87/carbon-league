"""cnet2_search_valcal -- calibration of the cnet2 value head (tanh(margin/3000) target) on held-out positions:
RMSE in margin units and R^2 by game phase.  python3 cnet2_search_valcal.py DATA_DIR WEIGHTS [N]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_np as NP, cnet2_search as S
d, wp = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
K = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")
D = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for k in K + ("value", "gid", "step", "teacher")}
meta = json.load(open(os.path.join(d, "meta.json")))
val = np.array([g["val"] for g in meta["games"]])[np.asarray(D["gid"])]
idx = np.nonzero(val)[0]; rs = np.random.RandomState(0); idx = np.sort(rs.choice(idx, min(N, len(idx)), replace=False))
pol = NP.Policy(S.load_weights(wp))
pv = []
for i in range(0, len(idx), 64):
    ii = idx[i:i + 64]
    X = {k: np.asarray(D[k][ii]) for k in K}
    X["scal"] = X["scal"].astype(np.float32); X["tok_cargo"] = X["tok_cargo"].astype(np.float32); X["hist_cargo"] = X["hist_cargo"].astype(np.float32)
    X["tok_pos"] = X["tok_pos"].astype(np.int64); X["tok_kind"] = X["tok_kind"].astype(np.int64)
    X["hist_cell"] = X["hist_cell"].astype(np.int64); X["hist_move"] = X["hist_move"].astype(np.int64)
    tok, ctx = S.encode_b(pol, X, np.asarray(D["teacher"][ii]).astype(np.int64))
    pv.append(np.tanh(ctx @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0]))
pv = np.concatenate(pv).astype(np.float64); tv = np.asarray(D["value"][idx], np.float64); stp = np.asarray(D["step"][idx])
mp = 3000 * np.arctanh(np.clip(pv, -0.999, 0.999)); mt = 3000 * np.arctanh(np.clip(tv, -0.999, 0.999))
out = {"n": len(idx)}
for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)):
    m = (stp >= lo) & (stp < hi)
    if m.sum() < 20:
        continue
    e = mp[m] - mt[m]
    out["%d-%d" % (lo, hi)] = {"n": int(m.sum()), "rmse_margin": round(float(np.sqrt((e ** 2).mean()))), "sd_true": round(float(mt[m].std())),
                               "R2": round(float(1 - (e ** 2).mean() / mt[m].var()), 3), "corr": round(float(np.corrcoef(mp[m], mt[m])[0, 1]), 3)}
print(json.dumps(out))

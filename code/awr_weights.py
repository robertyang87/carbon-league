"""awr_weights DS CORPUS_DIR [BETA] -- AWR weights on a build_awr dataset: every row of a game gets
w = clip(exp(adv / BETA), 0, WMAX) (adv from CORPUS_DIR/awr.jsonl; BETA default = std of the sampled advantages);
uw (per unit) = 1 for planters and base slots, 0 for collectors (the cars stay the net's own; only planters and
recruits learn). Greedy games (adv 0) get weight 1 -- they anchor the policy."""
import json, os, sys
import numpy as np
ds, corpus = sys.argv[1], sys.argv[2]
adv = {r["key"]: r["adv"] for r in map(json.loads, open(os.path.join(corpus, "awr.jsonl")))}
sampled = np.array([r["adv"] for r in map(json.loads, open(os.path.join(corpus, "awr.jsonl"))) if r["k"] > 0])
beta = float(sys.argv[3]) if len(sys.argv) > 3 else float(sampled.std())
WMAX = float(os.environ.get("AWR_WMAX", "20"))
meta = json.load(open(os.path.join(ds, "meta.json"))); gid = np.load(os.path.join(ds, "gid.npy"))
gw = np.array([min(WMAX, float(np.exp(adv[g["key"].replace(".json.gz", "")] / beta))) for g in meta["games"]], np.float32)
w = gw[gid].astype(np.float32)
kind = np.load(os.path.join(ds, "tok_kind.npy"), mmap_mode="r"); y = np.load(os.path.join(ds, "y.npy"), mmap_mode="r")
S = y.shape[1]
uw = np.ones((len(gid), S), np.float32)
uw[:, :10] = (np.asarray(kind[:, :10]) != 0).astype(np.float32)        # own collectors (kind 0) out of the loss
np.save(os.path.join(ds, "w.npy"), w); np.save(os.path.join(ds, "uw.npy"), uw)
print(json.dumps({"rows": len(gid), "games": len(gw), "beta": round(beta, 1), "w_mean": round(float(w.mean()), 3),
                  "w_max": round(float(w.max()), 2), "w_p90": round(float(np.percentile(gw, 90)), 2), "collector_tokens_masked": int((uw[:, :10] == 0).sum())}))

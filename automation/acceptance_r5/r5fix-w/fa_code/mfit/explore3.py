import pickle, pandas as pd, numpy as np, sys, gzip, json
sys.path.insert(0, "."); from mfit.extract import D, NBR, state_feats, step as stp
exec(open("mfit/explore2.py").read().split("m = P[")[0])
idx = json.load(open("../data/m2_corpus/index.json"))
# examine decisions at the *first* step a target becomes the target (target switch points), with full state
P["prev_t"] = P.groupby(["g", "uid"]).tpos.shift(1)
dec = P[(P.y != "STAY") & (P.tpos >= 0) & (P.tpos != P.prev_t)]
print("target-choice decisions", len(dec))
cache = {}
res = []
for gi, grp in dec.groupby("g"):
    e = idx[gi]; d = json.load(gzip.open("../data/m2_corpus/" + e["file"]))["steps"]
    for _, r in grp.iterrows():
        t = int(r.step); o = d[t][0]["observation"]; P_ = o["players"]
        me = [i for i, s in enumerate(e["order"]) if e["strategies"][s]["name"].startswith("RyanX")][0]
        ctx, arr = state_feats(P_, me, o["full_carbon"], t)
        pos = int(r.pos); tp = int(r.tpos); nb = arr["nb"]; occ = arr["occ"]; dist = D[pos]
        car = np.asarray(o["full_carbon"], float)
        tage = {int(p): a for p, a in P_[1 - me][3].values()}
        cand = [c for c in range(225) if occ[c] in (0, 2)]
        res.append(dict(dist=dist[tp], tocc=occ[tp], tnb=nb[tp], tcar=car[tp], age=tage.get(tp, -1),
                        rank_nb_free=sum(1 for c in cand if occ[c] == 0 and nb[c] > nb[tp]),
                        nearer_etree=sum(1 for c in cand if occ[c] == 2 and dist[c] < dist[tp]),
                        better_free_within=sum(1 for c in cand if occ[c] == 0 and dist[c] <= dist[tp] and nb[c] > nb[tp]),
                        bestnb_within=max(nb[c] for c in cand if occ[c] == 0 and dist[c] <= dist[tp]),
                        net=nb[tp] - 0 , cash=ctx["cash"], price=ctx["price"], step=t,
                        dbase=arr["dbase"][tp], depl=arr["depl"][tp]))
R = pd.DataFrame(res)
pd.set_option("display.width", 200)
for o_, g in R.groupby("tocc"):
    print("TARGET occ", o_, len(g)); print(g.describe().round(1).T[["mean", "25%", "50%", "75%"]])
R.to_pickle("mfit/targets.pkl")

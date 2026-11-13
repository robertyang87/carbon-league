"""load M-family games from data/m2_corpus into compact per-seat trajectories (cached in m2fit/games.pkl)"""
import gzip, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "data", "m2_corpus")

def load(names=("RyanX-AIPowercore-M2",), cache=True):
    fn = os.path.join(HERE, "games.pkl")
    if cache and os.path.exists(fn):
        G = pickle.load(open(fn, "rb"))
    else:
        idx = json.load(open(os.path.join(ROOT, "index.json")))
        G = []
        for gi, e in enumerate(idx):
            seats = [i for i, s in enumerate(e["order"]) if e["strategies"][s]["name"].startswith("RyanX-AIPowercore-M")]
            if not seats:
                continue
            st = json.load(gzip.open(os.path.join(ROOT, e["file"])))["steps"]
            obs = [s[0]["observation"] for s in st]
            for me in seats:
                nm = e["strategies"][e["order"][me]]["name"]
                opp = e["strategies"][e["order"][1 - me]]["name"]
                G.append(dict(g=gi, me=me, name=nm, opp=opp, score=e["scores"][e["order"][me]], oscore=e["scores"][e["order"][1 - me]],
                              P=[o["players"] for o in obs], car=[np.asarray(o["full_carbon"], float) for o in obs],
                              pcar=[np.asarray(s[me]["observation"]["carbon"], float) for s in st],
                              act=[(st[t + 1][me].get("action") or {}) for t in range(len(st) - 1)],
                              tinc=[sum(float(o["trees"].get(k, [0, 0])[1] or 0) for k in o["players"][me][3]) for o in obs],
                              oact=[(st[t + 1][1 - me].get("action") or {}) for t in range(len(st) - 1)]))
        pickle.dump(G, open(fn, "wb"))
    return [g for g in G if g["name"] in names] if names else G

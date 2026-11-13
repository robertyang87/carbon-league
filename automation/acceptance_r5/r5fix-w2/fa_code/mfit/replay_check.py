"""run m_rules (dummy inner) on M's replay states and measure agreement with M's actual planter/base actions"""
import json, gzip, sys, os
sys.path.insert(0, "."); import m_rules
idx = json.load(open("../data/m2_corpus/index.json"))
cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
ok = n = bok = bn = 0
for gi, e in enumerate(idx[:: 7]):
    d = json.load(gzip.open("../data/m2_corpus/" + e["file"]))["steps"]
    me = [i for i, s in enumerate(e["order"]) if e["strategies"][s]["name"].startswith("RyanX")][0]
    ag = m_rules.wrap(lambda o, c=None: {}, cfg)
    for t in range(100):
        o0 = d[t][0]["observation"]
        obs = {"step": t, "player": me, "players": o0["players"], "carbon": d[t][me]["observation"]["carbon"]}
        a = ag(obs); A = d[t + 1][me].get("action") or {}
        for u, w in o0["players"][me][2].items():
            if w[2] == "PLANTER":
                n += 1; ok += a.get(u) == A.get(u)
        for b in o0["players"][me][1]:
            bn += 1; bok += a.get(b) == A.get(b)
print(f"planter agreement {ok/n:.3f} (n {n})  base {bok/bn:.3f}")

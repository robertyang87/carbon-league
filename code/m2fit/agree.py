"""offline agreement of the PLAYING code path (m2_bot agent fed M2 replay observations) with M2's recorded actions
   python m2fit/agree.py NGAMES [cfgjson] [full]"""
import os, sys, json
import numpy as np
sys.path.insert(0, "."); from m2fit.load import load
import m2_bot
from collections import Counter
n = int(sys.argv[1]); cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
full = len(sys.argv) > 3
MV = {"UP", "DOWN", "LEFT", "RIGHT"}
ok = Counter(); tot = Counter()
G = load()
for g in G[-n:]:          # last games (used in fitting too; trees are fitted on all M2 games)
    ag = m2_bot.make_agent(cfg); me = g["me"]
    for t in range(299):
        obs = dict(step=t, player=me, players=g["P"][t], carbon=list(g["pcar"][t]))
        if full: obs["full_carbon"] = list(g["car"][t]); cfg["FULL"] = 1
        a = ag(obs, {}); m = g["act"][t]
        for u, w in g["P"][t][me][2].items():
            k = w[2][0] + ":" + ("move" if m.get(u) in MV else "stay")
            tot[k] += 1; ok[k] += (a.get(u) if a.get(u) in MV else None) == (m.get(u) if m.get(u) in MV else None)
            if m.get(u) in MV: tot[w[2][0] + ":move-or-stay"] += 1; ok[w[2][0] + ":move-or-stay"] += a.get(u) in MV
        for bid in g["P"][t][me][1]:
            k = "B:" + {"RECCOLLECTOR": "C", "RECPLANTER": "P"}.get(m.get(bid), "none")
            tot[k] += 1; ok[k] += a.get(bid) == m.get(bid)
for k in sorted(tot): print(f"{k:22s} {ok[k] / tot[k]:.3f}  n={tot[k]}")
for c in "CP":
    T_ = sum(v for k, v in tot.items() if k in (c + ":move", c + ":stay")); O = sum(v for k, v in ok.items() if k in (c + ":move", c + ":stay"))
    print(c, "overall action agreement", round(O / T_, 3))

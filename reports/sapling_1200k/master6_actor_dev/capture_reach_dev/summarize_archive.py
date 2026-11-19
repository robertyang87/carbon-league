"""Keep the two distance groups paired within terrain and report opponent and seat strata."""
from collections import Counter,defaultdict
import gzip
import json
import numpy as np
from screen import HERE

def summarize(group):
    totals=Counter()
    for r in group:
        totals.update(r["counts"])
    return dict(games=len(group),**totals)

rows=json.loads(gzip.decompress((HERE/"master6.json.gz").read_bytes()))["appearances"]
groups=defaultdict(list)
by_map=defaultdict(Counter)
for r in rows:
    diff=r["score"][r["seat"]]-r["score"][1-r["seat"]]
    for group in ("all","seat_"+str(r["seat"]),"W" if diff>0 else "L" if diff<0 else "D","opponent_"+r["opponent"]):
        groups[group].append(r)
    by_map[r["seed"]].update(r["counts"])
matrix=np.array([[v["two_approach"],v["two_capture"],v["far_approach"],v["far_capture"]] for v in by_map.values()])
rng=np.random.default_rng(621758323)
boots=[]
for _ in range(5000):
    a,b,c,d=matrix[rng.integers(len(matrix),size=len(matrix))].sum(axis=0)
    boots.append(d/c-b/a)
doc=dict(unique_seeds=len(matrix),far_minus_two_95_map_bootstrap=np.quantile(boots,[.025,.975]).tolist(),
         strata={k:summarize(v) for k,v in groups.items()})
(HERE/"master6_summary.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
print(len(matrix),doc["far_minus_two_95_map_bootstrap"])

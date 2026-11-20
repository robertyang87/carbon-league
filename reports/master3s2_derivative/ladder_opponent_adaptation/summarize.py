"""Aggregate each game first, then each modeled role, from frozen held receipts."""
import gzip
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
ROLES=["collector_all","planter_all","base_all"]

def group(games):
    out={}
    for role in ROLES+["collector_risk","planter_risk"]:
        rows=[g["roles"][role] for g in games if role in g["roles"]]
        if not rows:continue
        out[role]={"n":sum(r["n"] for r in rows),"maps":len(rows)}
        for key in ("nll","brier","correct","argmax_hits"):
            norm=key in ("correct","argmax_hits")
            out[role][key]=[mean(r[key][i]/(r["n"] if norm else 1) for r in rows) for i in range(2)]
        out[role]["hits"]=[sum(r["correct"][i] for r in rows) for i in range(2)]
    out["equal_role"]={key:[mean(out[k][key][i] for k in ROLES if k in out) for i in range(2)] for key in ("nll","correct","brier","argmax_hits")}
    out["team_exact"]=[sum(g["native"]["n_exact"][i] for g in games) for i in range(2)]
    out["legal_invalid"]=[sum(g["native"]["invalid_orders"][i] for g in games) for i in range(2)]
    out["roots"]=sum(g["native"]["roots"] for g in games)
    return out

def main():
    games=json.loads((HERE/'held.json').read_bytes())["games"]
    gs={"all":group(games),"wins":group([g for g in games if g["margin"]>0]),"losses":group([g for g in games if g["margin"]<0])}
    ids={g["opponent_id"] for g in games}
    gs["by_registration"]={i:group([g for g in games if g["opponent_id"]==i]) for i in sorted(ids)}
    native=json.loads(gzip.decompress((HERE/'native.json.gz').read_bytes()))
    t=gs["all"]["equal_role"]
    out=dict(qualification=t["nll"][1]<t["nll"][0] and t["correct"][1]>t["correct"][0],
             native=native["counts"],evaluation=gs)
    (HERE/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=="__main__":main()

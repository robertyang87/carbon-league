"""Compare settled games from the first public state to expiry; all differences are from ds-pro-v1's seat."""
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0,str(ROOT/"reports/sapling_1200k/all_top5"))
import economy_extract as economy
sys.path.insert(0,str(HERE.parent/"tree_payback"))
from labels import tree_rows

MATCHES = (("ClaudeFable5.2-v8","5cef1a63ea3166ee1c05611fca1cb910",1),
           ("哈基米基准","d43d676041c83c5d9380f9daf394d2ce",0),
           ("安生N33","798a279b8c4cc1de4d5ab84de34d61ce",0))
CUTS = (0,10,25,50,100,150,200,250,275,299)


def summarize(name, game, seat):
    path=HERE.parent/"replays"/(game+"-1.json.gz")
    raw=path.read_bytes()
    env=json.loads(gzip.decompress(raw))
    frames=env["steps"]
    assert len(frames)==300 and all(row["status"]=="DONE" for row in frames[-1])
    roster=[r[0]["observation"]["players"] for r in frames]
    cash=[round(p[seat][0]-p[1-seat][0],2) for p in roster]
    assert env["rewards"]==[p[0] for p in roster[-1]]
    timeline,loots,banks,_=economy.reconstruct(frames,env["configuration"])
    order=(seat,1-seat)
    def resources(t,s):
        pl=roster[t][s]
        return dict(cash=pl[0],cargo=round(sum(w[1] for w in pl[2].values()),2),
                    collectors=sum(w[2]=="COLLECTOR" for w in pl[2].values()),planters=sum(w[2]=="PLANTER" for w in pl[2].values()),
                    trees=len(pl[3]))
    def interval(start,end):
        totals={k:[round(sum(x[s][i] for x in timeline[start:end]),2) for s in order] for i,k in enumerate(economy.KEYS)}
        trees=[sum(totals[k][i] for k in ("tree","plant","seize")) for i in range(2)]
        totals["net_tree"]=[round(v,2) for v in trees]
        totals["robbed_cargo"]=[round(sum(x[s] for x in loots[start:end]),2) for s in order]
        diff={k:round(v[0]-v[1],2) for k,v in totals.items()}
        raw_change = sum(sum(row[seat]) - sum(row[1-seat]) for row in timeline[start:end])
        assert abs(raw_change - (cash[end] - cash[start])) < 1e-6
        assert abs(sum(diff[k] for k in economy.KEYS)-(cash[end]-cash[start])) < .13
        return dict(start=start,end=end,begin_margin=cash[start],end_margin=cash[end],
                    margin_change=round(cash[end]-cash[start],2),income=totals,difference=diff,
                    end_resources=[resources(end,s) for s in order])
    streaks=[]
    for sign,frames_group in itertools.groupby(range(1,len(cash)),lambda t:(cash[t]>0)-(cash[t]<0)):
        group=list(frames_group)
        streaks.append(dict(sign=sign,start=group[0],end=group[-1],frames=len(group)))
    holdings=tree_rows(env)
    planting=[]
    for s in order:
        trees=[{k:v for k,v in tree.items() if k!="cash"} for tree in holdings if tree["seat"]==s]
        planting.append(dict(planted=len(trees),repaid=sum(t["repaid"] for t in trees),
                             seized_before_payback=sum(t["seized_unrepaid"] for t in trees),
                             first_tenure_exit=dict(Counter(t["exit"] for t in trees)),individual=trees))
    return dict(opponent=name,match=game,seat=seat,planting=planting,official_scores=[env["rewards"][s] for s in order],
                replay_sha256=hashlib.sha256(raw).hexdigest(),sign_counts={str(i):sum(x["frames"] for x in streaks if x["sign"]==i) for i in (-1,0,1)},
                worst={"step":cash.index(min(cash)),"margin":min(cash)},longest_deficit=max((x for x in streaks if x["sign"]<0),key=lambda x:x["frames"]),
                all_streaks=streaks,whole=interval(0,299),phases=[interval(a,b) for a,b in zip(CUTS,CUTS[1:])],
                steps=[{"t":i,"margin":m,"teams":[resources(i,s) for s in order]} for i,m in enumerate(cash)],
                receipts_by_transition=[[{"seat":s,"worker":wid,"amount":round(v,2)} for s,wid,v in transfers] for transfers in banks])


if __name__=="__main__":
    results=[summarize(*row) for row in MATCHES]
    raw=(json.dumps(results,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    (HERE/"whole_games.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    for g in results:
        print(g["opponent"],g["official_scores"],g["sign_counts"],g["longest_deficit"],g["worst"])
        print([(p["end"],p["end_margin"],p["difference"]["bank"],p["difference"]["net_tree"]) for p in g["phases"]])
        print(g["whole"]["difference"])

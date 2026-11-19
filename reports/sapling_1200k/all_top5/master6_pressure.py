"""Audit early advantage and opponent economy across every frozen Master6 appearance."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import median
import sys

from defense_summary import ranking_lookup
from economy_extract import KEYS, reconstruct
from layout import layout

HERE=Path(__file__).resolve().parent
MASTER6="ecde65e596acbeb9ba5feca22e05f100"
STEPS=(10,30,60,100,150,200,250)
WINDOWS=((0,30),(30,60),(60,100),(100,200),(200,299))


def audit(card, path, lookup):
    src=json.loads(gzip.decompress(path.read_bytes()))
    frames=src["steps"];cfg=src["configuration"];s=card["seat_strategy_ids"].index(MASTER6);o=1-s
    led,loots,banks,_=reconstruct(frames,cfg)
    assets=[f[0]["observation"]["players"] for f in frames]
    gap=[a[s][0]-a[o][0] for a in assets]
    d=card["score_by_seat"][s]-card["score_by_seat"][o]
    def first(predicate):return next((i for i,x in enumerate(gap) if predicate(x)),None)
    final_lead=None if gap[-1]<=0 else 1+max([i for i,v in enumerate(gap) if v<=0],default=-1)
    four={}
    for t in STEPS:
        if t<len(frames):
            p=assets[t]
            four[str(t)]={"cash_gap":round(gap[t],2),
                "workers":[len(p[k][2]) for k in (s,o)],"trees":[len(p[k][3]) for k in (s,o)],
                "cargo":[round(sum(v[1] for v in p[k][2].values()),2) for k in (s,o)],
                "banked_by_now":[round(sum(a[k][0] for a in led[:t]),2) for k in (s,o)],
                "absorbed_by_now":[round(sum(a[k][2] for a in led[:t]),2) for k in (s,o)]}
    windows={}
    for a,b in WINDOWS:
        if len(led)<b:continue
        per=[]
        for k in (s,o):
            items={name:round(sum(turn[k][i] for turn in led[a:b]),2) for i,name in enumerate(KEYS)}
            items["lost_workers"]=sum(len(set(assets[t-1][k][2])-set(assets[t][k][2])) for t in range(a+1,b+1))
            per.append(items)
        windows[f"{a}_{b}"]=per
    first_economy={}
    for k in (s,o):
        first_economy["master" if k==s else "opponent"]={
            event:next((i+1 for i,r in enumerate(led) if predicate(r[k][index])),None)
            for event,index,predicate in [("hire",4,lambda v:v<0),("plant",5,lambda v:v<0),
                                         ("seize",6,lambda v:v<0),("normal_bank",0,lambda v:v>0)]}
    return {"id":card["id"],"source_sha256":card["sha256"],"seed":cfg["randomSeed"],"seat":s,
            "opponent":card["seat_names"][o],"tier":lookup(card["seat_strategy_ids"][o],card["seat_names"][o]),
            "steps":len(led),"result":"win" if d>0 else "loss" if d<0 else "draw","terminal_margin":round(d,2),
            "finished_permanent_lead":final_lead,"first_cash_lead_500":first(lambda x:x>=500),
            "landmarks":four,"windows":windows,"firsts":first_economy}


def summarize(rows):
    if not rows:return {}
    a={"games":len(rows),"opponents":len({r["opponent"] for r in rows}),"outcome":dict(Counter(r["result"] for r in rows)),
       "terminal_median_margin":median(r["terminal_margin"] for r in rows)}
    a["landmarks"]={}
    for t in STEPS:
        sample=[r for r in rows if str(t) in r["landmarks"]]
        if not sample:continue
        v=[r["landmarks"][str(t)] for r in sample]
        out={"games_reaching_step":len(v),"cash_ahead":sum(x["cash_gap"]>0 for x in v),"cash_plus500":sum(x["cash_gap"]>=500 for x in v),
             "median_cash_margin":round(median(x["cash_gap"] for x in v),2),
             "more_workers":sum(x["workers"][0]>x["workers"][1] for x in v),
             "more_trees":sum(x["trees"][0]>x["trees"][1] for x in v),
             "permanent_lead_set_by_now":sum(r["finished_permanent_lead"] is not None and r["finished_permanent_lead"]<=t for r in sample)}
        for key in ("workers","trees","cargo","banked_by_now","absorbed_by_now"):
            out["median_"+key]=[round(median(x[key][k] for x in v),2) for k in (0,1)]
        a["landmarks"][str(t)]=out
    a["windows"]={}
    for lo,hi in WINDOWS:
        key=f"{lo}_{hi}"; v=[r["windows"][key] for r in rows if key in r["windows"]]
        if v:
            a["windows"][key]={"games":len(v),"opponent_no_normal_bank":sum(x[1]["bank"]==0 for x in v),
                "opponent_positive_tree_income":sum(x[1]["tree"]>0 for x in v),
                "median_bank":[round(median(x[k]["bank"] for x in v),2) for k in (0,1)],
                "median_tree":[round(median(x[k]["tree"] for x in v),2) for k in (0,1)],
                "median_workers_lost":[median(x[k]["lost_workers"] for x in v) for k in (0,1)]}
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True);args=p.parse_args()
    cases=[]
    for f in (HERE/"manifest.json",HERE/"appendix_499886f1/manifest.json"):
        cases.extend(g for g in json.loads(f.read_bytes())["games"] if MASTER6 in g["seat_strategy_ids"])
    lookup,asof=ranking_lookup()
    rows=[audit(g,args.cache/g["path"],lookup) for g in cases]
    if len(rows)!=len({r["id"] for r in rows}) or len(rows)!=271:raise ValueError("unexpected Master6 corpus")
    groups=defaultdict(list)
    for r in rows:
        for key in ("all","tier/"+r["tier"],"result/"+r["result"],"seat/"+str(r["seat"])):
            groups[key].append(r)
    content={"master_id":MASTER6,"rank_as_of":asof,
             "definition":"Frozen heavy ranks 21–50 are the primary middle tier; missing/ambiguous strategy names remain unknown. Midgame windows only include games reaching both endpoints.",
             "groups":{k:summarize(v) for k,v in groups.items()},"rows":rows}
    (HERE/"master6_pressure.json.gz").write_bytes(gzip.compress(json.dumps(content,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    print({k:(v["games"],v["outcome"]) for k,v in content["groups"].items()})

if __name__=="__main__":main()

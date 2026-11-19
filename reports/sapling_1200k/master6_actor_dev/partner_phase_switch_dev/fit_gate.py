"""Fit a two-bin public switch and decide when a still-hidden development panel may be opened."""
from collections import Counter
from itertools import chain
from statistics import mean
import sys
import numpy as np
from register import HERE,read,seal,sha
from policy import estimate

def checked(stage):
    roots=read(HERE/"roots.json")["roots"];index={r["id"]:r for r in roots if r["split"]==stage}
    f=HERE/"responses"/(stage+"_labels.json");label=read(f)
    assert label["roots_sha256"]==sha(HERE/"roots.json")
    pairs=[]
    for p in label["pairs"]:
        r=index[p["id"]];a,b=p["it20"],p["it50"]
        assert p["stream"] in ("primary","repeat")
        assert abs(p["delta"]-(a["margin"]-b["margin"]))<1.e-6
        pairs.append({"id":p["id"],"regime":r["regime"],"stream":p["stream"],"map":r["map"],"seat":r["seat"],
            "t":r["t"],"delta":p["delta"],"it20":a,"it50":b})
    want={(r["id"],s) for r in index.values() for s in (("primary","repeat") if r['repeat'] else ("primary",))}
    assert len(pairs)==len(want) and {(p['id'],p['stream']) for p in pairs}==want
    return pairs,sha(f)

def payoff(cards,selected):
    assert len(cards)==len(selected)
    vals=[];matches=[]
    for r,choice in zip(cards,selected):
        x=r[choice]["margin"]
        vals.append([x-r['it50']['margin'],x-r['it20']['margin']])
        matches.append({"id":r["id"],"map":r["map"],"seat":r["seat"],"t":r["t"],"regime":r["regime"],"pick":choice,
            "margin":x,"win":r[choice]['win'],"it20_margin":r["it20"]["margin"],"it20_win":r["it20"]["win"],
            "it50_margin":r["it50"]["margin"],"it50_win":r["it50"]["win"],"it20_minus_it50":r["delta"]})
    a=np.array(vals);ix=np.random.default_rng(60155).integers(0,len(a),(4000,len(a)));ci=np.quantile(a[ix].mean(1),[.025,.975],axis=0).T
    return {"rows":matches,"wins":int(sum(r["win"] for r in matches)),"it20_wins":int(sum(r["it20_win"] for r in matches)),
            "it50_wins":int(sum(r["it50_win"] for r in matches)),"picks":dict(Counter(selected)),
            "mean_delta_vs_it50":float(a[:,0].mean()),"mean_delta_vs_it20":float(a[:,1].mean()),
            "map_bootstrap95_vs_it50":ci[0].tolist(),"map_bootstrap95_vs_it20":ci[1].tolist()}

def train():
    allrows,labels=checked("train");main=[r for r in allrows if r["stream"]=="primary"]
    sign_flip=[];folds=[]
    for r in allrows:
        if r["stream"]=="repeat":
            first=next(x for x in main if x["id"]==r["id"])
            sign_flip.append({"id":r["id"],"regime":r["regime"],"primary":first["delta"],"repeat":r["delta"],
                              "reversal":first["delta"]*r["delta"]<0})
    for root in main:
        fit=estimate([r for r in main if r["map"]!=root["map"]])
        folds.append(fit["regime_to_partner"][root["regime"]])
    cv=payoff(main,folds);model=estimate(main)
    conditions={"both_partners_in_leave_one_map_out":len(set(folds))==2,"higher_than_it50_in_leave_one_map_out":cv["mean_delta_vs_it50"]>0,
              "higher_than_it20_in_leave_one_map_out":cv["mean_delta_vs_it20"]>0,"not_both_repeat_signs_reverse":not all(r["reversal"] for r in sign_flip)}
    seal(HERE/"weights.json",{"roots_sha256":sha(HERE/"roots.json"),"training_labels_sha256":labels,**model})
    seal(HERE/"train_gate.json",{"decision":"PASS" if all(conditions.values()) else "STOP","roots_sha256":sha(HERE/"roots.json"),
        "train_labels_sha256":labels,"weights_sha256":sha(HERE/"weights.json"),"conditions":conditions,
        "loo":cv,"independent_stream_repeats":sign_flip,"training_fitted":payoff(main,[model["regime_to_partner"][r["regime"]] for r in main])})
    print(conditions,model)

def held():
    gate=read(HERE/"train_gate.json");assert gate['decision']=='PASS'
    main,labels=checked("held");m=read(HERE/"weights.json")
    assert sha(HERE/"weights.json")==gate["weights_sha256"]
    direct=payoff(main,[m["regime_to_partner"][r["regime"]] for r in main])
    conditions={"better_margin_vs_both":min(direct["mean_delta_vs_it20"],direct["mean_delta_vs_it50"])>0,
                "no_fewer_wins_than_either":direct["wins"]>=max(direct["it20_wins"],direct["it50_wins"])}
    seal(HERE/"held_gate.json",{"decision":"PASS" if all(conditions.values()) else "STOP",
        "held_labels_sha256":labels,"roots_sha256":sha(HERE/"roots.json"),"train_gate_sha256":sha(HERE/"train_gate.json"),
        "conditions":conditions,"direct":direct,"independent_full_games":0})
    print(conditions)
if __name__=="__main__":{"train":train,"held":held}[sys.argv[1]]()

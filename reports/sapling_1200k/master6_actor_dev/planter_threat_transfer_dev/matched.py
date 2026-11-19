"""The same selected planter at the same time under A, B and the original parent."""
from collections import Counter,defaultdict
import numpy as np
from registry import HERE,read,sha,save
import risk

def load():
    out=[]
    for g in read(HERE/"ladder_pairs.json.gz")["games"]:
        src=g["source"]
        seen={r["window"]:r for r in g["eligible"]}
        for w in (0,1):
            row=seen.get(w)
            out.append({"map":src["terrain"],"id":src["id"],"seat":src["seat"],"window":w,"split":src["split"],
                        "opponent":src["opponent"],"population":"ladder","cohort":src["cohort"],"choice":row})
    for f in sorted((HERE/"parent_panel/games").glob("*.json.gz")):
        g=read(f)["game"];seen={r["window"]:r for r in g["selected"]}
        for w in (0,1):
            r=seen.get(w);choice=None
            if r:
                reach=r["public_reach"]["existing"];pos=r["cell"]
                def dist(role):return min((int(v["from"]!=pos) for v in reach if v["role"]==role),default=2)
                def arm(k):
                    v=r[k];return {"survived":v["survived"],"paid":bool(v["booked"]),"expense":v["worker_debit"],"post_cash":v["cash_after"]}
                choice={"t":r["t"],"uid":r["uid"],"type":r["kind"],"changed":r["changed"],"enemy_collector_dist":dist("COLLECTOR"),
                        "enemy_planter_dist":dist("PLANTER"),"actual_enemy_arrivals":r["pay"]["enemy_real_existing_arrivals"],
                        "branches":{"A":arm("pay"),"parent":arm("parent")}}
            out.append({"map":g["map"],"id":g["seed"],"seat":g["seat"],"window":w,"split":"parent_original","population":"parent_original",
                        "cohort":"parent_original","opponent":"original-Master3s2","choice":choice})
    return out

def summarize(rows):
    saved=Counter();maps=defaultdict(Counter);rates=defaultdict(Counter)
    for r in rows:
        books=maps[r["map"]];saved["windows"]+=1;books["windows"]+=1
        if not r["choice"]:continue
        saved["chosen"]+=1;c=r["choice"];saved["changed"]+=int(c["changed"])
        parent,a=c["branches"]["parent"],c["branches"]["A"]; b=parent if c["veto"] else a
        saved["veto"]+=int(c["veto"])
        if c['changed']:
            g=rates["reachable" if c["near"] else "unreachable"]
            g["roots"]+=1;g["actual_enemy_arrived"]+=bool(c["actual_enemy_arrivals"])
            g["parent_survival"]+=parent["survived"];g["A_survival"]+=a["survived"];g["A_paid"]+=a["paid"]
            g["B_survival"]+=b["survived"];g["B_paid"]+=b["paid"]
        for k,x in (("parent",parent),("A",a),("B",b)):
            saved[k+"_paid"]+=int(x["paid"]);saved[k+"_survival"]+=int(x["survived"]);saved[k+"_expense"]+=x["expense"]
        saved["B_saved_lives"]+=int(b["survived"] and not a["survived"])
        saved["B_extra_deaths"]+=int(a["survived"] and not b["survived"])
        saved["B_forgone_successful_purchases"]+=int(a["paid"] and not b["paid"])
        books["delta_lives"]+=int(b["survived"])-int(a["survived"]);books["delta_pays"]+=int(b["paid"])-int(a["paid"])
    rng=np.random.default_rng(428781)
    mods=sorted(maps);ix=rng.integers(0,len(mods),(4000,len(mods)))
    ci={}
    for k in ("delta_lives","delta_pays"):
        a=np.array([maps[m][k]/maps[m]["windows"] for m in mods]);res=a[ix].mean(1)
        ci[k]={"mean_per_registered_window":float(a.mean()),"map_bootstrap95":[float(x) for x in np.quantile(res,[.025,.975])]}
    return {"physical_maps":len(maps),"counts":dict(saved),"changed_by_reach":{k:dict(v) for k,v in rates.items()},"B_minus_A":ci}

def main():
    m=read(HERE/"risk_weights.json");rows=load()
    for r in rows:
        c=r["choice"]
        if not c:continue
        dc,dp=c["enemy_collector_dist"],c["enemy_planter_dist"];kind=c["type"]
        c["near"]=min(dc,dp)<=1;c["death_prediction"]=risk.estimate(m,dc,dp,kind)
        c["veto"]=risk.pick_parent(m,c["changed"],dc,dp,kind)
    groups={}
    for k in ("new_opponent","known_new_map","parent_original"):
        ss=[r for r in rows if r["split"]==k];groups[k]=summarize(ss)
    rivals={r:summarize([x for x in rows if x["opponent"]==r]) for r in sorted({x["opponent"] for x in rows})}
    save(HERE/"matched.json.gz",{"model_sha256":sha(HERE/"risk_weights.json"),"ladder_pairs_sha256":sha(HERE/"ladder_pairs.json.gz"),
        "parent_receipt_sha256":sha(HERE/"parent_panel/summary.json"),"by_split":groups,"by_opponent":rivals,"rows":rows})
    for k,v in groups.items():print(k,v["counts"],flush=True)
if __name__=="__main__":main()

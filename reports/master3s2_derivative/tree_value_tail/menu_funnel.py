"""Count observed action opportunities at issuance, pre-pruning menu and final H8 selection."""
from collections import Counter
import itertools

from study import HERE,P, D, advance, check, dump, read, setup, source, unique_pair, views

def run():
    selection=read(HERE/"selection.json")
    scores=selection["scans"]
    result=[]; totals=Counter()
    for ticket in P["tickets"][:8]:
        frozen=[r for r in scores if r["seed"]==ticket["seed"]]
        if not frozen:continue
        _,actions,cfg,env,sim=setup(ticket,source(ticket))
        times={r["step"]:r for r in frozen}
        for t in range(max(times)+1):
            row=times.get(t)
            if row:
                obs=views(env)[ticket["seat"]]
                menu=row.get("menu",[]); allowable=[int(k) for k in row.get("native_scores",{})]
                pairs=[]
                for i,j in itertools.combinations(range(len(menu)),2):
                    diff=unique_pair(obs,cfg,menu[i],menu[j])
                    if diff:
                        pairs.append(dict(first=i,second=j,worker=diff["worker"],cell=diff["cell"],
                                          h8_both=row["complete"] and i in allowable and j in allowable))
                outcome=dict(seed=row["seed"],seat=row["seat"],step=t,affordable_planters=len(row["available"]),
                             public_call="clock" in row,H=row.get("clock",{}).get("H"),
                             menu_size=len(menu),finalists_if_complete=len(allowable) if row.get("complete") else None,
                             any_before_pruning=bool(pairs),any_in_H8=any(p["h8_both"] for p in pairs),compared_pairs=pairs)
                if "clock" in row:outcome["search_wall"]=row["clock"]["wall"]
                result.append(outcome)
            if t<max(times):advance(env,sim,actions[t])
    for r in result:
        totals["scanned"]+=1;totals["funded_empty_planter"]+=int(r["affordable_planters"]>0)
        totals["called"]+=int(r["public_call"]);totals["complete_h8"]+=int(r["finalists_if_complete"] is not None)
        totals["any_pure_pair_before_prune"]+=int(r["any_before_pruning"]);totals["any_pure_pair_at_h8"]+=int(r["any_in_H8"])
        totals["before_pair_and_native_H8_complete"]+=int(r["any_before_pruning"] and r["finalists_if_complete"] is not None)
    check(totals["scanned"]==len(scores),"Dropped root")
    dump("menu_funnel.json",{"totals":dict(totals),"rows":result,"unchecked_design_slots":88-len(scores)})

if __name__=="__main__":run()

"""Exact frozen Sapling A and its parent on source ladder boards; no new game and one shared enemy command."""
from collections import Counter
from copy import deepcopy
import gzip
import importlib.util
import sys
from pydantic_core import from_json
from registry import HERE,ACTOR,BASE,A,REPLAYS,read,sha,save

sys.path.insert(0,str(ACTOR/"lot_settlement_dev"))
import lots
spec=importlib.util.spec_from_file_location("frozen_pay_opportunities", A/"opportunities.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def dist(a,b):
    y,x=divmod(a,15);v,w=divmod(b,15)
    return min(abs(y-v),15-abs(y-v))+min(abs(x-w),15-abs(x-w))

def apply_branch(sim,s,uid,joint):
    branch=lots.FastCarbon(sim.config);branch.set_state(sim.get_state())
    # The official Board checks players, cash, trees, visible/precomputed carbon and UIDs.
    _,_,obs=lots.one_step(branch,joint,s,verify=True)
    purchases=[(k,-v) for k,p,v,*a in branch.cash.events if k in ("plant","seize") and p==s and a[0]==uid]
    assert len(purchases)<=1
    return {"survived":uid in obs["players"][s][2],"paid":bool(purchases),"expense":purchases[0][1] if purchases else 0.,
            "kind":purchases[0][0] if purchases else None,"post_cash":[p[0] for p in obs["players"]]}

def run(row,model):
    path=REPLAYS/(row["id"]+".json.gz")
    assert sha(path)==row["raw_sha256"]
    game=from_json(gzip.decompress(path.read_bytes()))
    frames=game["steps"]; s=row["seat"]; cfg=game["configuration"]
    init=frames[0][0]["observation"]
    import hashlib
    assert hashlib.sha256(bytes(int(x) for x in init["full_carbon"])).hexdigest()==row["terrain"]
    sim=lots.FastCarbon(cfg).load_observation(init)
    selected=[]; examined=0
    for t in range(min(len(frames)-1,236)):
        snapshot=frames[t][0]["observation"];current=sim.full_observation()
        assert current["players"]==snapshot["players"] and current["trees"]==snapshot["trees"]
        observed=[p["action"] or {} for p in frames[t+1]]
        obs=sim.obs(s)
        window=0 if 45<=t<=125 else 1 if 150<=t<=235 else None
        if t==0 or (window is not None and window not in model._PAY_USED):
            oldsize=0 if t==0 else len(model._PAY_TRACE)
            commands=model.agent(obs,cfg);examined+=1
            if len(model._PAY_TRACE)>oldsize:
                trace=model._PAY_TRACE[-1]
                focal=trace["uid"]; pos=trace["cell"]; orig=trace["base"]
                changed={k for k in orig.keys()|commands.keys() if orig.get(k)!=commands.get(k)}
                assert changed==({focal} if trace["changed"] else set())
                assert commands.get(focal,"NONE")=="NONE"
                us=obs["players"][s];enemy=obs["players"][1-s]
                distances={r:min((dist(pos,w[0]) for w in enemy[2].values() if w[2]==r),default=99) for r in ("COLLECTOR","PLANTER")}
                dest=sim._move_nbr
                actual=[wid for wid,w in enemy[2].items() if (dest[observed[1-s][wid]][w[0]] if observed[1-s].get(wid) in dest else w[0])==pos]
                both={}
                for name,action in (("parent",orig),("A",commands)):
                    orders=observed.copy();orders[s]=action
                    both[name]=apply_branch(sim,s,focal,orders)
                selected.append({"window":window,"t":t,"uid":focal,"pos":pos,"type":trace["kind"],"quoted":trace["quoted_cost"],
                                 "changed":trace["changed"],"parent_command":trace["pre"],"enemy_collector_dist":distances["COLLECTOR"],
                                 "enemy_planter_dist":distances["PLANTER"],"actual_enemy_arrivals":actual,"branches":both})
        elif len(model._PAY_USED)<2:model._FEAT.observe(obs)
        if len(model._PAY_USED)>=2:break
        sim.step(*observed)
    return {"source":row,"actual_frozen_calls":examined,"eligible":selected}

def main():
    plan=read(HERE/"protocol.json")
    assert sha(HERE/"panel.json")==plan["small_panel_sha256"] and sha(A/"candidate.py")==plan["A_sha256"]
    rows=read(HERE/"panel.json")["maps"];results=[]
    model=old.pack()
    outdir=HERE/"ladder_games";outdir.mkdir(exist_ok=True)
    for row in rows:
        path=outdir/(row["id"]+".json.gz")
        if not path.exists():
            save(path,run(row,model))
        d=read(path);assert d["source"]==row
        results.append(d)
    save(HERE/"ladder_pairs.json.gz",{"protocol_sha256":sha(HERE/"protocol.json"),"runner_sha256":sha(HERE/"ladder_pairs.py"),
                                     "games":results,"new_live_games":0,"scope":"A emits its own joint orders on the actual historical ladder public observations. Original and A branches share all nonfocal orders and the same logged simultaneous opponent response. Official one-step labels only."})
    print(len(results),sum(x["changed"] for d in results for x in d["eligible"]))

if __name__=="__main__":main()

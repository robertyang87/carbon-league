"""Training maps only. Trace real transfers to an enemy collector with its pre-collision burden."""
from collections import Counter,deque
import sys
from shared import HERE,ROOT,ACTOR,dist,move,read,save,sha,source
sys.path.insert(0,str(ROOT/"code"))
from fastsim import FastCarbon
sys.path.insert(0,str(ACTOR/"lot_settlement_dev"))
import lots

def snapshot(obs,side,uid,foeus,orders):
    own=obs["players"][side];enemy=obs["players"][1-side]
    u=own[2].get(uid);v=enemy[2].get(foeus)
    if u:
        near=[[i,w[0],round(w[1],6),dist(w[0],u[0])] for i,w in enemy[2].items()
              if w[2]=="COLLECTOR" and w[1]<=u[1] and dist(w[0],u[0])<=3]
    else:near=[]
    return {"step":obs["step"],"cargo_worker":u,"realized_receiver":v,
            "our_action":orders[side].get(uid,"NONE") if u else None,
            "receiver_action":orders[1-side].get(foeus,"NONE") if v else None,
            "distance":dist(u[0],v[0]) if u and v else None,
            "distance_home":min((dist(u[0],p) for p in own[1].values()),default=None) if u else None,
            "public_own_tile":obs["carbon"][u[0]] if u else None,
            "near_lighter_or_equal":sorted(near),"cash":[p[0] for p in obs["players"]]}

def panel(row):
    tape=source(row);side=row["seat"];sim=FastCarbon(tape["game_configuration"]).reset(seed=row["seed"])
    history=deque(maxlen=11);events=[];dismissed=Counter()
    for t,joint in enumerate(tape["actions"]):
        obs=sim.obs(side)
        history.append((obs,joint))
        # Official Board and simulator must give the same actions->cash/players/trees outcome.
        _,_,post=lots.one_step(sim,joint,side,verify=True)
        for uid,p,v,beneficiary,winning_side in sim.last_collisions:
            if p!=side:continue
            pre=obs["players"][side][2].get(uid)
            if not pre or pre[2]!="COLLECTOR" or pre[1]<80:continue
            if winning_side!=1-side:
                dismissed["friendly_relay" if winning_side==side else "no_worker_winner"]+=1;continue
            assert beneficiary==winning_side
            dest=move(pre[0],joint[side].get(uid,"NONE"))
            winner,=[u for u,w in post["players"][1-side][2].items() if w[0]==dest and w[2]=="COLLECTOR"]
            enemy=obs["players"][1-side][2].get(winner)
            enemy_order=joint[1-side].get(winner,"NONE") if enemy else "RECRUIT_HERE"
            enemy_cargo=enemy[1]*(1-tape["game_configuration"].get("moveCost",0.)*(enemy_order not in ("NONE","RECRUIT_HERE"))) if enemy else 0.
            if not enemy_cargo<v:raise ValueError("enemy winner must be strictly lighter")
            entries=list(history)
            back={str(n):snapshot(*[entries[-1-n][0],side,uid,winner,entries[-1-n][1]]) if len(entries)>n else None for n in (0,3,5,10)}
            events.append({"t":t,"uid":uid,"winner":winner,"root_principal":pre[1],"actual_transferred":v,
                    "destination":dest,"original":joint[side].get(uid,"NONE"),"enemy_pre_collision_cargo":enemy_cargo,
                    "receiver_command":enemy_order,"was_empty":enemy_cargo<=1e-9,"back":back})
    assert sim.step_count==tape["steps"]
    return {k:row[k] for k in ("seed","map","seat","raw_sha256")}|{"events":events,"other_high_cargo_losses":dict(dismissed)}

def main():
    inv=read(HERE/"source/inventory.json")
    cases=[panel(r) for r in inv["games"] if r["split"]=="train"]
    out={"source_sha256":sha(HERE/"source/inventory.json"),"threshold":80,"scope":"Entire baseline on training maps only. Event is direct source-worker loss to a surviving strictly lighter enemy collector. Winner identity and future outcomes are retrospective.","cases":cases}
    save(HERE/"historical_train_losses.json.gz",out)
    print("verified historical recorded transfers",len(cases),sum(len(c["events"]) for c in cases),flush=True)

if __name__=="__main__":main()

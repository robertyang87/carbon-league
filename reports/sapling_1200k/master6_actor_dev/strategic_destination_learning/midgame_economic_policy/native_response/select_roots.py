"""Select the first registered economic disagreement using only the acting seat's public history."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
sys.path.insert(0,str(STUDY/"policy_window_returns/runner"))
import run as common

SOURCES={621758540:(0,"550446ca58b03cd4a54525be372dbfc5743ed3351d9dc3f0cbc1560ed4740194"),
         621758541:(1,"ab7348a12f18b32c8e717a83940bb0372c5bc230540d6113fcf4758b0f4cd850")}
HASH={"uniform":"2261ff2c4c35f99e20ea942db808c172f963dbc3a773941d3ab7fad912f358fe",
      "weighted":"de4a920d6528a54b8f852d91c7f0f4b245104797baecdacf56339d63ccfa2916"}
MOVE={"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1),"NONE":(0,0)}


def destination(p,act):
    x,y=divmod(p,15)
    a,b=MOVE[act]
    return 15*((x+a)%15)+(y+b)%15


def distance(p,q):
    dy,dx=abs(p//15-q//15),abs(p%15-q%15)
    return min(dy,15-dy)+min(dx,15-dx)


def traits(obs,cmd,cfg):
    seat=obs["player"]
    own,enemy=(obs["players"][i] for i in (seat,1-seat))
    own_bases=set(own[1].values()); bases=own_bases|set(enemy[1].values())
    own_trees={v[0] for v in own[3].values()}
    rival_trees={v[0] for v in enemy[3].values()}
    num_trees=len(own_trees)+len(rival_trees)
    plant_cost=cfg["plantCost"]+(cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**num_trees
                                 if len(own_trees)>cfg["playerTreeProtectiveNumber"] else 0)
    foes=[(p,c) for p,c,k in enemy[2].values() if k=="COLLECTOR"]
    at_risk=[(p,c) for p,c,k in own[2].values() if k=="COLLECTOR" and c>=75
             and any(f<c and distance(q,p)<=3 for q,f in foes)]
    states={}
    for uid,(p,c,kind) in sorted(own[2].items()):
        action=cmd.get(uid,"NONE"); q=destination(p,action)
        status={"uid":uid,"position":p,"cargo":c,"role":kind,"action":action,"destination":q}
        flags={}
        if kind=="COLLECTOR":
            if c>0 and own_bases:
                flags["loaded_home_distance"]=min(distance(q,k) for k in own_bases)
            flags["visible_harvest"]=action=="NONE" and p not in bases|own_trees|rival_trees and obs["carbon"][p]>0
            threats=[e for e,v in foes if v<c and distance(e,p)<=3]
            flags["withdraw"]=bool(c>=75 and threats and
                                   min(distance(q,e) for e in threats)>min(distance(p,e) for e in threats))
            flags["shield"]=any(c<h and distance(q,v)<=1<distance(p,v) for v,h in at_risk)
        else:
            flags["plant_now"]=(action=="NONE" and p not in bases|own_trees|rival_trees and own[0]>=plant_cost)
            flags["seize_now"]=(action=="NONE" and p in rival_trees and own[0]>=cfg["seizeCost"])
        states[uid]={"status":status,"geometry":flags}
    return states


def select(views,cfg,actors):
    log=[]
    for obs in views:
        t=obs["step"]
        if t>100:
            break
        if t!=0 and t<40:
            for player in actors.values():
                common.observe_only(player,obs)
            continue
        orders={n: p.act(obs,cfg) for n,p in actors.items()}
        if t<40:
            continue
        facts={n:traits(obs,cmd,cfg) for n,cmd in orders.items()}
        reasons=[]
        for uid,x in facts["uniform"].items():
            y=facts["weighted"][uid]
            if x["status"]["action"]==y["status"]["action"]:
                continue
            fields=[k for k in x["geometry"] if x["geometry"][k]!=y["geometry"][k]]
            if fields:
                reasons.append({"uid":uid,"changed_geometry":fields,"uniform":x,"weighted":y})
        log.append({"step":t,"public":common.digest(obs),"joint_changed":len(set(map(common.digest,orders.values())))>1,
                    "qualifying_units":len(reasons)})
        if reasons:
            return {"selected":True,"root_step":t,"root_public_sha256":common.digest(obs),"orders":orders,
                    "reasons":reasons,"scan":log}
    return {"selected":False,"scan":log}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipts",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    files={n:HERE.parent/(n+".py") for n in HASH}
    for n,f in files.items():
        common.need(common.sha(f.read_bytes())==HASH[n],"Candidate drift")
    rows=[]
    for seed,(seat,sha) in SOURCES.items():
        directory=args.receipts/"fit"/f"{seed}_{seat}"
        path=directory/"source.json.gz"
        common.need(common.sha(path.read_bytes())==sha,"Historical source drift")
        raw=json.loads(gzip.decompress(path.read_bytes()))
        # Deliberately pass no opponent commands, hidden data, scores, current action or future rows into selection.
        views=(r["public"] for r in raw["rows"] if r["step"]<=100)
        cfg={k:v for k,v in raw["configuration"].items() if k!="randomSeed"}
        actors={n:common.collect.Player(f) for n,f in files.items()}
        chosen=select(views,cfg,actors)
        rows.append({"seed":seed,"seat":seat,"source_sha256":sha,"terrain":raw["terrain"],**chosen})
    common.save(args.out,{"selector_sha256":common.sha(Path(__file__).read_bytes()),"candidates":HASH,"tickets":rows})
    print([(r["seed"],r.get("root_step"),[z["changed_geometry"] for z in r.get("reasons",[])]) for r in rows])


if __name__=="__main__":
    main()

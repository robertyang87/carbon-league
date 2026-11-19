"""Freeze public tree decisions and run one paired fully responsive terminal label per root."""
import argparse
from itertools import product
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/"late_response_training"))
sys.path.insert(0,str(BASE/"root_recovery_branches"))
from branches import begin, clean, destination, digest, distances, MOVES, public, PACK_HASH
from batch import load_case
from live import save
import live
from importlib.util import spec_from_file_location,module_from_spec

P=HERE/"protocol.json"
def read(path): return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())
def sha(path):return digest(path.read_bytes())


def proposals(obs,cfg,acts,seed):
    s=obs["player"];t=obs["step"];n=cfg["size"];dist=distances(n)
    my,foe=obs["players"][s],obs["players"][1-s]
    homes=set(my[1].values());enemy_homes=set(foe[1].values())
    new={cell for uid,cell in my[1].items() if acts.get(uid) in ("RECCOLLECTOR","RECPLANTER")}
    global_workers=[w for side in obs["players"] for w in side[2].values()]
    reserve=0.
    for bid in my[1]:
        p=acts.get(bid)
        if p in ("RECCOLLECTOR","RECPLANTER"):
            typ="Collector" if p=="RECCOLLECTOR" else "Planter"
            reserve+=cfg["rec"+typ+"Cost"] + cfg["rec"+typ+"IncreaseCost"]*sum(w[2]==typ.upper() for w in global_workers)
    for uid,(pos,load,kind) in my[2].items():
        old=acts.get(uid,"NONE")
        enemies=[w for w in foe[2].values() if w[2]=="PLANTER"]
        near=min((dist[pos][w[0]] for w in enemies),default=n*2)
        for family,trees in (("seize_enemy_tree",foe[3]),("hold_own_tree",my[3])):
            if family=="seize_enemy_tree" and (kind!="PLANTER" or my[0]-reserve<cfg["seizeCost"]):continue
            if family=="hold_own_tree" and near>2:continue
            for tid,(cell,age) in trees.items():
                lifetime=cfg["treeLifespan"]-age
                if cell!=pos or lifetime<4:continue
                blocked=enemy_homes|new|{destination(z[0],acts.get(other,"NONE"),n) for other,z in my[2].items() if other!=uid}
                if pos in blocked:continue
                choices=[]
                for move in MOVES:
                    if (old=="NONE")== (move=="NONE") or move==old:continue
                    dest=destination(pos,move,n)
                    if dest in blocked or destination(pos,old,n) in blocked:continue
                    departure=dest if move!="NONE" else destination(pos,old,n)
                    tie=digest(f"midtree-1|{seed}|{s}|{t}|{uid}|{tid}|{move}".encode())
                    priority=(min(dist[departure][b] for b in homes),tie)
                    prior_income=obs["trees"][tid][1]
                    root=dict(t=t,seat=s,seed=seed,family=family,worker=uid,role=kind,origin=pos,cargo=load,old=old,alternative=move,
                              tree=tid,cell=pos,tree_age=age,tree_life=lifetime,tree_previous_cash=prior_income,previous_owner=s if family=="hold_own_tree" else 1-s,
                              nearest_enemy_planter=near if enemies else None,own_cash=my[0],enemy_cash=foe[0],hire_reserve=reserve,
                              original_landing=destination(pos,old,n),alternative_landing=dest,rank_hash=tie)
                    choices.append((priority,root))
                if choices:yield min(choices,key=lambda a:a[0])[1]


def standalone(root):
    return (-(root["t"]>=200),-(root["tree_previous_cash"]*root["tree_life"]),root["rank_hash"])


def register():
    cfg=read(P);out=HERE/"roots.json"
    if out.exists():
        r=read(out)
        if r["protocol_sha256"]!=sha(P) or r["script_sha256"]!=sha(Path(__file__)):raise ValueError("frozen contract changed")
        return r
    pools={key:[] for key in cfg["trees"]};hist={};records={};sources={}
    for seed,folder in cfg["sources"]:
        for seat in (0,1):
            raw,rec=load_case(ROOT/folder,seed,seat);env=begin(raw);count={k:0 for k in pools}
            records[seed,seat]=rec;sources[seed]=folder
            for t,j in enumerate(raw["actions"]):
                if t>cfg["phase"][1]:break
                if t>=cfg["phase"][0]:
                    for r in proposals(public(env,seat),raw["game_configuration"],clean(j[seat]),seed):
                        pools[r["family"]].append(r);count[r["family"]]+=1
                env.step(j)
            hist[f"{seed}/{seat}"]=count
    choices=[]
    for a,b in product(*pools.values()):
        if a["seed"]==b["seed"] or a["seat"]==b["seat"]:continue
        rank=(-sum(x["t"]>=200 for x in (a,b)),-(b["role"]=="COLLECTOR"),b["nearest_enemy_planter"],
              -sum(x["tree_previous_cash"]*x["tree_life"] for x in (a,b)),a["rank_hash"],b["rank_hash"])
        choices.append((rank,(a,b)))
    chosen=min(choices,key=lambda x:x[0])[1] if choices else next(([min(v,key=standalone)] for v in pools.values() if v),[])
    if not chosen:raise ValueError("no eligible tree decision")
    reg=dict(protocol_sha256=sha(P),script_sha256=sha(Path(__file__)),eligible=hist,selected=[],compatible_pairs=len(choices))
    for r in chosen:
        seed,seat=r["seed"],r["seat"];wd=HERE/r["family"];wd.mkdir(exist_ok=True)
        raw,rec=load_case(ROOT/sources[seed],seed,seat);env=begin(raw)
        for j in raw["actions"][:r["t"]]:env.step(j)
        obs=public(env,seat);cmd=clean(raw["actions"][r["t"]][seat])
        assert r in list(proposals(obs,raw["game_configuration"],cmd,seed))
        save(wd/"root_input.json.gz",{"public_observation":obs,"own_proposed_joint_action":cmd,"alternative":r["alternative"],"worker":r["worker"]})
        case=dict(seed=seed,seat=seat,family=r["family"],root_t=r["t"],root=r,source_dir=sources[seed],record=rec,
                  protocol_sha256=sha(P),script_sha256=reg["script_sha256"],root_input_sha256=sha(wd/"root_input.json.gz"),
                  full_search_runner_sha256=sha(Path(live.__file__)))
        save(wd/"live_protocol.json",case);reg["selected"].append(dict(**r,case_sha256=sha(wd/"live_protocol.json")))
    save(out,reg);return reg


def verify(wd,case):
    verifier=BASE/"planter_routes_dev/endgame_responses";sys.path.insert(0,str(verifier))
    spec=spec_from_file_location("locked_tree_economy_ledger",verifier/"verify.py");m=module_from_spec(spec);spec.loader.exec_module(m)
    s=case["seat"];o=1-s;r=case["root"];raw,rec=load_case(ROOT/case["source_dir"],case["seed"],s);assert rec==case["record"]
    results={}
    for arm in ("parent","alternative"):
        info=read(wd/f"live_{arm}.json");log=wd/info["action_tape"];tail=read(log)
        assert sha(log)==info["action_sha256"] and info["pack_sha256"]==PACK_HASH and info["source_sha256"]==live.MASTER_SHA
        assert (info["statuses"],info["shared_init_sha256"],info["protocol_sha256"])==(["DONE","DONE"],sha(wd/"live_shared_init.json"),sha(wd/"live_protocol.json"))
        root=raw["actions"][r["t"]];src,alt=clean(root[s]),clean(tail[0][s]);changed={u for u in src.keys()|alt.keys() if src.get(u)!=alt.get(u)}
        assert clean(root[o])==clean(tail[0][o]) and changed==({r["worker"]} if arm=="alternative" else set())
        assert alt.get(r["worker"],"NONE")==r["alternative" if arm=="alternative" else "old"]
        data=m.ledger(raw,r,tail)
        assert data["score_by_physical_seat"]==info["score"] and info["search_calls"]==len(tail)-1
        env=begin(raw)
        for a in raw["actions"][:r["t"]]:env.step(a)
        tracked=[];income=[0.,0.]
        for a in tail:
            env.step(a);obs=env.states[0].observation
            owner=next((k for k,side in enumerate(obs.players) if r["tree"] in side[3]),None)
            val=obs.trees[r["tree"]][1] if owner is not None else 0.
            tracked.append([obs.step,owner,val])
            if owner is not None:income[owner]+=val
        data.update(focal_tree_timeline=tracked,focal_tree_income=income,ledger_sha256=sha(verifier/"verify.py"),
                    official_search_calls=info["search_calls"],actual_searched=info["search_completed"])
        results[arm]=data
    a,b=results.values();eff={k:round(b["gross"][s][k]-b["gross"][o][k]-a["gross"][s][k]+a["gross"][o][k],4) for k in a["gross"][0]}
    dm=round(b["margin"]-a["margin"],4);assert abs(sum(eff.values())-dm)<.03
    out=dict(seed=case["seed"],seat=s,family=r["family"],root=r,root_input_sha256=case["root_input_sha256"],shared_init_sha256=sha(wd/"live_shared_init.json"),delta_margin=dm,delta_u=b["u"]-a["u"],cash_delta_by_channel=eff,arms=results)
    save(wd/"verified.json.gz",out)
    return out


def main():
    args=argparse.ArgumentParser(description=__doc__);args.add_argument("command",choices=["register","run","arm"]);args.add_argument("--family");args.add_argument("--arm",choices=("parent","alternative"));opts=args.parse_args()
    reg=register()
    if opts.command=="register":
        print(json.dumps(reg,ensure_ascii=False));return
    if opts.command=="arm":
        wd=HERE/opts.family;case=read(wd/"live_protocol.json")
        assert sha(Path(live.__file__))==case["full_search_runner_sha256"] and sha(wd/"root_input.json.gz")==case["root_input_sha256"]
        live.run(opts.arm,wd,case,resume=True);return
    with (HERE/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=[]
        for r in reg["selected"]:
            wd=HERE/r["family"];case=read(wd/"live_protocol.json");assert sha(wd/"live_protocol.json")==r["case_sha256"]
            live.prepare(wd)
            for arm in ("parent","alternative"):
                subprocess.run([sys.executable,str(Path(__file__).resolve()),"arm","--family",r["family"],"--arm",arm],check=True)
            v=verify(wd,case);result.append({k:x for k,x in v.items() if k!="arms"})
            save(HERE/"training_labels.json",result)


if __name__=="__main__":main()

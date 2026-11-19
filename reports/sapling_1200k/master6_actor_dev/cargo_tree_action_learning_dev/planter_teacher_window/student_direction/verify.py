"""Test released bundles on whole decoded commands and direction-only change at an identical prefix."""
from collections import defaultdict
import gzip
from pathlib import Path
import sys
import numpy as np
from dataset import HERE,BASE,PARENT,SEQ,read,write,sha,terr,policy


def normalized(z):
    z=z.astype(np.float64)-np.max(z)
    p=np.exp(z)
    return p/p.sum()


def root_replay(c,group,models,targets):
    cfg,actions,_,full=terr.load_game(c)
    sim=terr.FastCarbon(cfg).reset(seed=c["seed"],full_carbon=full)
    s=c["seat"];all_rows=[]
    for t,joint in enumerate(actions):
        if t>192:break
        view=sim.obs(s)
        if t==0 or t in group:
            commands={arm:model.agent(view,cfg) for arm,model in models.items()}
            for r in group.get(t,[]):
                uid=r["uid"]
                res=dict(map=c["terrain"],seed=c["seed"],source=c["source"],game=c["sha256"],t=t,seat=s,uid=uid,truth=r["action"])
                for arm,cmd in commands.items():
                    res[arm]=cmd.get(uid,"NONE")
                all_rows.append(res)
        else:
            for model in models.values():model._FEAT.observe(view)
        sim.step(*joint)
    return all_rows


def act12(models):
    decisions=[]
    for case in read(HERE.parent/'protocol.json')["cases"]:
        n=case["seed"];s=case["seat"]
        teacher=read(HERE.parent/f"{n}_{s}_teacher.json.gz");raw=read(terr.ROOT/case["path"])
        cfg=raw["game_configuration"];sim=terr.FastCarbon(cfg).reset(seed=n)
        record={r["t"]:r for r in teacher["decisions"]}
        for t,joint in enumerate(raw["actions"][:128]+teacher["tape"][:16]):
            view=sim.obs(s)
            if t==0 or 128<=t<144:
                cmds={arm:model.agent(view,cfg) for arm,model in models.items()}
                for uid in record.get(t,{}).get("changed_planters",[]):
                    row=record[t]
                    assert cmds["parent"].get(uid)==row["parent"].get(uid)
                    decisions.append(dict(map=case["terrain"],seed=n,seat=s,t=t,uid=uid,teacher=row["actual"][uid],
                           **{arm:cmd.get(uid,"NONE") for arm,cmd in cmds.items()}))
            else:
                for model in models.values():model._FEAT.observe(view)
            sim.step(*joint)
    assert len(decisions)==12
    return decisions


def main():
    rows=read(HERE/"rows.json.gz")
    dev=[(i,r) for i,r in enumerate(rows) if r["split"]=="test"]
    fit=read(HERE/"fit.json")
    paths=dict(parent=PARENT,candidate=HERE/"candidate.py",control=HERE/"control.py")
    assert all(sha(paths[a])==v["package"] for a,v in fit["arms"].items())
    models={a:policy.load(path) for a,path in paths.items()}
    f=np.load(HERE/"features.npz")
    validation=[]
    invariants={arm:dict(max_stay_mass_error=0.,max_move_mass_error=0.) for arm in ('candidate','control')}
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in dev:
        h,m,z=[f[k][i].astype(np.float64) for k in ('h','m','z')]
        orig=normalized(z)
        rec=dict(map=r['map'],source=r['source'],action=r['label'],weight=1/r['nplanters'],loss24=r['return24'],seat=r['seat'],uid=r['uid'],t=r['t'])
        groups[(r["game"],r["seat"])][r["t"]].append(r)
        for arm,model in models.items():
            adjusted=z if arm=="parent" else model._direction_scores(z,h,m)
            p=normalized(adjusted)
            if arm!="parent":
                item=invariants[arm]
                item["max_stay_mass_error"]=max(abs(p[0]-orig[0]),item["max_stay_mass_error"])
                item["max_move_mass_error"]=max(abs(p[1:].sum()-orig[1:].sum()),item["max_move_mass_error"])
            rec[arm]=-np.log(p[r["label"]]/sum(p[1:]))
        validation.append(rec)
    for v in invariants.values():
        assert v["max_stay_mass_error"]<1e-6 and v["max_move_mass_error"]<1e-6
    issued=[]
    for c in read(SEQ/"sources.json"):
        key=(c["sha256"],c["seat"])
        if c["split"]=="test" and key in groups:
            issued.extend(root_replay(c,groups[key],models,None))
    assert len(issued)==len(dev)
    qualitative=act12(models)
    metrics={}
    for source in ("all","Master3s2","Top5"):
        data=[r for r in validation if source=="all" or r["source"]==source]
        acts=[r for r in issued if source=="all" or r["source"]==source]
        maps=sorted({r["map"] for r in data})
        metrics[source]={}
        for arm in models:
            scores=[];hit=[]
            for mp in maps:
                rr=[r for r in data if r["map"]==mp]
                scores.append(np.average([r[arm] for r in rr],weights=[r["weight"] for r in rr]))
                orders=[r for r in acts if r["map"]==mp];hit.append(np.mean([a[arm]==a["truth"] for a in orders]))
            metrics[source][arm]=dict(conditional_nll=float(np.mean(scores)),published_hit_rate=float(np.mean(hit)),maps=len(maps),decisions=len(data))
    direct=[r for r in issued if r["parent"]!="NONE" and r["candidate"]!="NONE" and r["candidate"]!=r["parent"]]
    write(HERE/"held_actions.json.gz",issued);write(HERE/"conditional_losses.json.gz",validation);write(HERE/"teacher_examples.json",qualitative)
    write(HERE/"qualification.json",dict(candidate=sha(paths["candidate"]),weights=sha(HERE/"candidate.npz"),parent=sha(PARENT),control=sha(paths["control"]),
        sources=sha(SEQ/"sources.json"),fit=sha(HERE/"fit.json"),held_actions=sha(HERE/"held_actions.json.gz"),preservation=invariants,
        eligible=len({r["map"] for r in direct})>=2,actual_changed_directions=len(direct),changed_direction_terrains=len({r["map"] for r in direct}),
        actual_first_direction_changes=direct[:6],physical_deployment_metrics=metrics,qualitative_matches={arm:sum(r[arm]==r["teacher"] for r in qualitative) for arm in models}))
    print(metrics, len(direct),len(set(r["map"]for r in direct)),flush=True)


if __name__=="__main__":main()

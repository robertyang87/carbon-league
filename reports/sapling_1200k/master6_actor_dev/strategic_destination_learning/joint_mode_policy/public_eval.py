"""Replay eligible physical orders through each frozen candidate's ordinary entry."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import free_entry as previous
from free_entry import read,save,sha

ARMS=("parent","independent","joint")


def module(name,path):
    spec=importlib.util.spec_from_file_location("mode_"+name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def outcomes(rows):
    exact=Counter()
    right=Counter()
    total=0
    changed=Counter()
    for r in rows:
        roles=r["roles"]
        total+=len(roles)
        truth=r["truth"]
        orders=r["orders"]
        for arm,cmd in orders.items():
            exact[arm]+=cmd==truth
            right[arm]+=sum(cmd.get(w)==truth.get(w) for w in roles)
        changed["joint_vs_independent"]+=orders["joint"]!=orders["independent"]
        changed["joint_vs_parent"]+=orders["joint"]!=orders["parent"]
    return dict(n=len(rows),slots=total,correct=dict(right),exact=dict(exact),changed=dict(changed))


def main():
    study=HERE.parent
    protocol=read(HERE/"protocol.json")
    releases=read(HERE/"fit.json")["releases"]
    paths={a:((study/"bc.py") if a=="parent" else (HERE/(a+".py"))) for a in ARMS}
    for a,p in paths.items():
        expected=protocol["parent"] if a=="parent" else releases[a]["package_sha256"]
        if sha(p)!=expected:raise ValueError("changed candidate")
    if sha(study/"sources.json")!=protocol["sources"]:raise ValueError("source roster mismatch")
    models={a:module(a,p) for a,p in paths.items()}
    historic=read(study/"free_commands.json.gz")
    parent={(r["tape"],r["step"]):r["commands"]["bc"] for r in historic}
    cards=[c for c in read(study/"sources.json")["cards"] if c["split"] in ("val","dev")]
    rows=[]
    for c in cards:
        selected=set(c["registered_times"]);seat=c["expert_seat"]
        for step,obs,cfg,truth in previous.public_history(c):
            cmd={}
            for a,m in models.items():
                if step==0 or step in selected:
                    cmd[a]=m.agent(obs,{k:v for k,v in cfg.items() if k!="randomSeed"})
                else:
                    m._FEAT.observe(obs)
            if step in selected:
                if cmd["parent"]!=parent[c["sha256"],step]:
                    raise ValueError("historical BC public order mismatch")
                player=obs["players"][seat]
                roles={k:"base" for k in player[1]}
                roles.update({k:v[2] for k,v in player[2].items()})
                if any(set(a)-roles.keys() for a in [*cmd.values(),truth]):
                    raise ValueError("unknown actor")
                rows.append(dict(seed=c["seed"],split=c["split"],step=step,seat=seat,terrain=c["terrain"],
                                 tape=c["sha256"],roles=roles,truth=truth,orders=cmd))
        print(c["split"],c["seed"],seat,len(rows),flush=True)
    save(HERE/"free_orders.json.gz",rows)
    likelihood=read(HERE/"likelihood.json")
    summary={}
    for split in ("val","dev"):
        section=[r for r in rows if r["split"]==split]
        if len(section)!=192:raise ValueError("incomplete holdout")
        whole=outcomes(section)
        by_seat={s:outcomes([r for r in section if r["seat"]==s]) for s in (0,1)}
        by_map={t:outcomes([r for r in section if r["terrain"]==t]) for t in sorted({r["terrain"] for r in section})}
        per=likelihood[split]
        means={a:float(np.mean([v[a] for v in per.values()])) for a in ARMS}
        checks=dict(better_nll=means["joint"]<min(means["parent"],means["independent"]),
                    majority_nll=sum(v["joint"]<v["independent"] for v in per.values())>=3,
                    joint_orders=whole["exact"]["joint"]>whole["exact"]["parent"] and whole["exact"]["joint"]>=whole["exact"]["independent"],
                    seats=all(v["correct"]["joint"]>=v["correct"]["parent"] for v in by_seat.values()),
                    independent_disagreement=whole["changed"]["joint_vs_independent"]>=2)
        summary[split]=dict(whole=whole,by_map=by_map,by_seat=by_seat,mean_pre_guard_joint_nll=means,
                            checks=checks,passed=all(checks.values()))
    save(HERE/"assessment.json",dict(split=summary,qualifies=summary["dev"]["passed"],receipts=sha(HERE/"free_orders.json.gz")))
    print("historical development",summary["dev"]["whole"],summary["dev"]["checks"],flush=True)


if __name__=="__main__":
    main()

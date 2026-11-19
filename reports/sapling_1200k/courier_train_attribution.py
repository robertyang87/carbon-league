"""Compare the frozen V3 and V6b nets on the identical observations actually seen in each exam."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import external_3s2 as exam


def check(run, output):
    d=json.loads((run/"results.json").read_bytes())
    if d["status"]!="complete" or len(d["games"])!=2:
        raise ValueError("requires the complete two-seat development exam")
    root=exam.ROOT/"data/weights/sapling_1200k"
    parent=root/"cash_v3_agent.py"
    child=root/"courier_train_v6b_agent.py"
    if exam.prior.digest(child)!=d["header"]["pack_sha256"]:
        raise ValueError("candidate hash mismatch")
    rows=[]
    for r in d["games"]:
        path=run/r["raw_file"]
        if exam.prior.digest(path)!=r["raw_sha256"]:
            raise ValueError("raw transcript hash")
        raw=json.loads(gzip.decompress(path.read_bytes()))
        p=r["candidate_seat"];sim=exam.FastCarbon(raw["game_configuration"]).reset(seed=r["seed"])
        parent_agent=exam.prior.shared.module(parent,f"v3_shadow_{p}").agent
        child_agent=exam.prior.shared.module(child,f"v6b_shadow_{p}").agent
        changed=[]
        for t,actions in enumerate(raw["actions"]):
            obs=sim.obs(p)
            a0=parent_agent(obs,sim.config)
            a1=child_agent(obs,sim.config)
            if a1!=actions[p]:
                raise ValueError(f"deployed V6b command mismatch on seat {p} turn {t}")
            diffs=[]
            for uid in sorted(a0.keys()|a1.keys()):
                if a0.get(uid)==a1.get(uid):
                    continue
                w=sim.workers.get(uid)
                diffs.append(dict(uid=uid,parent=a0.get(uid,"NONE"),candidate=a1.get(uid,"NONE"),
                                  root_pos=w.pos if w else None,root_load=w.carbon if w else 0,
                                  role=("COLLECTOR" if w.is_collector else "PLANTER") if w else "BASE"))
            if diffs:
                changed.append(dict(t=t,units=diffs))
            sim.step(*actions)
        rows.append(dict(seed=r["seed"],seat=p,candidate_exact_replay_roots=len(raw["actions"]),changed_roots=len(changed),
                         changed_slots=sum(len(x["units"]) for x in changed),events=changed))
    with output.open("x") as f:
        json.dump(dict(candidate_sha256=exam.prior.digest(child),parent_sha256=exam.prior.digest(parent),games=rows),
                  f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();check(a.run,a.out)

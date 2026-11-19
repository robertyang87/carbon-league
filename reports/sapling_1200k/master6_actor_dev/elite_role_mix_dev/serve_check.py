"""Check normal deployed commands, legality and map-level differences on untouched maps."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"blended_actor_dev/serving"))
import probe


def main(directory=HERE):
    plan=json.loads((directory/"protocol.json").read_bytes())
    fitted=json.loads((directory/"fit.json").read_bytes())
    parent=HERE.parent/"actor_partner_dev/candidate.py"
    cand=directory/"candidate.py"
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=plan["parent"] or hashlib.sha256(cand.read_bytes()).hexdigest()!=fitted["candidate_sha256"]:
        raise ValueError("actor identity mismatch")
    models={k:probe.package(p,k) for k,p in (("parent",parent),("candidate",cand))}
    games=[probe.run_game(models,g,probe.STEPS) for g in plan["external_games"]]
    by_map=[]; total=Counter()
    for game in games:
        c=Counter()
        for root in game["roots"]:
            a=root["arms"]; c["roots"]+=1
            changed=a["parent"]["actual_command"]!=a["candidate"]["actual_command"]
            c["changed_joints"]+=changed
            if root["t"]==0 and changed:raise ValueError("opening action changed")
            c["t0"]+=root["t"]==0
            for slot in root["slots"].values():
                r=slot["role"]; out=slot["outputs"]; y=slot["demo_canonical"]
                # Restrict denominator to decisions offering >1 legal alternative in BOTH
                # histories so that changing the joint prefix cannot alter the scored sample.
                if any(len(out[arm]["legal"])<=1 for arm in out):
                    continue
                c["n"]+=1; c["n_"+r]+=1
                for arm,v in out.items():
                    c[arm]+=v["actual"]==y; c[arm+"_"+r]+=v["actual"]==y
            for arm in a:
                c[arm+"_whole"]+=a[arm]["actual_command"]==root["demonstration_command"]
        by_map.append(dict(id=game["id"],seed=next(g["seed"] for g in plan["external_games"] if g["id"]==game["id"]),
                           seat=game["seat"],wdl=game["wdl"],c=dict(c)))
        total+=c
    def delta(role):
        suffix="_"+role if role else ""
        values=np.array([(g["c"].get("candidate"+suffix,0)-g["c"].get("parent"+suffix,0))/g["c"]["n"+suffix]
                         for g in by_map if g["c"].get("n"+suffix,0)>0])
        boot=np.random.default_rng(4238741).choice(values,(5000,len(values)),replace=True).mean(1)
        return dict(maps=len(values),mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    root_fraction=total["changed_joints"]/total["roots"]
    official=delta(""); teacher=fitted["external"]["all"]["paired_map_delta"]
    decision=official["mean"]>0 and teacher>0 and root_fraction>=.05
    result=dict(parent_sha256=plan["parent"],candidate_sha256=fitted["candidate_sha256"],
                fixed_steps=list(probe.STEPS),eligible_for_fresh_research_match=bool(decision),
                stage="offline transfer only",teacher_forced_external=teacher,actual_delta=official,
                changed_root_rate=root_fraction,counts=dict(total),by_role={r:delta(r) for r in ("collector","planter","base")},
                by_map=by_map)
    with (directory/"serving.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    with (directory/"serving_roots.json.gz").open("xb") as f:
        f.write(gzip.compress(json.dumps(games,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    print(decision,official,root_fraction,flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else HERE)

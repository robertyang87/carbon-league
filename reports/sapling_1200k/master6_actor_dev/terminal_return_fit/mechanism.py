"""Blind functional parity and causal legal-prefix probes on two already exposed parent action tapes."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from build import HERE, ROOT, PLAN, digest, source, save
sys.path.insert(0,str(ROOT/"code"))
from fastsim import FastCarbon

PROBES=frozenset(range(4,299,25))


def run():
    ns={"__name__":"terminal_return_test","__file__":"single_file_residual_agent.py"}
    exec(compile(source([0]*4),ns["__file__"],"exec"),ns)
    times={"zero":[],"plus":[],"minus":[]}
    cases=[]
    changed=Counter()
    n=0
    for seat in (0,1):
        tape=HERE.parent/"hybrid_candidate"/f"raw_621458920_{seat}.json.gz"
        raw=json.loads(gzip.decompress(tape.read_bytes()))
        if raw["pack_sha256"]!=PLAN["parent"]:raise ValueError("different parent tape")
        sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        capture=[]
        original=ns["_mixed_actions"]
        def capture_base(x,aux):
            capture[:]=[(x,aux)]
            return original(x,aux)
        ns["_mixed_actions"]=capture_base
        for t,joint in enumerate(raw["actions"]):
            probe=t in PROBES
            obs=sim.obs(seat)
            start=time.perf_counter()
            base=ns["agent"](obs,raw["game_configuration"])
            times["zero"].append(time.perf_counter()-start)
            expected={u:a for u,a in joint[seat].items() if a not in ("NONE",None)}
            if base!=expected:raise ValueError(f"zero residual changed seat {seat} turn {t}")
            n+=1
            if probe:
                x,aux=capture[0]
                feat=ns["_FEAT"]
                parent_labels=feat.labels(base)
                parent_conflicts=ns["_return_conflicts"](x,parent_labels)
                row={"t":t,"seat":seat}
                for arm,sign in (("plus",1),("minus",-1)):
                    ns["_RETURN_THETA"][:]=np.array(PLAN["direction"])*sign*PLAN["perturbation"]
                    start=time.perf_counter()
                    labs=original(x,aux)
                    times[arm].append(time.perf_counter()-start)
                    act=feat.to_actions(labs)
                    diff={u:act.get(u,"NONE") for u in base.keys()|act.keys() if base.get(u)!=act.get(u)}
                    mine=obs["players"][seat][2]
                    if any(u not in mine or mine[u][2]!="COLLECTOR" for u in diff):
                        raise ValueError("protected unit changed")
                    if ns["_return_conflicts"](x,labs)-parent_conflicts:
                        raise ValueError("introduced new team collision or active recruit occupancy")
                    alias=ns["sys"].modules["cnet2_feat"]
                    prefix=alias.Prefix(x,aux)
                    for slot in np.flatnonzero(x["tok_mask"][:alias.OWN]):
                        if not prefix.feats(slot)[2][labs[slot]]:raise ValueError("illegal on real prefix")
                        prefix.advance(slot,labs[slot])
                    row[arm]=diff
                    changed[arm]+=int(bool(diff))
                ns["_RETURN_THETA"][:]=0
                cases.append(row)
            sim.step(*joint)
        ns["_mixed_actions"]=original
    if n!=598 or len(cases)!=24 or min(changed[k] for k in ("plus","minus"))==0:
        raise ValueError(f"insufficient coverage: {n}, {len(cases)}, {changed}")
    ret={"passed":True,"before_games":True,"map":621458920,"seats":[0,1],"zero_exact_turns":n,
         "sampled_roots_per_arm":len(cases),"roots_with_actual_changed_commands":dict(changed),
         "score_uses":"No terminal score read; compare emitted actions on original documented public trajectories.",
         "changed_on_fixed_roots":cases,"timing_s":{"note":"zero times full agent; plus/minus include both networks encodings and decodes, excluding the public Feat pass; concurrent host load, not deployment timing.",
             **{k:{"n":len(v),"mean":float(np.mean(v)),"p95":float(np.percentile(v,95))} for k,v in times.items()}},
         "source_sha256":{f:digest((HERE/f).read_bytes()) for f in ("build.py","runtime.py","protocol.json","mechanism.py")}}
    save(HERE/"mechanism.json",ret);print(n,len(cases),dict(changed),flush=True)


if __name__=='__main__':run()

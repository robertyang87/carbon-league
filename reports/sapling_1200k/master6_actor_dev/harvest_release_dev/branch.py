"""Select only from public, action-changing decisions. Pair with the real simultaneous opponent."""
import argparse
from collections import Counter
import fcntl
import hashlib
import subprocess
import sys
from fit import HERE, load, save, sources, split, FastCarbon
from policy import choose
sys.path.insert(0, str(HERE.parent/"capture_pressure_branches"))
import compare as runner
sr=runner.sr
live=sr.live
PACK=HERE.parent/"planter_capture_full_dev"

def prepare():
    weights=load(HERE/"weights.json")["weights"]
    parent_hash=sr.sha(PACK/"candidate.py")
    locations=[]
    rows=[]
    for c,raw in sources():
        if split(c["seed"])!="held" or raw["pack_sha256"]!=parent_hash:
            continue
        cfg=raw["game_configuration"];s=c["seat"]
        sim=FastCarbon(cfg).reset(seed=c["seed"])
        prev=None;found=[]
        for t,pair in enumerate(raw["actions"]):
            obs=sim.obs(s);proposal=choose(obs,prev,cfg,pair[s],weights)
            if proposal:
                p,uid=proposal
                root=dict(t=t,worker=uid,old=pair[s][uid],alternative="NONE",
                          visible_ground=obs["carbon"][obs["players"][s][2][uid][0]],
                          score=p,initial_cargo=obs["players"][s][2][uid][1])
                found.append(root)
            prev=obs;sim.step(*pair)
        rows.append(dict(source=c["path"],seed=c["seed"],seat=s,roots=found))
        if found: locations.append((s,c["seed"],c["path"],c,raw,found[0]))
    save(HERE/"disagreements.json.gz",rows)
    seen=set(); cases=[]
    for seat in (0,1):
        _,seed,path,c,raw,root=next(r for r in sorted(locations,key=lambda v:v[:3]) if r[0]==seat and r[1] not in seen)
        seen.add(seed);out=HERE/f"seat{seat}";out.mkdir(exist_ok=True)
        source=sr.ROOT/path
        receipt=load(source.with_name(source.name.replace("raw_","receipt_").replace(".json.gz",".json")))
        if receipt["game"]["raw_sha256"]!=c["source_sha256"]:raise ValueError("official source hash differs")
        sr.retain(out/"official_replay.json",dict(runs={out.name:dict(games=[receipt["official"]])}))
        q=dict(seed=seed,seat=seat,root_t=root["t"],family="safe_patch_hold",source_dir=str(out.relative_to(sr.ROOT)),
               record=dict(seed=seed,seat=seat,source=path,source_sha256=c["source_sha256"]),root=root,
               pack_sha256=raw["pack_sha256"],policy_sha=sr.sha(HERE/"policy.py"),
               weights_sha=sr.sha(HERE/"weights.json"),gate_sha=sr.sha(HERE/"protocol.json"))
        sr.retain(out/"live_protocol.json",q);live.prepare(out);cases.append(q)
    save(HERE/"selected.json",cases)
    print([(c["seat"],c["seed"],c["root"]) for c in cases],sum(len(r["roots"]) for r in rows))

def run(seat,arm):
    runner.HERE=HERE;runner.PRIOR=PACK
    read=sr.load
    def legacy_trace_schema(path):
        result=read(path)
        if path.name=="live_protocol.json":
            # The shared runner also supports planter trials and logs the owner of an optional target tree.
            result=dict(result,root=dict(result["root"],tree=None))
        return result
    sr.load=legacy_trace_schema
    runner.play(seat,arm)

if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare()
    elif len(sys.argv)>1:run(int(sys.argv[1]),sys.argv[2])
    else:
        for seat in (0,1):
            with (runner.TREE/".continuation.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                jobs=[subprocess.Popen(["taskset","-c",str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(("parent","hold"))]
                if any([j.wait() for j in jobs]):raise ValueError("paired native continuation failed")

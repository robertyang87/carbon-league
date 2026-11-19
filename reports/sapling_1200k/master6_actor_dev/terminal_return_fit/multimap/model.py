"""One scalar orthogonal to the frozen first-return update; emit the intact dual-network submission."""
import hashlib
import json
import math
from pathlib import Path
import sys

from clearance import HERE,FIT,BASE,ROOT,PLAN,PLAN_PATH,PLAN_SHA,sha,load,atomic

sys.path[:0]=[str(FIT),str(BASE)]
import build
import trial

V1 = ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"


def source_checks():
    assert sha(PLAN_PATH)==PLAN_SHA and sha(FIT/"learned_checkpoint.json")==PLAN["v1_checkpoint_sha256"]
    assert all(sha(FIT/k)==v for k,v in PLAN["source_file_sha256"].items())
    assert sha(build.PARENT)==PLAN["underlying_pack_sha256"] and sha(V1)==PLAN["parent_sha256"]
    assert sha(ROOT/"data/weights/sapling_1200k/master6_actor_v1.npz")==PLAN["student_sha256"]
    assert sha(ROOT/"data/master3s2_source.py")==PLAN["examiner_source_sha256"]
    assert hashlib.md5((ROOT/"data/master3s2_source.py").read_bytes()).hexdigest()==PLAN["examiner_md5"]
    assert load(FIT/"protocol.json")["direction"]==PLAN["base_direction"]
    assert len(PLAN["direction"])==len(PLAN["v1_theta"])==4 and sum(x*y for x,y in zip(PLAN["direction"],PLAN["base_direction"]))==0
    assert build.source(PLAN["v1_theta"])==V1.read_bytes()


def theta(alpha):
    if not math.isfinite(alpha) or abs(alpha)>PLAN["epsilon"]:
        raise ValueError("scalar outside frozen symmetric domain")
    coeff=[v+alpha*d for v,d in zip(PLAN["v1_theta"],PLAN["direction"])]
    if any(abs(v)>PLAN["coordinate_absolute_cap"] for v in coeff):
        raise ValueError("coefficient out of bounds")
    return coeff


def line(values):
    return ("_RETURN_THETA = np.array("+repr(values)+",dtype=np.float32)\n").encode()


def package(alpha):
    original=V1.read_bytes()
    assert hashlib.sha256(original).hexdigest()==PLAN["parent_sha256"]
    old=line(PLAN["v1_theta"])
    assert original.count(old)==1
    raw=original.replace(old,line(theta(alpha)))
    assert len(raw)<=10_000_000
    assert (raw==original)==(alpha==0.)
    assert raw.replace(line(theta(alpha)),old)==original
    compile(raw,"terminal_return_multimap_agent.py","exec")
    return raw


def manifest(alpha,data):
    if alpha==0:
        return load(FIT/"learned_checkpoint.json")["learned_pack"]
    return dict(parent_v1_sha256=PLAN["parent_sha256"],student_sha256=PLAN["student_sha256"],alpha=alpha,
                coefficients=theta(alpha),direction=PLAN["direction"],byte_sha256=hashlib.sha256(data).hexdigest(),
                architecture="Frozen dual-network policy and prefix decoder with four collector logit coefficients",
                protocol_sha256=PLAN_SHA)


def write_pack(directory,alpha):
    raw=package(alpha);p=directory/"agent.py";w=directory/"weights.json"
    p.write_bytes(raw)
    build.save(w,manifest(alpha,raw))
    if alpha==0:
        assert sha(w)==load(FIT/"replication/protocol.json")["pack_identity"]["learned"]["weight_sha256"]
    return p,w


def mechanism():
    from fastsim import FastCarbon
    import numpy as np
    source_checks()
    for v,p,m,d in zip(PLAN["v1_theta"],theta(PLAN["epsilon"]),theta(-PLAN["epsilon"]),PLAN["direction"]):
        assert math.isclose((p-m)/2,PLAN["epsilon"]*d,abs_tol=1e-15) and math.isclose((p+m)/2,v,abs_tol=1e-15)
    packs={label:package(alpha) for label,alpha in (("v1",0.),("plus",PLAN["epsilon"]),("minus",-PLAN["epsilon"]))}
    assert len(set(packs.values()))==3
    # Keep each installed package in a fresh namespace with its own full history.
    mods={}
    for label,b in packs.items():
        ns={"__name__":"frozen_direction_"+label,"__file__":"agent.py"}
        exec(compile(b,"agent.py","exec"),ns)
        mods[label]=ns
    counts={k:0 for k in ("plus_vs_v1","minus_vs_v1","plus_vs_minus")}
    trace=[]
    for s in (0,1):
        path=FIT.parent/"hybrid_candidate"/f"raw_621458920_{s}.json.gz"
        raw=load(path);assert raw["pack_sha256"]==PLAN["underlying_pack_sha256"]
        sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        for t,joint in enumerate(raw["actions"]):
            obs=sim.obs(s)
            if t in range(4,295,5):
                chosen={k:m["agent"](obs,raw["game_configuration"]) for k,m in mods.items()}
                for key in counts:
                    a,b=key.split("_vs_")
                    diff={w:[chosen[a].get(w,"NONE"),chosen[b].get(w,"NONE")] for w in chosen[a].keys()|chosen[b].keys()
                          if chosen[a].get(w)!=chosen[b].get(w)}
                    assert all(w in obs["players"][s][2] and obs["players"][s][2][w][2]=="COLLECTOR" for w in diff)
                    counts[key]+=bool(diff)
                    if diff:trace.append(dict(seat=s,t=t,comparison=key,commands=diff))
            elif t==0:
                for m in mods.values():m["agent"](obs,raw["game_configuration"])
            else:
                for m in mods.values():m["_FEAT"].observe(obs)
            sim.step(*joint)
    receipt=dict(protocol_sha256=PLAN_SHA,parent_exact_bytes=package(0)==V1.read_bytes(),
                 roots=2*len(range(4,295,5)),counts=counts,changed_cases=trace,passed=min(counts.values())>0,
                 pack_sha256={k:hashlib.sha256(v).hexdigest() for k,v in packs.items()})
    trial.frozen_file(HERE/"mechanism.json",receipt)
    print("mechanism",counts,receipt["passed"],flush=True)
    return receipt


if __name__=="__main__":
    sys.path.insert(0,str(ROOT/"code"))
    mechanism()

"""Replay both pure, separately packed policies on held public prefixes, then audit signed returns."""
import ast
import base64
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import zlib
import numpy as np
from propose import HERE,ROOT,ACTOR,APEX,NATIVE,PARENT,read,sha,check,clean,dump,identity


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    trial=read(HERE/"fit.json");proposals=read(HERE/"proposals.json.gz");lineage=identity()
    check(sha(HERE/"proposals.json.gz")==trial["proposals_sha256"] and sha(HERE/"public_features.npz")==proposals["feature_sha256"],"dossier source binding")
    check(sha(HERE/"train_actor.py")==trial["fit_source_sha256"] and sha(HERE/"propose.py")==proposals["source_sha256"],"implementation binding")
    check(lineage==proposals["identity"],"source lineage mismatch")
    assert all(not v for v in trial["gate"].values()) and not trial["eligible"]
    sys.path.insert(0,str(ROOT/"code"))
    import cnet2_pack as pack
    from fastsim import FastCarbon
    weights=ACTOR.parents[2]/"data/weights/sapling_1200k"
    parent=module("actual_pure_parent",weights/"master6_actor_v1.py")
    candp=HERE/"sapling_only.py";child=module("new_standalone",candp)
    check(sha(candp)==trial["standalone_sha256"],"candidate deploy artifact")
    target=HERE/"proposal.npz"
    check(sha(target)==trial["weights"]["proposal"]["sha256"],"independent student weights")
    with np.load(target) as z:
        q,binary,deq=pack.quantize(z)
        check(q==child._QMETA and zlib.decompress(base64.b64decode(child._W64))==binary and str(z['meta'])==child._WMETA,"code and weight artifact mismatch")
    released={k:hashlib.sha256(v.encode()).hexdigest() for k,v in child._SRC.items()}
    check(released=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in parent._SRC.items()} and
          all(n in ('cnet2_feat','cnet2_np','cnet2_agent') for n in child._SRC),"added teacher runtime")
    check(len(child._W64)<2_000_000 and not hasattr(child,"_WEIGHTS") and not hasattr(child,"_VALUE"),"unexpected teacher")
    full=[];rows=proposals['rows']
    by_source={}
    for r in rows:
        if r["split"]=="hold":by_source.setdefault(r["source"],[]).append(r)
    for src,rr in by_source.items():
        raw=read(ROOT/src);s=raw["candidate_seat"];sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        when={r["t"]:r for r in rr}
        for t in range(max(when)+1):
            obs=sim.obs(s)
            if t in when or t==0:
                parent_order=clean(parent.agent(deepcopy(obs),raw["game_configuration"]))
                child_order=clean(child.agent(deepcopy(obs),raw["game_configuration"]))
                if t in when:
                    r=when[t];source=ROOT/r["case"];v=read(source/"live_parent.json");a=read(source/"live_alternative.json")
                    check(r["native"]["parent"]["source_sha256"]==sha(source/"live_parent.json") and
                          r["native"]["alternative"]["source_sha256"]==sha(source/"live_alternative.json"),"native source re-read")
                    op0=clean(v["root_actions"][s]);op1=clean(a["root_actions"][s]);uid=r["uid"]
                    check(op0.get(uid,"NONE")==r["parent_action"] and op1.get(uid,"NONE")==r["alternative_action"],"focal mutation identity")
                    full.append({"case":r["case"],"seed":r["seed"],"seat":s,"t":t,"focal":uid,
                        "parent_order":parent_order,"candidate_order":child_order,
                        "orders_differ":child_order!=parent_order,"actual_focal_differ":parent_order.get(uid,"NONE")!=child_order.get(uid,"NONE"),
                        "candidate_focal":child_order.get(uid,"NONE"),"parent_focal":parent_order.get(uid,"NONE"),
                        "candidate_matches_native_parent_joint":child_order==op0,"candidate_matches_native_alt_joint":child_order==op1,
                        "parent_matches_native_parent_joint":parent_order==op0,"parent_matches_native_alt_joint":parent_order==op1})
            else:
                for runtime in (parent,child):
                    inspect.getclosurevars(runtime._AG).nonlocals['st']['f'].observe(deepcopy(obs))
            sim.step(*raw["actions"][t])
    check(len(full)==16,"held pure runtime missing")
    counts={k:sum(r[k] for r in full) for k in ("orders_differ","actual_focal_differ","candidate_matches_native_parent_joint",
             "candidate_matches_native_alt_joint","parent_matches_native_parent_joint","parent_matches_native_alt_joint")}
    out={"sources_sha256":released,"identity":lineage,"pure_parent_pkg":sha(weights/"master6_actor_v1.py"),"pure_candidate_pkg":sha(candp),
         "rows":full,"counts":counts,"held_maps":len({r["seed"] for r in full}),"native_jobs_started":0}
    dump(HERE/"serving_verification.json",out)
    print(counts,flush=True)


if __name__=="__main__":main()

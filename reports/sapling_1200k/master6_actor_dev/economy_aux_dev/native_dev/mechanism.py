"""Frozen public replay probes of package parity and the student's actual effect on orders."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

from build import ACTOR, HERE, PARENT, ROOT, STUDY, PLAN, read, encoded_student, seal, sha

sys.path.insert(0, str(ROOT/"code"))
from fastsim import FastCarbon

STEPS = {*range(0,300,30),298}
ARMS = ("control","candidate")


def package(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decode=module._mixed_actions
    def record(x,aux):
        module._mechanism_inputs=(x,aux)
        return decode(x,aux)
    module._mixed_actions=record
    return module


def run():
    release = read(HERE/"release.json")
    if release["protocol_sha256"]!=PLAN:
        raise ValueError("wrong release")
    source=ACTOR/"tree_response_training"
    tapes=read(source/"source_receipt.json")["games"]
    rows=[]
    counter={arm:dict(identity_errors=0,contributing_worker_orders=0,
                      trajectory_reproduced=0,changed_vs_v1_worker_orders=0) for arm in ARMS}
    restored={arm:encoded_student(STUDY/(arm+".npz"))[1] for arm in ARMS}
    for seat in (0,1):
        rec=next(g for g in tapes if g["seat"]==seat and g["seed"]==621471120)
        p=ROOT/rec["raw_path"]
        if sha(p)!=rec["raw_sha256"]:
            raise ValueError("source replay changed")
        with gzip.open(p,"rt") as stream:
            tape=json.load(stream)
        sim=FastCarbon(tape["game_configuration"]).reset(seed=621471120)
        parent=package(PARENT,"sapling_economic_ref")
        models={key:package(HERE/(key+".py"), "sapling_economic_"+key) for key in ARMS}
        references={}
        for t,joint in enumerate(tape["actions"]):
            obs=sim.obs(seat)
            if t not in STEPS:
                for m in [parent,*models.values()]:
                    m._FEAT.observe(obs)
                sim.step(*joint)
                continue
            original=parent.agent(obs,sim.config)
            if original!=joint[seat]:
                raise ValueError("existing official public history differs from v1")
            row={"step":t,"seat":seat,"v1":original}
            for arm,m in models.items():
                actual=m.agent(obs,sim.config)
                if arm not in references:
                    ref=type(parent._NETS[1])(restored[arm])
                    if set(ref.w)!=set(m._NETS[1].w) or any(not np.array_equal(ref.w[n],m._NETS[1].w[n]) for n in ref.w):
                        raise ValueError("embedded quantized student differs from direct NPZ quantization")
                    if any(not np.array_equal(m._NETS[0].w[n],v) for n,v in parent._NETS[0].w.items()):
                        raise ValueError("expert weights differ")
                    references[arm]=ref
                x,aux=m._mechanism_inputs
                # The untouched parent decoder serves as an independent reference, with only
                # its in-memory second Policy set to the separately reconstructed student.
                old=parent._NETS[1]
                try:
                    parent._NETS[1]=references[arm]
                    expect=parent._FEAT.to_actions(parent._mixed_actions(x,aux))
                finally:
                    parent._NETS[1]=old
                base_weights=(m._HYBRID["expert_weight"],m._HYBRID["student_weight"])
                if base_weights!=(.75,.25):
                    raise ValueError("policy mix coefficients changed")
                try:
                    m._HYBRID["expert_weight"],m._HYBRID["student_weight"]=1.,0.
                    ablated=m._FEAT.to_actions(m._mixed_actions(x,aux))
                finally:
                    m._HYBRID["expert_weight"],m._HYBRID["student_weight"]=base_weights
                workers=obs["players"][seat][2]
                tally=counter[arm]
                tally["identity_errors"]+=actual!=expect
                tally["trajectory_reproduced"]+=1
                tally["contributing_worker_orders"]+=sum(actual.get(uid)!=ablated.get(uid) for uid in workers)
                tally["changed_vs_v1_worker_orders"]+=sum(actual.get(uid)!=original.get(uid) for uid in workers)
                row[arm]={"package":actual,"int8_reference":expect,"student_weight_zero":ablated}
            rows.append(row)
            sim.step(*joint)
    valid=len(rows)==22 and all(v["identity_errors"]==0 and v["trajectory_reproduced"]==22
                                and v["contributing_worker_orders"]>0 for v in counter.values())
    doc={"passed":valid,"protocol_sha256":PLAN,"release_sha256":sha(HERE/"release.json"),
         "frozen_public_samples":len(rows),"by_arm":counter,"source_ledger_sha256":sha(source/"source_receipt.json"),
         "reference_impl_sha256":sha(Path(__file__)),"source_rows":rows}
    seal(HERE/"mechanism.json",doc)
    if not valid:
        seal(HERE/"mechanism_skip.json",{"protocol_sha256":PLAN,"mechanism_sha256":sha(HERE/"mechanism.json"),
                                        "native_games_started":0})
    print(valid,counter,flush=True)


if __name__=="__main__":
    run()

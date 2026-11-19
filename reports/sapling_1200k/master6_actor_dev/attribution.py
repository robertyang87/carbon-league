"""Replay candidate decisions on its actual seat-visible history and measure whole-policy action differences."""
import argparse
import base64
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import hybrid_build as build
from tree_audited_game import exam, prior
import tree_win_gate as gate

SEED = 621458920
ROLES = ("BASE","COLLECTOR","PLANTER")


def prove_mode(mode, pack):
    reference = build.PARENT if mode=="hybrid" else build.REFERENCE
    required = build.PARENT_SHA if mode=="hybrid" else build.REFERENCE_SHA
    if build.digest(reference)!=required or build.digest(pack)==required:
        raise ValueError("reference identity or updated candidate identity invalid")
    before=reference.read_text()
    restored=build.substitute(pack.read_text(),before)
    if mode=="hybrid":
        restored=restored.replace('"student": "Sapling Master6 actor V1"','"student": "Sapling cash_v3"')
    if restored!=before:
        raise ValueError("unexpected inference change beyond the registered Sapling payload")
    return reference


RECEIPT = HERE/"weight_receipt.json"
RECEIPT_SHA = "21fe203988d38cee5075469b7d4b40aca03dc6a5da088d7d88ac745235d09ad4"
HYBRID_MANIFEST = HERE/"hybrid_manifest.json"
TRAINED_AGENT = build.WEIGHTS/"master6_actor_v1.py"
TRAINED_NPZ = build.WEIGHTS/"master6_actor_v1.npz"
UPDATED_HYBRID = build.WEIGHTS/"master6_actor_hybrid.py"


def payload_segments(path):
    lines, fields=build.payloads(path.read_bytes().decode("utf-8"))
    return {field:"".join(lines[a:b]).encode("utf-8") for field,(a,b) in fields.items()}


def prove_artifacts(mode, pack, header, *, npz=TRAINED_NPZ, student=TRAINED_AGENT, manifest=HYBRID_MANIFEST):
    receipt=json.loads(RECEIPT.read_bytes())
    if build.digest(RECEIPT)!=RECEIPT_SHA:
        raise ValueError("frozen training receipt changed")
    npz_hash, agent_hash=build.digest(npz),build.digest(student)
    if npz_hash!=receipt.get("npz_sha256") or npz.name!=receipt.get("npz"):
        raise ValueError("actual FP32 NPZ differs from the training receipt")
    if agent_hash!=receipt.get("agent_sha256") or student.name!=receipt.get("agent"):
        raise ValueError("actual trained actor pack differs from the training receipt")
    deployed=build.digest(pack)
    if header.get("pack_sha256")!=deployed:
        raise ValueError("deployed pack hash differs from the game header")
    identity=dict(training_receipt_sha256=RECEIPT_SHA,trained_npz_sha256=npz_hash,trained_actor_sha256=agent_hash)
    if mode=="single":
        if (header.get("weight_sha256"),deployed)!=(npz_hash,agent_hash):
            raise ValueError("single-game pack or FP32 checkpoint differs from training receipt")
    elif mode=="hybrid":
        manifest_hash=build.digest(manifest)
        if header.get("weight_sha256")!=manifest_hash:
            raise ValueError("hybrid-game weight identity differs from the actual build manifest")
        data=json.loads(manifest.read_bytes())
        if (data.get("teacher_weight"),data.get("student_weight"),data.get("guard"),data.get("search"),
                data.get("development_seed"))!=(.75,.25,False,False,SEED):
            raise ValueError("hybrid build configuration differs from the frozen protocol")
        sources={build.REFERENCE:build.REFERENCE_SHA,build.PARENT:build.PARENT_SHA,TRAINED_AGENT:agent_hash,
                 TRAINED_NPZ:npz_hash,RECEIPT:RECEIPT_SHA,UPDATED_HYBRID:deployed,
                 Path(build.__file__).resolve():build.digest(Path(build.__file__))}
        wanted={str(path.relative_to(build.ROOT)):sha for path,sha in sources.items()}
        if wanted!=data.get("input_and_output_sha256"):
            raise ValueError("hybrid manifest does not bind both trained identities and generated pack")
        expected_segments=payload_segments(student)
        actual_segments=payload_segments(pack)
        for field in sorted(expected_segments):
            if expected_segments[field]!=actual_segments[field]:
                raise ValueError(f"hybrid student segment {field} differs from the trained actor")
        identity.update(hybrid_manifest_sha256=manifest_hash,student_payload_sha256={
            f:hashlib.sha256(b).hexdigest() for f,b in sorted(actual_segments.items())})
    else:
        raise ValueError("unknown audit mode")
    return identity


def expert_only(parent, seat):
    """Decode with the frozen int8 it50 alone, taking each next action on its own prefix."""
    code=parent.sys.modules
    w=parent._dequantize(zlib.decompress(base64.b64decode(parent._B1)), parent._Q1)
    w["meta"]=np.array(parent._M1)
    expert=code["cnet2_np"].Policy(w)
    feat=code["cnet2_feat"].Feat(seat)
    def choose(observation, configuration):
        x=feat.observe(observation)
        labels=expert.act(x,feat.aux,teacher=2)[0]
        return feat.to_actions(labels)
    return choose


def roster(observation, seat):
    _,bases,workers,_=observation["players"][seat]
    entities={uid:("BASE",position,None) for uid,position in bases.items()}
    entities.update({uid:(kind,position,amount) for uid,(position,amount,kind) in workers.items()})
    return entities


def check_game(directory, row, result_header, mode, updated_pack, reference):
    seat=row["candidate_seat"]
    if seat not in (0,1) or row["seed"]!=SEED:
        raise ValueError("unexpected map or physical seat")
    raw=json.loads(gzip.decompress((directory/row["raw_file"]).read_bytes()))
    gate.validate_game(row)
    gate.validate_examiner_configuration(raw["configuration"])
    if result_header["examiner_md5"]!=gate.EXAMINER or raw.get("audit_messages")!=[]:
        raise ValueError("uncertified opponent evidence")
    terrain=gate.map_identity.local_board(SEED)
    gate.verify_raw(directory,row,build.digest(updated_pack),result_header["weight_sha256"],expected_terrain=terrain)
    sim=exam.FastCarbon(raw["game_configuration"]).reset(seed=SEED)
    old=prior.shared.module(reference,"m6_attrib_parent_"+str(seat))
    current=prior.shared.module(updated_pack,"m6_attrib_updated_"+str(seat))
    policies={"old":old.agent,"updated":current.agent}
    contrasts={"old_to_updated":("old","updated")}
    if mode=="hybrid":
        for model in (old,current):
            opts=model._HYBRID
            if (opts["expert_weight"],opts["student_weight"],opts["style"],opts["guard"],opts["search"])!=(.75,.25,2,False,False):
                raise ValueError("incorrect hybrid configuration")
        policies["it50_only"]=expert_only(old,seat)
        contrasts.update(it50_to_updated=("it50_only","updated"),it50_to_old=("it50_only","old"))
    valid=Counter()
    counts={key:{"changed_roots":0,"changed_commands":0,"by_role":Counter(),"transitions":Counter()} for key in contrasts}
    observations=hashlib.sha256()
    roots=[]
    for t, joint_commands in enumerate(raw["actions"]):
        obs=sim.obs(seat)
        if obs["step"]!=t or obs["player"]!=seat or "full_carbon" in obs:
            raise ValueError("an agent received a wrong turn, seat or omniscient board")
        observations.update(prior.hashed(obs).encode()+b"\n")
        entities=roster(obs,seat)
        outputs={key:dict(policy(copy.deepcopy(obs),sim.config)) for key,policy in policies.items()}
        if outputs["updated"]!=joint_commands[seat]:
            raise ValueError(f"updated deployment policy differed from action tape at seat {seat} turn {t}")
        if any(set(a)-entities.keys() for a in outputs.values()):
            raise ValueError("a counterfactual policy issued commands for an unavailable unit")
        changed=[]
        changed_contrasts=set()
        for uid,(role,pos,load) in sorted(entities.items()):
            if role not in ROLES:
                raise ValueError(f"unknown unit role {role}")
            valid[role]+=1
            decisions={key:a.get(uid) or "NONE" for key,a in outputs.items()}
            for key,(left,right) in contrasts.items():
                a,b=decisions[left],decisions[right]
                if a!=b:
                    changed_contrasts.add(key)
                    counts[key]["changed_commands"]+=1
                    counts[key]["by_role"][role]+=1
                    counts[key]["transitions"][f"{role} {a}->{b}"]+=1
            if len(set(decisions.values()))>1:
                changed.append(dict(uid=uid,role=role,public_position=pos,public_cargo=load,commands=decisions))
        for key in changed_contrasts:
            counts[key]["changed_roots"]+=1
        roots.append(dict(t=t,executed_sha256=prior.hashed(joint_commands[seat]),
                          policy_command_sha256={key:prior.hashed(a) for key,a in outputs.items()},changes=changed))
        sim.step(*joint_commands)
    if prior.hashed(sim.get_state())!=row["final_state_sha256"]:
        raise ValueError("counterfactual inspection changed the original game state")
    return dict(seed=SEED,seat=seat,result=row["result"],official_score_by_seat=row["official_score_by_seat"],
                raw_sha256=row["raw_sha256"],initial_terrain_u8_sha256=terrain,
                exposed_observation_history_sha256=observations.hexdigest(),exact_match_roots=len(roots),
                possible_commands_by_role=valid,comparisons=counts,roots=roots)


def run(mode, directory, pack, output):
    report=json.loads((directory/"results.json").read_bytes())
    if (report["status"]!="complete" or len(report["games"])!=2
        or {(r["seed"],r["candidate_seat"]) for r in report["games"]}!={(SEED,0),(SEED,1)}
        or report["header"]["pack_sha256"]!=build.digest(pack)):
        raise ValueError("requires both complete registered-map seats from the exact updated artifact")
    training_identity=prove_artifacts(mode,pack,report["header"])
    reference=prove_mode(mode,pack)
    rows=[check_game(directory,r,report["header"],mode,pack,reference) for r in sorted(report["games"],key=lambda r:r["candidate_seat"])]
    totals={}
    for key in rows[0]["comparisons"]:
        merged={"changed_roots":0,"changed_commands":0,"by_role":Counter(),"transitions":Counter()}
        for row in rows:
            for field in merged:
                merged[field]+=row["comparisons"][key][field]
        totals[key]=merged
    data=dict(mode=mode,registered_seed=SEED,training_identity=training_identity,audit_source_sha256=build.digest(Path(__file__)),
              result_ledger_sha256=build.digest(directory/"results.json"),reference_pack_sha256=build.digest(reference),updated_pack_sha256=build.digest(pack),
              evidence_scope="Behavior on the updated candidate's recorded trajectories. Each shadow decodes a fresh joint action on its own legal prefixes.",
              possible_commands_by_role=sum((x["possible_commands_by_role"] for x in rows),Counter()),
              exact_match_roots=sum(x["exact_match_roots"] for x in rows),comparisons=totals,games=rows)
    payload=(json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    if output.suffix==".gz":
        payload=gzip.compress(payload,mtime=0)
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as dest:
        dest.write(payload)
        name=dest.name
    os.replace(name,output)
    print(json.dumps(dict(mode=mode,roots=data["exact_match_roots"],comparisons=totals),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("hybrid","single"))
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--pack",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.mode,args.run,args.pack,args.out)

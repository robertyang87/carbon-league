"""Hold the native root's opponent order fixed and compare two complete legal policy commands for one turn."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
sys.path.insert(0,str(DEV/"game_analysis"))
import analyze as evidence
official=evidence.official
check,read,sha=evidence.check,evidence.read,evidence.sha
PACKS=("candidate","control")


def dump(path,obj):
    raw=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode() if not isinstance(obj,str) else obj.encode()
    if path.suffix==".gz": raw=gzip.compress(raw,mtime=0)
    if path.exists():
        check(path.read_bytes()==raw,"existing artifact differs")
    else:
        with tempfile.NamedTemporaryFile(dir=HERE,delete=False) as out:
            out.write(raw)
        os.replace(out.name,path)


def package(arm):
    p=DEV/(arm+".py")
    spec=importlib.util.spec_from_file_location("sapling_paired_edits_"+arm,p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decoder=mod._mixed_actions
    def capture(x,aux):
        labels=decoder(x,aux)
        mod._audit_root=(x,aux,labels.copy())
        return labels
    mod._mixed_actions=capture
    return mod


def slots(pack,actions):
    x,aux,labels=pack._audit_root
    feat=pack._FEAT
    check(np.array_equal(labels,feat.labels(actions)) and actions==feat.to_actions(labels),"action round trip")
    pre=sys.modules["cnet2_feat"].Prefix(x,aux)
    log=[]
    for i,uid in enumerate(feat.own_uid):
        if uid is None: continue
        uf,mv,legal=pre.feats(i)
        chosen=int(labels[i])
        check(legal[chosen],"illegal action under actual preceding decisions")
        log.append({"slot":i,"uid":uid,"order":actions.get(uid,"NONE"),"canonical_label":chosen,
                    "physical_prefix_sha256":hashlib.sha256(uf.tobytes()+mv.tobytes()+legal.tobytes()).hexdigest(),
                    "legal_mask":legal.tolist()})
        pre.advance(i,chosen)
    return log


def public(state,seat,cfg):
    size=cfg["size"]
    mask=evidence.visibility(state["players"][seat][2],size)
    return {"players":state["players"],"trees":state["trees"],"player":seat,"step":state["step"],
            "remainingOverageTime":state.get("remainingOverageTime",60),
            "carbon":[v if yes else 0 for v,yes in zip(state["carbon"],mask,strict=True)]}


def advance(state,cfg,joint):
    with official.ledger() as ev,evidence.casualties() as deaths:
        result=official.helpers.Board(state,cfg,joint).next().observation
        payments=[list(e) for e in ev if e[0]!="rounding" and e[2]!=0]
        return result,{"payments":payments,"casualties":list(deaths)}


def assets(state):
    return [{"cash":p[0],"workers":p[2],"trees":p[3]} for p in state["players"]]


def differences(old,new):
    result={}
    for kind in ("workers","trees"):
        changes=[]
        for side in (0,1):
            a,b=old[side][kind],new[side][kind]
            for uid in sorted(set(a)|set(b)):
                if a.get(uid)!=b.get(uid):
                    changes.append({"id":uid,"physical_seat":side,"control":a.get(uid),"candidate":b.get(uid)})
        result[kind]=changes
    result["cash_by_physical_seat"]=[new[s]["cash"]-old[s]["cash"] for s in (0,1)]
    return result


def outcomes(source,cfg,both,s,enemy):
    # Both branches must see identical allocator state. A newly recruited worker/tree can already die on this turn.
    ids=official.idgen._id_dict.copy()
    out={}
    for arm in ("control","candidate"):
        official.idgen._id_dict=ids.copy()
        j=[None,None];j[s]=both[arm];j[1-s]=enemy
        post,ev=advance(source,cfg,j)
        out[arm]={"post":assets(post),"events":ev,"allocator_after":official.idgen._id_dict.copy()}
    # 'candidate' is the recorded native arm and is deliberately evaluated last.
    return out,post


def opp_divergence(candidate, other, s):
    for t,(a,b) in enumerate(zip(candidate,other,strict=True)):
        if a!=b: return {"pre":t,"own_changed":a[s]!=b[s],"opponent_changed":a[1-s]!=b[1-s],
                         "candidate_joint":a,"other_joint":b}
    return None


def run_game(g,r,release):
    seed,s=g["seed"],g["seat"]
    paths={arm:DEV/arm/r[arm]["raw_file"] for arm in ("parent","control","candidate")}
    for arm,p in paths.items():check(sha(p)==r[arm]["raw_sha256"],"tape digest")
    tapes={arm:read(p) for arm,p in paths.items()}
    tape=tapes["candidate"];cfg=tape["game_configuration"]
    signed={e["step"]:e["vs"]["control"] for e in g["events"] if "control" in e["vs"]}
    packs={arm:package(arm) for arm in PACKS}
    for arm in PACKS:
        check(sha(DEV/(arm+".py"))==release[arm+"_pack_sha256"]==tapes[arm]["pack_sha256"],"wrong package")
    sim=evidence.FastCarbon(cfg).reset(seed=seed)
    state=official.initial(tape)
    steps=[assets(state)]
    roots=[]
    for t,j in enumerate(tape["actions"]):
        pub=public(state,s,cfg)
        fast=sim.obs(s)
        check(pub["carbon"]==fast["carbon"] and pub["trees"]==fast["trees"] and pub["players"]==fast["players"],"public input differs")
        if t==0 or t in signed:
            actions={arm:p.agent(fast,cfg) for arm,p in packs.items()}
        else:
            for p in packs.values():p._FEAT.observe(fast)
        if t in signed:
            check(actions["candidate"]==j[s],"recorded candidate cannot be reproduced")
            mismatches=sorted([wid,actions["control"].get(wid,"NONE"),actions["candidate"].get(wid,"NONE")] for wid in set(actions["control"])|set(actions["candidate"])
                              if actions["control"].get(wid)!=actions["candidate"].get(wid))
            check(mismatches==sorted(signed[t]),"different edits from full 1196-turn package comparison")
            prefixes={arm:slots(packs[arm],actions[arm]) for arm in PACKS}
            candidate,control=prefixes["candidate"],prefixes["control"]
            diffs=[];same_so_far=True
            for a,b in zip(candidate,control,strict=True):
                check(a["uid"]==b["uid"] and a["slot"]==b["slot"],"changed unit ordering")
                if same_so_far:check(a["physical_prefix_sha256"]==b["physical_prefix_sha256"],"different decision features before first edit")
                if a["order"]!=b["order"]:
                    uid=a["uid"]
                    diffs.append({"unit":uid,"role":pub["players"][s][2][uid][2],"slot":a["slot"],
                                  "same_preceding_orders":same_so_far,"control":b["order"],"candidate":a["order"],
                                  "before":pub["players"][s][2][uid]})
                    same_so_far=False
            orig_pub=deepcopy(pub)
            explicit=evidence.visibility(state["players"][s][2],cfg["size"])
            outcomes_by_arm,state=outcomes(state,cfg,actions,s,j[1-s])
            c=outcomes_by_arm["control"]["post"];new=outcomes_by_arm["candidate"]["post"]
            roots.append({"pre":t,"edited":diffs,"public_input":orig_pub,"public_visibility_mask":explicit,
                          "complete_own_actions":actions,"observed_opponent_order":j[1-s],"slot_prefixes":prefixes,
                          "outcomes":outcomes_by_arm,"candidate_minus_control":differences(c,new)})
        else:
            state=official.helpers.Board(state,cfg,j).next().observation
        sim.step(*j)
        check(state["players"]==sim.obs(s)["players"],"candidate branch does not return to recorded public state")
        steps.append(assets(state))
    check([x["cash"] for x in steps[-1]]==tape["last_post_cash"] and tape["official_score_by_seat"]==r["candidate"]["official_scores"],"terminal outcome differs")
    for root in roots:
        t=root["pre"];now=steps[t]
        specific={e["unit"] for e in root["edited"]}
        future={}
        for h in (3,5,10):
            if t+h>=len(steps):
                future[str(h)]={"available":False,"remaining":len(steps)-1-t}
                continue
            z=steps[t+h]
            future[str(h)]={"available":True,"physical_cash":[p["cash"] for p in z],
                            "cash_difference_change":(z[s]["cash"]-z[1-s]["cash"])-(now[s]["cash"]-now[1-s]["cash"]),
                            "edited_workers":{uid:z[s]["workers"].get(uid) for uid in sorted(specific)},
                            "observed_joint_orders":tape["actions"][t:t+h]}
        root["recorded_candidate_trajectory_only"]=future
    divergence={a:opp_divergence(tape["actions"],tapes[a]["actions"],s) for a in ("control","parent")}
    for arm,d in divergence.items():
        signed_d=g["first_actual_trajectory_divergence"][arm]
        check((d["pre"],d["own_changed"],d["opponent_changed"])==(signed_d["step"],signed_d["candidate_seat_differs"],signed_d["opponent_differs"]),"different first divergence")
    return {"seed":seed,"seat":s,"raw_sha256":{a:sha(p) for a,p in paths.items()},
            "first_native_joint_divergence":divergence,"recorded_margin":r["candidate"]["margin"],"edit_roots":roots}


def main():
    src=read(DEV/"actual_attribution.json.gz");c=read(DEV/"comparison.json");release=read(DEV/"release.json")
    check(src["comparison_sha256"]==sha(DEV/"comparison.json"),"different official results")
    refs={(r["seed"],r["seat"]):r for r in c["paired_map_seats"]}
    games=[run_game(g,refs[g["seed"],g["seat"]],release) for g in src["games"]]
    roots=sum((g["edit_roots"] for g in games),[])
    check(len(roots)==src["different_steps_vs"]["control"]==8 and sum(len(x["edited"]) for x in roots)==src["different_orders_vs"]["control"]==10,"wrong cohort")
    result={"source_attribution_sha256":sha(DEV/"actual_attribution.json.gz"),"comparison_sha256":sha(DEV/"comparison.json"),
            "release_sha256":sha(DEV/"release.json"),"implementation_sha256":sha(Path(__file__)),
            "official_helpers_sha256":sha(Path(official.helpers.__file__)),"games":games}
    dump(HERE/"one_turn.json.gz",result)


if __name__=="__main__":
    main()

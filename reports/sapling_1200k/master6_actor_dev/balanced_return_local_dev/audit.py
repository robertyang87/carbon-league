"""Audit supported collector actions on historical observations and the two physical decoder prefixes."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "balanced_return_dev"
sys.path.insert(0, str(PREVIOUS))
import common
import worlds

read, sha = common.read, common.sha
WINDOWS = (205, 271)


def check(value, message):
    if not value:
        raise ValueError(message)


def support(receipts):
    groups = defaultdict(list)
    for header in receipts:
        item = header["choice"]
        if item is not None:
            load, distance, threat = item["context"]
            groups[header["t"], int(load >= 40), int(threat)].append((load, distance))
    return {key:(min(p[0] for p in a), max(p[0] for p in a), min(p[1] for p in a), max(p[1] for p in a),len(a))
            for key,a in groups.items()}


def eligible(x, aux, step, evidence):
    available=[]
    for i in range(10):
        if not x["tok_mask"][i] or int(x["tok_kind"][i])!=0:
            continue
        cell=int(x["tok_pos"][i]);load=float(x["tok_cargo"][i]);distance=float(aux["dob"][cell])
        ranges=evidence.get((step,int(load>=40),int(aux["en_min"][cell]<load)))
        if ranges and ranges[0]<=load<=ranges[1] and ranges[2]<=distance<=ranges[3]:
            available.append(i)
    return available


def highest_entropy(slots, trace):
    ranked=[]
    for i in slots:
        legal,z=trace[i]["legal"],trace[i]["combined"]
        if np.count_nonzero(legal)<2:continue
        # Match the documented temperature-1 transform, excluding impossible moves.
        z=z[legal];p=np.exp((z-z.max()).astype(np.float64));p/=p.sum();p=p[p>0]
        ranked.append((float(-p@np.log(p)), -i))
    return -max(ranked)[1] if ranked else -1


def heads(module):
    return module._NETS[1].w


def encoded(module,x):
    out=[]
    for net in module._NETS:
        tokens,ctx=net.encode(x,module._HYBRID["style"])
        out.append((net,tokens,ctx,net.lin(ctx,"h0")))
    return out


def independent_decode(parent,x,aux,enc,focal,residual,learned):
    feat=sys.modules["cnet2_feat"]
    states=[list(row) for row in enc];pref=feat.Prefix(x,aux);previous=8
    labels=np.full(feat.OWN,-1,np.int64); trace={}
    for i in range(feat.OWN):
        if not x["tok_mask"][i]:continue
        unit,move,legal=pref.feats(i);scores=[]
        for j,(net,tokens,ctx,h) in enumerate(states):
            obs=np.concatenate([tokens[i],ctx,unit,move.reshape(-1)])
            hidden=net.gru(net.lin(obs,"din")+net.w["aemb.weight"][previous],h);states[j][3]=hidden
            if i>=10:base=np.r_[net.lin(hidden,"bhead"),np.zeros(2,np.float32)]
            elif j==1 and i==focal:base=hidden@learned[0]+learned[1]
            else:base=net.lin(hidden,"whead")
            scores.append(base+net.lin(move,"V")@net.lin(hidden,"U"))
        z=parent._HYBRID["expert_weight"]*scores[0]+parent._HYBRID["student_weight"]*scores[1]
        if residual and i<10 and x["tok_kind"][i]==0:
            z=z+parent._return_basis(x,aux,i,move)@parent._RETURN_THETA
        action=int(np.argmax(np.where(legal,z,-np.inf)))
        trace[i]={"previous":previous,"action":action,"legal":legal.copy(),"combined":z.copy(),"unit":unit.copy(),"move":move.copy()}
        labels[i]=action;pref.advance(i,action);previous=action+(5 if i>=10 else 0)
    return labels,trace


@contextmanager
def capture_decoder(package):
    """Observe local variables in the actual loaded function without changing its operations."""
    fn=package._return_decode
    lines,start=inspect.getsourcelines(fn)
    offsets=[i for i,line in enumerate(lines) if line.strip()=="labels[slot] = action"]
    check(len(offsets)==1,"missing unique post-logit probe")
    target=start+offsets[0];all_passes=[]
    def enter(frame,event,arg):
        if event!="call" or frame.f_code is not fn.__code__:return None
        given=frame.f_locals
        stream={"focal":given["focal"],"residual":given["apply_residual"],"entropy_probe":given["entropy"] is not None,"steps":{}}
        all_passes.append(stream)
        def observe(fr,e,_):
            if e=="line" and fr.f_lineno==target:
                v=fr.f_locals;i=int(v["slot"])
                stream["steps"][i]={k:(v[k].copy() if isinstance(v[k],np.ndarray) else int(v[k]))
                                     for k in ("previous","action","legal","combined","unit","move")}
            return observe
        return observe
    prior=sys.gettrace();sys.settrace(enter)
    try:yield all_passes
    finally:sys.settrace(prior)


def same_trace(expected,observed):
    check(expected.keys()==observed.keys(),"active physical slots differ")
    for i in expected:
        for key in expected[i]:
            check(np.array_equal(expected[i][key],observed[i][key]),f"different {key} at physical slot {i}")


def check_root(candidate,parent,learned,feat,public_obs,public_feature,x,t,anchor,table):
    aux=feat.aux;enc=encoded(parent,x)
    parent_base,pb=independent_decode(parent,x,aux,enc,-1,False,learned)
    parent_adjusted,pa=independent_decode(parent,x,aux,enc,-1,True,learned)
    parent_y,parent_rollback=common.policy.guard(parent,x,parent_base,parent_adjusted)
    check(np.array_equal(parent._mixed_actions(x,aux),parent_y),"frozen parent disagrees with independent decoder")
    admissible=eligible(x,aux,t,table)
    check(list(candidate._local_candidates(x,aux,t))==admissible,"candidate physical support differs")
    focal=highest_entropy(admissible,pa)
    # Off-window, compute a diagnostic alternative while keeping the observed public state and remaining time.
    plausible=eligible(x,aux,anchor,table)
    check(list(candidate._local_candidates(x,aux,anchor))==plausible,"as-if support differs")
    as_if=highest_entropy(plausible,pa)
    d={(-1,False):(parent_base,pb),(-1,True):(parent_adjusted,pa)}
    def oracle(index,r):
        if (index,r) not in d:d[index,r]=independent_decode(parent,x,aux,enc,index,r,learned)
        return d[index,r]
    base,bt=oracle(focal,False); adjusted,at=oracle(focal,True)
    y,rolled=common.policy.guard(parent,x,base,adjusted)
    candidate._FEAT=public_feature
    with capture_decoder(candidate) as actual:
        physical=candidate.agent(public_obs,worlds.CFG)
    check(physical==feat.to_actions(y),"public agent failed the independent complete-prefix guard")
    execution=[p for p in actual if not p["entropy_probe"]]
    check([(p["focal"],p["residual"]) for p in execution]==[(focal,False),(focal,True)],"different execution slot or branch")
    switches=0
    for p in actual:
        selected=p["focal"]
        check(selected==-1 if p["entropy_probe"] else selected==focal,"learned head used to rank the focal slot")
        _,tr=oracle(selected,p["residual"]);same_trace(tr,p["steps"])
        if not p["entropy_probe"] and selected>=0:
            switches+=int(not np.array_equal(d[-1,p["residual"]][1][selected]["combined"],p["steps"][selected]["combined"]))
    post_diff=sorted(int(i) for i in np.flatnonzero(y!=parent_y))
    if focal<0:check(not post_diff and switches==0,"inactive gate changed a command")
    # A guard rollback can itself change the final chosen branch. Attribute prefix effects within each decoder.
    propagation=set()
    for branch,trace in ((False,bt),(True,at)):
        old=d[-1,branch][1]
        if focal>=0:
            for i in trace:
                if i>focal and trace[i]["action"]!=old[i]["action"]:propagation.add(i)
    hypothetic=False
    if as_if>=0:
        yy,_=common.policy.guard(parent,x,oracle(as_if,False)[0],oracle(as_if,True)[0])
        hypothetic=not np.array_equal(yy,parent_y)
    return {"step":t,"reference_window":anchor,"physical_eligible":admissible,"focal_slot":focal,
            "focal_uid":feat.own_uid[focal] if focal>=0 else None,"as_if_eligible":plausible,"as_if_focus":as_if,
            "as_if_different_joint_command":bool(hypothetic),"guard_rolled_back":bool(rolled),
            "parent_guard_rolled_back":bool(parent_rollback),"changed_final_slots":post_diff,
            "later_command_changes_inside_branches":sorted(propagation),"verified_execution_branches":2,
            "verified_active_slot_evaluations":sum(len(p["steps"]) for p in actual),
            "different_focal_logits_branches":switches}


def ranges_json(table):
    return {"/".join(map(str,k)):{"cargo":list(v[:2]),"distance":list(v[2:4]),"count":v[4]} for k,v in table.items()}


def input_bindings():
    protocol=read(HERE/"protocol.json");release=read(HERE/"release.json")
    check(sha(HERE/"candidate.py")==release["candidate_sha256"] and sha(HERE/"protocol.json")==release["protocol_sha256"]
          and sha(common.PARENT)==release["parent_sha256"]==protocol["references"]["parent"],"wrong release identity")
    check(sha(PREVIOUS/"candidate.npz")==protocol["references"]["weights"],"wrong trained weight identity")
    samples={}
    for name,digest in protocol["references"]["inputs"].items():
        path=PREVIOUS/"train"/name;check(sha(path)==digest,"changed training-root path")
        rec=read(path);hdr=rec.get("state",rec)["header"]
        key=hdr["seed"],hdr["original_seat"],hdr["t"];check(key not in samples,"duplicate training root")
        samples[key]=hdr
    sources=read(PREVIOUS/"maps.json");maps=sources["training"]
    check(set(samples)=={(s,i,t) for s in maps for i in (0,1) for t in WINDOWS},"training roots incomplete")
    table=support(samples.values());check(protocol["ranges"]==ranges_json(table),"target stratum not derived from the actual training worker")
    return release,sources,table,samples


def run():
    common.assert_cpu()
    release,sources,table,samples=input_bindings()
    parent=common.policy.load(common.PARENT);cand=common.policy.load(HERE/"candidate.py")
    with np.load(PREVIOUS/"candidate.npz") as dat:
        learned=(np.ascontiguousarray(dat["whead.weight"].T),dat["whead.bias"].copy())
    check(all(a.dtype==np.float32 for a in learned) and np.array_equal(learned[0],cand._LOCAL_WT) and np.array_equal(learned[1],cand._LOCAL_B),"incorrect exact FP32 expert update")
    for p,c in zip(parent._NETS,cand._NETS):
        check(p.w.keys()==c.w.keys() and all(np.array_equal(p.w[k],c.w[k]) for k in p.w),"nonfocal policy weights differ")
    check(np.array_equal(parent._RETURN_THETA,cand._RETURN_THETA) and parent._HYBRID==cand._HYBRID,"return residual or mixture changed")
    check(all(inspect.getsource(getattr(parent,n))==inspect.getsource(getattr(cand,n)) for n in ("_return_basis","_return_conflicts")),"return or guard basis changed")
    rows=[];maps=sources["training"]
    for idx,seed in enumerate(maps):
        for original_seat in (0,1):
            tape_path,digest=common.tapes()[seed,original_seat]
            check(sha(tape_path)==digest==sources["historical_maps"][str(seed)]["seat_raw_sha256"][str(original_seat)],"wrong source trajectory")
            tape=read(tape_path);world=worlds.World(seed,"training")
            check(world.terrain_sha256==tape["initial_terrain_u8_sha256"],"wrong terrain")
            # Each historical physical seat has its own feature history. Only sampled policy calls incur inference cost.
            feature=sys.modules["cnet2_feat"].Feat
            ff=[feature(s) for s in (0,1)];public_ff=[feature(s) for s in (0,1)]
            width=5 if idx<2 else 0
            schedule={t+dt:t for t in WINDOWS for dt in range(-width,width+1)}
            for t,joint in enumerate(tape["actions"]):
                if t>max(schedule):break
                anchor=schedule.get(t)
                for seat in (0,1):
                    obs=world.obs(seat);x=ff[seat].observe(obs)
                    actual_seat=original_seat if anchor==205 else 1-original_seat
                    if anchor and seat==actual_seat:
                        r=check_root(cand,parent,learned,ff[seat],obs,public_ff[seat],x,t,anchor,table)
                        r.update(seed=seed,original_seat=original_seat,physical_seat=seat,source_tape_sha256=digest)
                        if t==anchor:
                            old=samples[seed,original_seat,anchor];check(old["seat"]==seat,"recorded physical seat differs")
                            choice=old["choice"];r["original_training_slot"]=None if choice is None else choice["slot"]
                            r["original_training_uid"]=None if choice is None else choice["uid"]
                        rows.append(r)
                    else:public_ff[seat].observe(obs)
                world.advance(joint)
    inside=[r for r in rows if r["step"] in WINDOWS];outside=[r for r in rows if r["step"] not in WINDOWS]
    check(len(inside)==32 and len(outside)==80,"wrong root sample count")
    summary={"training_anchors":len(inside),"off_window_observations":len(outside),"active_training_anchors":sum(r["focal_slot"]>=0 for r in inside),
        "same_focus_as_original_training":sum(r["focal_uid"] is not None and r["focal_uid"]==r["original_training_uid"] for r in inside),
        "changed_joint_anchors":sum(bool(r["changed_final_slots"]) for r in inside),"off_window_focal_activations":sum(r["focal_slot"]>=0 for r in outside),
        "off_window_command_changes":sum(bool(r["changed_final_slots"]) for r in outside),"neighbors_with_as_if_focus":sum(r["as_if_focus"]>=0 for r in outside),
        "neighbors_with_as_if_changed_joint":sum(r["as_if_different_joint_command"] for r in outside),
        "anchors_with_new_prefix_changing_a_later_command":sum(bool(r["later_command_changes_inside_branches"]) for r in inside),
        "logged_slot_evaluations":sum(r["verified_active_slot_evaluations"] for r in rows),"different_focal_logits_branches":sum(r["different_focal_logits_branches"] for r in inside)}
    report={"passed":True,"candidate_sha256":release["candidate_sha256"],"parent_sha256":release["parent_sha256"],"protocol_sha256":release["protocol_sha256"],
            "audit_sha256":sha(Path(__file__)),"source_weights_sha256":sha(PREVIOUS/"candidate.npz"),"scope":"historical training observations, independent gated FP32 decoder with observed public production commands and exact per-slot logits/legality/features","training_ranges":ranges_json(table),"summary":summary,"rows":rows}
    common.write(HERE/"mechanism.json",report);print(summary,flush=True)
    return report


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()

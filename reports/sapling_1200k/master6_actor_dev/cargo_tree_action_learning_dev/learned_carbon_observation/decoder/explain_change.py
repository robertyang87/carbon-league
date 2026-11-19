"""Explain the chronologically first decoder-induced order difference on the first held terrain."""
import gzip
import inspect
import json
from pathlib import Path
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import F, PROTOCOL, ROOT, public_frames, read, save, sha


def tapped(path):
    module=types.ModuleType("explain_"+path.parent.name)
    module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),"exec"),vars(module))
    func=inspect.getsource(module._return_decode)
    marker="        action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    assert func.count(marker)==1
    student_move="policy_move" if "policy_move" in func else "move"
    func=func.replace(marker,marker+f"\n        tap(apply_residual,slot,action,combined,move,{student_move})")
    rows={}
    def tap(branch,slot,cmd,combined,shared,student):
        rows.setdefault(branch,{})[slot]=dict(action=int(cmd),logits=combined.tolist(),carbon_shared=(shared[:,7]*100).tolist(),carbon_sapling=(student[:,7]*100).tolist())
    module.tap=tap
    exec(compile(func,str(path)+":trace","exec"),vars(module))
    return module,rows


def extract(path,case,seat,t,expected,uid):
    m,track=tapped(path)
    for obs,truth,_,cfg in public_frames(case):
        now=obs[seat]
        if now["step"] in (0,t):
            track.clear()
            physical=m.agent(now,cfg)
            if now["step"]==t:
                assert physical==expected
                active=m._FEAT.own_uid
                f=m._FEAT
                target=active.index(uid)
                labels={}
                for side,rr in track.items():
                    arr=np.full(F.OWN,-1)
                    for k,z in rr.items():arr[k]=z["action"]
                    labels[side]=arr
                base,res=labels[False],labels[True]
                to_actions=f.to_actions
                if to_actions(base)!=to_actions(res):
                    branch=to_actions(res)==physical
                else:
                    # If the candidate actions match exactly, neither protected-slot nor extra-conflict guard fires.
                    branch=True
                assert to_actions(labels[branch])==physical
                final=track[branch]
                physical_order=[0,1,4,3,2] if seat else list(range(5))
                affected=[k for k,z in final.items() if k<=target and z["carbon_shared"]!=z["carbon_sapling"]]
                return dict(pack_sha256=sha(path.read_bytes()),accepted_return_path=branch,focal_slot=target,role=now["players"][seat][2][uid][2],
                     emitted=physical.get(uid,"NONE"), earlier_physical_labels=[(active[k],physical.get(active[k],"NONE")) for k in final if k<target],
                     earlier_or_focal_slots_with_sapling_carbon_difference=affected,
                     focal_physical_order_combined_logits=[final[target]["logits"][k] for k in physical_order],
                     focal_physical_order_student_carbon=[final[target]["carbon_sapling"][k] for k in physical_order],
                     focal_physical_order_partner_carbon=[final[target]["carbon_shared"][k] for k in physical_order])
        else:
            m._FEAT.observe(now)
    raise AssertionError("root missing")


def main():
    case=next(c for c in PROTOCOL["cards"] if c["split"]=="val")
    path=HERE/"held_turns.json.gz"
    rows=read(path); rows=rows["rows"] if isinstance(rows,dict) else rows
    choices=[]
    for r in rows:
        if r["seed"]!=case["seed"]:continue
        a,b=[r["commands"][k] for k in ("encoder_student","decoder_student")]
        if a!=b:
            choices.append((r["t"],r["seat"],a,b))
    if not choices:
        raise ValueError("no held first-map action differences")
    t,seat,a,b=min(choices,key=lambda z:(z[0],z[1]))
    changed=sorted(u for u in a.keys()|b.keys() if a.get(u)!=b.get(u))
    uid=changed[0]
    arms={k:extract(path,case,seat,t,cmd,uid) for k,path,cmd in (("encoder",HERE.parent/"candidate.py",a),("decoder",HERE/"candidate.py",b))}
    assert arms["encoder"]["earlier_physical_labels"]==arms["decoder"]["earlier_physical_labels"]
    save(HERE/"example.json",dict(seed=case["seed"],t=t,seat=seat,focal_uid=uid,physical_moves=["NONE","UP","RIGHT","DOWN","LEFT"],
         purpose="First difference in chronological then seat order on the first held card. Only after aggregate comparisons are frozen. No counterfactual payoff inferred.",arms=arms))

if __name__=="__main__":main()

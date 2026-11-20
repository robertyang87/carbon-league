"""A two-base state with one recruit left exercises route-dependent physical action mass."""
from unittest.mock import patch
import sys
import numpy as np
from train import HERE, STUDY, read, save, sha, load_source, entry, check


def run():
    candidate=HERE/"candidate.py"
    actor=load_source(candidate)
    card=read(STUDY/"sources.json")["cards"][0]
    _,observation,cfg,_=next(entry.public_history(card))
    actor.agent(observation,cfg)
    F=sys.modules["cnet2_feat"]
    f=F.Feat(card["expert_seat"])
    x=f.observe(observation)
    x["cash"]=30.
    x["slots"]=1
    x["tok_mask"][:]=False
    x["tok_mask"][10:12]=True
    x["tok_kind"][10:12]=2
    check(all(f.own_uid[i] is not None for i in [10,11]),"two physical bases required")
    prefix0,prefix1=F.Prefix(x,f.aux),F.Prefix(x,f.aux)
    u0,m0,l0=prefix0.feats(10)
    prefix1.feats(10)
    prefix0.advance(10,0)
    prefix1.advance(10,1)
    u11,m11,l11=prefix0.feats(11)
    u21,m21,l21=prefix1.feats(11)
    check(l11.tolist()==[True,True,True,False,False] and l21.tolist()==[True,False,False,False,False],
          "real Prefix did not consume shared slots and cash")
    calls=[]
    def stub(x,aux,encoded,residual,forced=None,trace=None):
        if forced is not None:
            calls.append(forced)
            raise AssertionError("An unrepresentable option reached the forced decoder")
        labels=np.full(F.OWN,-1,np.int64)
        labels[10:12]=[int(residual),0]
        trace[10]=(u0,m0,l0,np.array([12.,-12.,-12.,0.,0.]))
        trace[11]=((u21,m21,l21) if residual else (u11,m11,l11))+(np.zeros(5),)
        return labels
    with patch.object(actor,"_NETS",[]),patch.object(actor,"_native_decode",stub),patch.object(actor,"_FEAT",f):
        base,menu=actor._native_menu(x,f.aux,18)
        teams,probabilities=actor._native_distribution(base,menu)
    check(menu[0]==11 and len(teams)==1 and np.isclose(probabilities[0],1.) and not calls,
          "unsupported cross-route mass did not revert to guarded incumbent")
    save(HERE/"guard_check.json",dict(candidate_sha256=sha(candidate),public_source=card["sha256"],
        cases="Actual public base IDs and real Prefix legality at cash 30, one free slot. Stub forces a divergent greedy first-base decision; actual runtime carries the merged guard fallback.",
        default_second_base_legal=l11.tolist(),other_second_base_legal=l21.tolist(),
        pivot=int(menu[0]),attempted_forced_calls=calls,distinct_physical_orders=len(teams),
        merged_probability=float(probabilities[0]),physical_order=f.to_actions(teams[0])))
if __name__=="__main__":
    run()

"""Exercise the actual decoder in both modes, including team members downstream of a changed route."""
from collections import Counter
import inspect
import sys
import numpy as np
from dataset import HERE,BASE,SEQ,PARENT,terr,sha,read,write,policy

def main():
    m=policy.load(HERE/"candidate.py")
    p=policy.load(PARENT)
    # The disabled adapter must reproduce the complete frozen incumbent on all observed prefixes.
    f=sys.modules["cnet2_feat"]
    prefix=inspect.getsource(m._return_decode)
    start='        if direction and slot < 10 and x["tok_kind"][slot] == 1:'
    end='        if not apply_residual:'
    assert prefix.count(start)==1 and prefix.count(end)==1
    counts=Counter()
    def check(role,old,new,direction):
        if not direction:return
        if role=="planter":
            assert (old==0)==(new==0)
            counts["planter_directions_changed"]+=int(old!=new)
        else:
            assert old==new
        counts[role]+=1
    ns=dict(vars(m),check=check)
    src=prefix.replace(start,'        old_choice=action\n'+start)
    src=src.replace(end,'        check("base" if slot>=10 else ("planter" if x["tok_kind"][slot]==1 else "collector"),old_choice,action,direction)\n'+end)
    exec(src,ns)
    m._return_decode=ns["_return_decode"]
    cases=[]
    for source in ("Top5","Master3s2"):
        c=next(c for c in read(SEQ/"sources.json") if c["source"]==source and c["split"]=="test")
        cfg,orders,_,full=terr.load_game(c)
        sim=terr.FastCarbon(cfg).reset(seed=c["seed"],full_carbon=full)
        feat=f.Feat(c["seat"]);ref=f.Feat(c["seat"])
        cohort=0
        for t,joint in enumerate(orders[:193]):
            obs=sim.obs(c["seat"]);x=feat.observe(obs);oldx=ref.observe(obs)
            if t>=64 and t%4==0:
                enc=[]
                for net in m._NETS:
                    tok,ctx=net.encode(x,2);enc.append((net,tok,ctx,net.lin(ctx,"h0")))
                assert np.array_equal(m._direction_team(x,feat.aux,enc,False),p._mixed_actions(oldx,ref.aux))
                original=m._direction_team(x,feat.aux,enc,False)
                actual=m._mixed_actions(x,feat.aux)
                assert not m._return_conflicts(x,actual)-m._return_conflicts(x,original)
                cohort+=1
            sim.step(*joint)
        cases.append(dict(terrain=c["terrain"],reference_turns=cohort))
    assert counts["planter"] and counts["collector"] and counts["base"] and counts["planter_directions_changed"]
    write(HERE/"invariants.json",dict(package=sha(HERE/"candidate.py"),frozen_parent=sha(PARENT),cases=cases,emission_by_role=dict(counts)))


if __name__=="__main__":main()

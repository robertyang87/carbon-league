"""Verify how the exact derived file behaves if optional PyTorch cannot be imported."""
import sys
sys.modules["torch"]=None
from probe import D,HERE,Game,build,read,sha,module,need,ref
import numpy as np
import json,gzip
from pathlib import Path

def main():
    parent=(D/"cold_start/candidate.py").read_bytes();candidate=build()
    raw=(D/"ladder/replays/bdbe59b04f71e43e53696874c8fd5958-1.json.gz").read_bytes()
    data=json.loads(gzip.decompress(raw));game=Game(data);t=102;s=1;orders=[]
    receipt=read(HERE/"probe.json");ticket=next(r for c in receipt["cases"] if c["group"]=="Fable_V7" for r in c["roots"] if r["t"]==t)
    for src in (parent,candidate):
        m=module(src);track=m.BeliefTracker(m._Bel,s);prev=None
        for turn in range(t):
            obs=game._obs(turn,s);ps=obs["players"];track.step(ps,turn,obs["carbon"]);track.set_actions(ps,data["steps"][turn+1][s]["action"]);prev=ps
        m._ST=dict(tr=track,prev=prev,rng=np.random.RandomState(ticket["p_rng"]));need(not m._SEARCH_OK,"Expected torch unavailable")
        action=m.agent(game._obs(t,s),{k:v for k,v in data["configuration"].items() if k!="randomSeed"})
        orders.append(dict(sha=sha(src),search_available=m._SEARCH_OK,tree_head_dispatched=getattr(m,"_TREE_LAST",None) is not None,action=action))
    need(orders[0]["action"]==orders[1]["action"],"Unexpected fallback difference")
    (HERE/"numpy_only.json").write_text(json.dumps(dict(game=sha(raw),t=t,seat=s,modes=orders),indent=2)+"\n")
if __name__=="__main__":main()

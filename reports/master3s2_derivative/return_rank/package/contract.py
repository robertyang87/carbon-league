"""Synthetic physical safeguards for the search menu, independent of learned ranking quality."""
import copy
import json
import time
import types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]/"cold_start/candidate.py"


def main():
    # The contract checks only menu selection and game geometry, using injected
    # deterministic predictions. The real immutable model is separately probed.
    src = PARENT.read_text()
    mod=types.ModuleType("return_contract")
    ns=mod.__dict__
    ns.update(np=np,_time=time,_RETURN_WEIGHTS={})
    for start,end in (("SIZE = 15","def go_home("),):
        exec(src[src.index(start):src.index(end)],ns)
    exec((HERE/"offer.py").read_text(),ns)
    ns["_RETURN_FEATURES"]=lambda players,seat,uid,step,carbon:np.zeros((5,1))
    ns["_RETURN_SCORE"]=lambda _,w:np.array([0,1,0,0,0])
    # On row 1 column 1, UP deposits at cell 1. DOWN is the poor A0.
    players=[[300,{"home":1},{"worker":[16,150,"COLLECTOR"]},{}],
             [300,{"enemyhome":200},{},{}]]
    a0={"worker":"DOWN"}
    def evaluate(ps=None, orders=None, fixed=(),clock=0,most=2,cost=0):
        action=copy.deepcopy(a0 if orders is None else orders)
        joint=[action,{"worker":"LEFT"}]
        mod._RETURN_TRACE={}
        mod._return_menu(joint,ps or players,0,80,fixed,most,time.perf_counter()-clock,
                         {"size":15,"moveCost":cost},[50]*225)
        assert joint[0]==action
        return joint
    original=[a0,{"worker":"LEFT"}]
    assert evaluate()==[a0,{"worker":"UP"}]
    assert evaluate(most=8)==[a0,{"worker":"UP"}]
    single=[a0]
    mod._return_menu(single,players,0,80,(),8,time.perf_counter(),{},[50]*225)
    assert single==[a0]
    assert evaluate(fixed=["worker"])==original
    assert evaluate(clock=.6)==original
    assert evaluate(most=1)==original
    assert evaluate(cost=.001)==original
    for load in (149,150):
        ps=copy.deepcopy(players);ps[1][2]["foe"]=[2,load,"COLLECTOR"]
        assert evaluate(ps)==original
    ps=copy.deepcopy(players);ps[1][2]["foe"]=[2,151,"COLLECTOR"]
    assert evaluate(ps)[1]=={"worker":"UP"}
    ps=copy.deepcopy(players);ps[0][2]["friendly"]=[2,0,"COLLECTOR"]
    assert evaluate(ps,{"worker":"DOWN","friendly":"LEFT"})[1]=={"worker":"LEFT"}
    for recruit in ("RECCOLLECTOR","RECPLANTER"):
        assert evaluate(orders={"home":recruit,"worker":"DOWN"})[1]=={"worker":"LEFT"}
    # Duplicates preserve order. All diagnostic hooks are optional and O(1).
    m=[a0,{"worker":"UP"}]
    mod._return_menu(m,players,0,80,(),2,time.perf_counter(),{},[50]*225)
    assert m==[a0,{"worker":"UP"}] and mod._RETURN_TRACE["existing"] is True
    mod._RETURN_SCORE=lambda _,w:[0,np.nan,0,0,0]
    assert evaluate()==original
    def broken(*args,**kwargs):
        raise ValueError("Missing public observation")
    mod._RETURN_FEATURES=broken
    assert evaluate()==original
    print("PASS candidate width, parent identity, protected workers, wall cutoff, geometry, same/greater enemy load, friendly landing, recruitment, duplicate and finite/error fallback")


if __name__=="__main__":main()

"""Exercise the official ACTIVE->DONE transition against the native search's two unmodified callbacks."""
import ast
import copy
from types import SimpleNamespace
from visibility import D,HERE,read,report,data,sha
from zerosum_env import utils
from zerosum_env.envs.carbon.carbon import compute_next_board
from zerosum_env.envs.carbon.idgen import reset
import numpy as np

def native_functions():
    code=(D/"cold_start/candidate.py").read_text()
    fs=ast.parse(code)
    functions={n.name:n for n in fs.body if isinstance(n,ast.FunctionDef)}
    nested={n.name:n for n in functions["_search"].body if isinstance(n,ast.FunctionDef)}
    d={"np":np,"me":0,"op":1,"LAST_TURN":298}
    for n in (functions["_terminal"],nested["over"],nested["result"]):
        exec(compile(ast.Module(body=[n],type_ignores=[]),"native:"+n.name,"exec"),d)
    return d

def test(t=20,cash=29,has_tree=False,both=False):
    c=read(D/"ladder/ds-pro-v1-placement.json")["games"][0];source,_=data(c)
    cfg=utils.structify(source["configuration"]);counter=native_functions()
    ps=[[cash,{"b0":0},{"p0w":[1,0,"COLLECTOR"]},{}],
        [80,{"b1":2,"b2":200},{"p1w":[185,91,"COLLECTOR"]},{}]]
    tree={}
    if has_tree:
        ps[0][3]["p0t"]=[112,3];tree["p0t"]=["p0w",0.]
    if both:
        ps[1][0]=cash;ps[1][2]["p1w"]=[15,0,"COLLECTOR"]
    cmd=[{"p0w":"RIGHT"},{"p1w":"UP" if both else "DOWN"}];arr=[]
    carbon=[0]*225
    for s in (0,1):
        o=utils.structify(dict(step=t,carbon=carbon.copy(),full_carbon=carbon.copy(),player=s,
                               players=copy.deepcopy(ps),trees=copy.deepcopy(tree),remainingOverageTime=60.))
        arr.append(utils.structify(dict(observation=o,action=cmd[s],reward=ps[s][0],status="ACTIVE")))
    reset();official=compute_next_board(arr,SimpleNamespace(configuration=cfg))
    ob=official[0].observation
    post=[x[0] for x in ob.players]; b=[0,0,dict(step=ob.step,players=ob.players)]
    status=[x.status for x in official];rewards=[x.reward for x in official]
    return dict(t=t,cash=cash,has_tree=has_tree,both=both,root_cash=[p[0] for p in ps],orders=cmd,
                post_cash=post,official_status=status,official_rewards=rewards,native_would_stop=counter["over"](b),
                native_wallet_only_terminal_if_called=counter["result"](b),official_terminal_value_if_done=counter["_terminal"](rewards[0]-rewards[1]) if all(x=="DONE" for x in status) else None)

def main():
    cases=[test(),test(cash=30),test(has_tree=True),test(both=True),test(t=298)]
    assert cases[0]["official_rewards"]==[-280,80] and cases[0]["post_cash"]==[29,171] and not cases[0]["native_would_stop"]
    assert cases[1]["official_status"]==cases[2]["official_status"]==["ACTIVE","ACTIVE"]
    assert cases[3]["official_rewards"]==[-280,-280]
    assert cases[4]["native_would_stop"] and cases[4]["official_rewards"]==[-2,80]
    report(HERE/"terminal_examples.json",dict(source=sha((D/"cold_start/candidate.py").read_bytes()),cases=cases,
                source_note="At the environment's normal time limit DONE is applied by outer core; compute_next_board handles extinction independently. All samples above with DONE were produced by actual economic extinction. Native wallet-only value shown even when over() would actually call critic and continue."))
if __name__=="__main__":main()

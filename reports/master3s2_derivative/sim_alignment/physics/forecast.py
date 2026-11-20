"""Official-action eight-transition forecast from each contender's own causal carbon estimate."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import check
sys.path.insert(0,str(check.HERE.parent))
import visibility

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
TICKS=(5,14,34,70)
PROTOCOL="4a9d178f35d2a284961c31bb18b2a1508a4763e8376a8361891aa1d8bce3b7f3"
CANDIDATE="aba10c2c83c620f93166881c4e8c626a78233d2dffe0b864acfccaa1d9b50b5b"


def margin(pl,seat):
    return dict(cash=pl[seat][0]-pl[1-seat][0],
                cargo=sum(w[1] for w in pl[seat][2].values())-sum(w[1] for w in pl[1-seat][2].values()))

def identity(pl):
    return [([*sorted((uid,x[0],x[2]) for uid,x in p[2].items())],
             [*sorted((tid,tr[0],tr[1]) for tid,tr in p[3].items())]) for p in pl]

def fork(turn,history,est,frames,t,seat,cfg):
    root=history["_st"].copy()
    root.carbon=list(map(float,est))
    sim=dict(_st=root)
    for k in range(t,t+8):
        frame=frames[k+1];acts=[a["action"] or {} for a in frame]
        sim=turn(sim,cfg,acts)
        expected=frame[0]["observation"]
        if identity(sim["players"])!=identity(expected["players"]):
            before=identity(sim["players"]);after=identity(expected["players"]);delta=[]
            for side in (0,1):
                for slot,kind in ((0,"worker"),(1,"tree")):
                    a,b=set(before[side][slot]),set(after[side][slot])
                    if a!=b:delta.append(dict(side=side,kind=kind,only_model=sorted(a-b),only_official=sorted(b-a)))
            return dict(complete=False,first_structural_change=k+1,delta=delta)
    final=frames[t+8][0]["observation"]["players"]
    predicted=margin(sim["players"],seat);truth=margin(final,seat)
    return dict(complete=True,prediction=predicted,truth=truth,
                error={k:predicted[k]-v for k,v in truth.items()})

def group(rows):
    kept=[r for r in rows if r["both_complete"]]
    result=dict(total=len(rows),paired_complete=len(kept),
                failure_patterns=dict(Counter(f'{int(r["parent"]["complete"])}{int(r["candidate"]["complete"])}' for r in rows)))
    if kept:
        result["map_macro"]={}
        for key in ("cash","cargo"):
            errors={}
            for arm in ("parent","candidate"):
                bymap={}
                for r in kept:bymap.setdefault(r["game"],[]).append(r[arm]["error"][key])
                errors[arm]=dict(mae=float(np.mean([np.mean(np.abs(z)) for z in bymap.values()])),
                                 bias=float(np.mean([np.mean(z) for z in bymap.values()])))
            ds=[abs(r["parent"]["error"][key])-abs(r["candidate"]["error"][key]) for r in kept]
            errors.update(better_roots=sum(d>1e-8 for d in ds),worse_roots=sum(d< -1e-8 for d in ds),
                          tied_roots=sum(abs(d)<=1e-8 for d in ds),maps=len(bymap))
            result["map_macro"][key]=errors
    return result

def main():
    pr=(HERE.parent/"belief_protocol.json").read_bytes()
    assert check.sha(pr)==PROTOCOL
    bookpath=FAMILY/"ladder/ds-pro-v1-placement.json"
    assert check.sha(bookpath.read_bytes())==json.loads(pr)["placement_sha256"]
    cp=FAMILY/"sim_alignment/candidate.py";pp=FAMILY/"cold_start/candidate.py"
    assert check.sha(cp.read_bytes())==CANDIDATE and check.sha(pp.read_bytes())==check.SHA
    assert check.extract(cp.read_bytes())==check.extract(pp.read_bytes())
    turn=check.engine(); models=[visibility.load(pp),visibility.load(cp)]
    rows=[]
    for i,card in enumerate(json.loads(bookpath.read_bytes())["games"]):
        env,rs=visibility.data(card);cfg=env["configuration"];seat=card["candidate_seat"];frames=env["steps"]
        tr=[m.BeliefTracker(m._Bel,seat) for m in models]
        hist=None
        for t in range(max(TICKS)+1):
            step=frames[t];obs=visibility.obs(step,seat);pl=obs["players"]
            estimates=[tracker.step(pl,t,obs["carbon"])[0] for tracker in tr]
            if t in TICKS:
                assert hist is not None and hist["players"]==pl and hist["carbon"]==step[0]["observation"]["full_carbon"]
                a,b=[fork(turn,hist,est,frames,t,seat,cfg) for est in estimates]
                rows.append(dict(map_index=i,split="diagnostic" if i<16 else "confirmation",game=card["match_id"],replay_sha=rs,
                                 t=t,seat=seat,parent=a,candidate=b,both_complete=a["complete"] and b["complete"]))
            acts=[a["action"] or {} for a in frames[t+1]]
            for tracker in tr:tracker.set_actions(pl,acts[seat])
            if hist is None:
                hist={**obs,"carbon":step[0]["observation"]["full_carbon"]}
            hist=turn(hist,cfg,acts)
    assert len(rows)==128
    groups={s:group([r for r in rows if r["split"]==s]) for s in ("diagnostic","confirmation")}
    groups["all"]=group(rows)
    groups["by_t"]={str(t):group([r for r in rows if r["t"]==t]) for t in TICKS}
    report=dict(protocol=PROTOCOL,candidate=CANDIDATE,parent=check.SHA,groups=groups,rows=rows)
    b=gzip.compress(json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    (HERE/"forecast.json.gz").write_bytes(b)
    print(check.sha(b),groups["all"],groups["confirmation"],flush=True)

if __name__=="__main__":
    main()

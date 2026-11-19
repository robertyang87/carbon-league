"""Official final score decomposition and temporal cash gap on the four fixed trajectories."""
from collections import Counter
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import responses as r


def one(row):
    side=row["seat"];s=row["seed"];d=HERE/"games"/row["arm"]
    tape=r.load(d/f"raw_{s}_{side}.json.gz");sim=r.FastCarbon(tape["game_configuration"]).reset(seed=s)
    sim.cash=r.CashEvents(sim.cash); ledger=[Counter(),Counter()]; initial=list(sim.cash); milestones={}
    for t,orders in enumerate(tape["actions"],1):
        sim.cash.events.clear();sim.step(*orders)
        for kind,seat,delta,*_ in sim.cash.events:ledger[seat][kind]+=delta
        if t in (60,120,180,240,270,len(tape["actions"])):
            amount=sim.cash[side]-sim.cash[1-side]
            accounted=sum(ledger[side].values())-sum(ledger[1-side].values())+initial[side]-initial[1-side]
            if abs(amount-accounted)>.00001:raise ValueError("ledger divergence")
            milestones[str(t)]=dict(cash_margin=amount,net_by_kind={k:ledger[side][k]-ledger[1-side][k] for k in sorted(set(ledger[side])|set(ledger[1-side]))})
    assert tuple(sim.cash)==tuple(row["score_by_seat"])
    return milestones


if __name__=="__main__":
    plan=r.load(HERE/"scores.json")
    paths={row["arm"]+str(row["seat"]):one(row) for row in plan};deltas={}
    for side in (0,1):
        x=paths["parent"+str(side)];y=paths["candidate"+str(side)]
        for t in sorted(set(x)&set(y),key=int):
            a=x[t]["net_by_kind"];b=y[t]["net_by_kind"]
            deltas.setdefault(str(side),{})[t]=dict(cash_margin=y[t]["cash_margin"]-x[t]["cash_margin"],
                net_by_kind={k:b.get(k,0)-a.get(k,0) for k in sorted(set(a)|set(b))})
    r.write(HERE/"economy.json",dict(paths=paths,same_seat_candidate_minus_parent=deltas,
       note="Each arm induces a new trajectory against a naturally stochastic examiner. These stage-by-stage paid cash ledgers describe outcomes and cannot identify the cause of a change."))

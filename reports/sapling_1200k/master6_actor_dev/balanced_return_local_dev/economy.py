"""Read all three realized action tapes, retaining both own and opposing income and outlays."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "balanced_return_dev/economy_summary.py"
SHA = "7eed5f4453d18f28ab424c8ede35a723e8475e4029fdbdc8d16c866b9857c247"
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
    raise ValueError("the preregistered cash write reconstruction changed")
spec = importlib.util.spec_from_file_location("balanced_frozen_cash_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def timing(cases, ledgers, seat, base):
    raw_a,raw_b=cases[base],cases["local"]
    fields_a,flow_a=ledgers[base]
    fields_b,flow_b=ledgers["local"]
    limit=min(len(fields_a),len(fields_b))-1
    def cash_delta(t):
        return round(fields_b[t]["current_cash_margin"]-fields_a[t]["current_cash_margin"],2)
    first=None
    for t in range(limit):
        order_a,order_b=raw_a["actions"][t],raw_b["actions"][t]
        diff=[audit.differences(x,y) for x,y in zip(order_a,order_b)]
        if any(diff):
            same_obs=raw_a["shadow"][t]["public_observation_sha256"] == raw_b["shadow"][t]["public_observation_sha256"]
            if not same_obs or diff[1-seat]:
                raise ValueError("first actual action difference did not start at an identical physical state and foe action")
            first=dict(t=t,differences_by_physical_seat=diff,
                       focal_observation_sha256=raw_b["shadow"][t]["public_observation_sha256"])
            break
    first_cash=next((t for t in range(1,limit+1) if cash_delta(t) != 0.),None)
    if first_cash is not None and (first is None or first["t"] >= first_cash):
        raise ValueError("cash leads policy differences")
    window=[]
    if first_cash is not None:
        for t in (first_cash+d for d in (-10,-5,-3,-1,0,1,3,5,10)):
            if 0<=t<=limit:
                window.append(dict(state_step=t, post_action_cash_margin_delta=cash_delta(t),
                                   base_cash_by_seat=fields_a[t]["cash_by_physical_seat"],
                                   local_cash_by_seat=fields_b[t]["cash_by_physical_seat"],
                                   previous_transition_cash_writes=None if t==0 else {base:flow_a[t-1],"local":flow_b[t-1]}))
    return dict(first_different_joint=first,first_cash_margin_divergence=first_cash,window=window)


def case_pair(seed,seat,cases,arms):
    by_arm={a:audit.relative_summary(c["economy"]) for a,c in cases.items()}
    ledgers={a:audit.replay(c) for a,c in cases.items()}
    cmp={}
    for base in ("parent","ungated"):
        diff={side:{k:by_arm["local"][side][k]-by_arm[base][side][k] for k in by_arm[base][side]}
              for side in ("focal","enemy")}
        dm=arms["local"]["margin"]-arms[base]["margin"]
        if abs(diff["focal"]["official_score"]-diff["enemy"]["official_score"]-dm)>1e-6:
            raise ValueError("cash accounts do not explain official difference")
        cmp[base]=dict(delta=diff,official_delta_margin=dm,timing=timing(cases,ledgers,seat,base))
    return dict(seed=seed,seat=seat,by_arm=by_arm,new_relative_to=cmp,
                input_sha256={a:v["tape_sha256"] for a,v in arms.items()},
                definitions="Positive outlay means paid cash. Physical seat preserved. State t follows order t-1. Both sides continue reacting independently after the first differing joint.")

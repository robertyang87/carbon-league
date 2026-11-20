"""Replay the first qualified held root from both actual root orders to official terminal."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
HELD=HERE.parent/"held"
BASE=HERE.parent/"terminal_pairs"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(row):
    return (row["seed"],row["seat"],row["step"])


def verify_tickets():
    plan=json.loads((HERE/"protocol.json").read_bytes())
    held=(HELD/"protocol.json").read_bytes()
    assert sha(held)==plan["search_protocol_sha256"]
    schedule=json.loads(held)
    raw=(HELD/"search_roots.json.gz").read_bytes()
    panel=json.loads(gzip.decompress(raw))
    raw_settlement=(HELD/"search_settlement.json").read_bytes()
    summary=json.loads(raw_settlement)
    assert panel["protocol_sha256"]==sha(held) and panel["parent"]==plan["parent"] and panel["candidate"]==plan["candidate"]
    tickets=schedule["roots"]
    assert len(tickets)==32 and len(set(map(key,tickets)))==32
    selected=[]
    for i,(ticket,r,record) in enumerate(zip(tickets,panel["roots"],summary["roots"],strict=True)):
        assert key(ticket)==key(r)==key(record) and ticket["terrain"]==r["terrain"]
        physics=record["first_trial_physics"]
        eligible=(record["attributable_changed"] and record["all_original_and_new_abs_gt_1e_5"] and bool(physics)
                  and all(not j["same_official_next"] for j in physics))
        if eligible and summary["protocol_gate"] and not summary["admitted_over_058"]:
            selected.append((i,ticket,r))
    first=selected[0] if selected else None
    assert (key(first[1]) if first else None)==(key(summary["first_eligible_root"]) if summary["first_eligible_root"] else None)
    prev=json.loads((BASE/"selection.json").read_bytes())["selected"]
    assert len(prev)==1 and prev[0]["seed"]==621481220
    exclude={p["terrain"] for p in prev}
    assert all(t["seed"]!=621481220 and t["terrain"] not in exclude for t in tickets)
    return first,dict(registration_sha256=sha((HERE/"protocol.json").read_bytes()),search_protocol_sha256=sha(held),
                      panel_sha256=sha(raw),settlement_sha256=sha(raw_settlement),all_tickets=len(tickets),
                      all_independently_qualified=len(selected),qualifying_tickets=[key(t) for i,t,r in selected],
                      selected=(key(first[1]) if first else None),prior_map_excluded=sorted(exclude))


def save(filename,obj):
    raw=(json.dumps(obj,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()
    if filename.endswith(".gz"):raw=gzip.compress(raw,mtime=0)
    path=HERE/filename
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("Conflicting saved file "+filename)
    if not path.exists():
        tmp=path.with_suffix(".partial")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return sha(raw)


def main():
    first,ledger=verify_tickets()
    save("selection.json",ledger)
    if first is None:
        save("results.json",dict(qualified=0,completed_pairs=0,delta_margin=None,development_gate=False,
                                  reason="No distinct eligible held root from the complete registered panel"))
        return
    i,ticket,row=first
    os.sched_setaffinity(0,{0})
    os.environ["CARBON_DEBUG"]="1"
    for keystr in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[keystr]="1"
    sys.path.insert(0,str(BASE))
    import branches
    from audit import replay
    from history import setup,views,advance,token,freight
    lib=HERE.parents[1]/"opponent_policy/protocol.json"
    records=json.loads(lib.read_bytes())["games"]
    card=next(c for c in records if c["split"]=="val" and c["sha256"]==ticket["source_sha256"] and c["terrain"]==ticket["terrain"] and c["expert_seat"]==ticket["seat"])
    old,acts,_,env,sim=setup(ticket,card)
    t=ticket["step"]
    for command in acts[:t]: advance(env,sim,command)
    expected=dict(public_sha256_by_seat=[freight.digest(v) for v in views(env)],incoming_cash=list(sim.cash), prior_orders_sha256=token(acts[:t]))
    state=branches.restore_prefix(ticket,source=card,expected=expected)
    mods,cfg,env,sim,obs,_=state
    s=ticket["seat"]
    parent=mods[s]
    prefix=freight.original_state.h([m._ST for m in mods])
    parent._ST["rng"]=freight.np.random.RandomState(ticket["policy_rng"])
    parent._S_COST[:]=row["starting_cost"]
    initial_rng=freight.original_state.h(parent._ST["rng"])
    self_reply,menu=freight.root_call(parent,obs[s],cfg)
    arms=row["arms"]
    old_order=arms["P1"]["issued"]; new_order=arms["C1"]["issued"]
    admission=dict(cost=row["starting_cost"],menu_sha256=token(arms["P1"]["menu"]),parent=old_order,rotated=new_order)
    if token(menu)!=admission["menu_sha256"] or old_order not in menu or new_order not in menu or old_order==new_order:
        save("results.json",dict(qualified=1,completed_pairs=0,delta_margin=None,development_gate=False,
                                 reason="Registered root menu did not independently reproduce",root=list(key(ticket)),
                                 recovered=token(menu),expected=admission["menu_sha256"]))
        return
    order=["parent","rotated"] if i%2==0 else ["rotated","parent"]
    result=branches.continue_pair(ticket,admission,order,state,self_reply,prefix,initial_rng,save)
    audits=[replay(HERE/("%d_%d_%d_%s.json.gz"%(*key(ticket),a)),card,ticket) for a in ("parent","rotated")]
    effects=result["shifted_minus_parent_margin"]
    for idx,a in enumerate(("parent","rotated")):
        assert audits[idx]["sha256"]==result["arms"][a]["evidence_sha256"] and audits[idx]["official_margin"]==result["arms"][a]["margin"]
    save("results.json",dict(qualified=1, completed_pairs=1,delta_margin=effects,development_gate=effects>0,
                             result=result,official_rereads=audits,
                             interpretation="One controlled simultaneous root response and one naturally diverging official suffix per arm"))
    print("Independent terminal official margin difference",effects,flush=True)


if __name__=="__main__":main()

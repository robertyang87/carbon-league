"""Recover H1/H4/H8 of two already-archived native complete candidates without running new games."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DER=HERE.parent
DATA=DER/"action_rank_data"
sys.path.insert(0,str(DER/"opponent_policy"))
V2="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"


def sha(a):
    return hashlib.sha256(a).hexdigest()


def need(ok,msg):
    if not ok:raise ValueError(msg)


def patch(raw):
    text=raw.decode().replace("\r\n","\n")
    lookup={
        "    cands = [act0]; keys": "    most = _archived_k\n    cands = [act0]; keys",
        "        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT": "        return False  # offline reconstruction of a documented full H8 call",
        "        depth = t\n        if t == D2": """        depth = t
        if t in (1,4,8):
            for b in run:
                c, m = b[:2]
                if c in _selected and m == 0:
                    if over(b):
                        _emit(t,c,None,None,result(b),opp[m])
                    else:
                        _emit(t,c,view(b,me),_scalars(b[2]['players'],b[2]['step'],me),val[c,m],opp[m])
        if t == D2""",
    }
    for old,new in lookup.items():
        need(text.count(old)==1,"Capture anchor changed")
        text=text.replace(old,new)
    return text


def record_ticket(ticket,card,model):
    import numpy as np
    from original_policy import free_entry
    stem=f'{ticket["seed"]}_{ticket["seat"]}_{ticket["step"]}'
    source=DATA/ticket["split"]
    raw=(source/(stem+"_root.json.gz")).read_bytes()
    saved=json.loads(gzip.decompress(raw))
    root=dict(ordinal=ticket["ordinal"],seed=ticket["seed"],seat=ticket["seat"],step=ticket["step"],
              split=ticket["split"],terrain=ticket["terrain"],source_archive_sha256=ticket["source_sha256"],
              root_sha256=sha(raw),arms=[])
    need(saved["ticket"]==ticket and saved["protocol_sha256"]==sha((DATA/"protocol.json").read_bytes()),"Wrong root registration")
    if saved["status"]!="paired_eligible":
        root.update(status="original_skip",reason=saved["reason"])
        return root
    leaf=source/(stem+"_leaf.npz");root["leaf_sha256"]=sha(leaf.read_bytes())
    seat=ticket["seat"];steps=ticket["step"]
    tracker=model.BeliefTracker(model._Bel,seat);previous=None
    model._archived_k=saved["clock"]["K"]
    model._selected={arm["candidate_index"] for arm in saved["arms"]}
    collected={}; enemy=[]
    def emit(t,c,x,sc,v,opp):
        need((t,c) not in collected and x is not None,"Repeated or unexpected terminal observation")
        collected[t,c]=(x,np.array(sc,dtype=np.float32),float(v))
        enemy.append(opp)
    model._emit=emit
    for t,obs,cfg,issued in free_entry.public_history(card):
        if t!=steps:
            tracker.step(obs["players"],t,obs["carbon"])
            tracker.set_actions(obs["players"],issued)
            previous=obs["players"]
            continue
        archived=saved["public_root"]
        need(all(obs[k]==archived[k] for k in ("player","step","players","trees","carbon")),"Different public root history")
        model._ST=dict(tr=tracker,prev=previous,rng=np.random.RandomState(ticket["policy_rng"]))
        model._S_COST[:]=saved["cost_before_root"]
        model._S_REC=[]
        config={k:v for k,v in cfg.items() if k!="randomSeed"}
        order=model.agent(dict(obs,remainingOverageTime=archived["remainingOverageTime"]),config)
        break
    else:raise ValueError("No such decision")
    step,me,menu,depths,vals,best=model._S_REC[0]
    need((step,me)==(steps,seat) and menu==saved["menu"] and order==saved["issued"],"Different reconstructed menu/order")
    need(len(collected)==6 and all(z==enemy[0] for z in enemy),"Incomplete paired rollouts or nonshared modeled opponent")
    root.update(status="verified",menu_size=len(menu),modeled_root_opponent=enemy[0])
    with np.load(leaf,allow_pickle=False) as z:
        need(z["X"].shape==(2,1,54,15,15) and not z["exact_terminal"].any(),"Unsupported terminal/leaf")
        for idx,arm in enumerate(saved["arms"]):
            cid=arm["candidate_index"];nativeX,sc,v=collected[8,cid]
            need(arm["name"]==("first","second")[idx] and menu[cid]==arm["order"] and cid==z["candidate_index"][idx] and depths[cid]==8,"Different finalist")
            need(np.array_equal(nativex:=nativeX,z["X"][idx,0].astype(np.float32)*np.float32(.01)) and
                 sha(nativex.tobytes())==arm["leaves"][0]["x_float32_sha256"] and np.array_equal(sc,z["scalars"][idx,0]),
                 "H8 inputs disagree with original")
            orig=float(z["critic"][idx,0])
            need(orig==arm["original_value"] and abs(v-orig)<=1.e-5,"H8 value differs")
            timeline={}
            for t in (1,4,8):
                x,scalar,val=collected[t,cid]
                timeline[str(t)]=dict(value=val,x_sha256=sha(x.tobytes()),scalars_sha256=sha(scalar.tobytes()),scalars=scalar.tolist())
            root["arms"].append(dict(name=arm["name"],candidate_index=cid,order=arm["order"],depths=timeline,
                                     archived_H8=orig,H8_abs_error=abs(orig-v)))
    return root


def main(split):
    import numpy as np
    os.sched_setaffinity(0,{4})
    os.environ["CARBON_DEBUG"]="1"
    for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
    import torch
    torch.set_num_interop_threads(1)
    from original_policy import module,PROTOCOL
    raw=(DER/"cold_start/candidate.py").read_bytes()
    need(sha(raw)==V2,"Wrong agent identity")
    model=module(patch(raw).encode());torch.set_num_threads(1)
    need(model._RNET is model._NET and model._S_M==model._S_MMAX==1 and model._S_D2==8,"Unexpected native engine")
    protocol=(DATA/"protocol.json").read_bytes()
    tickets=[t for t in json.loads(protocol)["tickets"] if t["split"]==split]
    cards={r["sha256"]:r for r in PROTOCOL["games"] if r["split"]==("train" if split=="train" else "val")}
    results=[]
    for ticket in tickets:
        c=cards[ticket["source_sha256"]];need(c["expert_seat"]==ticket["seat"] and c["terrain"]==ticket["terrain"],"Source identity")
        try:
            row=record_ticket(ticket,c,model)
        except ValueError as e:
            row=dict(ordinal=ticket["ordinal"],seed=ticket["seed"],seat=ticket["seat"],step=ticket["step"],status="reconstruction_failed",reason=str(e),arms=[])
        results.append(row)
        print(row["ordinal"],row["status"],row.get("reason",""),flush=True)
    final=dict(schema="native-multidepth-finalists/1",source_sha256=V2,root_registration_sha256=sha(protocol),
               extractor=sha(Path(__file__).read_bytes()),instrumented_search_sha256=sha(patch(raw).encode()),roots=results)
    dest=HERE/split/"features.json";dest.parent.mkdir(exist_ok=True)
    with dest.open("x") as f:
        json.dump(final,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--split",choices=("train","held"),default="train");main(p.parse_args().split)

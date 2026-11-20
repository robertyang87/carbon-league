"""Time a full public request with exactly the source rollout inputs, actions, menu and budget decisions.

Teacher logits for the simulated opponent are replayed to keep untrained placeholder heads from changing
the physics. The three paths still execute their own advertised forward passes, including shallow heads.
"""
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import median
import sys
import time
from capture import setup,HERE,DATA,sha
from full_search_check import root_call


def insert_hooks(code):
    text=code.decode().replace("\r\n","\n")
    start=text.index("def _search(obs,")
    prefix,text=text[:start],text[start:]
    substitutions={
        "    net = _net()":"    net = _timed_forward",
        "    cands = [act0]":"    most = _budget_menu(most)\n    cands = [act0]",
        "        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT":"        return _budget_cut(_time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT)",
    }
    for old,new in substitutions.items():
        assert text.count(old)==1
        text=text.replace(old,new)
    return (prefix+text).encode()


def main():
    cfg,stock=setup()
    import numpy as np
    import torch
    from forward import evaluate
    from original_policy import module,original,PROTOCOL,free_entry
    raw=original()
    mod=module(insert_hooks(raw))
    torch.set_num_threads(1)
    net=mod._net()
    assert mod._RNET is mod._NET and mod._S_ABORT==.58
    panel=json.loads((HERE/"panel.json").read_bytes())
    old=json.loads(gzip.decompress((DATA/"compute/measurements.json.gz").read_bytes()))["roots"]
    arms=cfg["arms"]
    permutations=list(itertools.permutations(arms))
    who={(c["terrain"],c["expert_seat"]):c for c in PROTOCOL["games"] if c["split"]=="train"}
    rows=[]
    grouped={}
    for r in old:grouped.setdefault((r["terrain"],r["seat"]),{})[r["t"]]=r
    for (terrain,seat),sched in grouped.items():
        card=who[terrain,seat]
        tr=mod.BeliefTracker(mod._Bel,seat)
        prev=None
        for t,obs,conf,order in free_entry.public_history(card):
            if t in sched:
                s=sched[t]
                conf={k:v for k,v in conf.items() if k!="randomSeed"}
                capture={"cut":[],"menu":[],"batches":[]}
                def budget(key,truth):
                    if arm=="reference":
                        capture[key].append(truth)
                        return truth
                    n=counts[key];counts[key]+=1
                    return capture[key][n]
                mod._budget_cut=lambda v:budget("cut",v)
                mod._budget_menu=lambda v:budget("menu",v)
                def forward(x,sid,sc=None):
                    caller=sys._getframe(1)
                    kind={"_search":"root","look":"joint" if caller.f_locals.get("both") else "own","other":"other"}[caller.f_code.co_name]
                    if arm=="reference":
                        real=net(x,sid,sc)
                        capture["batches"].append((kind,x,sc,real))
                        return real
                    n=counts["batches"];counts["batches"]+=1
                    expected_kind,_,_,reference=capture["batches"][n]
                    assert kind==expected_kind
                    seen.append((x,sc))
                    ow,ob,ov,ew,eb=evaluate(net,x,sid,sc,kind,arm,net.hw,net.hb)
                    if kind in ("root","own"):
                        return (ow,ob,ov) if kind=="own" else (ew,eb,None)
                    if kind=="other":
                        return reference
                    # Same adapters and frozen opponent outputs in *all* timed paths.
                    w=np.empty_like(reference[0]);b=np.empty_like(reference[1]);v=np.empty_like(reference[2])
                    w[::2],b[::2],v[::2]=ow,ob,ov
                    w[1::2],b[1::2],v[1::2]=reference[0][1::2],reference[1][1::2],0.
                    return w,b,v
                mod._timed_forward=forward
                arm="reference"
                refer=root_call(mod,tr,prev,obs,conf,s["policy_rng"],s["starting_costs"])
                measurements={v:[] for v in arms}
                for round_no in range(3):
                    for arm in permutations[(len(rows)+round_no*2)%6]:
                        counts={"menu":0,"cut":0,"batches":0}
                        seen=[]
                        got=root_call(mod,tr,prev,obs,conf,s["policy_rng"],s["starting_costs"])
                        assert all(counts[k]==len(capture[k]) for k in counts)
                        for (x1,sc1),(_,x2,sc2,_) in zip(seen,capture["batches"],strict=True):
                            assert np.array_equal(x1,x2) and np.array_equal(sc1,sc2)
                        assert all(refer[k]==got[k] for k in ("acts","menu","issued","depth","rollouts","sorted_depths","best") if k in refer)
                        measurements[arm].append(dict(wall_s=got["wall_s"],cpu_s=got["cpu_s"]))
                summary={v:median(m["wall_s"] for m in ts) for v,ts in measurements.items()}
                rx=hashlib.sha256()
                for kind,x,sc,_ in capture["batches"]:
                    rx.update(kind.encode());rx.update(x.tobytes())
                    if sc is not None:rx.update(sc.tobytes())
                rows.append(dict(seed=card["seed"],seat=seat,t=t,menu_limit=capture["menu"][0],cuts=capture["cut"],
                                 menu=refer["candidates"],depth=refer["depth"],original_live_s=refer["wall_s"],frozen_input_sha256=rx.hexdigest(),
                                 cases=[(k,len(x)) for k,x,_,_ in capture["batches"]],identical_inputs_and_final_orders=True,
                                 measurements=measurements,median_times=summary))
                print(t,summary,flush=True)
            tr.step(obs["players"],t,obs["carbon"])
            tr.set_actions(obs["players"],order);prev=obs["players"]
    assert len(rows)==len(panel["public_roots"])==12
    tot={arm:sum(r["median_times"][arm] for r in rows) for arm in arms}
    report=dict(source=cfg["source_sha256"],source_instrumentation=sha(insert_hooks(raw)),script=sha(Path(__file__).read_bytes()),forward=sha((HERE/"forward.py").read_bytes()),
                torch=torch.__version__,threads=torch.get_num_threads(),affinity=list(os.sched_getaffinity(0)),aggregate_wall_s=tot,roots=rows)
    with (HERE/"full_request.json").open("x") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print(tot,flush=True)


if __name__=="__main__":
    main()

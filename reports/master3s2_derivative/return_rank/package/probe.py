"""Frozen local public-entry A/B probe on archived, already-exposed maps."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["CARBON_DEBUG"] = "1"
import numpy as np
from build import build, PARENT

GAMES = (("798a279b8c4cc1de4d5ab84de34d61ce",0),
         ("5cef1a63ea3166ee1c05611fca1cb910",1),
         ("d43d676041c83c5d9380f9daf394d2ce",0))
COST = [.001,.002,.001,.002]


def load(code):
    module = types.ModuleType("archived_ds_pro_dev")
    start = time.perf_counter()
    exec(compile(code, "ds_pro.py", "exec"), module.__dict__)
    return module,time.perf_counter()-start


def reset(m, tracker, previous, seed):
    m._ST = dict(tr=copy.deepcopy(tracker), prev=previous, rng=np.random.RandomState(seed))
    m._S_COST[:] = COST
    m._S_REC = []
    m._S_STATS.pop("last",None)


def public_trial(m, tag, tracker, previous, seed, obs, cfg):
    reset(m, tracker, previous, seed)
    trace = {}
    if tag != "P":
        m._RETURN_ENABLED = (tag == "E")
        m._RETURN_TRACE = trace
    start = time.perf_counter()
    issued = json.loads(json.dumps(m.agent(obs,cfg)))
    elapsed = time.perf_counter()-start
    entry = m._S_STATS.get("last")
    chosen = None
    if m._S_REC:
        _, _, menu, depths, values, choice = m._S_REC[0]
        chosen = dict(menu=menu, depths=depths, leaf=values, choice=choice)
        assert issued == menu[choice]
    if tag != "P":
        m._RETURN_TRACE = None
    return dict(tag=tag, issued=issued, wall_s=elapsed, trace=trace,
                effort=list(entry[:3]) if entry else None, search=chosen)


def identify(m, tracker, previous, seed, obs, cfg):
    """Capture original actor output from the public entry, before any outcome or root search."""
    reset(m, tracker, previous, seed)
    original = m._search
    captured = {}
    def capture(obs0, configuration, est, act0, lw0, lb0, prev, t0, home=()):
        captured.update(native=act0, home=sorted(home))
        return act0
    m._search = capture
    try:
        m.agent(obs, cfg)
    finally:
        m._search = original
    return captured


def check_groups(arms):
    parent = next(v for v in arms if v["tag"]=="P")
    off = next(v for v in arms if v["tag"]=="D")
    es = [v for v in arms if v["tag"]=="E"]
    def stable(a, b):
        return a["search"] is not None and a["search"]==b["search"] and a["effort"]==b["effort"] and a["issued"]==b["issued"]
    if not stable(parent,off) or not stable(*es):
        return {"matched":False,"joint_changed":None,"selected_new_offer":False,"physical_gain_claim":False}
    e = es[0]
    strict = e["effort"]==parent["effort"] and e["search"]["depths"]==parent["search"]["depths"]
    selected = (e["trace"].get("existing") is False and
                e["trace"].get("offer_index")==e["search"]["choice"])
    return dict(matched=True,matched_effort=e["effort"]==parent["effort"],matched_index_depths=strict,
                joint_changed=e["issued"]!=parent["issued"], selected_new_offer=selected,
                same_pivot_change=selected and e["issued"].get(e["trace"]["pivot"])!=parent["issued"].get(e["trace"]["pivot"]),
                # Pruning may assign different per-candidate depths after a genuinely different
                # candidate is offered. Use the registered whole-search K, depth and rolls.
                attributable_new_offer=bool(selected and e["issued"]!=parent["issued"] and e["effort"]==parent["effort"]),
                physical_gain_claim=False)


def run(core):
    if core not in os.sched_getaffinity(0):raise ValueError("Requested local CPU unavailable")
    os.sched_setaffinity(0,{core})
    code,manifest=build()
    assert code == (HERE/"candidate.py").read_bytes()
    parent, parent_cold=load(PARENT.read_bytes())
    candidate, candidate_cold=load(code)
    native_model, _=load((HERE.parent/"model.py").read_bytes())
    import torch
    torch.set_num_threads(1)
    groups=[]
    for game_index,(gid,seat) in enumerate(GAMES):
        path=D/"ladder/replays"/(gid+"-1.json.gz")
        if not path.exists():raise ValueError(f"Missing reusable replay {gid}")
        raw=path.read_bytes()
        tape=json.loads(gzip.decompress(raw));cfg={k:v for k,v in tape["configuration"].items() if k!="randomSeed"}
        frames=tape["steps"];tracker=parent.BeliefTracker(parent._Bel,seat);previous=None;census=[];probes=[]
        for t,frame in enumerate(frames[:-1]):
            flat=frame[0]["observation"];side=frame[seat]["observation"];ps=flat["players"]
            own_orders=frames[t+1][seat]["action"] or {}
            obs=dict(player=seat,step=t,players=ps,trees=flat["trees"],carbon=side["carbon"],
                     remainingOverageTime=side["remainingOverageTime"])
            named=game_index==2 and t==174
            requested= (game_index<2 and 20<=t<=287 and (t-20)%3==0) or named
            if requested:
                seed=331999+7*t+seat
                pre=identify(parent,tracker,previous,seed,obs,cfg)
                candidate._RETURN_TRACE={}
                ts=time.perf_counter()
                alt=candidate._return_offer(ps,seat,t,pre["native"],pre["home"],obs["carbon"])
                ms=(time.perf_counter()-ts)*1000
                details=candidate._RETURN_TRACE
                candidate._RETURN_TRACE=None
                rc=dict(t=t,eligible=alt is not None,ms=ms,pivot=details.get("pivot"),decision=details)
                if named:
                    uid="player-0-worker-15"
                    rr=candidate._RETURN_FEATURES(ps,seat,uid,t,carbon=obs["carbon"])
                    rs=candidate._RETURN_SCORE(rr,candidate._RETURN_WEIGHTS)
                    rc["named_collector"]=dict(uid=uid,position=ps[seat][2][uid],act0=pre["native"].get(uid,"NONE"),
                        homefixed=uid in pre["home"],chosen_pivot=details.get("pivot")==uid,scores=rs.tolist())
                if alt is not None:
                    uid=details["pivot"]
                    orig_rows=native_model.features(ps,seat,uid,t,carbon=obs["carbon"])
                    orig_scores=native_model.score(orig_rows,candidate._RETURN_WEIGHTS)
                    assert np.array_equal(orig_rows,candidate._RETURN_FEATURES(ps,seat,uid,t,carbon=obs["carbon"]))
                    assert np.array_equal(orig_scores,details["scores"])
                census.append(rc)
                if (alt is not None and len(probes)<6) or named:
                    order=("P","E","E","D") if len(probes)%2==0 else ("D","E","E","P")
                    arms=[public_trial(parent if tag=="P" else candidate,tag,tracker,previous,seed,obs,cfg) for tag in order]
                    probes.append(dict(t=t,seed=seed,native=pre["native"],archive=own_orders,
                                       menu_proposal=alt,arms=arms,conclusion=check_groups(arms)))
            tracker.step(ps,t,obs["carbon"])
            tracker.set_actions(ps,own_orders)
            previous=ps
        groups.append(dict(gid=gid,seat=seat,replay_sha256=hashlib.sha256(raw).hexdigest(),census=census,checks=probes))
        print(gid,"eligible",sum(r["eligible"] for r in census),"checked",len(probes),
              "changed",[(r["t"],r["conclusion"]) for r in probes],flush=True)
    return dict(manifest=manifest,platform=dict(affinity=core,python=sys.version,numpy=np.__version__,
                   cold_parent_s=parent_cold,cold_candidate_s=candidate_cold),groups=groups)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu",type=int,default=4)
    args=parser.parse_args()
    result=run(args.cpu)
    (HERE/"probe.json.gz").write_bytes(gzip.compress(json.dumps(result,sort_keys=True,allow_nan=False,separators=(",",":")).encode(),mtime=0))

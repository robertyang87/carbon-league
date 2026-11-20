"""Measure completed same-depth decisions from unmodified public v2, without reading the opponent's current action."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
PARENT=D/"ladder/prediction_feedback/package/candidate.py"
EXPECTED="8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43"

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def save(p,row):
    raw=(json.dumps(row,ensure_ascii=False,separators=(",",":"),sort_keys=True,allow_nan=False)+"\n").encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def measured(m,tracker,prev,obs,cfg,seed):
    m._ST=dict(tr=copy.deepcopy(tracker),prev=copy.deepcopy(prev),rng=np.random.RandomState(seed))
    m._S_COST[:]=[.001,.002,.001,.002];m._S_REC=None
    captured={}
    def hook(code,offset,result):captured.update(sys._getframe(1).f_locals)
    mon=sys.monitoring;tool=4;mon.use_tool_id(tool,"frozen v2 finalists on native return")
    try:
        mon.register_callback(tool,mon.events.PY_RETURN,hook);mon.set_local_events(tool,m._search.__code__,mon.events.PY_RETURN)
        w=time.perf_counter();p=time.process_time();act=m.agent(obs,cfg);wall=time.perf_counter()-w;cpu=time.process_time()-p
    finally:
        mon.set_local_events(tool,m._search.__code__,0);mon.register_callback(tool,mon.events.PY_RETURN,None);mon.free_tool_id(tool)
    f=captured
    rec=dict(wall=wall,cpu=cpu,action=act,depth=f.get("depth",0),rolls=f.get("rolls",0),cut=f.get("cut",True),menu=f.get("cands",[act]))
    if rec["depth"]!=8 or rec["cut"] or rec["rolls"]!=1:
        return dict(rec,eligible=False,reason="native_search_incomplete")
    if len(f["alive"])<2 or not all(f["dep"].get(i)==8 and (i,0) in f["val"] for i in f["alive"]):
        return dict(rec,eligible=False,reason="no_two_survivors")
    sort=sorted(f["alive"],key=f["mean"].__getitem__,reverse=True);a,b=sort[:2]
    assert f["cands"][f["best"]]==act
    if f["best"]!=a:
        return dict(rec,eligible=False,reason="incumbent_protected_by_native_epsilon")
    boards={v[0]:v for v in f["run"] if v[1]==0}
    arms=[]
    for c in (a,b):
        st=boards[c][2];assert st["step"]==obs["step"]+8
        cards=st["players"];side=obs["player"];cargo=[sum(w[1] for w in bank[2].values() if w[2]=="COLLECTOR") for bank in cards]
        arms.append(dict(index=c,command=f["cands"][c],value=float(f["mean"][c]),leaf_step=st["step"],leaf_cash=[p[0] for p in cards],
                         leaf_cargo=cargo,roster=cards))
    assert arms[0]["command"]!=arms[1]["command"]
    return dict(rec,eligible=True,reason="comparable",arms=arms)


def public_tape(c):
    raw=(D/c["rel"]).read_bytes();assert sha(raw)==c["archive_sha256"]
    with zipfile.ZipFile(D/c["rel"]) as z:
        arm=c["arm"];declaration=json.loads(z.read(arm+"/registration.json"));assert declaration["files"]["candidate"]["sha256"].startswith("7cf5c6c9")
        moves=z.read(arm+"/turns.jsonl");obs=gzip.decompress(z.read(arm+"/observations.jsonl.gz"));rc=json.loads(z.read(arm+"/receipt.json"))
        assert sha(moves)==rc["turns_sha256"]
        result=[]
        for rr,oo in zip(moves.splitlines(),obs.splitlines(),strict=True):
            mv,ov=json.loads(rr),json.loads(oo);side=c["seat"];x=ov["observations"][side]
            assert "full_carbon" not in x and sha((json.dumps(x,separators=(",",":"))+"\n").encode())==mv["public_observation_sha256"][side]
            result.append((x,mv["calls"][side]["response"]["action"]))
    return result


def generate():
    assert sha(PARENT.read_bytes())==EXPECTED
    m=load("cash_frozen_native",PARENT);m._torch.set_num_threads(1)
    cards=read(D/"opponent_rollout_calibration/protocol.json")["tickets"]
    schema=read(D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json")["configuration"]
    cfg={k:v["default"] for k,v in schema.items() if isinstance(v,dict) and "default"in v};cfg["episodeSteps"]=300
    rows=[]
    for c in cards:
        tape=public_tape(c);seat=c["seat"];tracker=m.BeliefTracker(m._Bel,seat);prev=None
        for t,(obs,order) in enumerate(tape):
            if t in (279,285):
                seed=(c['seed']+17*seat+19073+7*t)%2**32
                record=measured(m,tracker,prev,obs,cfg,seed)
                rows.append(dict(terrain=c['map'],map_seed=c['seed'],source_sha=c['archive_sha256'],seat=seat,t=t,policy_seed=seed,
                                 players=obs['players'],record=record,source_signed_candidate="7cf5c6c9"))
            tracker.step(obs["players"],t,obs["carbon"]);tracker.set_actions(obs["players"],order);prev=obs["players"]
        print(c['seed'],seat,len(rows),flush=True)
    save(HERE/"native_roots.json.gz",dict(parent_sha256=EXPECTED,cohort="six previously used maps vs complete original; whole trajectories belong to 7cf5; public replayed decisions made by current 8b4c4baf",affinity=sorted(os.sched_getaffinity(0)),rows=rows))
    print(len(rows),sum(r["record"]["eligible"] for r in rows))
if __name__=="__main__":generate()

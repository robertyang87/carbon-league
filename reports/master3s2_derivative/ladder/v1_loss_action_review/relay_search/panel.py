"""One observer per archived game. Current opposing actions are not loaded by the inference routines."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parents[2]
LADDER=D/"ladder"
sys.path.insert(0,str(D.parent/"sapling_1200k/all_top5"))
from context import Game


def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def digest(data):
    if not isinstance(data,bytes):
        data=json.dumps(data,separators=(",",":"),sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def save(p,data):
    raw=(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw)


def source_list():
    cards=json.loads((LADDER/'placement_refresh/summary.json').read_bytes())
    for card in cards["all_settled_games"]:
        if card["kind"]!="newrank-placement":continue
        gid=card["match"];name=gid+"-1.json.gz";path=LADDER/"replays"/name
        if not path.exists():path=LADDER/"placement_refresh/replays"/name
        data=json.loads(gzip.decompress(path.read_bytes()));g=Game(data);s=card["seat"]
        assert len(data["steps"])==300 and data['rewards'][s]==card['score'][0]
        views=[g._obs(t,s) for t in range(299)];commands=[f[s]["action"] or {} for f in data["steps"][1:]]
        yield dict(id=gid,stage="ladder",seat=s,path_sha=digest(path.read_bytes())),views,commands,data["configuration"]
    cohort=json.loads((D/"opponent_rollout_calibration/protocol.json").read_bytes())
    path=D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json"
    schema=json.loads(path.read_bytes())['configuration'];cfg={k:v['default'] for k,v in schema.items() if isinstance(v,dict) and 'default'in v}
    for card in cohort["tickets"]:
        p=D/card["rel"];s=card["seat"];assert digest(p.read_bytes())==card["archive_sha256"]
        with zipfile.ZipFile(p) as z:
            reg=json.loads(z.read(card['arm']+'/registration.json'));rc=json.loads(z.read(card['arm']+'/receipt.json'))
            calls=z.read(card["arm"]+"/turns.jsonl");view=gzip.decompress(z.read(card["arm"]+"/observations.jsonl.gz"))
            assert rc["turns_sha256"]==digest(calls) and reg["files"]["candidate"]["sha256"].startswith("7cf5c6c9")
            views=[json.loads(line)["observations"][s] for line in view.splitlines()]
            commands=[json.loads(line)["calls"][s]["response"]["action"] for line in calls.splitlines()]
            assert len(views)==len(commands)==299
        yield dict(id=str(card["seed"]),stage="archived_original",seat=s,path_sha=digest(p.read_bytes())),views,commands,cfg


def call(mod,enabled,history,prev,seed,obs,cfg):
    mod._RELAY_ON=enabled
    mod._RELAY_LAST={}
    mod._ST=dict(tr=copy.deepcopy(history),prev=prev,rng=np.random.RandomState(seed))
    mod._S_COST[:]=[.001,.002,.001,.002];mod._S_STATS.pop("last",None);mod._S_REC=[]
    start=time.perf_counter();cpu=time.process_time();result=mod.agent(obs,cfg)
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    trace=copy.deepcopy(mod._RELAY_LAST)
    rec=list(mod._S_REC);mod._S_REC=None
    menu,depths,final_idx=(rec[0][2],rec[0][3],rec[0][5]) if rec else (None,None,None)
    return dict(action=result,wall=wall,cpu=cpu,trace=trace,original_menu_sha=digest(menu),original_depths=depths,
                original_chosen=final_idx,original_size=len(menu) if menu else None)


def main():
    protocol=json.loads((HERE/'protocol.json').read_bytes());w=json.loads((HERE/'manifest.json').read_bytes())
    assert w["candidate"]==digest((HERE/"candidate.py").read_bytes())==protocol["candidate"]
    m=load('frozen_relay',HERE/'candidate.py');m._torch.set_num_threads(1)
    allrows=[];games=[]
    for ticket,views,commands,cfg in source_list():
        s=ticket['seat']; potential=[]; facts=[]
        for t in range(294,298):
            obs=views[t];assert obs["player"]==s and "full_carbon" not in obs
            if m._relay_choice(obs["players"],s,t):
                potential.append(t)
            fact=m._relay_choice(obs["players"],s,t,commands[t])
            if fact:
                assert t in potential
                facts.append(dict(step=t,archived_parent_order=commands[t],candidate=fact))
        games.append(dict(**ticket,screen=potential,archive_own_order_eligible=facts))
        if not potential:continue
        tracker=m.BeliefTracker(m._Bel,s);prev=None
        for t,obs in enumerate(views):
            if t in potential:
                seed=(41357+int(digest(ticket["id"])[:8],16)+17*t+s)%(2**32)
                def pair(repeat):
                    order=(False,True) if (len(allrows)+repeat)%2==0 else (True,False)
                    z={str(enable):call(m,enable,tracker,prev,seed,obs,cfg) for enable in order}
                    a,b=z['False'],z['True'];same=all(a[k]==b[k] for k in ('original_menu_sha','original_depths','original_chosen'))
                    z['same_parent_work']=same
                    z['attributable']=bool(same and b['trace'].get('emitted') and a['action']==b['trace']['parent'] and a['action']!=b['action'])
                    return z
                rs=[pair(0)]
                if rs[0]['True']['trace'].get('emitted'):rs.append(pair(1))
                allrows.append(dict(**ticket,t=t,public_sha=digest(obs),rng=seed,pairs=rs))
            tracker.step(obs["players"],t,obs["carbon"]);tracker.set_actions(obs["players"],commands[t]);prev=obs["players"]
        print(ticket["stage"],ticket['id'],s,potential,'root count',len(allrows),flush=True)
    save(HERE/'gates.json.gz',games);save(HERE/'public_entries.json.gz',allrows)
    summary={}
    for stage in ('ladder','archived_original'):
        xs=[x for x in allrows if x['stage']==stage]; gs=[g for g in games if g['stage']==stage]
        qualified=[x for x in xs if x['pairs'][0]['True']['trace']]
        hits=[x for x in qualified if x['pairs'][0]['True']['trace'].get('emitted')]
        stable=[x for x in hits if len(x['pairs'])==2 and all(p['attributable'] for p in x['pairs']) and x['pairs'][0]['True']['action']==x['pairs'][1]['True']['action']]
        stats=lambda xs:{k:float(v) for k,v in zip(('median','p95','max'),np.percentile(xs,[50,95,100]))} if xs else {}
        summary[stage]=dict(games=len(gs),eligible_window=4*len(gs),physical_maps=len({g['id'] for g in gs}),geom=len(xs),
             archive_selected=sum(len(g['archive_own_order_eligible']) for g in gs),actual_parent_anchors=len(qualified),
             reasons=dict(Counter(x['pairs'][0]['True']['trace'].get('condition') for x in qualified)),
             first_emissions=len(hits),stable=len(stable),stable_roots=[[x['id'],x['seat'],x['t']] for x in stable],
             wall_original=stats([x['pairs'][0]['False']['wall'] for x in xs]),wall_new=stats([x['pairs'][0]['True']['wall'] for x in xs]),
             extra_wall=stats([x['pairs'][0]['True']['trace'].get('elapsed',0) for x in qualified]))
    save(HERE/'results.json',dict(candidate=w['candidate'],panels=summary,affinity=sorted(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()))
    print(summary,flush=True)


if __name__=="__main__":
    main()

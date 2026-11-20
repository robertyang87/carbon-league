"""Exploratory replay of six already labelled action pairs; no new suffixes or fitting."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
DER = TASK.parent
CLOSE = DER / 'return_response_learning/close_games'
ECO = DER / 'v3_release/development/return_turns'
sys.path.insert(0, str(CLOSE))
import experiment as e

PAIR = {'v2': (e.V2, '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'),
        'child': (TASK/'v2_bc.py', 'cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a')}
read = lambda f: json.loads(f.read_bytes())
sha = e.sha
def cases():
    menus = CLOSE / 'menus.json'
    assert sha(menus.read_bytes()) == '3b737e03d58639eeca74303c8798b517172d69028ea9db3ac961669ec58b355a'
    for ix, entry in enumerate(read(menus)['rows']):
        r = entry['ticket']
        p = CLOSE/f'cases/{ix}.json'
        case = read(p)
        assert case['menu_sha256'] == sha(menus.read_bytes()) and case['root']==[r['seed'],r['seat'],r['step']]
        yield r, e.rng(r), entry['orders'], [case['arms'][k]['margin'] for k in ('A','B')], p, 'v2'
    settled = read(ECO.parent/'hosted/settlement.json')
    for seat, t in ((0,290),(1,271)):
        seed = 621758607
        p = ECO/f'{seed}_{seat}_{t}_result.json'
        case = read(p)
        f = case['focus']
        archive = next(a for a in settled['archives'] if (a['seed'],a['seat'])==(seed,seat))
        game = next(a for a in settled['games'] if (a['seed'],a['seat'],a['arm'])==(seed,seat,'challenger'))
        r = dict(seed=seed,seat=seat,step=t,arm='challenger',terrain=game['terrain'],sha256=archive['sha256'],
                 path=f"v3_release/development/hosted/{archive['id']}.zip",public_sha=f['public_digest'][seat])
        a = f['own_archived']
        b = a.copy()
        if f['new_move'] == 'NONE': b.pop(f['blocker'], None)
        else: b[f['blocker']] = f['new_move']
        key = int(sha(f'{seed}/{seat}/{t}/return-turn/v1'.encode())[:8],16)
        yield r, key, [a,b], [case['arms'][k]['margin'] for k in ('parent','block')], p, 'v3'

def priors(m,role,ob,orders):
    p,seat,t = ob['players'],ob['player'],ob['step']
    tr = copy.deepcopy(m._ST['tr'])
    est,known = tr.step(p,t,ob['carbon'])
    x = np.clip(np.rint(m.encode(p,t,seat,est,known,m._ST['prev'])*100.0),0,255)*.01
    fw = m._POLICY.forward
    w,b = fw(x,m._SID,own_root=True) if role=='child' else fw(x,m._SID)
    rows = {}
    for uid,pos in p[seat][1].items():
        rows[uid]=dict(index=pos,role='BASE',probs=dict(zip(m.BACT,map(float,m._probs(b[pos])))))
    homes=set(p[seat][1].values())
    for uid,(pos,load,kind) in p[seat][2].items():
        probs=m._probs(w[pos])
        home=m.go_home(pos,load,probs,x,homes,list(p[1-seat][1].values()),t,m._HOME) if kind=='COLLECTOR' else None
        rows[uid]=dict(index=pos,cargo=load,role=kind,probs=dict(zip(m.WACT,map(float,probs))),
                       home=None if home is None else m.WACT[home], orders=[cmd.get(uid,'NONE') for cmd in orders])
    return rows

def member_report(menu,order,goals,units):
    return dict(menu_matches=[cmd in menu for cmd in goals], issued_matches=[cmd==order for cmd in goals],
                units={u:dict(menu_matches=[any(a.get(u,'NONE')==cmd.get(u,'NONE') for a in menu) for cmd in goals],
                               issued=order.get(u,'NONE')) for u in units})

def main():
    assert len(os.sched_getaffinity(0))==1
    os.environ['CARBON_DEBUG']='1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models={}
    for name,(path,expected) in PAIR.items():
        assert sha(path.read_bytes())==expected
        m=e.module('response_probe_'+name,path)
        m._net()
        assert m._SEARCH_OK and m._SEED is None and m._TEMP==1 and m._S_ABORT==.58
        models[name]=m
    e.N.fresh_stats=copy.deepcopy(models['v2']._S_STATS)
    training = read(TASK/'protocol.json')
    rows=[]
    for r, registered, goals, margins, outcome, continuation in cases():
        s,t = r['seat'],r['step']
        state=e.setup(r,False)
        cfg=state['cfg']; ob=state['public'][s]
        assert t == ob['step'] and r['public_sha']==e.F.digest(ob)
        if continuation=='v3':
            _,turns,_=e.sourcefiles(r)
            assert json.loads(turns[t])['calls'][s]['response']['action']==goals[0]
        e.N.history_state(models['v2'],state,s,registered)
        snapshot=e.N.snap(models['v2'])
        parent=models['v2']
        assert parent._ST['prev']==state['prefix'][-1][0][s]['players']
        units=sorted(k for k in set(goals[0])|set(goals[1]) if goals[0].get(k,'NONE')!=goals[1].get(k,'NONE'))
        row=dict(seed=r['seed'],seat=s,step=t,terrain=r['terrain'],continuation=continuation,
                 dataset=next((v['split'] for v in training['tickets'] if v['map']==r['terrain']),'external'),
                 step_fit_schedule=t in training['steps'],input_sha256=r['public_sha'],archive_sha256=r['sha256'],
                 paired_label_source=str(outcome.relative_to(DER)),label_sha256=sha(outcome.read_bytes()),
                 full_orders=goals,margins=margins,best=int(margins[1]>margins[0]),changed_units=units,priors={}, draws=[])
        for name,m in models.items():
            e.N.restore(m,snapshot,cfg)
            row['priors'][name]=priors(m,name,ob,goals)
        entropy=[registered]+[int(sha(f"joint-probe/{r['seed']}/{s}/{t}/{i}".encode())[:8],16) for i in range(1,8)]
        for i,key in enumerate(entropy+[registered]):
            draws={}
            for mode in (('v2','child') if (i+len(rows))%2==0 else ('child','v2')):
                m=models[mode];e.N.restore(m,snapshot,cfg);m._ST['rng']=np.random.RandomState(key)
                cmd,clock,tr=e.sample(m,ob,cfg)
                draws[mode]=dict(cmd=cmd,menu=tr['menu'],depth=tr['depths'],clock=clock,
                                 **member_report(tr['menu'],cmd,goals,units))
            row['draws'].append(dict(key=key,repeat_control=i==8,arms=draws))
        rows.append(row)
        print(r['seed'],s,t,len(rows),flush=True)
    raw=json.dumps(dict(packages={n:a for n,(_,a) in PAIR.items()},rows=rows),allow_nan=False,separators=(',',':')).encode()
    (HERE/'observations.json.gz').write_bytes(gzip.compress(raw,mtime=0))

if __name__=='__main__': main()

"""Register close public positions, seal native H8 candidate pairs, then fork only after model forecasts."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
ROOT=D.parents[1]
LOCK='f8b83f5bd6f73c9116ffadd7921d3a14d70632c2363700f9ca49f60a91dc9793'
REG=json.loads((HERE/'protocol.json').read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha((HERE/'protocol.json').read_bytes())==LOCK
def dump(path,value):
    path=HERE/path;path.parent.mkdir(exist_ok=True,parents=True)
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError(f'inconsistent prior output {path}')
    else:
        tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path)
    return sha(raw)
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
R=module('close_recovery',D/'endgame_cash_projection/response_check/recovery.py')
F=R.freight
N=module('close_call_tools',D/'endgame_cash_projection/response_check/run.py')
dist=F.econ.distances(15)
V2=D/'ladder/prediction_feedback/package/candidate.py'
ORIG=ROOT/'data/master3s2_source.py'
MODEL='7a1270e3905e396d24acf79bb6b33f1308628e7892b809f6013e6b37c55fdecf'
def sourcefiles(row):
    arc=D/row['path'];arm=row['arm']; assert sha(arc.read_bytes())==row['sha256']
    with zipfile.ZipFile(arc) as z:
        reg=z.read(arm+'/registration.json');rc=z.read(arm+'/receipt.json');turns=z.read(arm+'/turns.jsonl')
        frames=z.read(arm+'/observations.jsonl.gz')
    rego=json.loads(reg);receipt=json.loads(rc)
    assert sha(reg)==receipt['protocol_sha256'] and sha(turns)==receipt['turns_sha256'] and rego['physical_map_sha256']==row['terrain']
    return reg,turns.splitlines(),gzip.decompress(frames).splitlines()
def moves(p):
    y,x=divmod(p,15);return {'NONE':p,'UP':15*((y-1)%15)+x,'DOWN':15*((y+1)%15)+x,'LEFT':15*y+(x-1)%15,'RIGHT':15*y+(x+1)%15}
def prescore():
    rows=[]
    for src in REG['sources']:
        reg,_,frames=sourcefiles(src);s=src['seat']
        for t in REG['time']:
            public=json.loads(frames[t])['observations'][s]
            assert public['player']==s and public['step']==t
            both=public['players']
            cash=[z[0] for z in both];cargo=[sum(w[1] for w in z[2].values() if w[2]=='COLLECTOR' and min(dist[w[0]][b] for b in z[1].values())<=299-t) for z in both]
            c=cash[s]-cash[1-s]; potential=c+cargo[s]-cargo[1-s]
            rows.append(dict(**src,step=t,public_sha=F.digest(public),cash=cash,remaining=299-t,cargo=cargo,
                  cash_margin=c,potential=potential,admitted=abs(c)<=1500 and abs(potential)<=900,
                  rank=[max(abs(c),abs(potential)),(abs(c)+abs(potential))/2,src['seed'],t]))
    return rows
def setup(r,env_required):
    reg,turns,frames=sourcefiles(r);seat=r['seat'];t=r['step'];seed=r['seed'];meta=json.loads(reg)
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td);fn=tmp/'reg.json';fn.write_bytes(reg);R.live.configure(fn,tmp/'nc',tmp/'lock');env,cfg=R.live.official()
    sim=F.econ.FastCarbon(cfg).reset(seed=seed);sim.cash=F.econ.CashEvents(sim.cash)
    assert sha(bytes(map(int,sim.carbon)))==r['terrain']
    prefix=[];bank=[meta['platform']['overage_seconds_per_player']]*2;bank=list(map(float,bank))
    for i in range(t+1):
        frame=json.loads(frames[i]);public=frame['observations']
        assert all(F.digest(v)==h for v,h in zip(public,frame['public_observation_sha256']))
        if env_required:
            assert public==[F.public(env,j,bank) for j in (0,1)] and list(sim.cash)==[z[0] for z in public[0]['players']]
        if i==t:break
        line=json.loads(turns[i]);assert line['public_observation_sha256']==frame['public_observation_sha256'] and all(v['status']=='ok' for v in line['calls'])
        actions=[v['response']['action'] for v in line['calls']]
        prefix.append((public,actions));bank=line['overage_remaining_seconds']
        if env_required:sim.cash.events.clear();sim.step(*actions);env.step(actions,[])
    assert F.digest(public[seat])==r['public_sha']
    return dict(prefix=prefix,cfg=cfg,public=public,env=env,sim=sim,bank=bank)
def agents():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ['CARBON_DEBUG']='1'
    assert sha(V2.read_bytes())=='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
    assert sha(ORIG.read_bytes())=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
    subj=module('close_original_v2',V2);opp=module('close_native_original',ORIG)
    N.fresh_stats=copy.deepcopy(subj._S_STATS)
    for m in (subj,opp):assert m._SEARCH_OK and m._TEMP==1 and m._SEED is None and m._S_ABORT==.58
    return subj,opp
def sample(mod,obs,cfg):
    trace={}
    def cap(code,offset,ret):
        f=sys._getframe(1).f_locals
        trace.update(menu=f.get('cands',[ret]),means={int(k):float(v) for k,v in f.get('mean',{}).items()},
                     depths={int(k):int(v) for k,v in f.get('dep',{}).items()},cut=f.get('cut'),alive=f.get('alive',[]))
    mon=sys.monitoring;tid=4;co=mod._search.__code__;mon.use_tool_id(tid,'one completed native public H8 call')
    try:
        mon.register_callback(tid,mon.events.PY_RETURN,cap);mon.set_local_events(tid,co,mon.events.PY_RETURN);cmd,clock=F.measured(mod,obs,cfg)
    finally:
        mon.set_local_events(tid,co,0);mon.register_callback(tid,mon.events.PY_RETURN,None);mon.free_tool_id(tid)
    return cmd,clock,trace
def meaning(players,seat,a,b):
    own=players[seat];homes=own[1].values();workers=own[2]
    dest=lambda c:{uid:moves(v[0])[c.get(uid,'NONE')] for uid,v in workers.items()}
    first,next_=dest(a),dest(b);reasons=[]
    for uid,v in workers.items():
        if v[2]!='COLLECTOR' or v[1]<=0 or first[uid]==next_[uid]:continue
        before=min(dist[first[uid]][h] for h in homes);after=min(dist[next_[uid]][h] for h in homes)
        same_a=[j for j in workers if j!=uid and first[j]==first[uid]]
        same_b=[j for j in workers if j!=uid and next_[j]==next_[uid]]
        if before!=after or same_a!=same_b:
            reasons.append(dict(uid=uid,cargo=v[1],position=v[0],first=first[uid],alternate=next_[uid],home_distance=[int(before),int(after)],conflicts=[same_a,same_b]))
    return reasons
def privfile(ticket):
    return Path(tempfile.gettempdir())/f"carbon-close-pair-{LOCK[:8]}-{ticket['seed']}-{ticket['seat']}-{ticket['step']}.pkl"
def rng(r):
    return int.from_bytes(hashlib.sha256(f"{r['seed']}/{r['seat']}/{r['step']}/close-returns-v1".encode()).digest()[:4],'big')
def menu(r,subj):
    state=setup(r,False);s=r['seat'];ob=state['public'][s]
    N.history_state(subj,state,s,rng(r))
    prefix_hash=F.original_state.h(F.original_state.dynamic(subj))
    entropy_hash=F.original_state.h(subj._ST['rng'])
    cmd,clock,tr=sample(subj,ob,state['cfg']);mean,dep=tr['means'],tr['depths'];cs=tr['menu']
    chosen=cs.index(cmd)
    ranking=sorted([i for i in tr['alive'] if dep.get(i)==8 and i in mean],key=lambda i:(-mean[i],i))
    doc=dict(ticket=r,public_sha=r['public_sha'],menu_sha=F.digest(cs),native_winner=chosen,menu=cs,
               ranking=ranking,depths=dep,native_values=mean,cost=clock,rolls=clock['rolls'],cut=tr['cut'],
               status='abstention',reason='native winner does not lead the H8 scored survivors')
    if not ranking or chosen!=ranking[0] or clock['H']!=8:return doc
    doc['reason']='no distinct positive-load collector with changed home distance or conflict'
    for j,i in enumerate(ranking[1:],2):
        changed=meaning(ob['players'],s,cmd,cs[i])
        if not changed:continue
        snapshot=copy.deepcopy(F.original_state.dynamic(subj))
        with privfile(r).open('wb') as fp:pickle.dump(dict(pre=prefix_hash,entropy=entropy_hash,post=snapshot,post_h=F.original_state.h(snapshot),cfg_cache=subj._RX_CFG[1]),fp,protocol=5)
        doc.update(status='qualified',reason=None,players=ob['players'],orders=[cmd,cs[i]],indices=[chosen,i],
                   dep=[dep[chosen],dep[i]],native=[mean[chosen],mean[i]],rank_B=j,native_top_two=j==2,
                   tags=['within_registered_7a1270e3_domain' if j==2 else 'extrapolation_non_top_two'],changed=changed,
                   snapshot_identity=F.original_state.h(snapshot));break
    return doc
def intake():
    assert len(os.sched_getaffinity(0))==1
    subj,_=agents();pool=prescore()
    dump('public_pool.json',dict(protocol=LOCK,rows=pool))
    used=set();accepted=[];trials=[]
    for s in (0,1,0,1,0,1):
        while True:
            available=sorted((r for r in pool if r['admitted'] and r['seat']==s and r['seed'] not in used),key=lambda r:r['rank'])
            if not available:break
            seed=available[0]['seed'];used.add(seed);ok=False
            for r in (x for x in available if x['seed']==seed):
                m=menu(r,subj);trials.append(m)
                print(r['seed'],s,r['step'],m['status'],m['reason'],m['cost']['H'],m.get('rank_B'),flush=True)
                if m['status']=='qualified':
                    accepted.append(m);ok=True;break
            if ok:break
    print('SEALED',dump('menus.json',dict(schema='close-returns-native-pairs/v1',protocol_sha256=LOCK,
       corpus_sha256=sha((HERE/'public_pool.json').read_bytes()),trials=trials,rows=accepted)),len(accepted),flush=True)
if __name__=='__main__':
    intake()

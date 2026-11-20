"""Select observable late-game blockade opportunities, then fork a single order with responsive agents."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from turns import DEV,ROOT,HERE,econ,load
DER=DEV.parents[1]
LOCK='87c466ba924c34e8b27fe3d745cfd20c4c6d5d9ebcc45b74c089d4473c3768d6'
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha((HERE/'intervention_protocol.json').read_bytes())==LOCK
def moves(cell):
    y,x=divmod(cell,15)
    return {'NONE':cell,'UP':15*((y-1)%15)+x,'DOWN':15*((y+1)%15)+x,
            'LEFT':15*y+(x-1)%15,'RIGHT':15*y+(x+1)%15}
import zipfile
def dump(name,obj):
    dest=HERE/name
    raw=(json.dumps(obj,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode()
    if dest.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if dest.exists() and dest.read_bytes()!=raw:raise ValueError(name)
    if not dest.exists():
        tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dest)
    return sha(raw)
def readseat(seat):
    settle=json.loads((DEV/'hosted/settlement.json').read_bytes())
    v=next(a for a in settle['archives'] if (a['seed'],a['seat'])==(621758607,seat))
    p=DEV/'hosted'/(str(v['id'])+'.zip')
    assert sha(p.read_bytes())==v['sha256']
    with zipfile.ZipFile(p) as z:
        reg=z.read('challenger/registration.json');rc=z.read('challenger/receipt.json')
        turns=z.read('challenger/turns.jsonl');frames=z.read('challenger/observations.jsonl.gz')
        assert sha(reg)==json.loads(rc)['protocol_sha256'] and sha(turns)==json.loads(rc)['turns_sha256']
        assert any(a['arm']=='challenger' and a['receipt_sha256']==sha(rc) for a in map(json.loads,z.read('execution.jsonl').splitlines()))
    return reg,list(map(json.loads,turns.splitlines())),list(map(json.loads,gzip.decompress(frames).splitlines()))
def select():
    dist=econ.distances(15);selection={}
    for seat in (0,1):
        reg,turns,frames=readseat(seat);eligible=[];near=0
        for t in range(270,295):
            pub=frames[t]['observations'][seat];prev=frames[t-1]['observations'][seat]['players'][1-seat][2]
            own,opp=pub['players'][seat],pub['players'][1-seat]
            issued=turns[t]['calls'][seat]['response']['action']
            for uid,(pos,load_,role) in opp[2].items():
                homes=list(opp[1].values())
                near_home=lambda p:min(dist[p][home] for home in homes)
                if role!='COLLECTOR' or load_<=0 or not 1<=near_home(pos)<=5:continue
                near+=1
                pred_prev=next((a for a,z in moves(prev[uid][0]).items() if z==pos),None) if uid in prev else None
                forecast=min(moves(pos).items(),key=lambda v:(near_home(v[1]),v[0]!=pred_prev,v[1]))
                target=forecast[1]
                if target in homes or target in own[1].values():continue
                for fid,(fpos,fc,fr) in own[2].items():
                    if fr!='COLLECTOR' or fc>100 or fc>=load_:continue
                    mov=next((a for a,x in moves(fpos).items() if x==target),None)
                    if mov is None or mov==issued.get(fid,'NONE'):continue
                    if any(moves(w[0])[issued.get(j,'NONE')]==target for j,w in own[2].items() if j!=fid):continue
                    eligible.append(dict(t=t,seat=seat,seed=621758607,uid=uid,enemy_load=load_,enemy_cell=pos,predict=forecast[0],target=target,closest_enemy_base=near_home(pos),
                          blocker=fid,own_cargo=fc,own_cell=fpos,base_move=issued.get(fid,'NONE'),new_move=mov,own_archived=issued,public_digest=frames[t]['public_observation_sha256']))
        eligible.sort(key=lambda a:(a['own_cargo']-a['enemy_load'],a['t'],a['uid'],a['blocker']))
        selection[seat]=dict(considered_enemy_worker_turns=near,opportunities=eligible,selected=eligible[0] if eligible else None)
    print(dump('locked_roots.json',dict(protocol=LOCK,by_seat=selection)),[(k,None if r['selected'] is None else (r['selected']['t'],r['selected']['enemy_load'])) for k,r in selection.items()])

def run(seat):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0))==1
    os.environ['CARBON_DEBUG']='1'
    locked=(HERE/'locked_roots.json').read_bytes()
    assert sha(locked)=='01bad423881899548a0566019dea8be98e5dccea5c5a535dfd100d0d7819fa45'
    r=json.loads(locked)['by_seat'][str(seat)]['selected']
    if r is None:return
    t,seed=r['t'],r['seed']
    prefix_tools=load('return_restore',DER/'endgame_cash_projection/response_check/recovery.py')
    native_tools=load('return_native',DER/'endgame_cash_projection/response_check/run.py')
    F=prefix_tools.freight
    reg,turns,frames=readseat(seat);meta=json.loads(reg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);p=tmp/'reg.json';p.write_bytes(reg)
        prefix_tools.live.configure(p,tmp/'empty',tmp/'lock')
        env,cfg=prefix_tools.live.official()
    sim=econ.FastCarbon(cfg).reset(seed=seed);sim.cash=econ.CashEvents(sim.cash)
    assert sha(bytes(map(int,sim.carbon)))==meta['physical_map_sha256']
    bank=[float(meta['platform']['overage_seconds_per_player'])]*2
    prefix=[]
    for i in range(t+1):
        frame,line=frames[i],turns[i]
        public=[F.public(env,j,bank) for j in (0,1)]
        assert [F.digest(v) for v in public]==frame['public_observation_sha256']==line['public_observation_sha256']
        assert public==frame['observations'] and list(sim.cash)==[p[0] for p in public[0]['players']]
        if i==t:break
        historical=[c['response']['action'] for c in line['calls']]
        prefix.append((frame['observations'],historical));bank=line['overage_remaining_seconds']
        sim.cash.events.clear();sim.step(*historical);env.step(historical,[])
    assert r['public_digest']==frame['public_observation_sha256']
    anc=line['calls'][seat]['response']['action']
    assert anc==r['own_archived']
    new=dict(anc)
    if r['new_move']=='NONE':new.pop(r['blocker'],None)
    else:new[r['blocker']]=r['new_move']
    assert [k for k in set(anc)|set(new) if anc.get(k)!=new.get(k)]==[r['blocker']]
    mpaths=[ROOT/'data/master3s2_source.py']*2;mpaths[seat]=DER/'v3_release/submission.py'
    modules=[F.load(f'return_src_{j}',p) for j,p in enumerate(mpaths)]
    assert [sha(p.read_bytes()) for p in mpaths]==[meta['files']['candidate' if j==seat else 'opponent']['sha256'] for j in (0,1)]
    assert modules[seat]._V3_AUDIT is None
    native_tools.fresh_stats=copy.deepcopy(modules[seat]._S_STATS)
    history=dict(cfg=cfg,public=public,prefix=prefix)
    rng=int.from_bytes(hashlib.sha256(f'{seed}/{seat}/{t}/return-turn/v1'.encode()).digest()[:4],'big')
    for j,m in enumerate(modules):
        assert m._SEARCH_OK and m._SEED is None and m._S_ABORT==.58 and m._TEMP==1
        native_tools.history_state(m,history,j,rng if seat==j else rng^0x7d4a1f6b)
    expected=dict(archive_header=meta['physical_map_sha256'],own_public_and_other=frame['public_observation_sha256'],
        history_sha256=F.digest([a for _,a in prefix]),banks=bank)
    incoming=F.original_state.h([F.original_state.dynamic(m) for m in modules])
    rng_h=F.original_state.h(modules[seat]._ST['rng'])
    self_reply=F.measured(modules[seat],public[seat],cfg)
    alternatives=dict(parent=anc,block=new,order_source='actual_frozen_V3_order_with_only_one_alternative_collector_destination')
    order=['parent','block'] if seat==0 else ['block','parent']
    raw=copy.deepcopy(sim)
    result=native_tools.continue_pair(dict(seed=seed,seat=seat,step=t),alternatives,order,
        (modules,cfg,env,sim,public,expected),self_reply,incoming,rng_h,dump)
    book={}
    for mode in order:
        path=HERE/f'{seed}_{seat}_{t}_{mode}.json.gz'
        assert sha(path.read_bytes())==result['arms'][mode]['evidence_sha256']
        evidence=json.loads(gzip.decompress(path.read_bytes()))
        sim=copy.deepcopy(raw);history=[];deposits=Counter()
        for row in evidence['rows']:
            sim.cash.events.clear();sim.step(*row['orders'])
            assert row['end_cash']==list(sim.cash)
            focus=[ev for ev in sim.cash.events if ev[0]=='own_delivery' and ev[3] in (r['uid'],r['blocker'])]
            for _,_,v,who,*_ in focus:deposits[who]+=v
            workers=sim.obs(0)['players']
            history.append(dict(step_after=sim.step_count,blocker=workers[seat][2].get(r['blocker']),enemy=workers[1-seat][2].get(r['uid']),
                          focus_deposits=focus,enemy_order=row['orders'][1-seat].get(r['uid'],'NONE')))
        book[mode]=dict(timeline=history,focused_deposits=deposits,root_opponent=evidence['root_shared_original_order'],clock=[dict(
             slow=sum(row['clocks'][j]['wall']>1 for row in evidence['rows']),max=max(row['clocks'][j]['wall'] for row in evidence['rows']),
             depths=dict(Counter(row['clocks'][j]['H'] for row in evidence['rows']))) for j in (0,1)])
    result.update(focus=r,protocol=LOCK,selected_sha256=sha(locked),observed=book,delta=result['arms']['block']['margin']-result['arms']['parent']['margin'])
    print('complete',seat,dump(f'{seed}_{seat}_{t}_result.json',result),result['delta'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','run']);p.add_argument('seat',type=int,nargs='?');a=p.parse_args()
    if a.command=='select':select()
    else:run(a.seat)

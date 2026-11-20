"""Build shared causal expert-prefix features, then join the same worker's official 12-step outcome."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from sources import CORPUS, EXP, HERE, ROOT, need, sha

sys.path.insert(0,str(CORPUS))
import common as c


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod


old=module('sapling_frozen_teacher_prefix',CORPUS/'prepare_data.py')
orig=module('sapling_archived_official',EXP/'replay.py')
SOURCES='f476fc829bc9ccb991d00d3066c1befce37f4bb2225578442b71763d8e2def34'


def projection(a,b):
    """Direction of net shortest toroidal displacement, horizontal on nonzero ties."""
    x=(b%15-a%15+7)%15-7;y=(b//15-a//15+7)%15-7
    if not x and not y:return 0
    if abs(x)>=abs(y):return 2 if x>0 else 4
    return 3 if y>0 else 1


def observed(env,seat):
    shared=env.states[0].observation
    return dict(step=shared.step,player=seat,carbon=env.states[seat].observation.carbon,players=shared.players,
                trees=shared.trees,remainingOverageTime=env.states[seat].observation.get('remainingOverageTime',60))


def materialize(card,m,teacher):
    path=ROOT/card['path'];raw=path.read_bytes();need(sha(raw)==card['sha256'],'Wrong historical tape')
    tape=json.loads(gzip.decompress(raw));seat=card['expert_seat'];n=card['transitions'];actions=tape['actions']
    need(len(actions)==n and tape['candidate_seat']==1-seat and orig.digest(actions)==card['actions_sha256']
         and orig.digest([a[seat] for a in actions])==card['expert_tape_sha256'],'Wrong source commands or physical expert')
    F=sys.modules['cnet2_feat'];f=F.Feat(seat);seed=card['seed'];np.random.seed(seed);random.seed(seed)
    env=orig.official_rules(ROOT)('carbon',configuration={**tape['game_configuration'],'randomSeed':seed})
    need(sha(bytes(env.states[0].observation.full_carbon))==card['terrain'],'Wrong physical terrain')
    sampled=set(card['registered_times']);frames=[];meta=[];all_slots=[];positions=[];cash=[]
    for t in range(n+1):
        # Only the expert's contemporary public inputs enter feature/decoder computations.
        obs=observed(env,seat)
        need(obs['step']==t and obs['player']==seat and 'full_carbon' not in obs,'Incorrect public state')
        workers=obs['players'][seat][2]
        positions.append({uid:v[0] for uid,v in workers.items()})
        cash.append([p[0] for p in obs['players']])
        if t==n:break
        x=f.observe(obs)
        if t in sampled:
            enc=[]
            for net in m._NETS:
                tokens,ctx=net.encode(x,2);enc.append((net,tokens,ctx,net.lin(ctx,'h0')))
            plain=m._return_decode(x,f.aux,enc,False);proposed=m._return_decode(x,f.aux,enc,True)
            sent,rejected=c.policy.guard(m,x,plain,proposed)
            # All after this point is label construction: only current gold-prefix actions reach later same-turn slots.
            r=c.actual_features(f,x,actions[t][seat],m);fixed=np.zeros_like(r['residual']);parent=np.zeros_like(fixed)
            cut=r.pop('cut');teacher(x,f.aux,enc,not rejected,r['y'],fixed,parent,cut)
            r.pop('residual');r.update(fixed=fixed,parent=parent,root_step=np.int16(t),prefix_cut=np.int16(cut))
            frames.append(r);all_slots.append(list(f.own_uid));meta.append(dict(t=t,cut=cut,expert_command_sha256=orig.digest(actions[t][seat]),
                uid=list(f.own_uid),guard_rejected=bool(rejected),parent_order_sha256=orig.digest(f.to_actions(sent))))
        env.step(actions[t],[])
    need(env.done and [s.reward for s in env.states]==tape['official_score_by_seat'] and
         orig.digest(cash)==card['official_cash_path_sha256'] and len(meta)==len(sampled),'Complete recorded game did not reproduce')
    # This independent second pass can see later physical locations solely to produce training targets.
    for r,info,uids in zip(frames,meta,all_slots):
        t,cut=info['t'],info['cut'];direction=np.full(F.OWN,-1,np.int8);alive=direction.copy()
        if t+12<=n:
            for slot,uid in enumerate(uids[:cut]):
                if uid not in positions[t]:continue
                last=positions[t+12].get(uid)
                alive[slot]=int(last is not None)
                if last is not None:direction[slot]=projection(positions[t][uid],last)
        r.update(dst12=direction,survive12=alive,window_complete=np.bool_(t+12<=n))
    data={k:np.stack([f[k] for f in frames]) for k in frames[0]}
    need(np.all((data['dst12']>=0)==(data['survive12']==1)) and np.all(data['survive12'][~data['window_complete']]==-1),
         'Incorrect death, terminal or displacement masking')
    return data,meta


def fingerprint():
    return {str(q.relative_to(ROOT)):c.sha(q) for q in (Path(__file__),HERE/'sources.py',HERE/'sources.json',CORPUS/'common.py',
                    CORPUS/'prepare_data.py',EXP/'replay.py',c.PACK,c.BASE)}


def one(card,m,teacher,version):
    work=HERE/'work';work.mkdir(exist_ok=True);key=card['sha256'][:20]
    path=work/(key+'.npz');recpath=work/(key+'.json')
    if recpath.is_file():
        rec=c.read(recpath);need(rec['version']==version and c.sha(path)==rec['sha256'],'Wrong cached materialization')
        return rec
    data,rows=materialize(card,m,teacher)
    with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**data)
    os.replace(path.with_suffix('.tmp'),path)
    item={k:card[k] for k in ('seed','terrain','split','candidate_seat','expert_seat','transitions')}
    dst,alive=data['dst12'],data['survive12']
    item.update(file=str(path.relative_to(HERE)),sha256=c.sha(path),game=card['sha256'],rows=len(rows),
                counts=dict(roots=len(rows),complete_windows=int(data['window_complete'].sum()),
                            actionable_prefix=int(data['eligible'].sum()),survivor_direction=int((dst>=0).sum()),
                            known_dead=int((alive==0).sum()),known_survival=int((alive>=0).sum()),
                            directions=np.bincount(dst[dst>=0],minlength=5).tolist()))
    rec=dict(**item,root_manifest=rows,version=version);c.save(recpath,rec)
    return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lane',type=int);ap.add_argument('--lanes',type=int,default=1);ap.add_argument('--merge',action='store_true');a=ap.parse_args()
    need(c.sha(HERE/'sources.json')==SOURCES,'Different map/sample registration')
    cards=c.read(HERE/'sources.json')['cards'];v=fingerprint()
    if a.merge:
        shards=[];all_rows=[]
        for card in cards:
            rec=c.read(HERE/'work'/(card['sha256'][:20]+'.json'));need(rec.pop('version')==v and c.sha(HERE/rec['file'])==rec['sha256'],'Different shard')
            rs=rec.pop('root_manifest');need([r['t'] for r in rs]==card['registered_times'],'Changed sampled turn')
            all_rows.extend(dict(game=rec['game'],split=rec['split'],**r) for r in rs);shards.append(rec)
        groups={}
        for group in ('train','val','dev'):
            subset=[s for s in shards if s['split']==group];co=Counter()
            for s in subset:co.update({k:v for k,v in s['counts'].items() if k!='directions'})
            groups[group]=dict(**co,maps=len({s['terrain'] for s in subset}),tapes=len(subset))
        c.save(HERE/'root_index.json.gz',all_rows)
        c.save(HERE/'data.json',dict(version=v,sources_sha256=SOURCES,root_index_sha256=c.sha(HERE/'root_index.json.gz'),
                                     direction_order=['NONE','UP','RIGHT','DOWN','LEFT'],groups=groups,shards=shards))
        print(groups,flush=True);return
    lane=0 if a.lane is None else a.lane
    need(0<=lane<a.lanes,'Invalid work lane')
    m=c.package();teacher=old.teacher_reader(m)
    for i,card in enumerate(cards):
        if i%a.lanes==lane:
            before=time.monotonic();r=one(card,m,teacher,v);print(i,card['split'],card['seed'],card['expert_seat'],r['counts'],round(time.monotonic()-before,2),flush=True)


if __name__=='__main__':main()

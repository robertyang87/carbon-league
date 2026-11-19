"""Extract pre-command opponent motion and candidate contact opportunities from signed original-game tapes."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parent
sys.path.insert(0,str(INTAKE/'endgame_curriculum'))
import roots
import features

PROTOCOL="2c904de2e1c31c49a9d8897123f0569e71728896b46da9f792ccbdb0b9e3c18c"
MOVES=("NONE","UP","RIGHT","DOWN","LEFT")
need=roots.need


def load_dataset(path=HERE/'dataset.npz'):
    """Return raw arrays where link_idx refers to global x/y rows, each target range is link_ptr[g:g+2]."""
    return np.load(path,allow_pickle=False)


def registered():
    p=(HERE/'protocol.json').read_bytes()
    need(roots.digest(p)==PROTOCOL and tuple(features.MOVES)==MOVES,"Unregistered feature protocol or action order")
    plan=json.loads(p);s=plan['sources'];m=(INTAKE/s['registration']).read_bytes()
    need(roots.digest(m)==s['sha256'],"Different training game manifest")
    manifest=json.loads(m);seeds=plan['fit_terrain_seeds']+plan['heldout_terrain_seeds']
    expected={(seed,side,arm) for seed in seeds for side in s['seats'] for arm in s['draws']}
    eps=sorted(manifest['episodes'],key=lambda e:(e['seed'],e['seat'],e['arm']))
    need(len(eps)==len(expected)==24 and {(e['seed'],e['seat'],e['arm']) for e in eps}==expected and
         all(e['episode_verified'] and (e['policy_sha256'],e['examiner_sha256'])==(s['student'],s['examiner']) for e in eps)
         and len({e['terrain'] for e in eps})==len(seeds)==6,"Different or incomplete map/seat/repetition set")
    return plan,eps


class Records:
    def __init__(self):
        self.cols=collections.defaultdict(list)
        self.x=[]
        self.symbols={}
        self.cols['link_ptr'].append(0)

    def symbol(self,name):
        return self.symbols.setdefault(str(name),len(self.symbols))

    def append(self,ep,idx,t,ids,x,all_targets,public,orders):
        n=len(ids);base=len(self.cols['y'])
        seat=ep['seat'];actors=public['players'][1-seat][2];own=public['players'][seat][2]
        need(set(ids)=={u for u,w in actors.items() if w[2]=='COLLECTOR'} and len(set(ids))==n,
             'Lost a currently visible opposing collector')
        if n:
            self.x.append(x)
        y=[MOVES.index(orders.get(uid,'NONE')) for uid in ids]
        for k,value in [('ep',idx),('step',t),('seed',ep['seed']),('seat',seat)]:
            self.cols[k].extend([value]*n)
        self.cols['uid'].extend(self.symbol(s) for s in ids)
        self.cols['y'].extend(y)
        counts=collections.Counter()
        for uid,intent,target,indices,moves in all_targets:
            pos=own[uid][0];intent=int(intent);target=int(target)
            need(own[uid][2]=='PLANTER' and target==features.DEST[pos,intent] and len(indices)==len(moves)==len(set(indices)),
                 'Wrong target cell or duplicate collector')
            counts[(uid,intent)]+=1
            # Independent public geometry census; reachable collectors all survive the join to labels, regardless of the outcome.
            expected={}
            for j,id_ in enumerate(ids):
                src=actors[id_][0]
                for a,dest in enumerate(features.DEST[src]):
                    if dest==target:expected[j]=a
            need(dict(zip(indices,moves))==expected,'Missing or extra reachable worker/move')
            self.cols['link_idx'].extend(base+j for j in indices)
            self.cols['link_action'].extend(moves)
            self.cols['link_ptr'].append(len(self.cols['link_idx']))
            truth=any(y[j]==m for j,m in zip(indices,moves))
            # Independent application of the recorded orders to the actor's public starting cell.
            actual_delivery=any(features.DEST[actors[who][0],move]==target for who,move in zip(ids,y))
            need(actual_delivery==truth,'Contact label disagrees with full physical submitted-order census')
            for key,val in [('g_ep',idx),('g_seed',ep['seed']),('g_seat',seat),('g_step',t),('g_intent',intent),
                            ('g_target',target),('g_planterpos',pos),('g_focal_uid',self.symbol(uid)),('g_truth',truth)]:
                self.cols[key].append(val)
        need(len(counts)==5*sum(w[2]=='PLANTER' for w in own.values()) and all(v==1 for v in counts.values()),
             'Every own planter must contribute all five distinct intents')

    def export(self,path,trains):
        packed={}
        for k,rows in self.cols.items():
            dtype=(np.int32 if k in ('seed','g_seed','link_idx','link_ptr') else
                   np.int16 if k in ('uid','step','g_step','g_focal_uid','g_target','g_planterpos') else
                   np.bool_ if k=='g_truth' else np.uint8)
            v=np.asarray(rows,dtype=dtype)
            if len(v):
                need(int(v.min())==int(min(rows)) and int(v.max())==int(max(rows)),'Integer width '+k)
            packed[k]=v
        packed['x']=np.concatenate(self.x)
        packed['uid_vocab']=np.asarray(list(self.symbols))
        packed['is_train']=np.isin(packed['seed'],trains)
        packed['g_is_train']=np.isin(packed['g_seed'],trains)
        need(packed['x'].shape[0]==len(packed['y']) and len(packed['link_ptr'])==len(packed['g_truth'])+1 and
             packed['link_ptr'][-1]==len(packed['link_idx'])==len(packed['link_action']),"Broken join offsets or row counts")
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            np.savez_compressed(f,**packed);tmp=Path(f.name)
        os.replace(tmp,path)
        return {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in packed.items()}


def process(ep,idx,data):
    ref=dict(seed=ep['seed'],seat=ep['seat'],path=ep['archive_path'],sha256=ep['archive_sha256'])
    before={k:len(data.cols[k]) for k in ('y','g_truth','link_idx')}
    with zipfile.ZipFile(roots.ROOT/ep['archive_path']) as z:
        receipt=z.read(ep['receipt_member'])
        need(roots.digest(receipt)==ep['receipt_sha256'] and json.loads(receipt)['turns_sha256']==ep['tape_sha256'],'Changed episode receipt')
    stream=hashlib.sha256();bank=[5.,5.];tracker=features.Tracks();n=0;seat=ep['seat']
    with roots.open_episode(ref,ep['arm']) as (env,live,tape,registration,reg):
        need(roots.digest(reg)==ep['registration_sha256'] and registration['physical_map_sha256']==ep['terrain'] and
             registration['files']['opponent']['sha256']==ep['examiner_sha256'] and roots.digest(bytes(env.states[0].observation.full_carbon))==ep['terrain'],
             'Different engine registration, opponent or opening terrain')
        while not env.done:
            t=env.states[0].observation.step
            public=live.observe(env,seat,bank[seat]);other=live.observe(env,1-seat,bank[1-seat])
            need(t==n and 'full_carbon' not in public and public['player']==seat,'Nonpublic feature request')
            tracker.observe(public)
            ids,x=features.board(public,tracker)
            ids=tuple(ids);x=np.asarray(x,dtype=np.float32).copy()
            need(x.ndim==2 and len(x)==len(ids) and np.isfinite(x).all(),'Bad pre-command feature rows')
            targets=[(u,a,c,tuple(i),tuple(m)) for u,a,c,i,m in features.target_links(public,ids)]
            observation_hashes=[None,None]
            for s,obs in ((seat,public),(1-seat,other)):
                observation_hashes[s]=roots.digest(live.encoded(obs))
            # Neither target candidates nor collector features can use the line read below.
            buf=tape.readline();need(buf,'Truncated historical episode');stream.update(buf);frame=json.loads(buf)
            need(frame['step']==t and frame['public_observation_sha256']==observation_hashes and
                 all(c['status']=='ok' and c['response']['step']==t for c in frame['calls']),'Replayed public turn disagrees with original')
            commands=[c['response']['action'] for c in frame['calls']]
            data.append(ep,idx,t,ids,x,targets,public,commands[1-seat])
            env.step(commands,[]);bank=frame['overage_remaining_seconds'];n+=1
        need(not tape.read() and stream.hexdigest()==ep['tape_sha256'] and n==ep['official_transitions'] and
             [s.reward for s in env.states]==ep['official_scores_by_seat'],'Historical physical tape or terminal mismatch')
    counts={k:len(data.cols[k])-v for k,v in before.items()}
    identity={k:ep[k] for k in ('seed','seat','terrain','arm','archive_sha256','registration_sha256','receipt_sha256','tape_sha256')}
    return dict(index=idx,**identity,turns=n,collector_rows=counts['y'],planter_intents=counts['g_truth'],links=counts['link_idx'])


def main():
    p,eps=registered()
    data=Records();episodes=[]
    for i,ep in enumerate(eps):
        result=process(ep,i,data);episodes.append(result)
        print(i,ep['seed'],ep['seat'],ep['arm'],result['collector_rows'],flush=True)
    assets=data.export(HERE/'dataset.npz',p['fit_terrain_seeds'])
    report=dict(schema='sapling-visible-opponent-motion-dataset/1',protocol_sha256=PROTOCOL,manifest_sha256=p['sources']['sha256'],
                files_sha256={k:roots.digest((HERE/k).read_bytes()) for k in ('dataset.py','features.py','dataset.npz')},
                arrays=assets,episodes=episodes,official_steps=sum(r['turns'] for r in episodes),move_order=MOVES,
                fit_seeds=p['fit_terrain_seeds'],heldout_seeds=p['heldout_terrain_seeds'],
                definitions={'uid':'index into uid_vocab; grouping only, never a model input',
                             'x':'exclusively from the student observer before any current tape line is consumed',
                             'link_idx':'global collector-row ID for one physically adjacent target and move in link_action',
                             'g_truth':'any reachable enemy collector issued the required move; includes subsequent deaths'})
    raw=json.dumps(report,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with tempfile.NamedTemporaryFile(dir=HERE,delete=False) as f:
        f.write(raw);tmp=Path(f.name)
    os.replace(tmp,HERE/'dataset.json')


if __name__=='__main__':
    main()

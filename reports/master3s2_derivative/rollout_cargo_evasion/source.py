"""Teacher-forced previous transition; recorded current opposing order labels only."""
import gzip
import importlib.util
import json
import zipfile
from collections import Counter

import numpy as np
from register import H,D,ROOT,read,sha
from runtime import _ev_units

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def init():
    path=D/'advantage_policy_update/v2_bc.py'; assert sha(path.read_bytes())==read(H/'protocol.json')['parent']
    m=load(path,'native_ev_parent');assert m._S_ABORT==.58 and m._TEMP==1 and m._SEARCH_OK
    m._torch.set_num_threads(1);m._net();return m

def tape(c):
    path=ROOT/c['source'];assert sha(path.read_bytes())==c['sha256']
    seat=c['seats'][0]
    if c['domain']=='ladder':
        a=json.loads(gzip.decompress(path.read_bytes()));frames=a['steps']; assert len(frames)-1<=299
        assert sha(bytes(frames[0][0]['observation']['full_carbon']))==c['terrain']
        out=[]
        for t,frame in enumerate(frames[:-1]):
            shared=frame[0]['observation']; priv=frame[seat]['observation']
            public=dict(step=t,player=seat,carbon=priv['carbon'],players=shared['players'],trees=shared['trees'],
                        remainingOverageTime=priv['remainingOverageTime'])
            out.append((public,[s['action'] or {} for s in frames[t+1]]))
        return out,a['configuration']
    with zipfile.ZipFile(path) as z:
        folder=c['repeat']+'/'
        reg=z.read(folder+'registration.json');rece=json.loads(z.read(folder+'receipt.json'))
        assert sha(reg)==rece['protocol_sha256'];reg=json.loads(reg);moves=z.read(folder+'turns.jsonl')
        assert sha(moves)==rece['turns_sha256'] and reg['candidate_seat']==seat and reg['physical_map_sha256']==c['terrain']
        fr=gzip.decompress(z.read(folder+'observations.jsonl.gz'))
    cfgfile=ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
    cfg=json.loads(gzip.decompress(cfgfile.read_bytes()))['game_configuration']
    out=[]
    for line,action in zip(fr.splitlines(),moves.splitlines(),strict=True):
        frame=json.loads(line);move=json.loads(action)
        assert frame['step']==move['step'] and frame['public_observation_sha256']==move['public_observation_sha256']
        pub=frame['observations'][seat]
        assert sha((json.dumps(pub,separators=(',',':'),allow_nan=False)+'\n').encode())==frame['public_observation_sha256'][seat]
        out.append((pub,[r['response']['action'] for r in move['calls']]))
    return out,cfg

def matching(p,real,opp):
    def slots(s):return {(w[0],w[2]):u for u,w in s[2].items()}|{(q,'BASE'):u for u,q in s[1].items()}
    x,y=slots(p[opp]),slots(real[opp])
    if set(x)!=set(y):return None
    return {uid:y[k] for k,uid in x.items()}

def collect(m,cards,held):
    output=[];seen=Counter(); delta=m._S_RTEMP if m._RNET is not m._NET else 1.
    for c in cards:
        tape0,cfg=tape(c);seat=c['seats'][0];op=1-seat;tr=m.BeliefTracker(m._Bel,seat); buf=[]; fallback=0
        def flush():
            if not buf:return
            lw,lb,_=m._RNET(np.stack([k[1] for k in buf]),m._SID)
            for j,(r,x) in enumerate(buf):
                r['lw']=lw[j]/delta;r['lb']=lb[j]/delta;output.append(r)
            buf.clear()
        for t,(pub,old_joint) in enumerate(tape0[:-1]):
            est,_=tr.step(pub['players'],t,pub['carbon'])
            if 2<=t<=294 and (t-2)%4==0:
                nxt,actions=tape0[t+1];seen['requested']+=1
                root=dict(players=pub['players'],step=t,player=seat,trees=pub['trees'],carbon=list(map(float,est)),remainingOverageTime=pub['remainingOverageTime'])
                p=m._turn(root,cfg,old_joint);units=_ev_units(p['players'],op,pub['players'])
                lookup=matching(p['players'],nxt['players'],op)
                if lookup is None:seen['mismatched_roster']+=1
                elif units or (held and fallback<24):
                    if not units:fallback+=1
                    x=m._quant(m.encode(p['players'],t+1,op,p['carbon'],[1.]*m.N,pub['players'])).astype(np.float32)
                    assert x.shape==(54,15,15)
                    labels={uid:m.WIDX[actions[op].get(lookup[uid],'NONE')] for uid in p['players'][op][2]}
                    root=dict(seed=c['seed'],terrain=c['terrain'],seat=seat,step=t+1,arm=c['repeat'],game=c['sha256']+'/'+c['repeat'],
                              players=p['players'],previous=pub['players'],units=units,labels=labels)
                    buf.append((root,x))
                    if len(buf)>=8:flush()
                else:seen['ordinary_not_sampled']+=1
            tr.set_actions(pub['players'],old_joint[seat])
        flush()
    return output,dict(seen)

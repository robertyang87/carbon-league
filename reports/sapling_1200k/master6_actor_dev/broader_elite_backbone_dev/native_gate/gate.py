"""Bind the unique final export, prove deployable action changes, and measure both natural-reply terminal branches."""
import ast
import base64
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent; A=TRAIN.parent; ROOT=A.parents[2]
sys.path.insert(0,str(A/'elite_role_mix_dev/conditional'))
import responses as r
sys.path.insert(0,str(ROOT/'code'))
import cnet2_pack
P=r.load(HERE/'protocol.json')
PACK={'parent':A/'actor_partner_dev/candidate.py','candidate':TRAIN/'candidate.py'}


def artifacts(p):
    src=p.read_text();lines=src.splitlines(keepends=True);meta={}
    for n in ast.parse(src).body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):continue
        k=n.targets[0].id
        if k in ('_W64','_QMETA','_WMETA'):
            meta[k]=ast.literal_eval(n.value);lines[n.lineno-1:n.end_lineno]=[f'{k} = 0\n']*(n.end_lineno-n.lineno+1)
    assert len(meta)==3
    return ''.join(lines),meta


def binding():
    assert r.sha(TRAIN/'protocol.json')==P['training_protocol_sha256'] and r.sha(TRAIN/'panel.json')==P['panel_sha256'] and r.sha(TRAIN/'shards.json')==P['shards_sha256']
    panel=r.load(TRAIN/'panel.json')['games']; shards=r.load(TRAIN/'shards.json')['games']
    assert len({v['gid'] for v in panel})==len(panel)==len(shards) and len({v['seed'] for v in panel})==len(panel)
    tr=[v for v in panel if v['split']=='train'];train_ids={v['gid'] for v in tr};n=sum(v['rows'] for v in shards if v['gid'] in train_ids)
    for e in P['exposures']:
        assert all(v['terrain_sha256']!=e['terrain'] and v['seed']!=e['seed'] for v in panel)
    trp=r.load(TRAIN/'protocol.json');fit=r.load(TRAIN/'fit.json');pkg=r.load(TRAIN/'package.json');recipe=r.load(TRAIN/'recipe.json')
    assert r.sha(TRAIN/'recipe.json')==fit['recipe_sha256'] and recipe['protocol_sha256']==P['training_protocol_sha256'] and r.sha(TRAIN/'train.py')==recipe['trainer_sha256']
    assert trp['seed']==7924059 and trp['train_epochs']==1 and trp['batch']==16 and fit['train_roots']==n and fit['map_count']==len(tr) and fit['updates']==int(np.ceil(n/16))
    weight=TRAIN/'candidate.npz';old=ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'
    assert r.sha(old)==P['parent_sapling']==fit['init_sha256'] and r.sha(weight)==fit['weight_sha256']==pkg['weights_sha256']
    assert r.sha(PACK['candidate'])==pkg['candidate_sha256'] and r.sha(PACK['parent'])==P['parent']==pkg['parent_sha256'] and r.sha(r.live.SOURCE)==P['original']
    student=artifacts(PACK['candidate']);previous=artifacts(PACK['parent'])
    assert student[0]==previous[0] and previous[1]['_W64']!=student[1]['_W64']
    with np.load(weight,allow_pickle=False) as z,np.load(old,allow_pickle=False) as parent:
        meta,raw,deq=cnet2_pack.quantize(z); assert meta==student[1]['_QMETA'] and json.loads(student[1]['_WMETA'])['cfg']==json.loads(str(z['meta']))['cfg']
        assert zlib.decompress(base64.b64decode(student[1]['_W64']))==raw
        assert set(z.files)==set(parent.files)
        changed=sum(np.count_nonzero(z[k]!=parent[k]) for k in z.files if k!='meta')
        assert changed>500000 and all(np.isfinite(z[k]).all() for k in z.files if k!='meta')
    out=dict(protocol=r.sha(HERE/'protocol.json'),files={n:r.sha(TRAIN/n) for n in ('recipe.json','fit.json','candidate.npz','candidate.py','standalone.py','package.json','train.py')},
        phase_panel=P['panel_sha256'],shards=P['shards_sha256'],maps=len(panel),train_maps=len(tr),train_roots=n,updates=fit['updates'],
        changed_sapling_parameters=int(changed),byte_identical_all_other_deployed_source=True,quantized_sapling_payload_sha256=hashlib.sha256(raw).hexdigest(),original=P['original'],parent=P['parent'])
    r.write(HERE/'binding.json',out)
    return out


def game(c):
    f=ROOT/c['source'];g=r.load(f)
    rec=r.load(f.with_name(f.name.replace('raw_','receipt_').replace('.json.gz','.json')))
    assert r.sha(f)==c['source_sha256']==rec['game']['raw_sha256'] and g['seed']==c['seed'] and g['candidate_seat']==c['seat'] and g['pack_sha256']==P['parent']
    assert rec['official']['statuses']==['DONE','DONE'] and g['initial_terrain_u8_sha256']==c['terrain'] and not g['errors']
    return g


def mechanism(c):
    raw=game(c);seat=c['seat'];cfg=raw['game_configuration'];sim=r.FastCarbon(cfg).reset(seed=raw['seed'])
    assert list(sim.carbon)==raw['initial_full_carbon']
    mods={name:r.live.module(p,'backbone_mech_'+name) for name,p in PACK.items()}
    for t,pair in enumerate(raw['actions'][:191]):
        v=sim.obs(seat)
        if t==0 or t>=140:
            cmds={n:r.live.clean(m.agent(v,cfg)) for n,m in mods.items()};assert cmds['parent']==r.live.clean(pair[seat])
            if t>=140 and cmds['parent']!=cmds['candidate']:
                base=v['players'][seat][1];workers=v['players'][seat][2]
                shift={k:dict(kind='base' if k in base else workers[k][2],parent=cmds['parent'].get(k,'NONE'),candidate=cmds['candidate'].get(k,'NONE')) for k in set(cmds['parent'])|set(cmds['candidate']) if cmds['parent'].get(k)!=cmds['candidate'].get(k)}
                return dict(**c,t=t,**cmds,opponent=pair[1-seat],changed=shift,candidate_public_sha256=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        else:
            for m in mods.values():m._FEAT.observe(v)
        sim.step(*pair)
    return dict(**c,t=None)


def trees(env):
    return [len(p[3]) for p in env.states[0].observation.players]


def play(job):
    c,name,cpu,warm=job;os.sched_setaffinity(0,{cpu})
    f=HERE/f"response_{c['seed']}_{c['seat']}_{name}.json.gz"
    b=dict(identity=r.sha(HERE/'binding.json'),source=c['source_sha256'],pack=r.sha(PACK[name]),cpu=cpu,shared_warm=warm)
    if f.exists():
        v=r.load(f);assert v['binding']==b and v['statuses']==['DONE','DONE'];return v
    raw=game(c);t0=c['t'];seat=c['seat'];op=1-seat;cfg=raw['game_configuration'];env=r.live.begin(raw)
    target=r.live.module(PACK[name],'native_own_'+name);master=r.live.module(r.live.SOURCE,'native_orig_'+name)
    assert master._SEED is None and master._TEMP==1 and master._SEARCH_OK and master._RX and master._S_ABORT==.58
    tr=master.BeliefTracker(master._Bel,op)
    for t,pair in enumerate(raw['actions'][:t0+1]):
        v=r.live.public(env,seat);o=r.live.public(env,op)
        tr.step(o['players'],t,o['carbon']);tr.set_actions(o['players'],pair[op])
        if t in (0,t0):
            a=r.live.clean(target.agent(v,cfg))
            if t==0 and name=='parent':assert a==r.live.clean(pair[seat])
        else: target._FEAT.observe(v)
        if t<t0:env.step(pair)
    assert a==c[name] and pair[op]==c['opponent']
    opening=list(pair);opening[seat]=a
    rng=int(hashlib.sha256(f"broader-final|{raw['seed']}|{seat}|{t0}".encode()).hexdigest()[:8],16)
    master._S_COST[:]=warm;master._ST.clear();master._ST.update(tr=tr,prev=o['players'],rng=np.random.RandomState(rng))
    recorder=r.native_clock.Recorder(master);opp=recorder.wrap(master.agent)
    tape=[opening];tree=[trees(env)];env.step(opening);tree.append(trees(env))
    while not env.done:
        pair=[None,None];pair[seat]=target.agent(r.live.public(env,seat),cfg);pair[op]=opp(r.live.public(env,op),cfg)
        env.step(pair);tape.append(pair);tree.append(trees(env))
    statuses=[s.status for s in env.states];scores=[s.reward for s in env.states];cash_terminal=[p[0] for p in env.states[0].observation.players]
    assert statuses==['DONE','DONE'] and not recorder.errors and len(recorder.calls)==len(tape)-1>0
    result=dict(binding=b,case=c,arm=name,rng=rng,statuses=statuses,scores=scores,margin=scores[seat]-scores[op],tape=tape,calls=recorder.calls,
        clock=r.native_clock.summarize(recorder.calls,master._S_ABORT,master._S_D2),cash=r.cash_summary(raw,seat,t0,tape,cash_terminal),tree_control=tree)
    r.write(f,result);return result


def run():
    b=binding();cases=[mechanism(c) for c in P['exposures']]
    r.write(HERE/'mechanism.json',dict(binding=r.sha(HERE/'binding.json'),cases=cases))
    if not all(c['t'] is not None for c in cases):
        r.write(HERE/'gate.json',dict(eligible=False,reason='no actual changed order in one or both frozen windows',cases=cases));return
    rows=[]
    with r.LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for c in cases:
            s=c['seat']; warmf=HERE/f"warm_{c['seed']}_{s}.json"
            if not warmf.exists():
                with ProcessPoolExecutor(1) as pool:warm=pool.submit(r.native_warm).result()
                r.write(warmf,dict(shared=warm,original=P['original'],binding=r.sha(HERE/'binding.json')))
            warm=r.load(warmf);assert warm['binding']==r.sha(HERE/'binding.json')
            with ProcessPoolExecutor(2) as pool:
                a,b=pool.map(play,[(c,'parent',s,warm['shared']),(c,'candidate',1-s,warm['shared'])])
            assert a['rng']==b['rng'] and a['binding']['shared_warm']==b['binding']['shared_warm']
            diff=[i+c['t'] for i,(x,y) in enumerate(zip(a['tape'],b['tape'])) if x[1-s]!=y[1-s]]
            row=dict(seed=c['seed'],seat=s,t=c['t'],genuine_changed=c['changed'],enemy_changed=diff,opponent_calls=[len(z['calls']) for z in (a,b)],
                parent_score_by_seat=a['scores'],candidate_score_by_seat=b['scores'],margins=[a['margin'],b['margin']],delta=b['margin']-a['margin'],
                delta_tree_income=b['cash']['tree_income_margin']-a['cash']['tree_income_margin'],delta_delivery=b['cash']['delivery_margin']-a['cash']['delivery_margin'],
                trees_by_seat_last=[z['tree_control'][-1] for z in (a,b)],tree_turns_by_seat=[np.sum(z['tree_control'][1:],axis=0).tolist() for z in (a,b)])
            rows.append(row);print('RESPONDED',c['seed'],row['margins'],row['delta'],flush=True)
    r.write(HERE/'gate.json',dict(protocol=r.sha(HERE/'protocol.json'),binding=r.sha(HERE/'binding.json'),
        eligible=all(row['delta']>0 and row['enemy_changed'] and row['genuine_changed'] for row in rows),rows=rows))


if __name__=='__main__':run()

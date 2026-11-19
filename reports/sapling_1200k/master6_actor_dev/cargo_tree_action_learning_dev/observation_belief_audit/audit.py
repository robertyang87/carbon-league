"""Compare the two frozen public belief estimators on identical observations and actual action histories."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=EXP.parents[3]
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon

def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())
def save(path,data):
    b=json.dumps(data,allow_nan=False,sort_keys=True,separators=(',',':'),default=lambda v:v.item()).encode()
    path.write_bytes(gzip.compress(b,mtime=0) if path.suffix=='.gz' else b)
def gitfile(name):return subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+name])

def modules():
    src=gitfile('sources/master3s2/master3s2.py')
    assert digest(src)==load(EXP/'territory_sequence/response/protocol.json')['examiner_sha256']
    m=types.ModuleType('verified_carbon_master');exec(compile(src,'master3s2.py','exec'),m.__dict__)
    actor=(EXP/'candidate.py').read_bytes();assert digest(actor)==load(HERE/'protocol.json')['parent_pack']
    sources=ast.literal_eval(next(n for n in ast.parse(actor).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SRC').value)
    f=types.ModuleType('exact_sapling_feat');exec(sources['cnet2_feat'],f.__dict__)
    assert 'x["maps"].astype(np.float32) / np.float32(255.0)' in sources['cnet2_np']
    return f,m

def cards():
    rules=load(HERE/'protocol.json');cfg=load(EXP/'direct_original/games/candidate/raw_621758346_1.json.gz')['game_configuration']
    for seed in rules['old_map_ids']:
        for seat in (0,1):
            pref='direct_original' if seed in (621758345,621758346) else 'replicate_original'
            raw=EXP/pref/f'games/candidate/raw_{seed}_{seat}.json.gz';d=load(raw)
            assert d['pack_sha256']==rules['parent_pack'] and d['examiner_md5'].startswith('57f9a594') and (d['seed'],d['candidate_seat'])==(seed,seat)
            yield dict(seed=seed,seat=seat,arm='parent',origin=pref,terrain=d['initial_terrain_u8_sha256'],src=str(raw.relative_to(ROOT)),sha=digest(raw.read_bytes())), d['game_configuration'],d['initial_full_carbon'],d['actions'],d['last_post_cash']
    for seed in rules['completed_map_ids']:
        for seat in (0,1):
            file=EXP/'territory_sequence/fresh_original/hosted'/f'{seed}-{seat}.zip'
            with zipfile.ZipFile(file) as z:
                for arm in ('parent','student'):
                    reg=json.loads(z.read(arm+'/registration.json'));receipt=json.loads(z.read(arm+'/receipt.json'));data=z.read(arm+'/turns.jsonl')
                    assert digest(data)==receipt['turns_sha256'] and reg['seed']==seed and reg['candidate_seat']==seat
                    assert reg['files']['opponent']['sha256']==load(EXP/'territory_sequence/response/protocol.json')['examiner_sha256']
                    assert receipt['status']=='complete' or receipt['official_transitions']==299
                    tape=[json.loads(b) for b in data.splitlines()]
                    actions=[[c['response']['action'] for c in t['calls']] for t in tape if all(c['status']=='ok' for c in t['calls'])]
                    assert len(actions)==receipt['official_transitions']
                    yield dict(seed=seed,seat=seat,arm=arm,origin='hosted',terrain=reg['physical_map_sha256'],src=str(file.relative_to(ROOT))+'#'+arm,sha=digest(file.read_bytes())),cfg,None,actions,receipt['official_scores_by_seat']


def view(f,m,feat,tr,obs,t):
    s=obs['player'];all_pl=obs['players'];assert set(obs)=={'players','step','carbon','trees','player','remainingOverageTime'}
    directly=np.zeros(225,bool);workers=all_pl[s][2]
    for pos,c,role in workers.values():directly[[pos] if role=='COLLECTOR' else f.DEST[pos,1:]]=True
    assert not np.any(np.asarray(obs['carbon'])[~directly])
    x=feat.observe(obs);raw=feat.est.copy();order=f.GRID if s==0 else f.CANON
    assert np.array_equal(np.array([feat.cpos(i) for i in range(225)]),order)
    unperm=x['maps'][0].reshape(-1)[order]
    assert np.array_equal(unperm,f.q8((raw/100.).astype(np.float32))), ('carbon plane orientation or quantization',s,t)
    assert np.array_equal(x['maps'][2].reshape(-1)[order]>0,feat.last_step>=0), ('history plane',s,t)
    actor=unperm.astype(float)*(100/255)
    decoded=x['tok_pos'][:10];labels=feat.labels({})
    for slot,uid in enumerate(feat.own_uid[:10]):
        if uid in workers:
            assert decoded[slot]==order[workers[uid][0]]
    v,known=tr.step(all_pl,t,obs['carbon']);v=np.asarray(v);known=np.asarray(known)
    # Master encode keeps board orientation physical for both seats; its own public inference rounds X*100.
    orig=np.clip(np.rint((v.astype(np.float32)/100)*100),0,255).astype(float)
    if t in (0,16,128):
        enc=m.encode(all_pl,t,s,v,known,None)
        assert np.array_equal(orig,np.clip(np.rint(enc[0].flatten()*100),0,255))
    if t==0:
        bases=np.zeros(225,bool);bases[[pos for side in all_pl for pos in side[1].values()]]=True
        assert not directly.any() and not (feat.last_step>=0).any() and np.all(raw[bases]==0) and np.all(raw[~bases]==11)
        assert np.array_equal(bases,known) and np.allclose(v[~bases],28.5/63*22)
        assert known.sum()==len(all_pl[0][1])+len(all_pl[1][1])
    return x,np.stack((raw,v,actor,orig)),directly,known


def record(card,cfg,full,tape,terminal,f,m,limit=None):
    sim=FastCarbon(cfg).reset(seed=card['seed'],full_carbon=full)
    assert digest(bytes(map(int,sim.carbon)))==card['terrain']
    F=[f.Feat(s) for s in (0,1)];M=[m.BeliefTracker(m._Bel,s) for s in (0,1)]
    sums=defaultdict(lambda:np.zeros(5));rank=[];checks=Counter();vis_age=defaultdict(Counter)
    grid=np.arange(225);combo=np.triu_indices(5,k=1);N=f.DEST[:,1:]
    for t,joint in enumerate(tape[:limit]):
        truth=np.asarray(sim.carbon,dtype=float)
        for s in (0,1):
            o=sim.obs(s);x,preds,vis,known=view(f,m,F[s],M[s],o,t);seen=F[s].last_step>=0
            assert np.allclose(preds[:2,vis],truth[vis],atol=.001,rtol=0)
            a,e=preds[2:]
            stage='opening' if t<100 else ('middle' if t<200 else 'end')
            own=o['players'][s];workers=own[2];pos=[[],[]]
            for i,c,role in workers.values():pos[int(role=='PLANTER')].append(i)
            near=[np.zeros(225,bool) if not b else f.DIST[:,b].min(axis=1)<=1 for b in pos]
            plan_ground=np.zeros(225,bool);plan_ground[N[np.flatnonzero(near[1])].flatten()]=True
            zones={'whole':np.ones(225,bool),'collector_options':near[0],'planter_options':near[1],
                   'planter_option_neighbors':plan_ground,'outside_all':~(near[0]|near[1]|plan_ground)}
            cats={'all':np.ones(225,bool),'direct':vis,'hidden':~vis,'hidden_known_both':~vis&seen&known,
                  'hidden_only_original_knows':~vis&~seen&known,'hidden_only_sapling_seen':~vis&seen&~known,'hidden_neither':~vis&~seen&~known}
            errs=np.abs(preds-truth)
            for zn,zm in zones.items():
                for nm,nm_mask in cats.items():
                    mask=zm&nm_mask
                    if mask.any():
                        k=(s,stage,zn,nm); v=sums[k];v[0]+=mask.sum();v[1:]+=errs[:,mask].sum(axis=1)
            tree=np.zeros(225,bool);own_tree=np.zeros(225,bool);centers=np.zeros(225,bool)
            for pid,team in enumerate(o['players']):
                centers[list(team[1].values())]=True
                for p,age in team[3].values():
                    tree[p]=True
                    if pid==s:own_tree[p]=True
            ntrees=sum(1 for team in o['players'] for _,age in team[3].values() if age<cfg['treeLifespan'])
            fee=cfg['plantCost']+(cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**ntrees if len(own[3])>cfg['playerTreeProtectiveNumber'] else 0)
            action_labels=F[s].labels(joint[s]);pre=f.Prefix(x,F[s].aux);vs=np.stack((truth,a,e))
            for slot in np.flatnonzero(x['tok_mask'][:f.OWN]):
                _,_,legal=pre.feats(slot);cmd=int(action_labels[slot])
                if slot<10:
                    uid=F[s].own_uid[slot];p,cargo,role=workers[uid];dest=f.DEST[p]
                    pl=role=='PLANTER'
                    # Strict count of available economic targets, independent of both map estimates.
                    feasible=(~centers[dest]&~own_tree[dest]&np.where(tree[dest],own[0]>=cfg['seizeCost'],own[0]>=fee)) if pl else (~tree[dest]&~centers[dest])
                    phys=f.DEST[x['tok_pos'][slot]];canon=f.GRID if s==0 else f.CANON;phys=canon[phys]
                    # Map physical order to canonical move legality using destination identity, including horizontal seat-1 reflection.
                    allowed=np.array([legal[np.flatnonzero(phys==d)[0]] for d in dest],dtype=bool)
                    real=int(np.flatnonzero(dest==phys[cmd])[0]);assert real==('NONE','UP','RIGHT','DOWN','LEFT').index(joint[s].get(uid,'NONE'))
                    if pl:
                        score=vs[:,N[dest]].mean(2);direct_opts=vis[N[dest]].all(1)
                    else:score=vs[:,dest];direct_opts=vis[dest]
                    # Three numbers per state per move preserve true and estimated alternatives; output binary uses float32.
                    rank.append((s,t,slot,uid,role,real,sum(1<<i for i in np.flatnonzero(allowed)),
                                 sum(1<<i for i in np.flatnonzero(feasible)),sum(1<<i for i in np.flatnonzero(direct_opts)),score.copy()))
                pre.advance(slot,cmd)
            M[s].set_actions(o['players'],joint[s])
        sim.step(*joint)
    if limit is None:
        assert np.allclose(sim.cash,terminal,atol=.011,rtol=0),(card,sim.cash,terminal)
    return sums,rank

def evidence():
    arg=argparse.ArgumentParser();arg.add_argument('--pilot',action='store_true');args=arg.parse_args()
    f,m=modules();results=[]; meta=[]; values=[]
    for card,cfg,full,actions,term in cards():
        if args.pilot and (card['seed'],card['seat'])!=(621758345,1):continue
        sums,rs=record(card,cfg,full,actions,term,f,m,48 if args.pilot else None)
        idx=len(results)
        for *info,score in rs:
            meta.append((idx,*info));values.append(score)
        results.append(dict(card=card,turns=48 if args.pilot else len(actions),cell_groups=[[*k,*v.tolist()] for k,v in sums.items()]))
        print(card['seed'],card['seat'],card['arm'],len(rs),flush=True)
        if args.pilot:break
    pref='pilot_' if args.pilot else ''
    save(HERE/(pref+'cell_groups.json.gz'),results);save(HERE/(pref+'opportunities.json.gz'),meta)
    np.savez_compressed(HERE/(pref+'ranks.npz'),values=np.stack(values).astype(np.float32))

if __name__=='__main__':evidence()

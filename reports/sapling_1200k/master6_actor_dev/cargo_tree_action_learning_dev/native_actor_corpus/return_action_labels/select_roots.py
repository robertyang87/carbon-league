"""Stream the frozen archive, compare deployed public commands and sign public return opportunities."""
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p/'.git').exists())
PLAN = 'ff9567c50fb7f39375e4151a2933bee0ccd19398401713a34e7b59a5f950af8b'


def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


A=module(EXP/'learned_carbon_observation/active_roots/select_roots.py','return_stream')
V=module(EXP/'learned_carbon_observation/active_roots/verify.py','return_geometry')
K=module(EXP/'learned_carbon_observation/native_response/run.py','return_native_api')


def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sig(x):return K.digest(K.compact(x))


def old_opportunities():
    times=set();hashes=defaultdict(set)
    for r in read(EXP/'native_actor_corpus/terminal_direction/commands.json.gz'):
        key=(r['seed'],r['seat']);times.add((*key,r['t']));hashes[key].update((r['prefix_sha256'],r['public_sha256']))
    for r in read(EXP/'signed_return_policy/inventory.json.gz'):
        times.add((r['seed'],r['seat'],r['t']))
    for name in ('signed','windows','earlier'):
        for r in read(EXP/f'paired_response_prefix/{name}.json.gz')['rows']:
            key=(r['seed'],r['seat']);times.add((*key,r['start']));hashes[key].add(r['prior_archived_joint_sha256'])
            if r.get('first_actual_intervention') is not None: times.add((*key,r['first_actual_intervention']))
    for r in read(EXP/'native_actor_corpus/original_response/release.json')['cases']:
        for t in range(r['t0'],r['t1_exclusive']):times.add((r['seed'],r['seat'],t))
    return times,hashes


class Policy:
    def __init__(self,ref,name):
        self.model=V.load_pack(ROOT/ref['path'],ref['sha256'])
        self.current=None
        native=self.model._mixed_actions
        def record(x,aux):
            self.current=x,aux
            return native(x,aux)
        self.model._mixed_actions=record

    def command(self,obs,cfg):return K.own_call(self.model,obs,cfg)[0]

    def legal_trace(self,orders):
        m=self.model;f=sys.modules['cnet2_feat'];x,aux=self.current
        labels=m._FEAT.labels(orders);assert m._FEAT.to_actions(labels)==orders
        pre=f.Prefix(x,aux);trace=[]
        for slot in np.flatnonzero(x['tok_mask'][:f.OWN]):
            act=int(labels[slot]);_,_,mask=pre.feats(slot);assert mask[act]
            trace.append(dict(slot=int(slot),uid=m._FEAT.own_uid[slot],label=act,legal=mask.astype(int).tolist(),cash=float(pre.cash),slots=int(pre.slots),reserved=np.flatnonzero(pre.reserved).tolist()))
            pre.advance(slot,act)
        return trace


def qualify(obs,cmd,immediate_only):
    side=obs['player']; team=obs['players'][side];bases=sorted(team[1].values()); options=[]
    for uid,(pos,cargo,role) in team[2].items():
        if role!='COLLECTOR' or cargo<20:continue
        a,b=(cmd[name].get(uid,'NONE') for name in ('parent','student'))
        if a==b:continue
        dest=[V.land(pos,z) for z in (a,b)]; da,db=(min(V.distance(p,h) for h in bases) for p in dest)
        immediate=(da==0)!=(db==0);near=1<=min(V.distance(pos,h) for h in bases)<=5
        if da==db or not immediate and (immediate_only or not near):continue
        focal=dict(uid=uid,cargo=cargo,position=pos,parent_command=a,student_command=b,parent_destination=dest[0],student_destination=dest[1],parent_distance=da,student_distance=db,own_bases=bases)
        features=[min(cargo/100,1)*(da-db)/2,min(cargo/100,1)*(int(db==0)-int(da==0))]
        options.append((not immediate,-cargo,uid,focal,features))
    return min(options,key=lambda o:o[:3]) if options else None


def run():
    assert __debug__ and sha(HERE/'protocol.json')==PLAN
    plan=read(HERE/'protocol.json')
    for ref in list(plan['inputs'].values())+list(plan['packages'].values()):assert sha(ROOT/ref['path'])==ref['sha256']
    old,seen=old_opportunities();roots=[];coverage=[]
    for i,card in enumerate(plan['maps']):
        seed,s=card['seed'],card['seat'];p=ROOT/card['source'];assert sha(p)==card['source_sha256'] and sha(ROOT/card['receipt'])==card['receipt_sha256']
        raw,archive=A.streaming_archive(p);assert raw['seed']==seed and raw['initial_terrain_u8_sha256']==card['terrain']
        env=K.checked_env(raw); models={k:Policy(ref,k) for k,ref in plan['packages'].items()}; hist=[]
        picked=set();stats=[Counter(),Counter()];cfg=raw['game_configuration']
        for t in range(266):
            own=K.public(env,s);cmd={k:m.command(own,cfg) for k,m in models.items()}
            if t>=210:
                wi=int(t>=238);begin,end=plan['windows'][wi];assert begin<=t<=end
                count=stats[wi];count['available_ticks']+=1
                if wi not in picked:
                    count['examined_before_choice']+=1
                    if cmd['parent']!=cmd['student']:
                        count['different_joint']+=1;choice=qualify(own,cmd,t<begin+14)
                        if choice:
                            key=(seed,s);pref=sig(hist);public=sig(own);short=sig({k:v for k,v in own.items() if k!='remainingOverageTime'})
                            if (seed,s,t) in old or {pref,short,public}&seen[key]:count['excluded_historical_overlap']+=1
                            else:
                                immediate,_,_,focal,features=choice; traces={k:models[k].legal_trace(a) for k,a in cmd.items()}
                                changed={u:[cmd['parent'].get(u,'NONE'),cmd['student'].get(u,'NONE')] for u in sorted(cmd['parent'].keys()|cmd['student'].keys()) if cmd['parent'].get(u)!=cmd['student'].get(u)}
                                order=['parent','student'] if (i+wi)%2==0 else ['student','parent']
                                roots.append(dict(**card,id=f'{seed}_{s}_{t}',t=t,window=wi,map_index=i,arm_order=order,**cmd,focal=focal,features=features,
                                    immediate_base=not immediate,all_changed=changed,legal_prefix=traces,prefix_sha256=pref,public_sha256=public,short_public_sha256=short,joint_pair_sha256=sig(cmd)))
                                picked.add(wi);seen[key].update((pref,short,public));count['enrolled']+=1
            if t<265:
                physical=next(archive);hist.append(physical);env.step(physical)
        archive.close();coverage.append(dict(seed=seed,seat=s,terrain=card['terrain'],windows=[dict(c) for c in stats]))
    out=dict(protocol_sha256=PLAN,selector_sha256=sha(Path(__file__)),roots=roots,coverage=coverage)
    target=HERE/'roots.json';data=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if target.exists():assert target.read_text()==data
    else:target.write_text(data)
    print('selection',len(roots),[(r['id'],r['features']) for r in roots],sha(target),flush=True)


if __name__=='__main__':run()

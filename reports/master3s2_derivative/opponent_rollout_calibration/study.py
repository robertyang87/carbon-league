"""Fit inverse temperatures and independently inspect native mixed strategy draws on old game histories."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from runtime import adjust

HERE=Path(__file__).resolve().parent
DERIV=HERE.parent
ROOT=DERIV.parents[1]
PARENT="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
ORIGINAL="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
PROTOCOL="79f726215aa3a8a8d6c72e915b2bfed2f7f2ec754f0ce3c7bd571d10b1897eef"
ROLES=("COLLECTOR","PLANTER","BASE")
DRAWS=16


def sha(x):
    return hashlib.sha256(x).hexdigest()


def put(path, data):
    b=(json.dumps(data,ensure_ascii=False,separators=(",",":"),sort_keys=True,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":
        b=gzip.compress(b,mtime=0)
    path.write_bytes(b)
    return sha(b)


def initialize():
    path=DERIV/"cold_start/candidate.py"
    assert sha(path.read_bytes())==PARENT
    spec=importlib.util.spec_from_file_location("frozen_rollout_model",path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._SEARCH_OK and m._TEMP==1.0 and m._S_ABORT==.58
    m._torch.set_num_threads(1);m._net()
    return m


def read(card):
    raw=(DERIV/card["rel"]).read_bytes()
    assert sha(raw)==card["archive_sha256"]
    arm=card["arm"]
    with zipfile.ZipFile(DERIV/card["rel"]) as z:
        r=json.loads(z.read(arm+"/registration.json")); c=json.loads(z.read(arm+"/receipt.json"))
        turns=z.read(arm+"/turns.jsonl"); obs=gzip.decompress(z.read(arm+"/observations.jsonl.gz"))
    seat=card["seat"]
    assert r['files']['candidate']['sha256']==PARENT and r['files']['opponent']['sha256']==ORIGINAL
    assert r['seed']==c['seed']==card['seed'] and c['candidate_seat']==seat
    assert r['physical_map_sha256']==card['map'] and c['official_transitions']==299 and c['counted']
    assert c['initialization'][1-seat]['response']['original']==dict(_TEMP=1.,_SEED=None,_SEARCH_OK=True,_S_ABORT=.58)
    assert sha(turns)==c["turns_sha256"]
    tape=[]
    for t,(ob,tm) in enumerate(zip(obs.splitlines(),turns.splitlines(),strict=True)):
        a=json.loads(ob); b=json.loads(tm)
        assert a["step"]==b["step"]==t and a['public_observation_sha256']==b['public_observation_sha256']
        public=a["observations"][seat]
        h=sha((json.dumps(public,separators=(",",":"),allow_nan=False)+"\n").encode())
        assert h==a['public_observation_sha256'][seat] and set(public)=={"step","player","players","trees","carbon","remainingOverageTime"}
        assert all(entered['status']=='ok' for entered in b['calls'])
        tape.append((public,[entered['response']['action'] for entered in b['calls']]))
    assert len(tape)==299
    return tape,c['winner_seat']==seat


def mapping(sim, actual):
    """Canonicalize newly born actor names, allowing cash/carbon to differ under belief projection."""
    pairs=[]
    for sp,rp in zip(sim,actual):
        a={(v, "BASE"):k for k,v in sp[1].items()}
        b={(v, "BASE"):k for k,v in rp[1].items()}
        a.update({(v[0],v[2]):k for k,v in sp[2].items()})
        b.update({(v[0],v[2]):k for k,v in rp[2].items()})
        if set(a)!=set(b) or len(a)!=len(sp[1])+len(sp[2]) or len(b)!=len(rp[1])+len(rp[2]):
            return None
        pairs.append({uid:b[key] for key,uid in a.items()})
    return pairs


def collect(m, cards):
    schema=json.loads((ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in schema['configuration'].items() if isinstance(v,dict) and 'default' in v}
    roots=[];miss=[]
    for ticket in cards:
        tape,won=read(ticket);seat=ticket['seat'];op=1-seat
        tr=m.BeliefTracker(m._Bel,seat)
        for t,(pub,command) in enumerate(tape):
            est,_=tr.step(pub['players'],t,pub['carbon'])
            if t>=2 and (t-2)%5==0 and t<=292:
                real,later=tape[t+1]
                common=dict(seed=ticket['seed'],map=ticket['map'],seat=seat,step=t+1,winner_candidate=won)
                root=dict(players=pub['players'],step=t,trees=pub['trees'],player=seat,
                          carbon=list(map(float,est)),remainingOverageTime=pub['remainingOverageTime'])
                projected=m._turn(root,cfg,command);ps=projected['players']
                ids=mapping(ps,real['players'])
                if ids is None:
                    miss.append(common|{"reason":"projected and actual occupied slots differ"})
                else:
                    x=m._quant(m.encode(ps,t+1,op,projected['carbon'],[1.]*m.N,pub['players'])).astype(np.float32)
                    lw,lb,_=m._RNET(x[None],m._SID)
                    lw,lb=lw[0],lb[0]
                    if m._RNET is not m._NET:
                        lw=lw/m._S_RTEMP;lb=lb/m._S_RTEMP
                    own,enemy=ps[seat],ps[op]
                    slots=[]
                    for k,(pos,cargo,role) in enemy[2].items():
                        target=m.WIDX[later[op].get(ids[op][k],'NONE')]
                        other_near=any(m.dist(pos,q[0])<=2 for q in own[2].values())
                        slots.append(dict(id=k,role=role,pos=pos,open=True,cargo=cargo,close=other_near,
                                          target=target,logits=np.array(lw[pos],np.float64)))
                    for k,pos in enemy[1].items():
                        slots.append(dict(id=k,role='BASE',pos=pos,open=enemy[0]>=30 and len(enemy[2])<10,
                                          target=m.BIDX[later[op].get(ids[op][k],'NONE')],logits=np.array(lb[pos],np.float64)))
                    own_dest={m.move(w[0],m.WIDX[later[seat].get(ids[seat][k],'NONE')]) for k,w in own[2].items()}
                    roots.append(common|dict(players=ps,op=op,lw=lw,lb=lb,slots=slots,own_dest=own_dest,
                            input_sha=sha(x.tobytes())))
            tr.set_actions(pub['players'],command[seat])
    return roots,miss


def fit(roots,roles):
    groups=defaultdict(list)
    for g in roots:
        groups[g["map"]].extend(s for s in g["slots"] if s["role"] in roles and s["open"])
    categories=[]
    for rows in groups.values():
        for role in roles:
            r=[s for s in rows if s["role"]==role]
            if r:
                categories.append((np.stack([s['logits'] for s in r]),np.array([s['target'] for s in r]),len(r)/len(rows)/len(groups)))
    def loss(beta):
        val=.001*(beta-1)**2
        for a,y,f in categories:
            z=beta*a
            val+=f*float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),y]))
        return val
    opt=minimize_scalar(loss,bounds=(.10,2.),method='bounded',options={'xatol':1e-9})
    return float(opt.x),dict(logit_NLL_penalized_before=loss(1.),after=loss(float(opt.x)),maps=len(groups),units=sum(map(len,groups.values())))


def events(m,g,slots,order):
    home=set(g['players'][g['op']][1].values());bank=intersect=recruit=False
    for s,a in zip(slots,order,strict=True):
        if s["role"]=="BASE":
            recruit|=bool(a)
        else:
            z=m.move(s["pos"],a)
            bank|=s["role"]=="COLLECTOR" and s["cargo"]>=50 and z in home
            intersect|=z in g["own_dest"]
    return int(bank)*4+int(intersect)*2+int(recruit)


def evaluate(m,roots,book):
    raw=[]
    for g in roots:
        vals={};slots=g["slots"]
        target=[s["target"] for s in slots]
        event=events(m,g,slots,target)
        for method,beta in book.items():
            lw,lb=adjust(g['players'],g['op'],g['lw'],g['lb'],beta)
            ys=[]; ev=[]
            seed=(1000003*g['step']+17*g['seat'])%(2**31)
            for rep in range(DRAWS):
                rng=np.random.RandomState(seed+7919+31*rep+2)
                cmd=m._sample(g['players'],g['op'],lw,lb,rng,g['step'])
                y=[(m.BIDX if s['role']=='BASE' else m.WIDX)[cmd.get(s['id'],'NONE')] for s in slots]
                ys.append(y); ev.append(events(m,g,slots,y))
            ys=np.array(ys)
            joint_freq=np.bincount(ev,minlength=8)/DRAWS
            joint_brier=float(np.square(joint_freq-np.eye(8)[event]).sum())
            sg=defaultdict(list)
            for j,s in enumerate(slots):
                if not s['open']:
                    continue
                n=len(s['logits']);y=s['target'];freq=np.bincount(ys[:,j],minlength=n)/DRAWS
                brier=float(np.square(freq-np.eye(n)[y]).sum())
                z=beta[s['role']]*s['logits']
                top=int(freq.argmax()==y)
                score=(brier,top,float(logsumexp(z)-z[y]))
                groups=['all',s['role'],'phase'+str(g['step']//100)]
                if s['role']!='BASE' and s['close']:groups.append('near_enemy')
                if s['role']=='COLLECTOR' and s['cargo']>=50:groups.append('loaded')
                for key in groups:sg[key].append(score)
            vals[method]=dict(group={key:dict(n=len(v),scores=np.mean(v,axis=0).tolist()) for key,v in sg.items()},
                              joint_brier=joint_brier,joint_freq=joint_freq.tolist(),
                              exact_draws=int(np.all(ys==np.array(target)[None],axis=1).sum()))
        raw.append({k:g[k] for k in ('seed','seat','map','step','winner_candidate','input_sha')}|
                  dict(physical_slots=len(slots),target_event=event,methods=vals))
    return raw


def aggregate(raw):
    per_map={}
    for b in sorted({r['map'] for r in raw}):
        rs=[r for r in raw if r['map']==b]
        per_map[b]={}
        for method in rs[0]['methods']:
            val=[r['methods'][method] for r in rs]
            segment={}
            for key in sorted(set().union(*(v['group'] for v in val))):
                elig=[v['group'][key] for v in val if key in v['group']]
                segment[key]=dict(roots=len(elig),units=sum(v['n'] for v in elig),
                    values=np.mean([v['scores'] for v in elig],axis=0).tolist())
            per_map[b][method]=dict(group=segment,joint_brier=float(np.mean([v['joint_brier'] for v in val])),
                    exact_draws=sum(v['exact_draws'] for v in val),roots=len(val))
    result={}
    for method in ('parent','shared','roles'):
        value=[d[method] for d in per_map.values()]
        keys=sorted(set.intersection(*(set(d['group']) for d in value)))
        result[method]=dict(group={k:np.mean([d['group'][k]['values'] for d in value],axis=0).tolist() for k in keys},
                            joint_brier=float(np.mean([d['joint_brier'] for d in value])),
                            exact=sum(d['exact_draws'] for d in value))
    return per_map,result


def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=('train','held'));p.add_argument('--weights-sha');a=p.parse_args()
    pr=(HERE/'protocol.json').read_bytes();assert sha(pr)==PROTOCOL
    cards=[c for c in json.loads(pr)['tickets'] if c['split']==a.split]
    if a.split=='held':
        wb=(HERE/'weights.json').read_bytes();assert a.weights_sha==sha(wb)
        w=json.loads(wb)
    m=initialize()
    roots,miss=collect(m,cards)
    if a.split=='train':
        beta,receipt=fit(roots,ROLES)
        by_role={r:fit(roots,[r]) for r in ROLES}
        result=dict(protocol=PROTOCOL,source=PARENT,backend=m._S_STATS['rnet'],
          roots=len(roots),skipped=miss,training_maps=sorted({g['map'] for g in roots}),
          betas=dict(shared=beta,roles={k:v[0] for k,v in by_role.items()}),
          fit=dict(shared=receipt,roles={k:v[1] for k,v in by_role.items()}))
        print('weights',put(HERE/'weights.json',result),result['betas'],len(roots),len(miss))
    else:
        book={'parent':dict.fromkeys(ROLES,1.),'shared':dict.fromkeys(ROLES,w['betas']['shared']),'roles':w['betas']['roles']}
        raw=evaluate(m,roots,book);maps,summary=aggregate(raw)
        win_maps=sum(v['roles']['group']['all']['values'][0] < v['parent']['group']['all']['values'][0] for v in maps.values())
        coverage=len(raw)/(len(raw)+len(miss))
        assert len(maps)==4 and len({(g['seed'],g['seat']) for g in raw})==8
        checks=dict(coverage=coverage>=.95,three_maps=win_maps>=3,
            better_parent=summary['roles']['group']['all'][0]<summary['parent']['group']['all'][0],
            no_worse_shared=summary['roles']['group']['all'][0]<=summary['shared']['group']['all'][0],
            no_worse_joint=summary['roles']['joint_brier']<=summary['parent']['joint_brier'])
        result=dict(weight_sha256=a.weights_sha,protocol=PROTOCOL,source=PARENT,backend=m._S_STATS['rnet'],roots=len(raw),
          skipped=miss,coverage=coverage,draws_per_root=DRAWS,metrics=['physical Brier','top1 empirical probability','predecode unit NLL'],
          by_map=maps,summary=summary,improved_maps=int(win_maps),criteria=checks,verdict='PASS' if all(checks.values()) else 'FAIL',
          panels={})
        for outcome in (False,True):
            sub=[r for r in raw if r['winner_candidate']==outcome]
            if sub:
                result['panels'][str(outcome)]=dict(roots=len(sub),maps=len({r['map'] for r in sub}),metrics=aggregate(sub)[1])
        put(HERE/'held_roots.json.gz',raw)
        print('held',put(HERE/'held.json',result),result['verdict'],coverage,
              {k:(d['group']['all'],d['joint_brier']) for k,d in summary.items()})
if __name__=='__main__':main()

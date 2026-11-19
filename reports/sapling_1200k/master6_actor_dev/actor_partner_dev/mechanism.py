"""Observe each physical seat independently and check actual package entry and same-prefix legality."""
from collections import Counter
import gzip
import importlib.util
import json
import sys

import numpy as np
from build import BASE,ROOT,HERE,CAND,PARENT,build,digest,out

sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def module(path,name,weight):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m._HYBRID['student_weight']=weight
    orig=m._mixed_actions
    def checked_actions(x,aux):
        labels=orig(x,aux)
        prefix=sys.modules['cnet2_feat'].Prefix(x,aux)
        active=np.flatnonzero(x['tok_mask'][:sys.modules['cnet2_feat'].OWN])
        roles={}
        for k in active:
            move=int(labels[k]);_,_,legal=prefix.feats(k)
            if not legal[move]:raise ValueError('illegal joint prefix')
            roles[int(k)]={'role':'base' if k>=10 else 'planter' if x['tok_kind'][k] else 'collector','action':move}
            prefix.advance(k,move)
        m.audited=roles
        return labels
    m._mixed_actions=checked_actions
    return m


def verify():
    release=build()
    plan=read(HERE/'protocol.json')
    ledger=read(HERE.parent/'tree_response_training/source_receipt.json')
    roots=[];totals=Counter()
    keys = ['parent','candidate','no_sapling']
    for r in ledger['games']:
        if r['seed'] not in plan['mechanism_maps']:continue
        mods=[module(p,name,w) for p,name,w in [(PARENT,'old',.25),(CAND,'new',.25),(CAND,'without',0)]]
        path=ROOT/r['raw_path']
        if digest(path)!=r['raw_sha256']:raise ValueError('exposed trajectory changed')
        raw=read(path);seat=r['seat']
        sim=FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
        for t,joint in enumerate(raw['actions']):
            obs=sim.obs(seat)
            if t in plan['mechanism_steps']:
                acts=[m.agent(obs,sim.config) for m in mods]
                if acts[0]!=joint[seat]:raise ValueError(f'parent disagrees {seat}/{t}')
                line=dict(seed=r['seed'],seat=seat,t=t,source_sha256=r['raw_sha256'],commands=dict(zip(keys,acts)),slots=mods[1].audited)
                for other,kind in [(0,'switch'),(2,'student')]:
                    changes=[(i,x['role']) for i,x in mods[1].audited.items() if x['action']!=mods[other].audited[i]['action']]
                    line[kind+'_changed_slots']=changes
                    totals[f'{seat}/{kind}/changed_roots']+=bool(changes)
                    for _,role in changes:totals[f'{seat}/{kind}/{role}']+=1
                roots.append(line)
            else:
                for m in mods:m._FEAT.observe(obs)
            sim.step(*joint)
    if len(roots)!=len(plan['mechanism_steps'])*2:raise ValueError('incomplete seats or times')
    passed=all(totals[f'{s}/{k}/changed_roots']>0 for s in (0,1) for k in ('switch','student'))
    out(HERE/'mechanism.json',dict(passed=passed,protocol_sha256=release['protocol_sha256'],candidate_sha256=digest(CAND),
       source_sha256=digest(HERE/'mechanism.py'),totals=dict(totals),roots=roots))
    print('mechanism',passed,dict(totals))


if __name__=='__main__':
    verify()

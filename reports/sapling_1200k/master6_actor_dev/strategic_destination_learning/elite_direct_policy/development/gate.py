"""Read-only wakeup decision for the exact elite policy. This program neither generates maps nor dispatches."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

repo=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
strategy=Path('reports/sapling_1200k/master6_actor_dev/strategic_destination_learning')
lane=strategy/'elite_direct_policy'
protocol_sha='d373ccae0808919790786132824c893d762410747e95c45fea9fe4f333fd9aae'
control_sha='ede48ad9770d357cbe4a4be6c24dafebc910be1e46a95c508c706e009fd713a0'
original_sha='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
h=lambda data:hashlib.sha256(data).hexdigest()

def read(path,rev):
    if rev:
        v=subprocess.run(['git','-C',str(repo),'show',rev+':'+str(path)],capture_output=True)
        if v.returncode:raise FileNotFoundError(str(path))
        return v.stdout
    return (repo/path).read_bytes()

def need(x,s):
    if not x:raise ValueError(s)

def validate(rev):
    rd=lambda p:read(lane/p,rev)
    raw=rd('protocol.json');p=json.loads(raw)
    need(h(raw)==protocol_sha and p['starting_actor_package']==control_sha,'Wrong elite learner preregistration')
    data=rd('assessment.json');a=json.loads(data)
    fb=rd('fit.json');fit=json.loads(fb);hr=rd('assessment_actions.json.gz');targets=json.loads(gzip.decompress(hr))
    cand=rd('trained.py');weight=rd('weights.npz')
    need(a['fit_sha256']==h(fb) and a['assessment_actions_sha256']==h(hr) and
         h(cand)==fit['package']==a['package_sha256']!=control_sha and len(cand)<=10_000_000 and
         h(weight)==fit['weights'] and a['protocol_sha256']==fit['protocol_sha256']==protocol_sha and
         fit['base']==control_sha and h(read(strategy/'bc.py',rev))==control_sha and
         h(rd('fit.py'))==fit['trainer_sha256'] and fit['roots']==320 and fit['updates']==40,
         'Different signed training, weights, assessment, or package')
    need(a['original_matches']==0 and a['protected_tapes_opened']==0 and
         len(p['entries']['fit'])==5 and len(p['entries']['assessment'])==6 and len(targets)==192,
         'Different data, protected exposure or prior matches')
    regs={(x['seed'],x['seat'],t) for x in p['entries']['assessment'] for t in x['roots']}
    need(len(regs)==192 and {(x['seed'],x['seat'],x['t']) for x in targets}==regs,'Mismatched assessment roots')
    v=a['assessment']; cs=Counter();summary=[]
    for m in p['entries']['assessment']:
        sr=[x for x in targets if x['seed']==m['seed']]
        av={};ac={}
        for k in ('bc','trained'):
            av[k]=sum(sum(r['policies'][k]['nll'].values()) for r in sr)/len(sr)
            ac[k]=sum(r['target'].get(uid,'NONE')==r['policies'][k]['action'].get(uid,'NONE') for r in sr for uid in r['roles'])/sum(len(r['roles']) for r in sr)
        cs[m['seat']]+=sum(r['policies']['bc']['action']!=r['policies']['trained']['action'] for r in sr)
        summary.append((av['trained']-av['bc'],ac['trained']-ac['bc']))
    checks={'majority':sum(x<0 for x,y in summary)>3,'map_equal_nll':sum(x for x,y in summary)<0,
            'map_equal_entity':sum(y for x,y in summary)>0,'both_seats_changed':cs[0]>0 and cs[1]>0}
    need(a['checks']==checks and a['eligible_for_development_consideration'] is all(checks.values()),'Released gate differs from 192 signed rows')
    return {'status':('ELIGIBLE' if rev else 'ELIGIBLE_LOCALLY_AWAIT_PUBLISHED_MAIN') if all(checks.values()) else 'HOLD_GATE_FAILED',
            'checks':checks,'package_sha256':h(cand),'weights_sha256':h(weight),'base':control_sha,
            'assessor':original_sha,'assessment_sha256':h(data),'fit_sha256':h(fb),'release':rev,'map_generation':0,'games':0}

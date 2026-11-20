"""Diagnostic search replay with the old live call's admitted candidate count, depth and rolls."""
import copy
import gzip
import json
import os
import numpy as np
import torch
from probe import HERE, PAIR, cases, e, member_report, sha

def main():
    assert len(os.sched_getaffinity(0))==1
    os.environ['CARBON_DEBUG']='1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evidence=HERE/'observations.json.gz'
    records=json.loads(gzip.decompress(evidence.read_bytes()))['rows']
    models={}
    for name,(path,expected) in PAIR.items():
        assert sha(path.read_bytes())==expected
        m=e.module('fixed_draw_'+name,path)
        m._net()
        models[name]=(m,e.N.control_fn(m,path.read_bytes()))
    parent=models['v2'][0]
    e.N.fresh_stats=copy.deepcopy(parent._S_STATS)
    roots=[]
    for (r,reg,goals,_,_,_), src in zip(cases(),records,strict=True):
        s=r['seat'];t=r['step']
        assert (r['seed'],s,t)==(src['seed'],src['seat'],src['step'])
        state=e.setup(r,False)
        e.N.history_state(parent,state,s,reg)
        saved=e.N.snap(parent)
        results=[]
        for i,obsdraw in enumerate(src['draws'][:-1]):
            observed=obsdraw['arms']['v2'];clock=observed['clock']
            cap,deep,rolls=(clock[k] for k in ('K','H','rolls'))
            draws={}
            for key in (('v2','child') if i%2==0 else ('child','v2')):
                m,controlled=models[key]
                e.N.restore(m,saved,state['cfg'])
                m._ST['rng']=np.random.RandomState(obsdraw['key'])
                old=m._search; max_rolls=m._S_MMAX
                try:
                    m._search=controlled
                    m._response_cap=cap;m._response_depth=deep;m._S_MMAX=rolls
                    cmd,actual,tr=e.sample(m,state['public'][s],state['cfg'])
                finally:
                    m._search=old;m._S_MMAX=max_rolls
                costs=[actual[n] for n in ('K','H','rolls')]
                draws[key]=dict(cmd=cmd,menu=tr['menu'],costs=costs,depth=tr['depths'],
                                **member_report(tr['menu'],cmd,goals,src['changed_units']))
            a,b=[draws[k] for k in ('v2','child')]
            restore=(a['cmd']==observed['cmd'] and a['menu']==observed['menu']
                      and {str(k):v for k,v in a['depth'].items()}==observed['depth'])
            equal=(a['costs']==b['costs']==[cap,deep,rolls]
                   and sorted(a['depth'].values())==sorted(b['depth'].values()))
            results.append(dict(key=obsdraw['key'],arms=draws,parent_reproduced=restore,matched_work=equal))
        sel=src['best']
        usable=[z for z in results if z['parent_reproduced'] and z['matched_work']]
        info=dict(registry=[r['seed'],s,t],n=len(usable),diff=sum(z['arms']['v2']['cmd']!=z['arms']['child']['cmd'] for z in usable),
                  hits={k:sum(z['arms'][k]['issued_matches'][sel] for z in usable) for k in models},
                  menu={k:sum(z['arms'][k]['menu_matches'][sel] for z in usable) for k in models})
        print(info,flush=True)
        roots.append(dict(report=info,draws=results))
    raw=json.dumps(dict(native_evidence_sha256=sha(evidence.read_bytes()),rows=roots),allow_nan=False,separators=(',',':')).encode()
    (HERE/'fixed_work.json.gz').write_bytes(gzip.compress(raw,mtime=0))
if __name__=='__main__':main()

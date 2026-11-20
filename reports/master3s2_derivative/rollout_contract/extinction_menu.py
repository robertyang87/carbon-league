"""Score all 14 last-decision public seats in the archived seven natural extinctions."""
import copy,csv,gzip,json,os,sys,types
from pathlib import Path
from extinction import HERE,build
from probe import base,rr,fp
def public(fr,i):
    shared=fr[0]['observation']
    v=dict(step=shared['step'],player=i,players=shared['players'],trees=shared['trees'],carbon=fr[i]['observation']['carbon'],
           remainingOverageTime=fr[i]['observation']['remainingOverageTime'])
    assert 'full_carbon' not in v;return v
def diagnostics(raw,name):
    src=raw.decode().replace('\r\n','\n')
    if name=='repair':
        key='            c = [penalty if gone[s] else float(b[3][s][0]) for s in (0, 1)]'
        assert src.count(key)==1
        src=src.replace(key,key+"\n            _EX_SEEN.append((int(b[0]),int(b[1]),state['step'],c))")
    m=types.ModuleType(name);exec(compile(src,name,'exec'),m.__dict__);m._EX_SEEN=[];return m
def main(cache):
    assert os.sched_getaffinity(0)=={4}
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    master=diagnostics((HERE.parent/'cold_start/candidate.py').read_bytes(),'stock');alt=diagnostics(build(),'repair')
    data=json.loads((HERE/'extinction_check.json').read_bytes())
    panel=[]
    for r in data['rows']:
        raw=(cache/(r['id']+'.json.gz')).read_bytes();assert fp(raw)==r['sha256']
        j=json.loads(gzip.decompress(raw));fr=j['steps'];seed=j['configuration']['randomSeed'];t=r['t'];cfg=j['configuration']
        for s in (0,1):
            tracker=master.BeliefTracker(master._Bel,s);prev=None;cost=[0.]*4
            for step in range(t):
                obs=public(fr[step],s);p=obs['players']
                if step>=t-4:
                    replay=rr.call(master,(tracker,prev),obs,cfg,(seed+step*193+s)%2**32,cost,'P')
                    cost=replay['clocks']['cost']
                tracker.step(p,step,obs['carbon']);tracker.set_actions(p,fr[step+1][s]['action'] or {});prev=p
            calls=[];ob=public(fr[t],s)
            for m in [master,alt,alt,master] if s==0 else [alt,master,master,alt]:
                m._EX_SEEN=[];rec=rr.call(m,(tracker,prev),ob,cfg,(seed+t*193+s)%2**32,cost,'P')
                calls.append(dict(variant='stock' if m is master else 'repair',extinctions_seen=m._EX_SEEN,**{k:rec[k] for k in ['action','menu','depths','values','selected','clocks']}))
            q=[x for x in calls if x['variant']=='stock'];a=[x for x in calls if x['variant']=='repair']
            agree=lambda ls: all(ls[0][k]==ls[1][k] for k in ['action','selected','menu','depths','values'])
            matched=all(c['menu']==calls[0]['menu'] and c['clocks']['K']==calls[0]['clocks']['K'] and c['clocks']['H']==calls[0]['clocks']['H'] for c in calls)
            row=dict(id=r['id'],t=t,seat=s,public_hash=base.digest(ob),costs=cost,detected=any(c['extinctions_seen'] for c in a),matched=matched,
                     stable=agree(q) and agree(a),changed=q[0]['action']!=a[0]['action'],calls=calls)
            panel.append(row);print(r['id'][:8],s,row['detected'],row['matched'],row['stable'],row['changed'],flush=True)
    rr.save(HERE/'extinction_menu.json.gz',dict(candidate=fp(build()),scope='All 14 historical pre-extinction final public seat observations, 4 preceding source calls calibrate cost, local native .58s ABBA.',panel=panel))
if __name__=='__main__':main(Path(sys.argv[1]))

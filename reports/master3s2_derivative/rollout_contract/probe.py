"""Measure archived own-seat public roots without full games or retuning the opponent."""
import copy,gzip,hashlib,importlib.util,json,os,sys,zipfile
from pathlib import Path
from build import HERE,PARENT,build
def src(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
rr=src('contracts_pair',HERE.parent/'return_rank/development/root_responses.py')
base=rr.base;np=base.np
def fp(x):return hashlib.sha256(x).hexdigest()
def main(path):
    p=(HERE/'protocol.json').read_bytes();plan=json.loads(p)
    assert os.sched_getaffinity(0)=={4} and fp(build())==fp(path.read_bytes())==plan['candidate']
    import torch
    torch.set_num_threads(1);os.environ['CARBON_DEBUG']='1'
    parent=base.load('own_future_original',PARENT);alt=base.load('own_future_return',path)
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    rows=[]
    for archive_id in plan['archives']:
        archive=HERE.parent/'cold_start/development/hosted'/(archive_id+'.zip')
        with zipfile.ZipFile(archive) as z:
            reg=json.loads(z.read('cold_repair/registration.json'));receipt=json.loads(z.read('cold_repair/receipt.json'))
            frames=gzip.decompress(z.read('cold_repair/observations.jsonl.gz')).splitlines();orders=z.read('cold_repair/turns.jsonl')
            assert fp(orders)==receipt['turns_sha256'] and reg['files']['candidate']['sha256']==plan['parent']
        seat=reg['candidate_seat'];seed=reg['seed'];commands=orders.splitlines()
        tracker=parent.BeliefTracker(parent._Bel,seat);prev=None
        for t,raw in enumerate(frames):
            ob=json.loads(raw)['observations'][seat];q=json.loads(commands[t]);a=q['calls'][seat]['response']['action']
            assert base.digest(ob)==q['public_observation_sha256'][seat] and 'full_carbon' not in ob
            if t in plan['steps']:
                reseed=seed+t*193+seat;calls=[]
                for name in ('PAAP' if len(rows)%2==0 else 'APPA'):
                    m=parent if name=='P' else alt
                    r=rr.call(m,(tracker,prev),ob,cfg,reseed,plan['costs'],'P')
                    calls.append({k:v for k,v in r.items() if k not in ('mode','trace')}|dict(variant=name))
                pa=[c for c in calls if c['variant']=='P'];aa=[c for c in calls if c['variant']=='A']
                stable=lambda ls: all(ls[0][k]==ls[1][k] for k in ['menu','selected','action','depths'])
                first=pa[0];new=aa[0]
                same_work=all(r['clocks']['K']==first['clocks']['K'] and r['clocks']['H']==first['clocks']['H'] and r['clocks']['rolls']==first['clocks']['rolls'] and r['menu']==first['menu'] for r in calls)
                strict=same_work and all(r['depths']==first['depths'] for r in calls)
                row=dict(seed=seed,seat=seat,t=t,public_sha256=base.digest(ob),source=fp(archive.read_bytes()),rng=reseed,action_real=a,calls=calls,
                          stable=stable(pa) and stable(aa),same_root_menu_and_depth=same_work,same_each_candidate_depth=strict,
                          changed=first['action']!=new['action'],value_diff_any=first['menu']==new['menu'] and first['values']!=new['values'])
                rows.append(row)
                print(seed,seat,t,row['changed'],row['stable'],same_work,strict,flush=True)
            tracker.step(ob['players'],t,ob['carbon']);tracker.set_actions(ob['players'],a);prev=ob['players']
    rr.save(HERE/'roots.json.gz',dict(protocol=fp(p),rows=rows))
if __name__=='__main__':main(Path(sys.argv[1]))

import sys, subprocess, json, gzip, time, hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from replay import digest, receive, FastCarbon
import numpy as np

def measure(p,arm,limit):
    tape=H.parents[1]/'direct_original/games/candidate/raw_621758345_1.json.gz'
    pack=H.parents[1]/'candidate.py' if arm=='initial' else H/'candidate.py'
    assert digest(pack)==p[arm] and digest(tape)==p['tape']
    signed=json.loads(gzip.decompress(tape.read_bytes()))
    cfg=signed['game_configuration']
    sim=FastCarbon(cfg).reset(full_carbon=signed['initial_full_carbon'])
    cmd=['docker','run','--rm','-i','--network','none','-e','OPENBLAS_NUM_THREADS=4','-v',f'{pack}:/candidate.py:ro','-v',f'{H}/worker_dual.py:/worker.py:ro',p['image'],'python','-u','/worker.py','/candidate.py',str(limit)]
    start=time.perf_counter()
    r=dict(arm=arm,limit_mib=limit,steps=[])
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0) as proc:
        try:
            ready=receive(proc,25)
            r.update(ready=ready,external_start_s=time.perf_counter()-start)
            assert ready['sha256']==p[arm] and ready['inherited']=='4'
            for i, pair in enumerate(signed['actions'][:32]):
                obs=sim.obs(1)
                obs['remainingOverageTime']=5-sum(max(0.,x['s']-1) for x in r['steps'])
                t=time.perf_counter()
                proc.stdin.write(json.dumps(dict(observation=obs,configuration=cfg),separators=(',',':')).encode()+b'\n')
                act=receive(proc,9)
                dt=time.perf_counter()-t
                assert act['step']==i
                r['steps'].append(dict(t=i,s=dt,action=act['action'],match=act['action']==pair[1],rss_kib=act['rss_kib']))
                sim.step(*pair)
            proc.stdin.close()
            r['exit']=proc.wait(timeout=10)
        except (RuntimeError,TimeoutError,BrokenPipeError) as err:
            r['exception']=str(err)
            if proc.poll() is None:
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill()
            r['exit']=proc.wait()
        r['stderr']=proc.stderr.read().decode(errors='replace')
    v=[x['s'] for x in r['steps']]
    r.update(wall_s=time.perf_counter()-start,complete=(len(v)==32 and r['exit']==0))
    if v:
        r.update(match=sum(x['match'] for x in r['steps']),median=float(np.median(v)),max=max(v),p95=float(np.percentile(v,95)),rss_kib=max(x['rss_kib'] for x in r['steps']))
    path=H/f'dual_{arm}_{limit}.json'
    assert not path.exists()
    path.write_text(json.dumps(r,indent=2)+'\n')
    print(arm,limit,r['complete'],r.get('ready',{}).get('actual_threads'),r.get('p95'),r.get('max'),r['stderr'][-500:],flush=True)
    return r

if __name__=='__main__':
    p=json.loads((H/'protocol_dual.json').read_text())
    cap=[measure(p,a,512) for a in ('initial','fixed')]
    if not all(r['complete'] for r in cap):
        for a in ('fixed','initial'):measure(p,a,0)

"""Local JSONL external latency from new container startup through a signed full tape."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT/'code'))
from fastsim import FastCarbon

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def receive(proc, timeout):
    with selectors.DefaultSelector() as sel:
        sel.register(proc.stdout, selectors.EVENT_READ)
        if not sel.select(timeout):
            raise TimeoutError('container protocol reply')
        line = proc.stdout.readline()
    if not line:
        raise RuntimeError('container EOF')
    return json.loads(line)

def collect(path, p):
    signed = json.loads(gzip.decompress(path.read_bytes()))
    s = signed['candidate_seat']
    cfg = signed['game_configuration']
    engine = FastCarbon(cfg).reset(full_carbon=signed['initial_full_carbon'])
    pack = HERE.parent/'candidate.py'
    worker = HERE/'worker.py'
    assert digest(path)==p['raw'][path.name] and digest(pack)==p['candidate'] and digest(worker)==p['worker']
    cmd = ['docker','run','--rm','-i','--network','none','--cpuset-cpus','2,3',
           '--cpus','1.5','--memory','512m','--memory-swap','512m',
           '-v',f'{pack}:/candidate.py:ro','-v',f'{worker}:/worker.py:ro',
           p['image'],'python','-u','/worker.py','/candidate.py','2','512']
    start = time.perf_counter()
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0) as child:
        try:
            ready = receive(child,25)
            init = time.perf_counter()-start
            assert ready['sha256']==p['candidate'] and ready['numpy']=='1.26.4' and ready['affinity']==[2]
            rows=[]
            extra=0.
            for t, pair in enumerate(signed['actions']):
                obs=engine.obs(s)
                obs['remainingOverageTime']=max(0.,5-extra)
                before=time.perf_counter()
                payload=json.dumps(dict(observation=obs,configuration=cfg),separators=(',',':'),allow_nan=False).encode()+b'\n'
                child.stdin.write(payload)
                child.stdin.flush()
                reply=receive(child,8)
                elapsed=time.perf_counter()-before
                assert reply['step']==t
                extra+=max(0.,elapsed-1)
                rows.append(dict(t=t,external_s=elapsed,obs_and_cfg_bytes=len(payload),
                                 matches_tape=reply['action']==pair[s],peak_rss_kib=reply['peak_rss_kib']))
                engine.step(*pair)
            child.stdin.close()
            exitcode=child.wait(timeout=15)
        finally:
            if child.poll() is None: child.kill()
    wall=time.perf_counter()-start
    assert exitcode==0 and np.allclose(engine.cash,signed['last_post_cash'],rtol=0,atol=.011)
    sec=[r['external_s'] for r in rows]
    slow=sum(v>1 for v in sec)
    return dict(seed=signed['seed'],seat=s,ready=ready,wall_start_to_process_exit_s=wall,init_external_s=init,
                max_s=max(sec),p95_s=float(np.percentile(sec,95)),p99_s=float(np.percentile(sec,99)),
                slow_steps=slow,sum_excess_s=extra,peak_rss_kib=max(r['peak_rss_kib'] for r in rows),n=len(rows),
                matches_tape=sum(r['matches_tape'] for r in rows),final_cash=engine.cash,
                local_thresholds_pass=(init<=20 and max(sec)<=3 and slow<=3 and extra<=5 and wall<=1500),
                steps=rows)

if __name__=='__main__':
    p=json.loads((HERE/'protocol.json').read_text())
    for file in p['raw']:
        out=HERE/file.replace('raw_','timing_')
        assert not out.exists(),out
        rec=collect(HERE.parent/'direct_original/games/candidate'/file,p)
        tmp=out.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
        tmp.replace(out)
        print(rec['seed'],rec['seat'],rec['n'],rec['matches_tape'],rec['max_s'],rec['init_external_s'],rec['local_thresholds_pass'],flush=True)

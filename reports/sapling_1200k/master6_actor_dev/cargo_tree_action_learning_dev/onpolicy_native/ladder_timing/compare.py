"""Freeze and replay public observation tapes against any deterministic single-file agent."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'ladder_timing'
sys.path.insert(0,str(OLD))
from replay import ROOT, FastCarbon, digest, receive

def write(path, data):
    encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if path.suffix=='.gz':
        encoded=gzip.compress(encoded,mtime=0)
    with path.open('xb') as stream:
        stream.write(encoded)

def bind(path):
    path=path.resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path))

def verified(binding):
    path=ROOT/binding['path']
    assert digest(path)==binding['sha256'],path
    return path

def register(args):
    pack=bind(args.pack)
    if pack['sha256']!=args.expect:raise ValueError('unexpected candidate bytes')
    cards=[]
    for path in args.tapes:
        row=json.loads(gzip.decompress(path.read_bytes()))
        if len(row['actions'])!=row['steps']:raise ValueError('truncated tape')
        cards.append(dict(**bind(path),seed=row['seed'],seat=row['candidate_seat'],steps=row['steps'],
                          historical_agent=row['pack_sha256']))
    if len({(c['seed'],c['seat']) for c in cards})!=len(cards):raise ValueError('duplicate tape')
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    write(args.out/'protocol.json',dict(registered=datetime.now(timezone.utc).isoformat(),pack=pack,
          cards=cards,image=image,worker=bind(OLD/'worker.py'),tool=bind(Path(__file__)),
          replay_library=bind(OLD/'replay.py'),sim=bind(ROOT/'code/fastsim.py'),host_ref_cpu=3,child_cpu=2,
          reg='Each tape from step0 to its stated terminal. Selection fixed before calls. New, persistent container per tape. No warmup. Exact frozen public JSON through IPC. After timed container exit, load same signed bytes in fresh ordinary host agent module and call every step on the exact same JSON in temporal order. Retain all differences against independent reference and against actual historic policy. Time from cold container launch to READY, each call from before serialization to decoded action, total until child exits. No new opponent or official win. 1s soft,3s hard,3 slow steps,5s overage; init20s, wall1500s; affinity and process AS cap only.'))

def run_one(p,card):
    pack=verified(p['pack'])
    raw=json.loads(gzip.decompress(verified(card).read_bytes()))
    worker=verified(p['worker'])
    engine=FastCarbon(raw['game_configuration']).reset(full_carbon=raw['initial_full_carbon'])
    cli=['docker','run','--rm','-i','--network','none','--cpus','1.5','--memory','512m',
         '--memory-swap','512m','--cpuset-cpus','2,3','-v',f'{pack}:/candidate.py:ro',
         '-v',f'{worker}:/worker.py:ro',p['image'],'python','-u','/worker.py','/candidate.py',str(p['child_cpu']),'512']
    cfg=raw['game_configuration']
    rows=[];inputs=[];over=0
    start=time.perf_counter()
    with subprocess.Popen(cli,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0) as proc:
        try:
            ready=receive(proc,30)
            init=time.perf_counter()-start
            assert ready['sha256']==p['pack']['sha256']
            for t,pair in enumerate(raw['actions']):
                observation=engine.obs(card['seat'])
                observation['remainingOverageTime']=max(0.,5-over)
                wall=time.perf_counter()
                msg=json.dumps(dict(observation=observation,configuration=cfg),separators=(',',':'),allow_nan=False).encode()+b'\n'
                proc.stdin.write(msg);proc.stdin.flush()
                reply=receive(proc,8)
                elapsed=time.perf_counter()-wall
                assert reply['step']==t
                over+=max(0.,elapsed-1)
                rows.append(dict(t=t,external_s=elapsed,peak_rss_kib=reply['peak_rss_kib'],action=reply['action'],
                                 historical=pair[card['seat']],request_sha256=hashlib.sha256(msg).hexdigest()))
                inputs.append(msg)
                engine.step(*pair)
            proc.stdin.close()
            assert proc.wait(timeout=15)==0
        finally:
            if proc.poll() is None:proc.kill()
    duration=time.perf_counter()-start
    # All independent reference calls happen after the timed process exited.
    ref=ModuleType('ordinary_public_reference')
    ref.__file__=str(pack)
    sys.modules[ref.__name__]=ref
    exec(compile(pack.read_bytes(),str(pack),'exec'),ref.__dict__)
    for r,msg in zip(rows,inputs,strict=True):
        q=json.loads(msg);expected=ref.agent(q['observation'],q['configuration'])
        r.update(reference=expected,matches_reference=(r['action']==expected),matches_historical=(r['action']==r['historical']))
    durations=[r['external_s'] for r in rows]
    peak=max(ready['peak_rss_kib'],max(r['peak_rss_kib'] for r in rows))
    slow=sum(x>1 for x in durations)
    n=len(rows)
    observed=dict(n=n,ref_matches=sum(r['matches_reference'] for r in rows),historical_matches=sum(r['matches_historical'] for r in rows),
                  slow=slow,excess_s=over,init_s=init,replay_wall_s=duration,rss_mib=peak/1024,
                  first_s=durations[0],p50_s=float(np.median(durations)),p95_s=float(np.percentile(durations,95)),
                  p99_s=float(np.percentile(durations,99)),max_s=max(durations))
    checks=dict(coverage=n==card['steps']==raw['steps'] and all(r['t']==i for i,r in enumerate(rows)),ref=observed['ref_matches']==n,
                horizon=bool(np.allclose(engine.cash,raw['last_post_cash'],rtol=0,atol=.011)),
                init=init<=20,slow=slow<=3,excess=over<=5,hard=max(durations)<=3,total=duration<=1500,
                rss=peak<=512*1024,size=0<pack.stat().st_size<=10000000,
                runtime=ready['python'].startswith('3.12.') and ready['numpy']=='1.26.4',
                local_resources=ready['affinity']==[p['child_cpu']] and ready['rlimit_as']==[512*1024**2]*2)
    return dict(card=card,ready=ready,ref_env=dict(python=sys.version,numpy=np.__version__,affinity=sorted(os.sched_getaffinity(0))),
                stats=observed,checks=checks,local_pass=all(checks.values()),formal_live_match=False,formal_cgroup=False,rows=rows)

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('phase',choices=('register','run'));args.add_argument('--out',type=Path,required=True)
    args.add_argument('--pack',type=Path);args.add_argument('--expect');args.add_argument('--image',default='sapling-ladder-312:local')
    args.add_argument('--tapes',nargs='+',type=Path)
    a=args.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=='register':register(a);sys.exit()
    p=json.loads((a.out/'protocol.json').read_text())
    for k in ('tool','replay_library','sim'):verified(p[k])
    assert os.sched_getaffinity(0)=={p['host_ref_cpu']}
    for c in p['cards']:
        label=f"receipt_{c['seed']}_{c['seat']}.json.gz"
        receipt=run_one(p,c)
        write(a.out/label,receipt)
        print(label,receipt['stats'],receipt['checks'],flush=True)

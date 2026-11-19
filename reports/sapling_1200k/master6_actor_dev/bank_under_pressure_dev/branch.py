"""Pair each seat's latest actionable development root against its original order."""
import sys
import subprocess
import fcntl
from bank import HERE
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr

def prepare():
    for seat in (0,1):
        t,_,case,root=max((r['t'],c['seed'],c,r) for c in sr.load(HERE/'audit.json') if c['seat']==seat for r in c['hits'])
        path=sr.ROOT/case['source']
        receipt=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert sr.sha(path)==receipt['game']['raw_sha256']==case['sha256']
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=case['seed'],seat=seat,root_t=t,family='bank_under_pressure',
            source_dir=str(out.relative_to(sr.ROOT)),record=dict(seed=case['seed'],seat=seat,source=case['source'],
            source_sha256=case['sha256']),root=root,pack_sha256=sr.load(path)['pack_sha256']))
        sr.live.prepare(out)

if __name__=='__main__':
    runner.HERE=HERE; runner.PRIOR=HERE.parent/'planter_capture_full_dev'
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:runner.play(int(sys.argv[1]),sys.argv[2])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([p.wait() for p in jobs]):raise RuntimeError('worker failed')

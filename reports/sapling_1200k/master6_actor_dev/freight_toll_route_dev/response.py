"""Paired diagnostic for each available seat using its actual source parent."""
from pathlib import Path
import sys,subprocess,fcntl
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr
def setup():
    groups=sr.load(HERE/'opportunities.json.gz')
    for s in (0,1):
        r=next(row for row in groups if row['seat']==s)
        p=sr.ROOT/r['path'];raw=sr.load(p);rec=sr.load(p.with_name(p.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert sr.sha(p)==rec['game']['raw_sha256']==r['source_sha256']
        out=HERE/f'seat{s}';out.mkdir(exist_ok=True);c=r['choice']
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[rec['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=r['seed'],seat=s,root_t=r['t'],family='freight_toll',
                 source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=raw['pack_sha256'],
                 record=dict(seed=r['seed'],seat=s,source=r['path'],source_sha256=sr.sha(p)),
                 root=dict(t=r['t'],worker=c['worker'],target='',tree='',old=c['old'],alternative=c['alternative'])))
        sr.live.prepare(out)
def play(s,arm):
    runner.HERE=HERE;runner.PRIOR=HERE.parent/('planter_capture_full_dev' if s==0 else 'midgame_role_value_dev')
    runner.play(s,arm)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:setup()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for s in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((i+s)%2),sys.executable,__file__,str(s),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([j.wait() for j in jobs]):raise RuntimeError('original continuation failed')

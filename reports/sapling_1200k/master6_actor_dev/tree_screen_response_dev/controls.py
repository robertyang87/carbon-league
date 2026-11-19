"""The first unconditional trigger and an opposite-seat case on another map, run to terminal."""
from pathlib import Path
import fcntl
import subprocess
import sys
from screen_trial import HERE,sr,runner

def select(label):
    rows=sr.load(HERE.parent/'tactical_exchange_dev/approach.json.gz')
    if label=='first_trigger':return rows[0]
    return next(r for r in rows if r['qualifies'] and r['seed']!=621758328 and r['seat']==1
                and sr.load(sr.ROOT/r['source'])['pack_sha256']==sr.sha(runner.PRIOR/'candidate.py'))

def setup(label):
    r=select(label);out=HERE/label/f"seat{r['seat']}";out.mkdir(parents=True,exist_ok=True)
    path=sr.ROOT/r["source"];raw=sr.load(path)
    receipt=sr.load(path.with_name(path.name.replace("raw_","receipt_").replace(".json.gz",".json")))
    assert sr.sha(path)==r["source_sha256"]==receipt["game"]["raw_sha256"]
    assert raw["pack_sha256"]==sr.sha(runner.PRIOR/"candidate.py")
    c=r['choice'];sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
    sr.retain(out/'live_protocol.json',dict(seed=r['seed'],seat=r['seat'],root_t=r['t'],family=label,
         source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=raw['pack_sha256'],
         record=dict(seed=r['seed'],seat=r['seat'],source=r['source'],source_sha256=sr.sha(path)),
         root=dict(t=r['t'],worker=c['uid'],target=c['enemy'],tree=c['tree'],old=c['before'] or 'NONE',alternative=c['after'])))
    sr.live.prepare(out)
def run():
    for label in ('first_trigger','opposite_map'):
        r=select(label);seat=r['seat']
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,label,arm]) for i,arm in enumerate(('parent','approach'))]
            codes=[j.wait() for j in jobs]
            if any(codes):raise RuntimeError(codes)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:
        for label in ('first_trigger','opposite_map'):setup(label)
    elif len(sys.argv)>2:
        label,arm=sys.argv[1:];r=select(label);runner.HERE=HERE/label;runner.play(r['seat'],arm)
    else:run()

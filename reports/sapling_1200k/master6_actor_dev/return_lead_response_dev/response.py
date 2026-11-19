"""Use the earliest physically attainable two-turn path for an exposed tactical diagnosis."""
from pathlib import Path
import sys,fcntl,subprocess
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr
SOURCE=HERE.parent/'freight_toll_route_dev'
def setup():
    rows=sr.load(SOURCE/'taped_lead_time.json')
    r=next(r for r in rows if any(v['confirmed'] for v in r['by_horizon']))
    ex=next(v['examples'][0] for v in r['by_horizon'] if v['confirmed']);seat=r['seat']
    path=next((SOURCE/'games/candidate').glob(f'raw_*_{seat}.json.gz'))
    raw=sr.load(path);rc=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
    assert rc['game']['raw_sha256']==sr.sha(path) and raw['pack_sha256']==sr.sha(SOURCE/'candidate.py')
    out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
    sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[rc['official']]}}})
    root=dict(t=ex['start'],worker=ex['worker'],target=ex['target'],tree='',old=ex['actual'][0] or 'NONE',
              alternative=ex['moves'][0],follow=ex['moves'][1])
    sr.retain(out/'live_protocol.json',dict(seed=raw['seed'],seat=seat,root_t=root['t'],family='exposed_two_turn_collision',
                source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=raw['pack_sha256'],
                record=dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path)),root=root))
    sr.live.prepare(out)
def run(arm):
    runner.PRIOR=SOURCE;runner.HERE=HERE
    q=sr.load(HERE/'seat1/live_protocol.json');r=q['root'];native=sr.live.module
    def mod(p,name):
        m=native(p,name)
        if name=='recovery_live_student' and arm=='approach':
            old=m.agent
            def new(obs,cfg):
                cmd=old(obs,cfg)
                if obs['step']==r['t']+1 and r['worker'] in obs['players'][1][2] and r['target'] in obs['players'][0][2]:
                    cmd=dict(cmd);cmd[r['worker']]=r['follow']
                return cmd
            m.agent=new
        return m
    sr.live.module=mod
    runner.play(1,arm)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:setup()
    elif len(sys.argv)>1:run(sys.argv[1])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=[subprocess.Popen(['taskset','-c',str(i),sys.executable,__file__,arm]) for i,arm in enumerate(('parent','approach'))]
            if any([j.wait() for j in jobs]):raise RuntimeError('original response failed')

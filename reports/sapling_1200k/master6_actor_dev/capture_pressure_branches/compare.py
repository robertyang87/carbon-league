"""Change one planter order and let both original policies respond until the official end."""
import fcntl
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'capture_reach_dev'
TREE=HERE.parent/'tree_response_training'
sys.path.insert(0,str(TREE))
import select_roots as sr
from select_roots import live
import native_clock

def setup():
    selections=[]
    for record in sr.load(PRIOR/'outcomes.json'):
        seat=record['seat']
        edits=record['edits']
        eligible=[e for e in edits if e.get('prior_enemy_tree') and e.get('prior') and
                  sum(v['t']==e['t'] for v in edits)==1]
        e=max(eligible,key=lambda v:v['t'])
        rawpath,=list((PRIOR/'games/candidate').glob(f'raw_*_{seat}.json.gz'))
        raw=sr.load(rawpath)
        src=PRIOR/'games/candidate'
        receipt=sr.load(src/rawpath.name.replace('raw_','receipt_').replace('.json.gz','.json'))
        assert sr.sha(rawpath)==receipt['game']['raw_sha256']
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
        case=dict(seed=raw['seed'],seat=seat,root_t=e['t'],family='capture_pressure',
            source_dir=str(out.relative_to(sr.ROOT)),record=dict(seed=raw['seed'],seat=seat,
            source=str(rawpath.relative_to(sr.ROOT)),source_sha256=sr.sha(rawpath)),
            root=dict(t=e['t'],worker=e['worker'],tree=e['prior_tree'],old=e['actual'] or 'NONE',alternative=e['prior']),
            pack_sha256=sr.sha(PRIOR/'candidate.py'))
        sr.retain(out/'live_protocol.json',case)
        live.prepare(out)
        selections.append(dict(seed=raw['seed'],step=e['t'],seat=seat))
    sr.retain(HERE/'selection.json',selections)

def play(seat,arm):
    out=HERE/f'seat{seat}'
    q=sr.load(out/'live_protocol.json')
    live.PACK=PRIOR/'candidate.py';live.PACK_HASH=q['pack_sha256']
    receipt=out/f'clock_{arm}.json.gz'
    if receipt.exists() and (out/f'live_{arm}.json').exists():
        live.run(arm,out,q,resume=True);return
    original=live.module
    capture=[]
    checkpoint=out/f'live_{arm}_checkpoint.json.gz'
    previous=sr.load(checkpoint).get('clock',[]) if checkpoint.exists() else []
    def load(path,name):
        mod=original(path,name)
        if name=='recovery_live_master':
            rec=native_clock.Recorder(mod)
            rec.calls.extend(previous)
            observed=rec.wrap(mod.agent)
            def call(obs,cfg):
                if len(rec.calls)==len(previous):
                    rec.before=native_clock.counters(mod._S_STATS)
                return observed(obs,cfg)
            mod.agent=call;capture.append(rec)
        return mod
    save=live.save
    def persist(path,value):
        if path==checkpoint:
            value=dict(value,clock=capture[0].calls)
        if path==out/f'live_{arm}.json':
            rec=capture[0];assert not rec.errors and len(rec.calls)==value['search_calls']
            save(receipt,dict(calls=rec.calls,start=rec.start,finish=native_clock.host_snapshot(rec.master),
                summary=native_clock.summarize(rec.calls,rec.master._S_ABORT,rec.master._S_D2),
                action_sha256=value['action_sha256']))
        save(path,value)
    live.module=load;live.save=persist
    live.run(arm,out,q,resume=True)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:setup()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        with (TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm])
                      for i,arm in enumerate(('parent','approach'))]
                codes=[j.wait() for j in jobs]
                if any(codes):raise RuntimeError(codes)

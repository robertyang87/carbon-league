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
    held=[r for r in sr.load(HERE/'rows.json.gz') if r['batch']>6 and r['selected']]
    selections=[]
    for r in (max(held,key=lambda r:r['observed']),min(held,key=lambda r:r['observed'])):
        src=TREE/r['case']
        case=sr.load(src/'live_protocol.json')
        seat=case['seat']
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        case.update(pack_sha256=sr.sha(sr.SOURCE),original_case=r['case'],observed_original=r['observed'])
        sr.retain(out/'live_protocol.json',case);live.prepare(out)
        selections.append(dict(seat=seat,seed=case['seed'],step=case['root_t'],historical=r['observed']))
    assert {r['seat'] for r in selections}=={0,1}
    sr.retain(HERE/'retest_selection.json',selections)

def play(seat,arm):
    out=HERE/f'seat{seat}'
    q=sr.load(out/'live_protocol.json')
    live.PACK=sr.SOURCE;live.PACK_HASH=q['pack_sha256']
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
                      for i,arm in enumerate(('parent','alternative'))]
                codes=[j.wait() for j in jobs]
                if any(codes):raise RuntimeError(codes)

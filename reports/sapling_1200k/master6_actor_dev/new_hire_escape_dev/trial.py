import sys,fcntl,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr;live=sr.live;PACK=HERE.parent/'midgame_role_value_dev'
def prepare():
    case=sr.load(PACK/'new_worker_trace.json')[0];r=case['trace'][-1]
    assert not r['survived'] and r['safe_options']==['UP','DOWN']
    # Fixed physical order UP, RIGHT, DOWN, LEFT breaks ties on this exposed diagnostic only.
    root=dict(t=r['t'],worker=case['focal'],target=r['enemies'][1]['uid'],tree='',alternative=r['safe_options'][0],old=r['orders'])
    path=next((PACK/'games/candidate').glob('raw_*_0.json.gz'));raw=sr.load(path)
    receipt=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
    out=HERE/'seat0';out.mkdir(exist_ok=True)
    assert sr.sha(path)==receipt['game']['raw_sha256'] and raw['pack_sha256']==sr.sha(PACK/'candidate.py')
    sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
    sr.retain(out/'live_protocol.json',dict(seed=raw['seed'],seat=0,root_t=r['t'],family='new_hire_escape',
        source_dir=str(out.relative_to(sr.ROOT)),record=dict(seed=raw['seed'],seat=0,source=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path)),
        root=root,pack_sha256=raw['pack_sha256']))
    live.prepare(out)
def play(arm):
    runner.HERE=HERE;runner.PRIOR=PACK
    original=live.module
    def load(path,name):
        mod=original(path,name)
        if name=='recovery_live_student':
            base=mod.agent
            def agent(obs,cfg):
                # Shared driver reconstructs all feature history but skips intermediate calls to the pack.
                # The sole role substitution already executed at t121 on this exact taped prefix.
                if obs['step']==124:mod._ROLE_USED[0]=True
                return base(obs,cfg)
            mod.agent=agent
        return mod
    live.module=load;runner.play(0,arm)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(sys.argv[1])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=[subprocess.Popen(['taskset','-c',str(i),sys.executable,__file__,a]) for i,a in enumerate(('parent','approach'))]
            if any([p.wait() for p in jobs]):raise RuntimeError('escape worker failed')

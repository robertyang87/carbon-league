"""Match the actual deployed prefix and run each exact-parent response pair under the native examiner lock."""
from fit import HERE,ACTOR,sr
import sys,fcntl,subprocess
sys.path.insert(0,str(ACTOR/'capture_pressure_branches'))
import compare as runner
runner.HERE=HERE;runner.PRIOR=ACTOR/'actor_partner_dev'
def prepare():
    plan=sr.load(HERE/'protocol.json');assert sr.sha(runner.PRIOR/'candidate.py')==plan['parent'] and sr.sha(HERE/'candidate.py')==plan['candidate']
    for g in plan['roots']:
        src=ACTOR/g['file'];raw=sr.load(src);s=g['seat'];out=HERE/f'seat{s}';out.mkdir(exist_ok=True)
        receipt=sr.load(src.with_name(src.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert raw['pack_sha256']==plan['parent'] and sr.sha(src)==receipt['game']['raw_sha256'] and raw['seed']==g['seed'] and raw['candidate_seat']==s
        model=sr.live.module(HERE/'candidate.py','deployed_ground_root')
        shadow=sr.live.module(runner.PRIOR/'candidate.py','deployed_ground_parent')
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        for t,commands in enumerate(raw['actions']):
            obs=sim.obs(s);parent=shadow.agent(obs,sim.config);actual=model.agent(obs,sim.config)
            assert parent==commands[s]
            if actual!=parent:
                c=model._GROUND_SELECTION.seats[s]['selected'];uid=c['worker']
                assert set(k for k in set(actual)|set(parent) if actual.get(k)!=parent.get(k))=={uid} and uid not in actual
                break
            sim.step(*commands)
        else:raise ValueError('registered map had no changed command')
        sr.retain(out/'prefix.json',dict(pack=plan['candidate'],parent=plan['parent'],steps=t+1,site=c,exact_same_earlier=t))
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=g['seed'],seat=s,root_t=t,family='new_tree_ground_payback',source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=plan['parent'],
            record=dict(seed=g['seed'],seat=s,source=str(src.relative_to(sr.ROOT)),source_sha256=sr.sha(src)),
            root=dict(t=t,worker=uid,target='',tree='',old=parent[uid],alternative='NONE',cell=c['cell'])))
        sr.live.prepare(out)
        assert all(float(v)>0 for v in sr.load(out/'live_shared_init.json')['warm_cost'])
def run():
    for s in (0,1):
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=[subprocess.Popen(['taskset','-c',str((s+i)%2),sys.executable,__file__,str(s),arm]) for i,arm in enumerate(('parent','approach'))]
            if any([j.wait() for j in jobs]):raise RuntimeError('native continuation failed')
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:runner.play(int(sys.argv[1]),sys.argv[2])
    else:run()

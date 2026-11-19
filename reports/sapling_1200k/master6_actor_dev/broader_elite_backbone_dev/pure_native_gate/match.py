"""Four pure-Sapling official games, two frozen policies in both seats, no sequential score stopping."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;T=HERE.parent;A=T.parent
sys.path.insert(0,str(A/'first_win_dev'))
import native
native.HERE=HERE;native.PLAN=HERE/'protocol.json';native.OUT=HERE/'games';native.REG=native.sha(native.PLAN)
plan=native.read(native.PLAN);driver=native.driver


def bound():
    d=driver();d.PARENT=native.ROOT/plan['parent_path'];d.CANDIDATE=T/'standalone.py';return d


native.driver=bound


def one(job):
    native.clock.run_one(bound(),job)
    return dict(arm=job[0],**native.check(job))


def run():
    b=native.read(HERE/'binding.json');res=native.read(HERE/'responses.json');actions=native.read(HERE/'held_actions.json.gz')
    assert b['protocol']==native.REG and b['same_runtime'] and b['parameters']==1206476 and actions['changed']>0
    assert actions['arms_binding']==native.sha(HERE/'binding.json') and res['registrations']==native.REG and len(res['rows'])==2
    assert res['held_actual_changed']==actions['changed'] and all(z['original_calls'][0]>0 and z['original_calls'][1]>0 for z in res['rows'])
    s=plan['seeds'][0];assert s==621758340
    d=bound();assert native.sha(d.CANDIDATE)==b['new_pack'] and native.sha(d.PARENT)==b['old_pack']==plan['parent'] and native.sha(d.arena.prior.SOURCE)==plan['original']
    with native.SHARED.open('a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        terrain=native.clearance()['physical_boards'][str(s)]
        assert all(z['seed']!=s and z['terrain_sha256']!=terrain for z in native.read(T/'panel.json')['games'])
        for x in plan['held_roots']+plan['exposures']: assert x['seed']!=s and x['terrain']!=terrain
        for f in A.rglob('clearance.json'):
            if f.parent==HERE:continue
            v=native.read(f)
            if isinstance(v,dict):
                known=v.get('physical_boards',{})
                if isinstance(known,dict):assert str(s) not in known and terrain not in known.values()
        native.OUT.mkdir(exist_ok=True);rows=[]
        auth=dict(protocol=native.REG,old_npz=b['old_checkpoint'],new_npz=b['new_checkpoint'],candidate=b['new_pack'],parent=b['old_pack'],original=plan['original'],terrain=terrain,
            legal_held_actions=native.sha(HERE/'held_actions.json.gz'),short_response=native.sha(HERE/'responses.json'))
        for wave in [[('candidate',0),('parent',1)],[('parent',0),('candidate',1)]]:
            jobs=[(arm,s,seat,seat,terrain,auth) for arm,seat in wave]
            with ProcessPoolExecutor(2) as pool:
                for z in pool.map(one,jobs):
                    rows.append(z);print('PURE FULL',z['arm'],z['seat'],z['score_by_seat'],z['margin'],flush=True)
                    native.save(HERE/f"complete_{z['arm']}_{z['seat']}.json",z)
        native.save(HERE/'scores.json',rows)


if __name__=='__main__':run()

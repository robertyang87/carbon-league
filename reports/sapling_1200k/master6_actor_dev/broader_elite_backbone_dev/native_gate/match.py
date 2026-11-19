"""Run the four full original-opponent games only after both frozen continuation gates pass."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent;ACTOR=TRAIN.parent
sys.path.insert(0,str(ACTOR/'first_win_dev'))
import native
native.HERE=HERE;native.PLAN=HERE/'protocol.json';native.OUT=HERE/'games';native.REG=native.sha(native.PLAN)
p=native.read(native.PLAN)
base_driver=native.driver


def bound():
    d=base_driver();d.PARENT=ACTOR/'actor_partner_dev/candidate.py';d.CANDIDATE=TRAIN/'candidate.py'
    return d


native.driver=bound


def play(job):
    native.clock.run_one(bound(),job)
    return dict(arm=job[0],**native.check(job))


def main():
    gate=native.read(HERE/'gate.json')
    if not gate['eligible']:raise SystemExit('Development gate closed')
    assert gate['protocol']==native.REG and gate['binding']==native.sha(HERE/'binding.json')
    b=native.read(HERE/'binding.json');s=p['seeds'][0];d=bound();cand=b['files']['candidate.py']
    assert s==621758339 and native.sha(d.CANDIDATE)==cand and native.sha(d.PARENT)==p['parent'] and native.sha(d.arena.prior.SOURCE)==p['original']
    assert len(gate['rows'])==2 and all(z['delta']>0 and z['enemy_changed'] and z['genuine_changed'] for z in gate['rows'])
    with native.SHARED.open('a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX);terrain=native.clearance()['physical_boards'][str(s)]
        assert all(g['seed']!=s and g['terrain_sha256']!=terrain for g in native.read(TRAIN/'panel.json')['games'])
        for f in ACTOR.rglob('clearance.json'):
            if f.parent==HERE: continue
            item=native.read(f)
            if isinstance(item,dict):
                known=item.get('physical_boards',{})
                if isinstance(known,dict):assert terrain not in known.values() and str(s) not in known
        native.OUT.mkdir(exist_ok=True);rows=[]
        auth=dict(protocol=native.REG,candidate=cand,parent=p['parent'],original=p['original'],terrain=terrain,final_weights=b['files']['candidate.npz'],gate=gate['binding'])
        for wave in [[('candidate',0),('parent',1)],[('parent',0),('candidate',1)]]:
            jobs=[(arm,s,seat,seat,terrain,auth) for arm,seat in wave]
            with ProcessPoolExecutor(2) as pool:
                for row in pool.map(play,jobs):
                    rows.append(row);print(row['arm'],row['seat'],row['score_by_seat'],row['margin'],flush=True)
                    native.save(HERE/f"complete_{row['arm']}_{row['seat']}.json",row)
        native.save(HERE/'scores.json',rows)


if __name__=='__main__':main()

from data import sr,HERE,base
from role import choose
import sys
sys.path.insert(0,str(HERE.parent/'bank_under_pressure_dev'))
import periods,timing
def check(r):
    seat=r['seat'];path=HERE/'games'/r['arm']/f"raw_{r['seed']}_{seat}.json.gz"
    raw=sr.load(path);raw['candidate_seat']=seat;sim=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
    raw['initial_terrain_u8_sha256']=base.H(bytes(map(int,sim.carbon)))
    pack=sr.live.module(HERE.parent/'planter_capture_full_dev/candidate.py','hire_parent_verify');root=None;ablation=[]
    original=pack._mixed_actions
    def compare(x,aux):
        full=original(x,aux)
        if t%17==0 and r['arm']=='candidate':
            weight=pack._HYBRID['student_weight'];pack._HYBRID['student_weight']=0.0
            try:masked=original(x,aux)
            finally:pack._HYBRID['student_weight']=weight
            a=pack._FEAT.to_actions(full);b=pack._FEAT.to_actions(masked)
            diffs=[k for k in obs['players'][seat][2] if a.get(k)!=b.get(k)]
            ablation.append(dict(t=t,worker_count=len(obs['players'][seat][2]),changed_worker_ids=diffs))
        return full
    pack._mixed_actions=compare
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(seat);expected=pack.agent(obs,raw['game_configuration'])
        if root is None and r['arm']=='candidate' and 120<=t<=200:
            root=choose(obs,expected,raw['game_configuration'])
            if root:expected[root['worker']]=root['alternative']
        assert expected==pair[seat],(r['arm'],seat,t)
        sim.step(*pair)
    result=base.audit(raw,dict(official_rewards=r['score_by_seat']));focal=None
    if root:
        b,=[v for v in result['base_rows'] if v['base']==root['worker'] and v['t']==root['t']]
        focal=result['workers'][b['uid']]
        assert focal['birth_visible_after_step'] and focal['role']=='C' and focal['price']==root['price']
    return dict(root=root,focal=focal,student_ablation=ablation,periods=periods.compare(path,seat)['windows'])

def intermediate(arm,seat):
    plan=sr.load(HERE/'protocol.json');seed=plan['seeds'][0]
    path=HERE/'games'/arm/f'raw_{seed}_{seat}.json.gz'
    raw=sr.load(path)
    r=dict(seed=seed,arm=arm,seat=seat,score_by_seat=raw['official_score_by_seat'])
    sr.retain(HERE/f'audit_{arm}_{seat}.json.gz',dict(raw_sha256=sr.sha(path),details=check(r)))
def main():
    rows=[]
    for r in sr.load(HERE/'scores.json'):
        key=HERE/f"audit_{r['arm']}_{r['seat']}.json.gz"
        if not key.exists():intermediate(r['arm'],r['seat'])
        doc=sr.load(key)
        assert r['raw_sha256']==doc['raw_sha256']
        rows.append(dict(**r,**doc['details']))
    sr.retain(HERE/'full_ledger.json.gz',rows);timing.HERE=HERE;timing.check()
if __name__=='__main__':
    if len(sys.argv)==3:intermediate(sys.argv[1],int(sys.argv[2]))
    else:main()

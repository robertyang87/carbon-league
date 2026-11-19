"""Recompute the update, independent-map decision and reacting opponent evidence from saved complete games."""
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def read(p):
    buf=p.read_bytes()
    return json.loads(gzip.decompress(buf) if p.suffix=='.gz' else buf)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=read(H/'protocol.json');r=read(H/'result.json');selection=read(H/'selection.json')
    rows={}
    for path in sorted((H/'games').rglob('*.json.gz')):
        x=read(path);v=x['identity'];o=x['official'];key=(v['arm'],v['seed'],v['seat'])
        assert key not in rows and v['protocol']==sha(H/'protocol.json') and v['terrain']==o['terrain_sha256']
        assert len(x['actions'])==o['steps']<=299
        m=o['scores'][v['seat']]-o['scores'][1-v['seat']]
        wd=1 if m>0 else .5 if m==0 else 0
        utility=.5*wd+.5/(1+np.exp(-np.clip(m/1000.,-600,600)))
        assert abs(utility-o['u_by_seat'][v['seat']])<1.e-10
        rows[key]=x
    assert len(rows)==64
    g=[]
    for axis in range(3):
        diff=[]
        for s in p['train']:
            for seat in (0,1):
                arms=[rows[(f'axis{axis}_{tag}',s,seat)]['official']['u_by_seat'][seat] for tag in ['plus','minus']]
                diff.append(arms[0]-arms[1])
        g.append(float(np.mean(diff)))
    np.testing.assert_allclose(g,selection['gradient'],atol=1.e-12,rtol=0)
    coeff=.30*np.array(g)/np.linalg.norm(g) if np.linalg.norm(g)>1e-8 else np.zeros(3)
    np.testing.assert_allclose(coeff,selection['coefficients'],atol=1.e-12,rtol=0)
    tapes_echo=[]
    economy=[]
    symmetry=[]
    for s in p['holdout']:
        seatvals=[]
        for seat in (0,1):
            a=rows[('parent',s,seat)];b=rows[('selected',s,seat)]
            first=next((t for t,(v,w) in enumerate(zip(a['actions'],b['actions'])) if v[seat]!=w[seat]),None)
            ec=[t for t,(v,w) in enumerate(zip(a['actions'],b['actions'])) if v[1-seat]!=w[1-seat]]
            # Simultaneous actions, separate deterministic policies on identical initial public histories.
            assert first is not None or not ec
            assert not ec or ec[0]>first
            so=b['official']['scores'][seat]-a['official']['scores'][seat]
            en=b['official']['scores'][1-seat]-a['official']['scores'][1-seat]
            seatvals.append([so,en])
            for t in (0,60,100,150,200,250):
                key=str(t)
                if key in a['history'] and key in b['history']:
                    for actor,side in [('own',seat),('opponent',1-seat)]:
                        prev,now=a['history'][key][side],b['history'][key][side]
                        economy.append(dict(seed=s,seat=seat,t=t,actor=actor,delta={k:now[k]-prev[k] for k in ('cash','workers','trees')}))
            tapes_echo.append(dict(seed=s,seat=seat,first_own_difference=first,first_opponent_response_difference=(ec[0] if ec else None),response_changed_turns=len(ec),delta_own_score=so,delta_opponent_score=en))
        symmetry.append(dict(seed=s,mirror_identical_terminal_changes=bool(np.allclose(seatvals[0],seatvals[1],rtol=0,atol=1e-9))))
    score_pairs={s:np.mean([rows[('selected',s,i)]['official']['u_by_seat'][i]-rows[('parent',s,i)]['official']['u_by_seat'][i] for i in (0,1)]) for s in p['holdout']}
    assert abs(r['holdout']['mean_delta_U']-np.mean(list(score_pairs.values())))<1e-10
    pause=set()
    for phase in ('axes','comparison'):
        file=H/f'worker_{phase}.json'
        if file.exists():
            fold,a,s,seat=read(file)['paused_ticket'];pause.add((a,s,seat))
    normal=[v['seconds'] for k,v in rows.items() if k not in pause]
    cost=dict(unpaused_tickets=len(normal),unpaused_ticket_wall_sum_seconds=sum(normal),median_unpaused_seconds=float(np.median(normal)),
              estimated_one_cpu_equivalent_hours=(sum(normal)+len(pause)*float(np.median(normal)))/3600,
              imputation='Each unpaused ticket ran single-threaded with one CPU affinity. Impute paused-ticket compute by median unpaused wall. Excludes overhead outside play(), not a billed time.')
    ans=dict(machine_cost=cost,official_games=len(rows),distinct_training_maps=len(p['train']),distinct_held_maps=len(p['holdout']),registered_only=True,gradient=g,terminal_score_decomposition=tapes_echo,paired_observed_economy=economy,symmetry=symmetry,
             selected_sha256=sha(H/'selected.py'),observed_result_sha256=sha(H/'result.json'),reference_training_and_seats_sha256=sha(H/'protocol.json'),result_source='Original rules/interpreter final scores and submitted command tapes')
    (H/'independent_audit.json').write_text(json.dumps(ans,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()

"""Register all native rank flips and budget the two latest distinct physical maps, before computing outcomes."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def register():
    parent=json.loads((BASE/'protocol.json').read_bytes())
    rows=[]
    for card in parent['tickets']:
        seed,seat=card['seed'],card['seat']
        raw=(BASE/'roots'/f'{seed}_{seat}.json.gz').read_bytes()
        history=json.loads(gzip.decompress(raw))
        assert history['protocol']==sha((BASE/'protocol.json').read_bytes())
        for r in history['rows']:
            m=r['M2']
            if m['individual_winners'] is None:continue
            i=m['individual_winners'][0];j=m['chosen']
            if i==j:continue
            assert len(m['values'][str(i)])==len(m['values'][str(j)])==2
            assert m['depth_per'][str(i)]==m['depth_per'][str(j)]==m['clock']['H']
            assert i in m['alive'] and j in m['alive'] and m['menu'][j]==m['issued'] and m['menu'][i]!=m['menu'][j]
            rows.append(dict(seed=seed,seat=seat,step=r['step'],remaining=299-r['step'],terrain=card['terrain'],
                             panel_sha=sha(raw),public_sha=r['public_sha'],single0_index=i,mean2_index=j,
                             single0=m['menu'][i],mean2=m['issued'],native_pair_values={str(x):m['values'][str(x)] for x in [i,j]},
                             work=m['clock'],rollout0_also_M1=r['M1']['issued']==m['menu'][i]))
    rows.sort(key=lambda r:(r['remaining'],r['seed'],r['seat']))
    chosen=[]
    for r in rows:
        if len(chosen)<2 and all(r['terrain']!=x['terrain'] for x in chosen):
            chosen.append(dict(seed=r['seed'],seat=r['seat'],step=r['step'],remaining=r['remaining'],terrain=r['terrain'],
                               order=['single0','mean2'] if not chosen else ['mean2','single0'],panel_sha=r['panel_sha']))
    assert len(rows)==6 and [(r['seed'],r['seat'],r['step']) for r in chosen]==[(621758607,1,293),(621758604,1,282)]
    doc=dict(parent_protocol_sha=sha((BASE/'protocol.json').read_bytes()),cohort=rows,selected=chosen,
             exact_selection='All six completed same-M2-survivor-set same native depth pairs where one-response rollout0 winner differs from two-response winner. Already public roots and response scores. Select latest decisions first, smaller seed then seat, distinct physical terrain. No selection using counterfactual outcome. Remaining four excluded for cost; all listed.',
             endpoint='Both use same true root examiner move and exact archived M2 menu, completed depths, scores, physical root actions and root clock. Both sides use same frozen M1 v2 vs original (.58s at T=1) naturally from their branch afterward. One shared future initial random state per side, official natural terminal, intact known benefits/costs and WDL. Local one-CPU diagnostic, not new-map win rate. Outcome cannot isolate prunes affected by M2; this test isolates final aggregation on its fixed survivors.',
             max_joint_transitions=2*sum(r['remaining'] for r in chosen),contaminated_map_note='Earlier reported one-step cash deltas compare adaptive M1 against M2. Neither earlier two-arm t293 suffix used either of these same-map/seat/step pairs, and in both selected roots the physical single0 command differs from actual M1. New paired labels do not exist among prior complete signed branches.')
    (HERE/'protocol.json').write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(sha((HERE/'protocol.json').read_bytes()))
if __name__=='__main__':register()

"""Reconcile v4 and both v2 games against the same registered d5 using original rules."""
import itertools
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
LADDER=HERE.parent
sys.path.insert(0,str(LADDER/'placement_refresh'))
import refresh
sys.path.insert(0,str(LADDER/'tree_payback'))
from labels import tree_rows

def charts():
    m=refresh.read(HERE/'match.json')
    fl=refresh.read(LADDER/'nn_patch_cutoff/v2_pages.json.gz')
    ids=('66854074e015e3ad3854ffdda19633c8','4a3dc96034a40fad17f2221ab06fe30f')
    old={}
    for page in fl:
        for x in page['data']['matches']:
            if x['id'] not in ids: continue
            own='a' if x['b']==m['opponent_id'] else 'b';other='a' if own=='b' else 'b'
            assert x[other]==m['opponent_id'] and x['name_'+own]=='ds-pro-v2'
            g=x['result'][0];old[x['id']]=(g['order'].index(own),[g['scores'][k] for k in (own,other)])
    assert set(old)==set(ids)
    refs=[(m['match'],m['seat'],m['score'],HERE/m['replay_file'])]
    refs.extend((k,*old[k],LADDER/f/'replays'/(k+'-1.json.gz')) for k,f in zip(ids,('v2_loss_review','nn_patch_cutoff')))
    rows=[]
    for key,seat,score,path in refs:
        env=refresh.read(path);frames=env['steps'];refresh.compare.HERE=path.parents[1]/'analysis'
        detail=refresh.review.losing_game({'opponent':m['opponent'],'match_id':key,'seat':seat,'steps':299},path)
        assert [env['rewards'][s] for s in (seat,1-seat)]==score
        if key==m['match']:refresh.store(HERE/'ledger.json.gz',detail)
        plant=tree_rows(env)
        body=[]
        for s in (seat,1-seat):
            trees=[v for v in plant if v['seat']==s]
            body.append(dict(n=len(trees),paid_before_first_loss=sum(v['repaid'] for v in trees),
                stolen_before_repayment=sum(v['seized_unrepaid'] for v in trees),cost=sum(v['cost'] for v in trees),
                first_tenure_gross=sum(v['gross'] for v in trees),first_tenure_net=sum(v['net'] for v in trees),
                stolen=sum(v['exit']=='seized' for v in trees)))
        rows.append(dict(match=key,seat=seat,score=score,field_sha=hashlib.sha256(bytes(frames[0][0]['observation']['full_carbon'])).hexdigest(),
            signed_replay_sha=hashlib.sha256(path.read_bytes()).hexdigest(),official=detail['official'],whole=detail['whole'],trees=body,
            phases=detail['phases'],last_strict_lead=detail['last_strict_lead'],
            windows={k:{z:v for z,v in q.items() if z not in ('returns','events','routes')} for k,q in detail['worst_windows'].items()}))
        if key==m['match']:
            refresh.store(HERE/'plant_holds.json.gz',plant)
    refresh.store(HERE/'three_games.json',rows)

if __name__=='__main__':charts()

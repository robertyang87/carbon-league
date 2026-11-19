"""Compare the public forecast API with the signed working matrices; prove hidden-board mutations inert."""
import gzip
import json
from collections import deque
import numpy as np
from data import CashEvents,FastCarbon,ROOT,SOURCES,raw_ladder
from model import Forecast,HERE,TARGETS,public_rows,intent
from collision_risk import CollisionRisk


def with_hidden_mutation(obs):
    seats=obs['players'];s=obs['player'];visible=set()
    for p,_,kind in seats[s][2].values():
        visible.update(map(int,intent.DEST[p,:1] if kind=='COLLECTOR' else intent.DEST[p,1:]))
    carbon=[v if i in visible else 10000+i for i,v in enumerate(obs['carbon'])]
    return dict(obs,carbon=carbon,full_carbon=[-1]*225,opponent_actual_command={'not_available':True},
                future_labels={'intentionally_invalid':True})


def run():
    move=Forecast();risk=CollisionRisk();cards=SOURCES['native']+SOURCES['ladder']
    cases=[];negative=0;examples=[]
    for panel in ('native','ladder'):
        with np.load(HERE/(panel+'_features.npz'),allow_pickle=False) as fd: rec=dict(fd)
        chosen=[]
        for split in (('held','recent_evaluation') if panel=='native' else ('ladder',)):
            for seat in (0,1):
                chosen.append(next(i for i,g in enumerate(cards) if g['split']==split and g['seat']==seat))
        for gi in chosen:
            g=cards[gi];seat=g['seat']
            if panel=='native':
                raw=json.loads(gzip.decompress((ROOT/g['source']).read_bytes()))
                sim=FastCarbon(raw['game_configuration']).reset(seed=g['seed']);acts=raw['actions']
            else:
                raw=raw_ladder(g);first=raw['steps'][0][0]['observation'];sim=FastCarbon(raw['configuration']).load_observation(first)
                acts=[[s['action'] or {} for s in pair] for pair in raw['steps'][1:]]
            sim.cash=CashEvents(list(sim.cash));history=deque(maxlen=11);checked=0;acted=[]
            for t,act in enumerate(acts):
                history.append(sim.obs(seat))
                if t in (37,79,131,250):
                    obs=list(history);rr=public_rows(obs);pred=move.predict(obs);prisk=risk.predict(obs)
                    altered=[with_hidden_mutation(o) for o in obs]
                    assert pred==move.predict(altered) and prisk==risk.predict(altered)
                    inds=np.flatnonzero((rec['game']==gi)&(rec['t']==t));lookup={str(rec['uid'][i]):i for i in inds}
                    assert set(lookup)==set(pred)==set(prisk)==set(r[0] for r in rr)
                    for uid,x,prior,previous,group in rr:
                        idx=lookup[uid]
                        assert np.array_equal(x,rec['x'][idx]) and np.array_equal(prior,rec['prior'][idx])
                        assert previous==rec['last'][idx] and group==rec['group'][idx]
                        q,y=move.arrays(x[None,:],prior[None,:],[group])
                        assert np.allclose(q[0],pred[uid]['move']) and np.allclose(y[0],[pred[uid][k] for k in TARGETS])
                        z=prisk[uid];assert 0<=z['lose3']<=z['lose5']<=1 and 0<=z['take3']<=z['take5']<=1
                    for bad in (obs[::-1],obs[:-2]+obs[-1:],obs[:-1]+[dict(obs[-1],player=1-seat)]):
                        try:move.predict(bad)
                        except ValueError:negative+=1
                        else:raise AssertionError('mixed/incomplete/descending history was accepted')
                    checked+=len(rr);acted.append(t)
                if t==250:break
                sim.step(*act)
            cases.append({'split':g['split'],'seed':g['seed'],'observing_seat':seat,'at':acted,'carriers':checked})
        if panel=='native':
            held=[i for i,g in enumerate(cards) if g['split'] in ('held','recent_evaluation')]
            mask=np.isin(rec['game'],held);p=move.arrays(rec['x'][mask],rec['prior'][mask],rec['group'][mask])[1][:,1]
            indices=np.flatnonzero(mask)
            for paid in (True,False):
                subset=indices[(p>=.8)&(rec['x'][indices,0]>=100)&((rec['bank'][indices,1]>0)==paid)]
                i=min(subset,key=lambda j:(cards[rec['game'][j]]['seed'],cards[rec['game'][j]]['seat'],rec['t'][j],rec['uid'][j]))
                g=cards[rec['game'][i]];x=rec['x'][i];h=move.arrays(x[None,:],rec['prior'][i,None,:],[rec['group'][i]])[1][0]
                examples.append({'rule':'first (seed,seat,t,ID) held native courier with cargo>=100 and forecast bank5>=.8 per outcome',
                    'seed':g['seed'],'seat':g['seat'],'t':int(rec['t'][i]),'uid':str(rec['uid'][i]),'cargo':float(x[0]),'home_distance':int(x[1]),
                    'safe_shortest_next_cells':int(x[7]),'home_progress_prior3':int(x[16]),'home_progress_prior5':int(x[19]),'p_bank5':float(h[1]),'actual_bank5':float(rec['bank'][i,1])})
    result={'archived_row_and_public_api_match':True,'unseen_carbon_mutations_and_extra_hidden_fields_inert':True,'invalid_histories_rejected':negative,'cases':cases,'illustrations_only':examples}
    (HERE/'input_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()

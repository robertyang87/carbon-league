"""Resume registered pre-outcome native snapshots; compare full live responses to a locked forecast."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time

from experiment import HERE,ROOT,D,F,N,LOCK,MODEL,sha,dump,agents,rng,setup
from snapshot import read as read_state
BLIND='8d90d61ac02281b59a67818116f840e10ba94d9245835b0183a71d5efb438489'
MENU='3b737e03d58639eeca74303c8798b517172d69028ea9db3ac961669ec58b355a'
def main(ordinal):
    begun=time.time()
    worker_cpu=time.process_time()
    assert len(os.sched_getaffinity(0))==1
    b=(HERE/'model_blind.json').read_bytes();m=(HERE/'menus.json').read_bytes()
    assert sha(b)==BLIND and sha(m)==MENU and json.loads(b)['menus_sha256']==MENU
    assert sha((HERE.parent/'weights.json').read_bytes())==MODEL and (HERE/'model_blind.json').stat().st_mtime<=begun
    row=json.loads(m)['rows'][ordinal];r=row['ticket'];seed,s,t=r['seed'],r['seat'],r['step']
    subj,opp=agents();state=setup(r,True);cfg,env,sim,pub,bank=[state[k] for k in ('cfg','env','sim','public','bank')]
    assert F.digest(pub[s])==r['public_sha']
    # menus.json was serialized with sorted JSON object keys. Its verified file digest binds
    # the complete published menu. The capture-time menu checksum used native dict order.
    assert row['orders']==[row['menu'][i] for i in row['indices']]
    assert row['native_winner']==row['indices'][0] and row['dep']==[8,8]
    N.history_state(opp,state,1-s,rng(r)^0x7d4a1f6b)
    # The portable, data-only snapshot carries the precise post-root RNG, belief and search cost.
    restored=read_state(r,subj,HERE/'state')
    assert F.original_state.h(restored['post'])==row['snapshot_identity']==restored['post_h']
    subj._net()
    for key,val in restored['post'].items():setattr(subj,key,copy.deepcopy(val))
    subj._RX_CFG[:]=[cfg,copy.deepcopy(restored['cfg_cache'])]
    assert F.original_state.h(F.original_state.dynamic(subj))==row['snapshot_identity']
    assert row['orders'][0]==row['menu'][row['native_winner']]
    mods=[opp,opp];mods[s]=subj
    arms={'A':row['orders'][0],'B':row['orders'][1],'order_source':'two_real_same_H8_native_orders_sealed_before_response'}
    sequence=['A','B'] if ordinal%2==0 else ['B','A']
    expected={'history_orders':F.digest([order for _,order in state['prefix']]),'public_hashes':[F.digest(o) for o in pub],
              'bank':bank,'terrain':r['terrain'],'archive':r['sha256'],'historical_subject':r['historical_subject']}
    raw=copy.deepcopy(sim)
    def save(filename,obj):return dump('evidence/'+filename,obj)
    case=N.continue_pair(r,arms,sequence,(mods,cfg,env,sim,pub,expected),(row['orders'][0],row['cost']),
        restored['pre'],restored['entropy'],save)
    for name,value in case['arms'].items():
        path=HERE/'evidence'/f'{seed}_{s}_{t}_{name}.json.gz'
        assert sha(path.read_bytes())==value['evidence_sha256'];log=json.loads(gzip.decompress(path.read_bytes()))
        shadow=copy.deepcopy(raw);h=min(12,299-t)
        init=[{uid:z[1] for uid,z in p[2].items() if z[2]=='COLLECTOR' and z[1]>0} for p in pub[s]['players']]
        first=[{},{}];allpaid=[0.,0.];records=[]
        for pos,turn in enumerate(log['rows']):
            shadow.cash.events.clear();shadow.step(*turn['orders'])
            assert list(shadow.cash)==turn['end_cash']
            for kind,side,amount,*detail in shadow.cash.events:
                if kind=='own_delivery' and amount>0 and pos<h:
                    uid=detail[0];records.append(dict(t=turn['t']+1,side=side,uid=uid,amount=amount,base=detail[1]))
                    allpaid[side]+=amount
                    if uid in init[side]:first[side].setdefault(uid,amount)
        old=[sum(min(v,first[i].get(uid,0.)) for uid,v in mass.items()) for i,mass in enumerate(init)]
        clocks=[[step['clocks'][side] for step in log['rows']] for side in (0,1)]
        value.update(actual_initial_same_carrier_cash_by_seat=old,total_first12_own_delivery_by_seat=allpaid,
            bank_events_within_first12=records,root_carriers_by_seat=init,first_credited_payments_by_seat=first,
            first12_net_delivery=allpaid[s]-allpaid[1-s],win_indicator=int(value['scores'][s]>value['scores'][1-s]),
            native_transformed_terminal=subj._terminal(value['margin']),clock_by_seat=[dict(
                total_cpu=sum(z['cpu'] for z in turn),total_wall=sum(z['wall'] for z in turn),max_wall=max(z['wall'] for z in turn),
                hard_steps=sum(z['wall']>3 for z in turn),slow_steps=sum(z['wall']>1 for z in turn)) for turn in clocks])
    a,bb=case['arms']['A'],case['arms']['B']
    case.update(forecast_sha256=BLIND,menu_sha256=MENU,protocol=LOCK,seed=seed,seat=s,step=t,ordinal=ordinal,
      delta_A_minus_B=dict(same_carrier_own=a['actual_initial_same_carrier_cash_by_seat'][s]-bb['actual_initial_same_carrier_cash_by_seat'][s],
           same_carrier_enemy=a['actual_initial_same_carrier_cash_by_seat'][1-s]-bb['actual_initial_same_carrier_cash_by_seat'][1-s],
           own_team_bank12=a['total_first12_own_delivery_by_seat'][s]-bb['total_first12_own_delivery_by_seat'][s],
           net_team_bank12=a['first12_net_delivery']-bb['first12_net_delivery'],
           terminal_margin=a['margin']-bb['margin'],terminal_native=a['native_transformed_terminal']-bb['native_transformed_terminal']),
      fresh_arm_order=sequence,started_unix=begun,ended_unix=time.time(),elapsed_with_setup=time.time()-begun,
      cpu_seconds_including_setup=time.process_time()-worker_cpu)
    print('COMPLETE',ordinal,dump(f'cases/{ordinal}.json',case),case['delta_A_minus_B'],flush=True)
if __name__=='__main__':main(int(sys.argv[1]))

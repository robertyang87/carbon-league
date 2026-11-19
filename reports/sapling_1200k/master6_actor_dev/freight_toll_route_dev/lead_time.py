"""Shortest taped interception from a past public position. Future landings used only for offline diagnosis."""
from itertools import product
from measure import sr,HERE,_D,_DIRECTION,_TO

def run():
    result=[]
    for r in sr.load(HERE/'turning_points_candidate.json.gz'):
        seat=r['seat'];e=1-seat
        target=max((v for v in r['deliveries'] if v['seat']==e),key=lambda v:v['amount'])
        raw=sr.load(HERE/f"games/candidate/raw_{r['seed']}_{seat}.json.gz")
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        alt=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);states=[]
        for t,joint in enumerate(raw['actions'][:target['t']+1]):
            if t>=r['start']-10:states.append((t,sim.get_state(),sim.obs(seat)))
            sim.step(*joint)
        summaries=[]
        for horizon in (1,2,3):
            alternatives=[]
            for i,(t,state,obs) in enumerate(states):
                if i+horizon>len(states):break
                end=t+horizon-1;uid=target['worker'];v=states[i+horizon-1][2]['players'][e][2].get(uid)
                if not v:continue
                landing=int(_TO[v[0],_DIRECTION.index(raw['actions'][end][e].get(uid))])
                if landing in obs['players'][e][1].values():continue
                for k,w in obs['players'][seat][2].items():
                    if w[2]!='COLLECTOR' or w[1]>=v[1] or _D[w[0],landing]>horizon:continue
                    for moves in product(range(5),repeat=horizon):
                        pos=w[0]
                        for a in moves:pos=int(_TO[pos,a])
                        if pos!=landing:continue
                        alt.set_state(state);actions=[];dead=False
                        for j,a in enumerate(moves):
                            taped=raw['actions'][t+j]
                            pair=[dict(m) for m in taped];m=_DIRECTION[a]
                            if m:pair[seat][k]=m
                            else:pair[seat].pop(k,None)
                            actions.append(m);before=alt.workers.get(uid)
                            alt.step(*pair)
                            if k not in alt.workers:break
                            if before and uid not in alt.workers:
                                dead=any(row[0]==uid and row[4]==seat for row in alt.last_collisions)
                                break
                        if dead and k in alt.workers:
                            alternatives.append(dict(start=t,finish=t+len(actions)-1,worker=k,start_cargo=w[1],
                                target=uid,moves=actions,actual=[raw['actions'][t+j][seat].get(k) for j in range(len(actions))],
                                end_cargo=alt.workers[k].carbon))
            summaries.append(dict(horizon=horizon,confirmed=len(alternatives),examples=alternatives[:6]))
            if alternatives:break
        result.append(dict(seat=seat,deposit=target,by_horizon=summaries))
    sr.retain(HERE/'taped_lead_time.json',result);print(result)
if __name__=='__main__':run()

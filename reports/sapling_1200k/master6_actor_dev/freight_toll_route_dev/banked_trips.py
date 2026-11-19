"""Account for harvests between positive own-base payments, preserving lost and unbanked trips."""
from collections import defaultdict
from measure import HERE,sr
import sys
sys.path.insert(0,str(HERE.parent/'harvest_contact_dev'))
from production import measuring_engine

def run():
    all_games=[];engine=measuring_engine()
    for game in sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz'):
        raw=sr.load(sr.ROOT/game['path']);sim=engine(raw['game_configuration']).reset(seed=raw['seed'])
        trips={};complete=[]
        def open_trip(uid,side,t):
            if uid not in trips:trips[uid]=dict(start=t,side=side,uid=uid,harvest=[],frames=0)
            return trips[uid]
        def finish(uid,t,type_,value=0):
            tr=trips.pop(uid);tr.update(end=t,type=type_,payment=value);complete.append(tr)
        for t,joint in enumerate(raw['actions']):
            sim.harvest=[[],[]];sim.banked=[[],[]]
            for w in sim.workers.values():
                if w.is_collector:open_trip(w.id,w.player,t)['frames']+=1
            sim.step(*joint)
            for s in (0,1):
                for uid,v in sim.banked[s]:
                    if v<=0:continue
                    receiver_is_owner=uid.startswith(f'player-{s}-')
                    worker_side=s if receiver_is_owner else 1-s
                    open_trip(uid,worker_side,t)
                    finish(uid,t,'own_deposit' if receiver_is_owner else 'donated',v)
                for uid,pos,amount in sim.harvest[s]:
                    open_trip(uid,s,t)['harvest'].append([t,pos,amount])
            for uid in list(trips):
                if uid not in sim.workers:finish(uid,t,'lost')
        for uid in list(trips):finish(uid,len(raw['actions'])-1,'unsettled')
        assert list(sim.cash)==raw['official_score_by_seat']
        all_games.append(dict(source=game['path'],seed=game['seed'],seat=game['seat'],trips=complete))
    sr.retain(HERE/'banked_trips.json.gz',all_games)
    summary=[]
    for lo,hi in ((0,100),(100,200),(200,299)):
        for side in (0,1):
            selected=[t for g in all_games for t in g['trips'] if t['type']=='own_deposit' and lo<=t['end']<hi and t['side']==(g['seat']^side)]
            total=sum(t['payment'] for t in selected);picks=sum(len(t['harvest']) for t in selected)
            harvested=sum(amt for t in selected for _,_,amt in t['harvest'])
            summary.append(dict(start=lo,side=side,paid_trips=len(selected),payments=total,
                   mean_payment=total/len(selected),picks_per_trip=picks/len(selected),amount_per_pick=harvested/picks,
                   other_net_per_trip=(total-harvested)/len(selected)))
    sr.retain(HERE/'banked_trip_summary.json',summary)
    print(summary)
if __name__=='__main__':run()

"""Pre-action denominators for bank conversion and midgame purchases, with subsequent fees/receipts as separate labels."""
from collections import Counter,defaultdict
from analyze import *
DIRECTIONS=('STAY','UP','RIGHT','DOWN','LEFT')
D=np.asarray([[dist(i,j) for j in range(225)] for i in range(225)],dtype=np.int8)
TO=np.asarray([[i,((i//15-1)%15)*15+i%15,(i//15)*15+(i%15+1)%15,((i//15+1)%15)*15+i%15,(i//15)*15+(i%15-1)%15] for i in range(225)])
def predict_fee(obs,cfg):
    ps=obs['players'];own=ps[obs['player']]
    return cfg['plantCost']+(cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**sum(a<cfg['treeLifespan'] for v in ps for _,a in v[3].values()) if len(own[3])>cfg['playerTreeProtectiveNumber'] else 0.)
def cohort():
    decisions=[];rows=[]
    for c in index():
        g,steps,ev,obs,orders=process(c,True); seat=c['seat'];events=defaultdict(list);deposits=defaultdict(list)
        for e in ev:
            events[(e['t'],e['recipient'],e['kind'])].append(e)
            if e['kind']=='own_delivery':deposits[e['ident'][0]].append((e['t'],e['value']))
        sums=[Counter(),Counter()];seen={}
        for e in ev:
            if e['kind']=='own_delivery' and 200<=e['t']<299 and e['value']>0:
                z=sums[int(e['recipient']!=seat)];z['late_delivery_events']+=1;z['late_delivery_cash']+=e['value']
                if e['value']>=200:
                    z['late_200plus_events']+=1;z['late_200plus_cash']+=e['value']
        for t,views in enumerate(obs):
            for s,o in enumerate(views):
                x=sums[int(s!=seat)];own=o['players'][s];opp=o['players'][1-s]
                homes=list(own[1].values());enemy=[w[0] for w in opp[2].values()];cash=own[0]
                cashpaid=events[t,s,'plant'];tr={c for p in o['players'] for c,a in p[3].values()};bases={c for p in o['players'] for c in p[1].values()}
                for uid,(pos,load,role) in own[2].items():
                    command=orders[t][s].get(uid) or 'STAY';dest=int(TO[pos,DIRECTIONS.index(command)])
                    if role=='COLLECTOR' and 200<=t<=min(289,len(orders)-10):
                        h=min(int(D[pos,k]) for k in homes)
                        if load<80 or not 1<=h<=4 or t-seen.get(uid,-100)<10:continue
                        seen[uid]=t;threat=min((int(D[pos,w[0]]) for w in opp[2].values() if w[2]=='COLLECTOR' and w[1]<load),default=99)
                        receipts=[a for tt,a in deposits[uid] if t<=tt<t+10];got=sum(receipts);toward=min(D[dest,k] for k in homes)<h
                        x['carrier_roots']+=1;x['cargo_mass']+=load;x['toward_actual']+=int(toward);x['credited_10']+=int(got>0)
                        x['credited_amount10']+=got;x['pressure']+=threat<=1;x['pressure_credited']+=(threat<=1 and got>0)
                        x['unpressured_credited']+=(threat>1 and got>0);x['mean_home_numerator']+=h
                        decisions.append(dict(id=c['sha256'],seed=c['seed'],seat=s,t=t,uid=uid,task='delivery10',pos=pos,cargo=load,dist_home=h,
                            nearest_lighter_enemy=threat,actual=command,toward=bool(toward),own_cash=cash,outcome_cash10=got))
                    elif role=='PLANTER' and 100<=t<200:
                        if pos in tr or pos in bases:continue
                        if enemy and min(D[pos,k] for k in enemy)<=1:continue
                        carbon=sum(o['carbon'][int(k)] for k in TO[pos,1:]);fee=predict_fee(o,DEFAULT_CONFIG)
                        if carbon<120 or cash<fee:continue
                        paid=next((e for e in cashpaid if e['ident'][0]==uid),None)
                        x['safe_fertile_planter_turns']+=1;x['stayed']+=int(command=='STAY');x['planted']+=int(paid is not None)
                        decisions.append(dict(id=c['sha256'],seed=c['seed'],seat=s,t=t,uid=uid,task='rich_safe_plant',pos=pos,visible_soil=carbon,
                             nearest_enemy=min((int(D[pos,k]) for k in enemy),default=99),own_cash=cash,quote=fee,actual=command,actual_debit=-paid['value'] if paid else 0.))
        rows.append(dict(seed=c['seed'],seat=seat,arm=c['arm'],source=c['source'],raw_sha=c['sha256'],own=sums[0],opponent=sums[1]))
    write('opportunity_summaries.json',rows);write('public_opportunities_and_receipts.json.gz',decisions)
if __name__=='__main__':cohort()

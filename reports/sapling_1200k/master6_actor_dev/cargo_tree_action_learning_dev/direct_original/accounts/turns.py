"""Trace the same retrospective last lead change and late ten-turn cash swing in every arm."""
import gzip
import json
from collections import Counter
from pathlib import Path

from reconcile import HERE, ROOT, process, read

def rounded(x):
    return {k:round(v,6) for k,v in x.items()}

def account_diff(accounts,seat):
    names=set(accounts[seat]).union(accounts[1-seat])
    # The remainder after splitting absorption is exactly the soil component plus enemy tax.
    return rounded({k:accounts[seat].get(k,0)-accounts[1-seat].get(k,0) for k in names if k!='tree_absorption'})

def describe(states,orders,steps,events,seat,t):
    x=states[t][seat]['players']
    teams={}
    relevant=[e for e in events if e['t']==t and e['kind'] in ('own_delivery','plant','seize','base_collision_cash','tree_collision_cash','hire_collector','hire_planter')]
    for role,p in [('own',seat),('opponent',1-seat)]:
        roster={}
        for uid,(pos,load,kind) in x[p][2].items():
            roster[uid]={'cell':pos,'cargo':load,'role':kind,'order':orders[t][p].get(uid) or 'STAY'}
        # Each side's private current carbon is drawn from its own observation.
        teams[role]={'pre_cash':x[p][0],'bases':x[p][1],'active_workers':roster,'trees':x[p][3],
                     'public_joint_order':orders[t][p],'visible_carbon_from_own_seat':states[t][p]['carbon'],
                     'cash_post':steps[t]['balances'][p]}
    return {'t_pre':t,'sides':teams,'transactions':relevant,'delta_margin_categories':account_diff(steps[t]['accounts'],seat)}

def safe_paid(states,events,seat):
    n=len(states)
    def dist(a,b):
        dy,dx=abs(a//15-b//15),abs(a%15-b%15)
        return min(dy,15-dy)+min(dx,15-dx)
    ground={(e['t'],e['recipient'],e['ident'][0]):e['value']-e['ident'][2] for e in events if e['kind']=='tree_absorption'}
    rows=[]
    for e in events:
        h=e['t'];s=e['recipient']
        if e['kind']!='plant' or not 90<=h<190:continue
        uid,p=e['ident'];view=states[h][s];nears=[((p//15+dy)%15)*15+(p%15+dx)%15 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]]
        richness=sum(view['carbon'][i] for i in nears);danger=min((dist(p,w[0]) for w in view['players'][1-s][2].values()),default=99)
        if richness<120 or danger<2:continue
        tree,=[tid for tid,(where,age) in states[h+1][s]['players'][s][3].items() if where==p and age==1]
        soil=0.
        for u in range(h+1,min(n,h+26)):
            if tree not in states[u][s]['players'][s][3]:break
            soil+=ground.get((u,s,tree),0.)
        rows.append(dict(side='own' if s==seat else 'opponent',t=h,uid=uid,tree=tree,cell=p,
                         public_richness=richness,nearest_enemy=danger,fee=-e['value'],soil25=soil,net25=soil+e['value']))
    return rows

def audit(card):
    _, steps, events, states, orders=process({k:card[k] for k in ('seat','terrain','path','seed')},events=True)
    seat=card['seat']; sign=1 if card['result']=='win' else -1
    assert card['result'] in ('win','loss') and not any(v['extinct'] for v in card['sides'].values())
    net=lambda cash:cash[seat]-cash[1-seat]
    margin=[net(row['balances']) for row in steps]
    last=max((t for t,d in enumerate(margin) if sign*d<=0),default=-1)+1
    assert sign*margin[last]>0
    spans=[(last,last,'last sustained lead after this action')]
    late=max(range(200,len(steps)-9),key=lambda t:sign*(margin[t+9]-margin[t-1]))
    spans.append((late,late+9,'maximum ten-action swing in final winner direction starting t>=200, earliest tie'))
    out=[]
    for start,end,selection in spans:
        acc=[Counter(),Counter()]
        for row in steps[start:end+1]:
            for q in (0,1):
                acc[q].update(row['accounts'][q])
        snap=states[start][seat]['players']
        pre=snap[seat][0]-snap[1-seat][0]
        out.append({'start':start,'end':end,'pre_margin':pre,'post_margin':margin[end],'own_minus_opponent_window_cash_by_source':account_diff(acc,seat),
                    'criterion':selection,'details':[describe(states,orders,steps,events,seat,z) for z in sorted({max(0,start-j) for j in [10,5,3,0]}|{end})]})
    return {'seed':card['seed'],'seat':seat,'arm':card['arm'],'result':card['result'],'source':card['raw_sha256'],'turns':out,'safe_rich_purchases_t90_189':safe_paid(states,events,seat)}

def main():
    for f in sorted(HERE.glob('*_621*_[01].json')):
        dest=HERE/(f.stem+'_turns.json.gz')
        if not dest.exists() or 'safe_rich_purchases_t90_189' not in read(dest):
            record=audit(read(f))
            b=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            temp=dest.with_suffix('.tmp')
            temp.write_bytes(gzip.compress(b,mtime=0))
            temp.replace(dest)

if __name__=='__main__':
    main()

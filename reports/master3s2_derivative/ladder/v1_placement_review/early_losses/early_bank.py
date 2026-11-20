"""Replay the first materially loaded near-home collector whose original order defers returning."""
import copy
import gzip
import hashlib
import json
import random
from collections import Counter
from early_tree import HERE, LADDER, MATCHES, VECTORS, dest, ref


def distance(a,b):
    return sum(min((i-j)%15,(j-i)%15) for i,j in zip(divmod(a,15),divmod(b,15)))


def one(mid):
    raw=(LADDER/"replays"/f"{mid}-1.json.gz").read_bytes()
    replay=json.loads(gzip.decompress(raw));frames=replay["steps"];cfg=replay["configuration"]
    def orders(t):return [dict(v["action"] or {}) for v in frames[t+1]]
    def p(t):return frames[t][0]["observation"]["players"]
    matches=[]
    for t in range(1,100):
        players=p(t);commands=orders(t)[1]
        for uid,(cell,load,role) in players[1][2].items():
            if role!="COLLECTOR" or load<80:continue
            homes=players[1][1].values()
            d,b=min((distance(cell,h),h) for h in homes)
            checked=commands.get(uid,"NONE")
            if 1<=d<=2 and min(distance(dest(cell,checked),h) for h in homes)>=d:
                matches.append(dict(t=t,worker=uid,cell=cell,cargo=load,base=b,d=d,logged=checked))
    case=matches[0];root=case['t'];uid=case['worker']
    def check(env,sim,t,original):
        o=env.states[0].observation
        q=sim.obs(0)
        assert o.full_carbon==list(sim.carbon) and o.trees==q["trees"] and o.players==q["players"] and sim.step_count==o.step==t
        if original:
            v=frames[t][0]["observation"]
            assert all(o[k]==v[k] for k in ("step","players","full_carbon","trees"))
    def banking(players):
        if uid not in players[1][2]:return None
        cell,carried,_=players[1][2][uid];base=case['base']
        if carried==0 or cell==base:return None
        options=[]
        for i,a in enumerate(VECTORS):
            target=dest(cell,a)
            if distance(target,base)>=distance(cell,base):continue
            threats=sum(role=="COLLECTOR" and load<=carried and distance(c,target)<=1 for c,load,role in players[0][2].values())
            options.append((threats,i,a))
        return min(options)[-1]
    random.seed(cfg["randomSeed"]);ref.np.random.seed(cfg["randomSeed"])
    env=ref.make("carbon",configuration=cfg);initial=frames[0][0]["observation"]["full_carbon"];env.states[0].observation.full_carbon=copy.deepcopy(initial)
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=initial);sim.cash=ref.econ.CashEvents(sim.cash)
    historic=[];checkpoint=None;shadow=None
    for t in range(100):
        check(env,sim,t,True)
        if t==root:
            env.steps[:-1]=[None]*t;env.logs=[];checkpoint=ref.official_state.take(env,[0,0]);shadow=copy.deepcopy(sim)
        pair=orders(t);sim.cash.events.clear();env.step(pair,[]);sim.step(*pair)
        historic.append(sim.cash.events[:])
    check(env,sim,100,True)
    def marginal(t):return round(p(t)[1][0]-p(t)[0][0],2)
    start=min(range(91),key=lambda i:marginal(i+10)-marginal(i))
    categories=[Counter(),Counter()];deposits=[]
    for t in range(start,start+10):
        for k,s,value,*others in historic[t]:
            categories[s][k]+=value
            if k=='own_delivery':deposits.append(dict(t=t,seat=s,amount=value,uid=others[0],cell=others[1]))
    detail=[]
    for intervene in (False,True):
        ref.official_state.restore(env,checkpoint);sim=copy.deepcopy(shadow);rows=[];completed=False
        for t in range(root,root+20):
            pair=orders(t);players=env.states[0].observation.players;original=pair[1].get(uid,"NONE")
            if intervene and t<root+case['d'] and not completed:
                cmd=banking(players)
                if cmd is not None:pair[1][uid]=cmd
            sim.cash.events.clear();sim.step(*pair);env.step(pair,[])
            check(env,sim,t+1,not intervene)
            new=env.states[0].observation.players
            completed |= any(ev[0]=='own_delivery' and ev[3]==uid for ev in sim.cash.events)
            rows.append(dict(t=t,original=original,issued=pair[1].get(uid,'NONE'),cargo_after=new[1][2].get(uid),
                             cash=[r[0] for r in new],margin=round(new[1][0]-new[0][0],2),
                             deposits=[ev for ev in sim.cash.events if ev[0]=='own_delivery'],lost=sim.last_collisions[:]))
        detail.append(rows)
    return dict(match_id=mid,sha256=hashlib.sha256(raw).hexdigest(),selected=case,matches=matches,lookback=[dict(t=t,worker=p(t)[1][2].get(uid),act=orders(t)[1].get(uid,"NONE")) for t in range(root-10,root+1)],
                largest_early_official_cash_reversal=dict(start=start,end=start+10,margin_start=marginal(start),margin_end=marginal(start+10),flow=[dict(v) for v in categories],arrivals=deposits),control=detail[0],early_return=detail[1])
if __name__=="__main__":
    cases=[one(k) for k in MATCHES]
    data=dict(method="First action t1..99 where a collector carrying at least 80, distance 1 or 2 from nearest home, issues an order not reducing distance. Early return uses only current public positions; ties choose step with fewer reachable equal-or-lighter enemy collectors, then fixed direction order. Change only this collector for at most the initial distance; all other commands and later commands follow the tape. Regions selected with retrospective knowledge of the economic pattern; no out-of-sample or terminal claim. Compare short-run swings to the previously archived full-game context.",cases=cases)
    (HERE/"early_bank.json.gz").write_bytes(gzip.compress((json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    for c in cases:
        print(c['match_id'][:8],c['selected'],[(h,round(c['early_return'][h-1]['margin']-c['control'][h-1]['margin'],2)) for h in (1,2,5,10,20)])

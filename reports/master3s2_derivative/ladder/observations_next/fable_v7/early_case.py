"""A pre-action candidate route against a visible loaded enemy in the first v2 Fable game."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
src=HERE.parents[2]/"freight_response/run.py"
spec=importlib.util.spec_from_file_location("freight",src); h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
replay=HERE.parents[1]/"replays/bdbe59b04f71e43e53696874c8fd5958-1.json.gz"
directions={"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1),"NONE":(0,0)}
def move(c,a):
    y,x=divmod(c,15);dy,dx=directions[a];return ((y+dy)%15)*15+(x+dx)%15
def dist(a,b):return sum(min((i-j)%15,(j-i)%15) for i,j in zip(divmod(a,15),divmod(b,15)))
def public(player_rows,previous):
    if not previous:return []
    hits=[]
    for foe,(c,l,role) in player_rows[0][2].items():
        if role!='COLLECTOR' or l<120 or foe not in previous[0][2]:continue
        homes=sorted((dist(c,b),b) for b in player_rows[0][1].values())
        d,b=homes[0]
        if d not in (2,3,4,5) or homes[1][0]==d or dist(previous[0][2][foe][0],b)<=d:continue
        gates=[move(b,a) for a in directions if a!='NONE' and dist(c,move(b,a))==d-1]
        if len(gates)!=1:continue
        gate=gates[0]
        for mine,(pos,cargo,kind) in player_rows[1][2].items():
            if kind=='COLLECTOR' and cargo<=35 and dist(pos,gate)<=d-1:
                hits.append(dict(mine=mine,foe=foe,start=pos,foe_start=c,cargo=cargo,foe_cargo=l,base=b,gate=gate,d=d,eta=dist(pos,gate)))
    return hits
def patrol(p,ticket):
    mine=ticket['mine'];occ=p[1][2].get(mine)
    if not occ:return None
    cell,load,_=occ;goal=ticket['gate']
    if cell==goal:return "NONE"
    candidates=[]
    for i,(cmd,_) in enumerate(directions.items()):
        n=move(cell,cmd)
        if n in p[0][1].values() or dist(n,goal)>=dist(cell,goal):continue
        threat=sum(role=='COLLECTOR' and cargo<=load and dist(pos,n)<=1 for pos,cargo,role in p[0][2].values())
        candidates.append((threat,i,cmd))
    return min(candidates)[-1] if candidates else 'NONE'
def setorder(orders,uid,cmd):
    if cmd=='NONE':orders.pop(uid,None)
    else:orders[uid]=cmd
def safer_reply(p,uid,recorded,base):
    cell,cargo,_=p[0][2][uid]
    landing=move(cell,recorded)
    def risk(where):
        return sum(role=='COLLECTOR' and load<cargo and dist(pos,where)<=1 for pos,load,role in p[1][2].values())
    if risk(landing)==0:return recorded
    choices=[(risk(move(cell,cmd)),dist(move(cell,cmd),base),cmd!=recorded,i,cmd) for i,cmd in enumerate(directions)]
    return min(choices)[-1]
def main():
    raw=replay.read_bytes();g=json.loads(gzip.decompress(raw));r=g['steps'];cfg=g['configuration'];me=1
    def roster(t):return r[t][0]['observation']['players']
    def acts(t):return [dict(row['action'] or {}) for row in r[t+1]]
    def board_ok(env,sim,t,must_original):
        o=env.states[0].observation
        assert o.players==sim.obs(0)['players'] and o.trees==sim.obs(0)['trees'] and o.full_carbon==list(sim.carbon)
        if must_original:
            v=r[t][0]['observation']
            assert o.step==t and all(o[key]==v[key] for key in ('players','trees','full_carbon'))
    # Take the earliest trigger; assignment reads present and previous rosters only.
    all_hits=[dict(t=t,**v) for t in range(20,201) for v in public(roster(t),roster(t-1))]
    ticket=all_hits[0];root=ticket['t'];end=root+ticket['d']+3
    random.seed(cfg['randomSeed']);h.np.random.seed(cfg['randomSeed'])
    env=h.make("carbon",configuration=cfg);initial=r[0][0]['observation']['full_carbon']
    env.states[0].observation.full_carbon=copy.deepcopy(initial)
    sim=h.econ.FastCarbon(cfg).reset(full_carbon=initial);sim.cash=h.econ.CashEvents(sim.cash)
    for t in range(root):
        board_ok(env,sim,t,True)
        orders=acts(t);sim.step(*orders);env.step(orders,[])
    board_ok(env,sim,root,True);env.steps[:-1]=[None]*root;env.logs=[]
    frozen=h.official_state.take(env,[0,0]);out=[]
    for block,reply in ((False,"historical"),(True,"historical"),(False,"avoid"),(True,"avoid")):
        h.official_state.restore(env,frozen);shadow=copy.deepcopy(sim);rows=[]
        for t in range(root,end):
            p=env.states[0].observation.players;a=acts(t)
            own=ticket['mine'];foe=ticket['foe'];old=a[1].get(own,"NONE")
            if block and foe in p[0][2] and p[0][2][foe][1]>0 and own in p[1][2]:
                setorder(a[1],own,patrol(p,ticket))
            e=a[0].get(foe,"NONE")
            if reply=="avoid" and foe in p[0][2] and p[0][2][foe][1]>0:
                setorder(a[0],foe,safer_reply(p,foe,e,ticket['base']))
            shadow.cash.events.clear();shadow.step(*a);env.step(a,[])
            board_ok(env,shadow,t+1,not block and reply=="historical")
            p2=env.states[0].observation.players
            rows.append(dict(t=t,old_own=old,own=a[1].get(own,'NONE'),old_foe=e,foe=a[0].get(foe,'NONE'),
                             after_mine=p2[1][2].get(own),after_foe=p2[0][2].get(foe),cash=[s[0] for s in p2],
                             collisions=shadow.last_collisions[:],cash_events=shadow.cash.events[:]))
        out.append(dict(block=block,reply=reply,rows=rows,margin=round(p2[1][0]-p2[0][0],2)))
    prior=[]
    for t in range(root-10,root+1):
        p=roster(t);a=acts(t);prior.append(dict(t=t,own=p[1][2].get(ticket['mine']),enemy=p[0][2].get(ticket['foe']),
                         own_order=a[1].get(ticket['mine'],'NONE'),enemy_order=a[0].get(ticket['foe'],'NONE')))
    result=dict(sha256=hashlib.sha256(raw).hexdigest(),seat=me,actual_official_scores=g['rewards'],event=ticket,
                geometry_only_roots=all_hits,lookback=prior,end_state=end,arms=out,
                limits="Exploratory on one exposed replay. Trigger uses two visible rosters and public rules; chose candidate threshold after seeing game economics but without using future moves to assign the blocking cell. Enemy avoid arm sees only previous public positions and recorded simultaneous baseline, then picks a minimum-risk step toward same public base (recorded command breaks a tie) if its historical destination is threatened. All uninvolved units keep archived commands. Horizon is distance+3 turns, no terminal utility estimate.")
    (HERE/'early_case.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(ticket,[(a['block'],a['reply'],a['margin'],[v['own'] for v in a['rows']],[v['foe'] for v in a['rows']]) for a in out])
if __name__=='__main__':main()

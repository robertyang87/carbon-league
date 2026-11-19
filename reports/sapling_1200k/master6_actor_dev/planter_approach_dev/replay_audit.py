"""Audit selected examples in the official board from t0. Keep every joint action in both short branches."""
import hashlib
import json
from screen import HERE,ROOT,PARENT,FastCarbon,CashEvents,live,clean,read,retain,sha


def main():
    cfg=read(HERE/'protocol.json');screen=read(HERE/'screen.json.gz')
    census=read(HERE/'diagnostic.json.gz');book={}
    for z in screen['games']:
        book[(z['seed'],z['seat'])]=z['first']['t']
    keys=['players','trees','carbon','step','player']
    rows=[]
    for row in census['roots']:
        s=row['seat'];t0=row['t'];seed=row['seed']
        primary=book[seed,s]==t0
        if not (primary or row['approach']['paid']>0 or row['parent']['paid']>0):continue
        game=next(g for g in cfg['public_source_games'] if (g['seed'],g['seat'])==(seed,s))
        raw=read(ROOT/game['raw_path']);config=raw['game_configuration'];out={}
        for tag in ('parent','approach'):
            agent=live.module(HERE/'agent.py' if tag=='approach' else PARENT,'economic_native_audit')
            sim=FastCarbon(config).reset(seed=seed);sim.cash=CashEvents(sim.cash)
            engine=live.begin(raw);branch=[]
            for t in range(t0+5):
                pair=raw['actions'][t]
                obs=live.public(engine,s)
                fast=sim.obs(s)
                if not all(obs[k]==fast[k] for k in keys):
                    raise ValueError(f'preaction board or visibility disagrees at {seed}/{s}/{t}/{tag}')
                if t<t0:
                    if t==0:
                        if clean(agent.agent(obs,config))!=clean(pair[s]):raise ValueError('first parent command differs')
                    else:
                        agent._FEAT.observe(obs)
                        if tag=='approach':agent._APP.observe(obs,config)
                    engine.step(pair);sim.step(*pair);continue
                joint=[dict(a) for a in pair];joint[s]=clean(agent.agent(obs,config))
                if tag=='parent' and joint[s]!=clean(pair[s]):raise ValueError('parent action differs')
                sim.cash.events.clear()
                sim.step(*joint);engine.step(joint)
                off=live.public(engine,s);fast=sim.obs(s)
                if not all(off[k]==fast[k] for k in keys):raise ValueError('post-action board differs')
                w=off['players'][s][2].get(row['uid'])
                observed=[list(e) for e in sim.cash.events if e[0] in ('plant','seize') and e[1]==s and e[3]==row['uid'] and e[4]==row['target']]
                known=row[tag]['steps'][t-t0]
                if known['cell']!=(w[0] if w else None) or known['cash']!=observed or joint[s].get(row['uid'],'NONE')!=known['cmd']:
                    raise ValueError('official branch differs from census')
                if observed and not any(tree[0]==row['target'] for tree in off['players'][s][3].values()):
                    raise ValueError('charged transaction absent from official buyer tree roster')
                pre_sha=hashlib.sha256(json.dumps({k:obs[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
                branch.append({'t':t,'pre_own_public_sha256':pre_sha,'both_commands':joint,'focus_after':w,'receipt':observed,
                               'official_cash_after_by_seat':[p[0] for p in off['players']]})
            out[tag]=branch
        rows.append(dict(seed=seed,seat=s,t=t0,uid=row['uid'],target=row['target'],primary=primary,arms=out))
    retain(HERE/'replay_audit.json.gz',dict(version=1,coverage='Every predeclared first launch plus every census root charged at target by either arm',
            census_sha256=sha(HERE/'diagnostic.json.gz'),script_sha256=sha(HERE/'replay_audit.py'),roots=rows))
    print(len(rows),'separate roots replayed from t0 with both engines')


if __name__=='__main__':main()

"""Replay eight sealed games, reconciling cash and recording both sides' public decisions around turns."""
from pathlib import Path
from collections import Counter
import hashlib,json,gzip,sys,os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from turning_point_ledger import CashEvents
from study import World

def total(items):
    r=Counter()
    for x in items:r.update(x)
    return dict(r)
def difference(x,y):
    return {k:x.get(k,0)-y.get(k,0) for k in sorted(set(x)|set(y))}
def sums(seq,l,r):return total(seq[l:r])
def grouping(kind):
    if kind in ('plant','seize'):return 'tree_purchase'
    if kind.startswith('hire_'):return 'recruitment'
    return kind
def replay(path,seal):
    raw=path.read_bytes();g=json.loads(gzip.decompress(raw));assert {k:v for k,v in g.items() if k!='actions'}==seal
    w=World(g['seed']);assert w.terrain_sha256==g['terrain'];sim=w._sim;sim.cash=CashEvents(sim.cash);initial=list(sim.cash)
    s=g['seat'];flows=[];steps=[];cum=[0.]
    for t,cmd in enumerate(g['actions']):
        players=[]
        for j in (s,1-s):
            p=w.obs(j)['players'][j]
            players.append(dict(cash=p[0],bases=p[1],trees=p[3],workers={uid:[*val,cmd[j].get(uid,'STAY')] for uid,val in p[2].items()},
                                base_commands={uid:cmd[j].get(uid,'NONE') for uid in p[1]}))
        sim.cash.events.clear();w.advance(cmd);ledger=[Counter(),Counter()];events=[]
        for kind,j,v,*attr in sim.cash.events:
            if kind=='tree_absorption':
                tax=attr[2];ledger[j]['tree_ground']+=v-tax;ledger[j]['tree_cargo_tax']+=tax
            else:ledger[j][grouping(kind)]+=v
            if abs(v)>=.5 and kind not in ('tree_absorption','round'):events.append([kind,j,round(v,4),*attr])
        tick=difference(ledger[s],ledger[1-s]);flows.append(tick)
        steps.append(dict(t=t,own_enemy_public=players,realized_ledger_by_own_enemy=[dict(ledger[s]),dict(ledger[1-s])],payments=events))
        cum.append(sim.cash[s]-sim.cash[1-s])
    terminal=w.terminal;assert list(terminal['scores'])==g['score'] and g['steps']==len(g['actions']) and max(terminal['extinct_by_seat'])==False
    err=abs(sum(total(flows).values())-g['margin']);assert err<1e-6 and initial[0]==initial[1]
    return dict(flows=flows,steps=steps,cum=cum,error=err,sha256=hashlib.sha256(raw).hexdigest())
def pair(seed,seat,g):
    a,b=g['candidate'],g['parent'];n=len(a['steps']);assert n==len(b['steps'])
    dif=[x-y for x,y in zip(a['cum'],b['cum'])];f=[difference(x,y) for x,y in zip(a['flows'],b['flows'])]
    cuts=[0,50,100,150,200,250,n]
    phase=[dict(actions=[l,r-1],delta=dif[r]-dif[l],channels=sums(f,l,r)) for l,r in zip(cuts,cuts[1:])]
    turns={}
    for side in ['gain','loss']:
        sign=1 if side=='gain' else -1
        t=max(range(10,n+1),key=lambda i:sign*(dif[i]-dif[i-5]))
        turns[side]=dict(after_actions=t,net_delta_last_5=dif[t]-dif[t-5],windows={str(k):dict(delta=dif[t]-dif[t-k],channels=sums(f,t-k,t)) for k in (3,5,10)},
                        actions={arm:v['steps'][t-10:t] for arm,v in g.items()})
    return dict(seed=seed,seat=seat,total=dif[-1],phase=phase,turns=turns,net_channels=total(f),
                ledger_abs_errors={k:v['error'] for k,v in g.items()},sources={k:v['sha256'] for k,v in g.items()})
def main():
    assert os.sched_getaffinity(0)=={4} and os.getpriority(os.PRIO_PROCESS,0)==19
    dev=json.loads((HERE/'development.json').read_bytes());data={}
    for row in dev['rows']:
        seed,seat,arm=(row[k] for k in ('seed','seat','arm'))
        data.setdefault((seed,seat),{})[arm]=replay(HERE/'games'/f'{seed}_{seat}_{arm}.json.gz',row)
    pairs=[pair(*k,g) for k,g in sorted(data.items())];maps={}
    for s in sorted({k[0] for k in data}):
        cases=[c for c in pairs if c['seed']==s]
        def mean(x):return {k:v/len(x) for k,v in total(x).items()}
        maps[s]=dict(channels=mean([c['net_channels'] for c in cases]),phases=[dict(actions=cases[0]['phase'][i]['actions'],mean_delta=sum(c['phase'][i]['delta'] for c in cases)/len(cases)) for i in range(len(cases[0]['phase']))])
    obj=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),development_sha256=hashlib.sha256((HERE/'development.json').read_bytes()).hexdigest(),
          definitions='On each sealed trajectory candidate minus parent of (focal minus opponent). Windows are trailing 3,5,10 actions at greatest and smallest five-action differential changes, earliest exact tie. Cash events and timestamps are retrospective; pre-action workers carry [cell,cargo,role,recorded command], grouped focal then opponent from own observations. Actions cannot identify counterfactual effect. Mirrored seats share one terrain.',maps=maps,pairs=pairs)
    (HERE/'economy.json').write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
if __name__=='__main__':main()

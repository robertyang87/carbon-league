"""Public-observation-only fork coverage and deterministic deployment selector."""
from collections import Counter
from functools import lru_cache
import hashlib
MOVES={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}
N=15
def destination(p,act,n):
    dr,dc=MOVES[act];return ((p//n+dr)%n)*n+(p%n+dc)%n
def _dist(i,j):
    a=abs(i//N-j//N);b=abs(i%N-j%N);return min(a,N-a)+min(b,N-b)
D=tuple(tuple(_dist(i,j) for j in range(N*N)) for i in range(N*N))
def home(x,bases):return min(D[x][b] for b in bases)
@lru_cache(None)
def routes(pos,bases):
    d=home(pos,bases);length=max(0,min(3,d-1))
    path=[()]
    for h in range(1,length+1):
        nxt=[]
        for pre in path:
            e=pre[-1] if pre else pos
            for act in list(MOVES)[1:]:
                z=destination(e,act,N)
                if home(z,bases)==d-h:nxt.append(pre+(z,))
        path=nxt
    return tuple(p for p in path if p)
@lru_cache(None)
def coverage(z,paths):
    all_=Counter();reach=Counter()
    for path in paths:
        group=path[0];all_[group]+=1
        if any(D[z][p]<=i for i,p in enumerate(path)):reach[group]+=1
    frac={g:reach[g]/n for g,n in all_.items()}
    return sum(frac.values())/len(frac) if frac else 0.,min(frac.values(),default=0),frac
def geometry(obs,a,eid,include_fails=False):
    s=obs['player'];me,opp=obs['players'][s],obs['players'][1-s];e=opp[2].get(eid); paths=routes(e[0],tuple(sorted(opp[1].values()))) if e else ()
    if not e or not paths:return [],dict(reason='enemy_absent_on_bank_or_one_step_from_bank',paths=[])
    path_list=[list(p) for p in paths];fork=len({p[0] for p in paths})>=2;h=home(e[0],opp[1].values())
    dests={u:destination(v[0],a.get(u) or 'NONE',N) for u,v in me[2].items()}
    recruit={v for uid,v in me[1].items() if a.get(uid) in ('RECCOLLECTOR','RECPLANTER')}
    diag=[];eligible=[]
    for uid,(pos,cargo,role) in me[2].items():
        if role!='COLLECTOR' or cargo>=e[1]:continue
        old=a.get(uid) or 'NONE';base=coverage(dests[uid],paths)[0]
        reserved={v for u,v in dests.items() if u!=uid}|set(opp[1].values())|recruit
        best=0.;possible=set()
        for act in MOVES:
            z=destination(pos,act,N)
            if z in reserved or any(u!=eid and w[2]=='COLLECTOR' and w[1]<=cargo and D[z][w[0]]<=1 for u,w in opp[2].items()):continue
            cov,worst,fracs=coverage(z,paths);best=max(best,cov);possible.update(k for k,f in fracs.items() if f)
            gain=cov-base
            if act==old or not fork or not 2<=h<=7 or e[1]<60 or worst<=0 or gain<.25-1e-10:continue
            eligible.append(dict(t=obs['step'],seat=s,worker=uid,target_worker=eid,old=old,alternative=act,tree='no-focal-tree',
                own_cell=pos,own_load=round(cargo,3),enemy_cell=e[0],enemy_load=round(e[1],3),enemy_distance=h,landing=z,
                coverage_after=cov,coverage_before=base,gain=gain,minimum_exit_coverage=worst,reachable_fractions=fracs,enemy_paths=path_list))
        if include_fails:diag.append(dict(worker=uid,cell=pos,cargo=round(cargo,3),distance=D[e[0]][pos],best_coverage=best,
                            reachable_first_exit_groups=sorted(possible),archived_coverage=base))
    return eligible,dict(paths=path_list,exit_count=len({p[0] for p in paths}),lighter_units=diag,changed_cover_count=len(eligible),
                         candidate_caps=[(x['worker'],x['alternative']) for x in eligible])

def geometry_for_target(obs,base_actions,eid):
    if not 219<=obs["step"]<=289:return []
    return geometry(obs,base_actions,eid)[0]

def ranked_actions(obs,cfg,seat,base_actions,forecasts):
    if seat!=obs['player'] or int(cfg.get('size',15))!=N or not 219<=obs['step']<=289:return []
    actions=[]
    for uid,(pos,load,kind) in obs['players'][1-seat][2].items():
        if kind!="COLLECTOR" or load<60 or uid not in forecasts:continue
        q=forecasts[uid];prob=q["p_deposit_10"];cash=q["expected_root_cash_10"]
        if not (0.4<=prob<=1 and 40<=cash<=load+1e-4):continue
        for action in geometry_for_target(obs,base_actions,uid):
            text=f"freight-fork-v1|{seat}|{obs['step']}|{action['worker']}|{uid}|{action['alternative']}"
            action.update(model={k:float(q[k]) for k in ('p_deposit_10','expected_root_cash_10','expected_root_fraction_10')},
                          rank_score=float(cash)*action['gain'],tie=hashlib.sha256(text.encode()).hexdigest())
            actions.append(action)
    return sorted(actions,key=lambda row:(-row['rank_score'],-row['minimum_exit_coverage'],row['tie']))

def choose_intervention(obs,cfg,seat,base_actions,forecasts):
    """Select at most one legal change. The caller owns the base action dict."""
    ranked=ranked_actions(obs,cfg,seat,base_actions,forecasts)
    return ranked[0] if ranked else None

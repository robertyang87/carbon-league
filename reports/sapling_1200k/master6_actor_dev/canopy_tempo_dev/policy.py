"""Public conditional probability that a top player pays for a fresh tree now."""
from math import exp, log1p

NAMES=('intercept','elapsed','remaining_tree_lifetime','owned_trees','expiring_in_15','rival_tree_advantage',
       'own_planters','price','available_cash','nearby_soil','competing_tree_edges','enemy_planter_distance',
       'enemy_collector_distance','home_distance')
MOVES=('NONE','UP','RIGHT','DOWN','LEFT')
DEST=[[15*((p//15+dy)%15)+(p%15+dx)%15 for dx,dy in ((0,0),(0,-1),(1,0),(0,1),(-1,0))] for p in range(225)]
DIST=[[sum(min(abs(a-b),15-abs(a-b)) for a,b in ((i//15,j//15),(i%15,j%15))) for j in range(225)] for i in range(225)]


def near(pos,cells):
    return min((DIST[pos][p] for p in cells),default=15)


def stage(t):
    return 'opening' if t<=50 else 'middle' if t<=250 else 'late'


def context(obs,cfg,commands):
    s=obs['player']; own,enemy=obs['players'][s],obs['players'][1-s]
    places={q for p in (own,enemy) for q in p[1].values()}
    trees={q:(owner,age) for owner,p in enumerate(obs['players']) for q,age in p[3].values()}
    life=cfg['treeLifespan'];price=cfg['plantCost']
    if len(own[3])>cfg['playerTreeProtectiveNumber']:
        price+=cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**sum(age<life for _,age in trees.values())
    counts={role:sum(w[2]==role for p in (own,enemy) for w in p[2].values()) for role in ('COLLECTOR','PLANTER')}
    fees={a:cfg['rec'+a.title()+'Cost']+k*cfg['rec'+a.title()+'IncreaseCost'] for a,k in counts.items()}
    cash=own[0];free=cfg['workerLimit']-len(own[2]);hires=set()
    for u,p in own[1].items():
        a=commands.get(u)
        if a in ('RECCOLLECTOR','RECPLANTER') and free>0 and cash>=fees[a[3:]]:
            hires.add(p);free-=1;cash-=fees[a[3:]]
    dest={};pay={}
    for uid,(p,_,typ) in own[2].items():
        a=commands.get(uid) or 'NONE';dest[uid]=DEST[p][MOVES.index(a)]
        if typ!='PLANTER' or a!='NONE' or p in places:continue
        tr=trees.get(p)
        if tr is None or tr[1]>=life:pay[uid]=price
        elif tr[0]!=s:pay[uid]=cfg['seizeCost']
    ep=[w[0] for w in enemy[2].values() if w[2]=='PLANTER'];ec=[w[0] for w in enemy[2].values() if w[2]=='COLLECTOR']
    return dict(own=own,enemy=enemy,seat=s,t=obs['step'],carbon=obs['carbon'],cfg=cfg,price=price,
                hires=hires,dest=dest,available=cash-sum(pay.values()),pay=pay,bases=places,trees=trees,
                ep=ep,ec=ec,min_hire=min(fees.values()),life=life,nbases=set(enemy[1].values()))


def proposal(st,u):
    own,en=st['own'],st['enemy'];p=own[2][u][0]
    if own[2][u][2]!='PLANTER' or p in st['trees'] or p in st['bases']:return None
    price=st['price'];cash=st['available']+st['pay'].get(u,0)
    if cash<price+st['min_hire'] or p in st['hires'] or any(c==p and k!=u for k,c in st['dest'].items()):return None
    dp=near(p,st['ep']);dc=near(p,st['ec'])
    if min(dp,dc,near(p,st['nbases']))<=1:return None
    life=st['life'];t=st['t'];nbs=DEST[p][1:];visible=st['carbon']
    if any(visible[n] is None or visible[n]<0 for n in nbs):return None
    soil=[n for n in nbs if n not in st['trees'] and n not in st['bases']]
    edges=sum(q in st['trees'] for n in soil for q in DEST[n][1:])
    left=st['cfg']['episodeSteps']-1-t
    feat=(1.,t/299,min(life,max(0,left-1))/life,min(len(own[3]),12)/12,
      min(6,sum(life-a<=15 for _,a in own[3].values()))/6,max(-1,min(1,(len(en[3])-len(own[3]))/8)),
      min(8,sum(w[2]=='PLANTER' for w in own[2].values()))/8,min(2,price/100),
      min(1,log1p(cash-price)/8),sum(visible[n] for n in soil)/400,min(edges,8)/8,
      min(dp,8)/8,min(dc,8)/8,near(p,own[1].values())/14)
    return tuple(feat)


def bounds(st,u):
    """Gross direct till horizon, with a lower bound conditional on the match remaining active.

    For j future absorptions exclude any neighbor reachable by another worker or recruit;
    all four adjacent trees together remove at most 4*rate of its initial soil per turn.
    The focal planter can contaminate at most floor(j/2) distinct neighbors by planting
    on them (first future move must reach a neighbor and second must stay to plant).
    Conservative upper bound drops future mobile tax only when no collector or new hire
    can reach the tree's 3x3 neighborhood by horizon. A newborn earns zero immediately.
    """
    p=st['own'][2][u][0];cfg=st['cfg'];left=max(0,cfg['episodeSteps']-2-st['t'])
    rate=cfg['cellAbsorptionRate'];neighbors=[n for n in DEST[p][1:] if n not in st['trees'] and n not in st['bases']]
    other=[w[0] for pl in (st['own'],st['enemy']) for uid,w in pl[2].items() if uid!=u]+list(st['bases'])
    lower=0.
    if 0<=rate<=.25:
        for j in range(1,min(left,5)+1):
            if near(p,st['ep']+list(st['nbases']))<=j+1:break
            isolated=sorted(st['carbon'][n] for n in neighbors if near(n,other)>j+1)
            keep=len(isolated)-j//2
            if keep>0:lower+=sum(isolated[:keep])*rate*(1-4*rate)**j
    upper=None
    if left==0:upper=0.
    elif left<=10 and near(p,st['ec']+list(st['nbases']))>left+3:
        upper=4*max(cfg['maxCellCarbon'],cfg['co2FrmWithered'])*rate*left
    return lower,upper


def probability(weights,x):
    if tuple(weights['features'])!=NAMES:raise ValueError('wrong canopy feature order')
    z=max(-40,min(40,sum(a*b for a,b in zip(weights['beta'],x))))
    return 1/(1+exp(-z))


def order(st,u,weight,release):
    x=proposal(st,u)
    if x is None:return None
    t=st['t'];phase=stage(t);p=st['own'][2][u][0]
    if phase=='opening' or not release.get(phase):return None
    est=probability(weight,x); lower,upper=bounds(st,u);price=st['price']
    if phase=='middle' and st['dest'][u]!=p and est>=.5 and lower>price+.02:return 'NONE'
    if phase=='late' and st['dest'][u]==p and est<.5 and upper is not None and upper+.02<price:
        reserved=st['hires']|st['bases']|{q for v,q in st['dest'].items() if v!=u}
        options=[(near(q,st['ec']+st['ep']+list(st['nbases'])),i,m) for i,(m,q) in enumerate(zip(MOVES,DEST[p])) if i and q not in reserved]
        safe=[o for o in options if o[0]>1]
        if safe:return min(safe,key=lambda v:(-v[0],v[1]))[2]
    return None

"""Select a publicly observable carrier and choose low-tax homeward moves on either physical seat."""
from functools import lru_cache
import math

DIRECTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT')
OFFSETS = ((-1,0),(0,1),(1,0),(0,-1))


def move(position, action, size):
    y, x = divmod(position,size)
    dy, dx = OFFSETS[DIRECTIONS.index(action)] if action in DIRECTIONS else (0,0)
    return ((y+dy)%size)*size+(x+dx)%size


def distance(a,b,size):
    y,x=divmod(a,size); v,u=divmod(b,size)
    return min(abs(y-v),size-abs(y-v))+min(abs(x-u),size-abs(x-u))


def select_carrier(players, seat, step, cfg):
    own=players[seat]; remaining=cfg['episodeSteps']-1-step; size=cfg['size']
    eligible=[uid for uid,(pos,cargo,kind) in own[2].items() if kind=='COLLECTOR'
              and any(distance(pos,b,size)<=remaining for b in own[1].values())]
    return min(eligible,key=lambda uid:(-own[2][uid][1],int(uid.rsplit('-',1)[1]))) if eligible else None


def least_tax(players, seat, uid, orders, step, cfg):
    own,enemy=players[seat],players[1-seat]; position=own[2][uid][0]; size=cfg['size']
    bases=set(own[1].values()); enemy_bases=set(enemy[1].values()); trees=list(enemy[3].values())
    remaining=cfg['episodeSteps']-1-step
    blocked={move(state[0],orders.get(other,'NONE'),size) for other,state in own[2].items() if other!=uid}
    blocked.update(pos for b,pos in own[1].items() if orders.get(b,'NONE')!='NONE')

    @lru_cache(maxsize=None)
    def tax(cell,elapsed):
        if cell in bases:return 0.
        y,x=divmod(cell,size); near=(0,1,size-1); n=0
        for pos,age in trees:
            yy,xx=divmod(pos,size)
            if age+elapsed-1<cfg['treeLifespan'] and (xx-x)%size in near and (yy-y)%size in near:n+=1
        return -math.log(max(1e-9,1-n*cfg['collectorAbsorptionRate']))

    @lru_cache(maxsize=None)
    def tail(cell,base,elapsed):
        if cell in bases:return 0.
        length=distance(cell,base,size)
        if length+elapsed>remaining:return math.inf
        nearer=(move(cell,a,size) for a in DIRECTIONS)
        return min((tax(n,elapsed+1)+tail(n,base,elapsed+1) for n in nearer
                    if n not in enemy_bases and distance(n,base,size)<length),default=math.inf)

    options=[]
    for index,cmd in enumerate(DIRECTIONS):
        cell=move(position,cmd,size)
        if cell in enemy_bases or cell in blocked:continue
        for base in bases:
            risk=tax(cell,1)+tail(cell,base,1)
            if math.isfinite(risk):options.append((round(risk,9),distance(cell,base,size),index,cmd))
    return min(options)[-1] if options else orders.get(uid,'NONE')

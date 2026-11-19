"""One equal-price late hire choice using only a single seat's observation and its actual commands."""
from collections import deque


def _lw_move(p,cmd,n):
    dx,dy={'UP':(0,-1),'RIGHT':(1,0),'DOWN':(0,1),'LEFT':(-1,0)}.get(cmd,(0,0))
    return ((p//n+dy)%n)*n+(p%n+dx)%n


def _lw_neighbours(p,n):
    return [_lw_move(p,d,n) for d in ('UP','RIGHT','DOWN','LEFT')]


def _lw_dist(p,q,n):
    a=abs(p%n-q%n);b=abs(p//n-q//n)
    return min(a,n-a)+min(b,n-b)


class LateRecruit:
    def __init__(self):
        self.seat=None;self.last=-1;self.known={};self.used=False;self.event={}

    def observe(self,obs,cfg):
        n=int(cfg['size']);t=obs['step'];s=obs['player']
        if s!=self.seat or t<=self.last:
            self.known={};self.used=False
        self.seat=s;self.last=t
        self.known={k:v for k,v in self.known.items() if t-v[0]<=6}
        for p,c,role in obs['players'][s][2].values():
            for cell in ([p] if role=='COLLECTOR' else _lw_neighbours(p,n)):
                self.known[cell]=(t,float(obs['carbon'][cell]))

    def propose(self,obs,cfg,command):
        self.observe(obs,cfg)
        t=obs['step'];n=int(cfg['size']);s=obs['player'];left=cfg['episodeSteps']-1-t
        info={'step':t,'seat':s,'reason':'outside_window','changed':False};self.event=info
        if self.used or not 235<=t<=285:return command,info
        ours,other=obs['players'][s],obs['players'][1-s];workers=list(ours[2].values());enemy=list(other[2].values())
        tree={p:a for side in (ours,other) for p,a in side[3].values()}
        grounds=set(ours[1].values())|set(other[1].values())
        collectors=[p for p,_,r in workers if r=='COLLECTOR'];planters=[p for p,_,r in workers if r=='PLANTER']
        all_units=workers+enemy
        prices={r:cfg['rec'+r.title()+'Cost']+sum(w[2]==r for w in all_units)*cfg['rec'+r.title()+'IncreaseCost']
                for r in ('COLLECTOR','PLANTER')}
        occ={_lw_move(unit[0],command.get(uid),n) for uid,unit in ours[2].items()}
        visible={p:value*(.90**(t-age)) for p,(age,value) in self.known.items()}
        rate=max(0.,cfg['initialCollectRate']-(len(collectors)+1)*cfg['collectDecreaseRate'])
        epos=[w[0] for w in enemy]
        near=lambda p, group: min((_lw_dist(p,q,n) for q in group),default=2*n)
        life=cfg['treeLifespan'];living=sum(a<life for a in tree.values())
        plant_price=cfg['plantCost']+(cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**living
                                     if len(ours[3])>cfg['playerTreeProtectiveNumber'] else 0)
        cash=ours[0];free=cfg['workerLimit']-len(workers)
        info['reason']='no_planter_recruit'
        for uid,origin in ours[1].items():
            action=command.get(uid,'NONE');role={'RECPLANTER':'PLANTER','RECCOLLECTOR':'COLLECTOR'}.get(action)
            if role is None:continue
            if role=='COLLECTOR':
                if cash>=prices[role] and free>0:cash-=prices[role];free-=1
                continue
            info.update(base=uid,price=prices['COLLECTOR'],available_cash=cash,free_slots=free,reason='eligible')
            if prices['PLANTER']!=prices['COLLECTOR']:info['reason']='different_price'
            elif cash<prices[role] or free<=0 or not planters:info['reason']='not_affordable_or_no_spare_role'
            elif origin in occ or near(origin,epos)<=1:info['reason']='base_traffic'
            elif any(_lw_dist(origin,p,n)<=2 and near(p,planters)>1 and min(left,life-age)>=8
                     for p,age in other[3].values()):info['reason']='nearby_enemy_tree'
            if info['reason']=='eligible':
                for place in range(n*n):
                    if (_lw_dist(origin,place,n)>2 or place in tree or place in grounds or place in occ
                            or place in epos or near(place,planters)<=1):continue
                    neighbor_stock=sum(visible.get(a,0.) for a in _lw_neighbours(place,n))
                    if neighbor_stock*(1-(1-cfg['cellAbsorptionRate'])**min(12,left))>=plant_price+5:
                        info['reason']='nearby_funded_tree_site';break
            if info['reason']=='eligible':
                queue=deque([(origin,())]);seen={origin};best=None
                while queue:
                    p,path=queue.popleft();dist=len(path)
                    if (dist and p not in grounds and p not in tree and near(p,collectors)>1 and near(p,epos)>2
                            and 2*dist+3+3<=left and p in visible):
                        drain=max(0.,1-cfg['cellAbsorptionRate']*sum(q in tree for q in _lw_neighbours(p,n)))
                        collected=visible[p]*drain**(dist+3)*(1-(1-rate)**3)
                        bank=collected*(1-cfg['moveCost'])**dist-cfg['smeltCost']
                        if bank>=prices[role]+5 and (best is None or bank>best['bank_estimate']):
                            best=dict(target=p,path=list(path),visible_carbon=visible[p],age=t-self.known[p][0],bank_estimate=bank)
                    if dist>=3:continue
                    for nxt in _lw_neighbours(p,n):
                        if nxt in seen or nxt in occ or nxt in grounds or near(nxt,epos)<=1:continue
                        if _lw_dist(origin,nxt,n)!=dist+1:continue
                        seen.add(nxt);queue.append((nxt,(*path,nxt)))
                if best:
                    info.update(**best,reason='proposed');return {**command,uid:'RECCOLLECTOR'},info
                info['reason']='no_public_safe_payback'
            if cash>=prices[role] and free>0:cash-=prices[role];free-=1
        return command,info

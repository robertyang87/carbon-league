"""Route one endgame collector through observed carbon, with a fresh safety check before every order."""
from collections import defaultdict
from functools import lru_cache

DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
OFFSETS = ((-1,0),(0,1),(1,0),(0,-1))


@lru_cache(maxsize=1024)
def adjacent(pos, size):
    y,x=divmod(pos,size)
    return {cmd:((y+dy)%size)*size+(x+dx)%size for cmd,(dy,dx) in zip(DIRECTIONS,OFFSETS)}


def dest(pos, cmd, n):
    return pos if not cmd or cmd=="NONE" else adjacent(pos,n)[cmd]


def dist(a,b,n):
    ay,ax=divmod(a,n); by,bx=divmod(b,n)
    dy,dx=abs(ay-by),abs(ax-bx)
    return min(dy,n-dy)+min(dx,n-dx)


def home_distance(pos, homes, n):
    return min((dist(pos,home,n) for home in homes),default=n*n)


def visible(obs,n):
    result=set()
    for pos,_,kind in obs["players"][obs["player"]][2].values():
        if kind=="COLLECTOR":
            result.add(pos)
        else:
            result.update(adjacent(pos,n).values())
    return result


class Context:
    def __init__(self,obs,past,own,cfg):
        self.obs=obs; self.n=cfg["size"]; self.own=dict(own); self.cfg=cfg
        self.t=obs["step"]; self.h=cfg["episodeSteps"]-1-self.t; self.s=obs["player"]
        self.me,self.opp=obs["players"][self.s],obs["players"][1-self.s]
        self.homes=tuple(self.me[1].values()); self.enemyhomes=set(self.opp[1].values())
        self.cells={}; self.where=defaultdict(set)
        for uid,(pos,_,_) in self.me[2].items():
            self.where[dest(pos,own.get(uid),self.n)].add(uid)
        for uid,pos in self.me[1].items():
            if own.get(uid): self.where[pos].add(uid)
        for dt in (2,1,0):
            old=obs if dt==0 else past.get(self.t-dt)
            if old and old.get("step")==self.t-dt and old.get("player")==self.s:
                for cell in visible(old,self.n):
                    self.cells[cell]=(old["carbon"][cell]*.7**dt,dt)
        self.trees={cell:age for side in obs["players"] for cell,age in side[3].values()}
        self.collect_rate=max(0.,cfg["initialCollectRate"]-
                              sum(w[2]=="COLLECTOR" for w in self.me[2].values())*cfg["collectDecreaseRate"])
        self.enemies=[]
        previous=past.get(self.t-1,{})
        prev=previous["players"][1-self.s][2] if previous.get("step")==self.t-1 else {}
        for uid,(pos,cargo,kind) in self.opp[2].items():
            if kind!="COLLECTOR":
                continue
            last=prev.get(uid)
            heading=next((c for c,v in adjacent(last[0],self.n).items() if v==pos),None) if last else None
            self.enemies.append((pos,cargo,dest(pos,heading,self.n)))
        self.tree_zones=defaultdict(list)
        for pos,age in self.opp[3].values():
            y,x=divmod(pos,self.n)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    self.tree_zones[((y+dy)%self.n)*self.n+(x+dx)%self.n].append(age)

    def danger(self,pos,eta,cargo,uid):
        if pos in self.enemyhomes:
            return None
        if eta==1 and self.where.get(pos,set())-{uid} and not(pos in self.homes and self.h==1):
            return None
        # This side receives cargo brought onto its own base even when a rival wins the collision.
        if pos in self.homes:
            return 0.
        risk=0.
        for enemy,enemy_cargo,projected in self.enemies:
            if enemy_cargo>cargo:
                continue
            d=dist(enemy,pos,self.n)
            if eta==1 and d<=1:
                return None
            future_d=dist(projected,pos,self.n)
            if d<=eta and future_d<=2:
                risk=max(risk,.22*(cargo+30))
        return risk

    def tax(self,pos,eta,cargo):
        if pos in self.homes:
            return 0.
        count=sum(age+eta-1 < self.cfg["treeLifespan"] for age in self.tree_zones.get(pos,()))
        return min(cargo,cargo*count*self.cfg["collectorAbsorptionRate"])


def bank_paths(ctx, uid, start, home, used, cargo):
    """Best scored shortest path for this bank. Time, occupancy, threat and tax enter at every cell."""
    n=ctx.n
    remaining=dist(start,home,n)
    if remaining+used>ctx.h or (start in ctx.homes and remaining):
        return None
    possibilities={start:(0.,(),cargo,0.)}
    for step in range(remaining):
        new={}
        for pos,(penalty,commands,load,levy) in possibilities.items():
            for cmd,cell in adjacent(pos,n).items():
                if dist(cell,home,n)!=remaining-step-1 or (cell in ctx.homes and cell!=home):
                    continue
                eta=used+step+1
                moved=load*(1-ctx.cfg["moveCost"])
                risk=ctx.danger(cell,eta,moved,uid)
                if risk is None:
                    continue
                lost=ctx.tax(cell,eta,moved)
                key=penalty+risk+2*lost
                if cell not in new or key<new[cell][0]:
                    new[cell]=(key,commands+(cmd,),moved-lost,levy+lost)
        possibilities=new
    answer=possibilities.get(home)
    if not answer: return None
    penalty,commands,load,levy=answer
    # Cost is in cash-margin units; taxes both reduce our cargo and raise the rival's cash.
    return dict(commands=list(commands),home=home,delivered=max(0.,load-ctx.cfg["smeltCost"]),
                enemy_tax=levy,penalty=penalty-2*levy)


def route(ctx, uid, target, holds, reserve, prefer_home=None):
    pos,initial,kind=ctx.me[2][uid]
    if kind!="COLLECTOR" or (holds and target not in ctx.cells):
        return None
    n=ctx.n; commands=[]; load=initial; risk=0.; tax=0.; harvested=0.
    if holds and pos!=target:
        approach=next((a for a,p in adjacent(pos,n).items() if p==target),None)
        if not approach or target in ctx.homes: return None
        commands.append(approach);load*=1-ctx.cfg["moveCost"]
        hazard=ctx.danger(target,1,load,uid)
        if hazard is None: return None
        lost=ctx.tax(target,1,load);load-=lost;tax+=lost;risk+=hazard;pos=target
    ground=ctx.cells.get(pos,(0,0))[0]
    for _ in range(holds):
        if pos in ctx.homes or pos in ctx.enemyhomes: return None
        commands.append("NONE");eta=len(commands)
        hazard=ctx.danger(pos,eta,load,uid)
        if hazard is None: return None
        if pos in ctx.trees and ctx.trees[pos]+eta-1 < ctx.cfg["treeLifespan"]:
            return None
        mined=ground*ctx.collect_rate
        ground-=mined; load+=mined;harvested+=mined
        lost=ctx.tax(pos,eta,load);load-=lost;tax+=lost;risk+=hazard
    best=None
    homes=sorted(ctx.homes,key=lambda h:(h!=prefer_home,h))
    for home in homes:
        if len(commands)+dist(pos,home,n)+reserve>ctx.h: continue
        bank=bank_paths(ctx,uid,pos,home,len(commands),load)
        if not bank: continue
        paths=commands+bank["commands"]
        if not paths: continue
        enemy_tax=tax+bank["enemy_tax"]
        score=bank["delivered"]-enemy_tax-risk-bank["penalty"]-2*len(paths)
        option=dict(route=paths,home=home,score=score,harvest=harvested,enemy_tax=enemy_tax,
                    delivery=bank["delivered"],risk_cost=risk+bank["penalty"],target=target,holds=holds)
        if best is None or best["score"]<score: best=option
    return best


def selected(own,uid,cmd):
    answer=dict(own)
    if cmd=="NONE": answer.pop(uid,None)
    else: answer[uid]=cmd
    return answer


def choose(obs,past,own,cfg,plan=None,allow_start=True):
    t,s=obs["step"],obs["player"]
    h=cfg["episodeSteps"]-1-t
    if plan is None and (not allow_start or not 5<=h<=12):
        return dict(own),None,{"status":"fallback","stage":"out_of_window","target_worker":None}
    ctx=Context(obs,past,own,cfg)
    if plan is not None:
        uid=plan["worker"]
        info={"status":"abort","stage":"lost_worker_or_history","target_worker":uid}
        w=ctx.me[2].get(uid)
        if h<1 or not w or plan["seat"]!=s or plan["last_t"]+1!=t or w[0]!=plan["expected"]:
            return dict(own),None,info
        if w[0] in ctx.homes:
            info["stage"]="settled";info["status"]="finish"
            return dict(own),None,info
        direct=route(ctx,uid,w[0],0,0,plan["home"])
        if direct is None:
            info["stage"]="safe_return_unavailable"
            return dict(own),None,info
        option=None
        # When remaining harvest no longer compensates its risks or delay, resume a safe route to either bank immediately.
        for holds in range(plan["holds"],0,-1):
            candidate=route(ctx,uid,plan["target"],holds,0,plan["home"])
            if candidate and candidate["score"]>=direct["score"]+2:
                option=candidate;break
        if option is None:option=direct
        status="continue"
    else:
        best=None; eligible=0; paths=0
        for uid,(pos,cargo,kind) in sorted(ctx.me[2].items()):
            if kind!="COLLECTOR" or cargo<=0 or pos in ctx.homes:continue
            parent_cmd=own.get(uid) or "NONE"
            if home_distance(dest(pos,parent_cmd,ctx.n),ctx.homes,ctx.n)>=home_distance(pos,ctx.homes,ctx.n):continue
            direct=route(ctx,uid,pos,0,0)
            if direct is None:continue
            eligible+=1
            for cell in (pos,*adjacent(pos,ctx.n).values()):
                if cell not in ctx.cells or ctx.cells[cell][0]<=0:continue
                for holds in (1,2,3):
                    opt=route(ctx,uid,cell,holds,1)
                    if not opt or opt["route"][0]==parent_cmd: continue
                    paths+=1;gain=opt["score"]-direct["score"]
                    if gain>=12 and (best is None or gain>best[0]):
                        best=(gain,uid,opt,direct)
        if best is None:
            return dict(own),None,{"status":"fallback","stage":"no_safe_gain","target_worker":None,
                                   "eligible":eligible,"paths":paths}
        gain,uid,option,direct=best;status="start"
    cmd=option["route"][0];pos=ctx.me[2][uid][0]
    future={"worker":uid,"seat":s,"last_t":t,"expected":dest(pos,cmd,ctx.n),"target":option["target"],
            "holds":option["holds"]-(cmd=="NONE"),"home":option["home"]}
    info={"status":status,"target_worker":uid,
          "stage":"harvest" if cmd=="NONE" else ("approach" if option["holds"] else "return"),
          "command":cmd,"parent":ctx.own.get(uid) or "NONE","route":option,"direct":direct,
          "predicted_gain":option["score"]-direct["score"]}
    return selected(own,uid,cmd),future,info

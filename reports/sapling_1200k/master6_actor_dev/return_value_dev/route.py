"""Rank equal-length routes a loaded, already-returning Sapling worker can take to the same base."""
from math import prod
from functools import lru_cache

MOVE = ("NONE","UP","RIGHT","DOWN","LEFT")
D = tuple(tuple(min(abs(x%15-y%15),15-abs(x%15-y%15))+min(abs(x//15-y//15),15-abs(x//15-y//15)) for y in range(225)) for x in range(225))
N = tuple(tuple((i%15+dx)%15+15*((i//15+dy)%15) for dx,dy in ((0,0),(0,-1),(1,0),(0,1),(-1,0))) for i in range(225))
ADJ = tuple(tuple(j for j in range(225) if min(abs(i%15-j%15),15-abs(i%15-j%15))<=1 and min(abs(i//15-j//15),15-abs(i//15-j//15))<=1) for i in range(225))


def tree_key(pos, planters, h):
    return str(h)+"/"+str(min(4,min((D[pos][p] for p in planters),default=15)))


def retention(weights, owner, players, life, rate):
    # P(tree remains with this owner at the absorption phase of the future turn).
    vals=[[1.]*225 for _ in range(4)]
    opp_plants=[p for p,c,k in players[1-owner][2].values() if k=="PLANTER"]
    for pos,age in players[owner][3].values():
        for h in (1,2,3):
            if age+h>life:
                continue
            key=tree_key(pos,opp_plants,h)
            chance=weights["p"].get(key,weights["base"][str(h)])
            for nb in ADJ[pos]:
                vals[h][nb] -= rate*chance
    return [[max(0.,v) for v in vlist] for vlist in vals]


def score_paths(cell, bank, tree_retention, turn=1):
    if cell==bank:
        return 1.,(cell,)
    factor=tree_retention[turn][cell]
    if turn==3:
        return factor,(cell,)
    remaining=D[cell][bank]
    options=[score_paths(nb,bank,tree_retention,turn+1) for nb in N[cell][1:] if D[nb][bank]<remaining]
    best,route=max(options,key=lambda p:p[0])
    return factor*best,(cell,)+route


def ranked(obs, cmd, cfg, weights):
    seat,t=obs["player"],obs["step"]
    own,enemy=obs["players"][seat],obs["players"][1-seat]
    if cfg["size"]!=15 or seat not in (0,1):
        raise ValueError("unsupported board")
    home=tuple(own[1].values())
    opposing_bases=set(enemy[1].values())
    commits={u:N[pos][MOVE.index(cmd.get(u) or "NONE")] for u,(pos,_,_) in own[2].items()}
    # Deliberately reserve even an unaffordable attempted hire rather than crowd a legal production prefix.
    recruits={pos for bid,pos in own[1].items() if cmd.get(bid) in ("RECPLANTER","RECCOLLECTOR")}
    if not enemy[3] or not home:
        return []
    out=[]
    rem=cfg["episodeSteps"]-1-t
    keep=retention(weights,1-seat,obs["players"],cfg["treeLifespan"],cfg["collectorAbsorptionRate"])
    for uid,(start,cargo,role) in own[2].items():
        old=cmd.get(uid) or "NONE"
        length=min(D[start][b] for b in home)
        if role!="COLLECTOR" or cargo<100 or length<2 or rem<length or old=="NONE":
            continue
        oldcell=commits[uid]
        banks=[b for b in home if D[start][b]==length and D[oldcell][b]==length-1]
        if not banks:
            continue
        blocks={pos for u,pos in commits.items() if u!=uid}|recruits|opposing_bases
        light=[p for p,c,k in enemy[2].values() if k=="COLLECTOR" and c*(1-cfg["moveCost"])<=cargo*(1-cfg["moveCost"])]
        if any(D[oldcell][p]<=1 for p in light):
            continue
        for bank in banks:
            old_keep,old_path=score_paths(oldcell,bank,keep)
            for move,newcell in zip(MOVE[1:],N[start][1:]):
                if move==old or D[newcell][bank]!=length-1 or newcell in blocks or newcell in home:
                    continue
                if any(D[newcell][p]<=1 for p in light):
                    continue
                alt_keep,alt_path=score_paths(newcell,bank,keep)
                predicted=cargo*(alt_keep-old_keep)
                first=cargo*(keep[1][newcell]-keep[1][oldcell])
                if predicted>=max(8.,.02*cargo) and first>=-1e-9:
                    out.append(dict(uid=uid,old=old,alternative=move,cargo=cargo,origin=start,old_cell=oldcell,new_cell=newcell,
                                    bank=bank,distance=length,old_path=old_path,alternative_path=alt_path,
                                    expected_first_step_retained_gain=first,expected_three_step_retained_gain=predicted))
    return sorted(out,key=lambda r:(-r["expected_three_step_retained_gain"],r["uid"],MOVE.index(r["alternative"]),r["bank"]))

"""Six comparable action-before checkpoints, independently accumulated for both physical seats."""
from collections import Counter
from functools import cache
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TICKS=(30,60,100,150,200,250,299)
MOVES={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}


@cache
def dest(cell, move):
    y,x=divmod(cell,15)
    v,u=MOVES[move]
    return ((y+v)%15)*15+(x+u)%15


@cache
def dist(a,b):
    y,x=divmod(a,15)
    v,u=divmod(b,15)
    return min(abs(y-v),15-abs(y-v))+min(abs(x-u),15-abs(x-u))


@cache
def routes(pos,bases):
    h=min(dist(pos,b) for b in bases)
    return tuple(dest(pos,m) for m in MOVES if min(dist(dest(pos,m),b) for b in bases)<h)


def opportunities(players,orders,after):
    # A second friendly collector can screen an exposed alternative lane while
    # the target takes a different clear lane. No two allies are sent to one cell.
    out=[Counter(),Counter()]
    reachable=[{} for _ in (0,1)]
    for seat,p in enumerate(players):
        for uid,(pos,cargo,role) in p[2].items():
            if role=="COLLECTOR":
                for m in MOVES:
                    reachable[seat].setdefault(dest(pos,m),[]).append((uid,cargo))
    for seat,p in enumerate(players):
        rival=1-seat
        for uid,(pos,cargo,role) in p[2].items():
            if role!="COLLECTOR" or cargo<100:
                continue
            score=out[seat]
            lanes=routes(pos,tuple(sorted(p[1].values())))
            if not lanes:
                continue
            score["loaded_turns"]+=1
            danger={tile:[(k,w) for k,w in reachable[rival].get(tile,()) if w<cargo] for tile in lanes}
            hot=[tile for tile in lanes if danger[tile]]
            clear=[tile for tile in lanes if not danger[tile]]
            score["one_or_more_threatened"]+=bool(hot)
            score["all_routes_threatened"]+=not clear
            score["multiple_home_lanes"]+=len(lanes)>1
            score["choice_of_hot_and_clear"]+=bool(hot and clear)
            screen=[tile for tile in hot if any(k!=uid and w < min(d[1] for d in danger[tile])
                                               for k,w in reachable[seat].get(tile,()))]
            score["screen_plus_separate_clear"]+=bool(screen and clear)
            actual=dest(pos,orders[seat].get(uid,"NONE"))
            score["takes_clear_when_choice"]+=bool(hot and actual in clear)
            score["screens_and_takes_clear"]+=bool(actual in clear and any(
                k!=uid and w<min(d[1] for d in danger[tile]) and
                dest(p[2][k][0],orders[seat].get(k,"NONE"))==tile
                for tile in screen for k,w in reachable[seat].get(tile,())))
            # One turn outcome conditional on actual simultaneous enemy movement.
            enters=[k for k,w in reachable[rival].get(actual,()) if w<cargo and
                    dest(players[rival][2][k][0],orders[rival].get(k,"NONE"))==actual]
            score["lost_on_threatened_home_move"]+=bool(actual in hot and enters and uid not in after[seat][2])
    return out


def snapshot(p,economic,window):
    cash,bases,workers,trees=p
    carried=[(w[1],min(dist(w[0],cell) for cell in bases.values())) for w in workers.values() if w[2]=="COLLECTOR"]
    f=dict(cash=cash,workers=len(workers),collectors=len(carried),planters=len(workers)-len(carried),
           trees=len(trees),cargo=sum(c for c,d in carried),cargo_steps=sum(c*d for c,d in carried),
           heavy_units=sum(c>=100 for c,d in carried),delivery=economic["own_delivery"],
           ground_rent=economic["ground_rent"],tree_investment=-(economic["plant"]+economic["seize"]),
           tree_net=economic["ground_rent"]+economic["cargo_tax"]+economic["plant"]+economic["seize"],
           cargo_tax=economic["cargo_tax"],tree_plant=-economic["plant"],tree_seize=-economic["seize"])
    f["cargo_distance"]=f["cargo_steps"]/f["cargo"] if f["cargo"] else None
    return dict(state=f,income=dict(economic),previous_window_opportunities=dict(window))


def profile(meta,steps):
    econ=[Counter(),Counter()]
    window=[Counter(),Counter()]
    ticks={}
    last=None
    for t,before,orders,flow,after in steps:
        if t in TICKS:
            ticks[t]=[snapshot(before[s],econ[s],window[s]) for s in (0,1)]
            window=[Counter(),Counter()]
        opp=opportunities(before,orders,after)
        for s in (0,1):
            if abs(sum(flow[s].values())-(after[s][0]-before[s][0]))>1e-5:
                raise ValueError(f"{meta['id']} {t} ledger and office cash differ")
            econ[s].update(flow[s])
            window[s].update(opp[s])
        last=(t+1,after)
    if last is None:
        raise ValueError("Empty source")
    end,after=last
    if end in TICKS:
        ticks[end]=[snapshot(after[s],econ[s],window[s]) for s in (0,1)]
    scores=meta["official_scores_by_seat"]
    terminal_adjust=[scores[s]-after[s][0] for s in (0,1)]
    return dict(meta=meta,transitions=end,ticks=ticks,terminal_adjustment=terminal_adjust)


def main(side):
    if side=="top5":
        from top5 import games
    elif side=="sapling":
        from sapling import games
    else:
        raise ValueError(side)
    out=[]
    for m,steps in games():
        out.append(profile(m,steps))
    dest_path=HERE/(side+"_profile.json.gz")
    tmp=dest_path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(out,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    tmp.replace(dest_path)
    print(side,len(out),len(set(x['meta']['terrain'] for x in out)),flush=True)


if __name__=="__main__":
    main(sys.argv[1])

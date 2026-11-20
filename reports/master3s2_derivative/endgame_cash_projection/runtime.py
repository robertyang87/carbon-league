"""Antisymmetric cash projections from a single public Carbon board."""

from functools import lru_cache
import math

AGGREGATE = ("cash_difference", "cargo_difference", "tree_difference", "worker_difference")
BASE = AGGREGATE + tuple("time_x_" + name for name in AGGREGATE)
PHYSICAL = (
    "reachable_cargo", "impossible_cargo", "tight_one_turn", "tight_three_turns",
    "reachable_cargo_time_fraction", "onestep_danger_on_shortest_paths",
    "has_safer_shortest_step", "cargo_within_two_of_home", "workers_home_distance",
    "collector_count", "planter_count", "reach_travel_time_fraction",
    "live_tree_turns", "tree_turns_before_nearest_visible_capture",
)
FULL = BASE + PHYSICAL


@lru_cache(maxsize=4)
def lattice(size):
    def distance(a, b):
        ay, ax = divmod(a,size)
        by, bx = divmod(b,size)
        x, y = abs(ax-bx), abs(ay-by)
        return min(x,size-x)+min(y,size-y)
    def neighbors(cell):
        y,x = divmod(cell,size)
        return (size*((y-1)%size)+x, size*y+(x+1)%size,
                size*((y+1)%size)+x, size*y+(x-1)%size)
    cells=size*size
    return (tuple(tuple(distance(a,b) for b in range(cells)) for a in range(cells)),
            tuple(neighbors(a) for a in range(cells)))


def _side(players, seat, turns, boardsize, tree_life):
    own,enemy = players[seat],players[1-seat]
    distances,adj = lattice(boardsize)
    bases=list(own[1].values())
    if not bases:
        raise ValueError("Missing physical home base")
    own_distance = lambda p: min(distances[p][b] for b in bases)
    enemy_collectors = [(w[0],w[1]) for w in enemy[2].values() if w[2]=="COLLECTOR"]
    enemy_planters = [w[0] for w in enemy[2].values() if w[2]=="PLANTER"]
    own_cargo = sum(w[1] for w in own[2].values())
    agg=[own[0]/1000,own_cargo/1000,len(own[3])/10,len(own[2])/10]
    phys=[0.0]*len(PHYSICAL)
    for pos,cargo,role in own[2].values():
        d=own_distance(pos)
        phys[8]+=d/(10*boardsize)
        if role=="PLANTER":
            phys[10]+=0.1
            continue
        if role!="COLLECTOR":
            raise ValueError("Unknown worker occupation")
        phys[9]+=0.1
        c=cargo/1000.0
        if d>turns:
            phys[1]+=c
            continue
        phys[0]+=c
        if turns-d<=1:
            phys[2]+=c
        if turns-d<=3:
            phys[3]+=c
        fraction=d/max(1,turns)
        phys[4]+=c*fraction
        phys[11]+=fraction/10.0
        if d<=2:
            phys[7]+=c
        if d==0:
            phys[6]+=c
            continue
        shortest=[q for q in adj[pos] if own_distance(q)==d-1]
        if not shortest:
            raise ValueError("No descent path on torus")
        # On our base the owner can bank cargo even if the original bearer dies.
        threatened=sum(q not in bases and any(load<=cargo and distances[e][q]<=1
                                             for e,load in enemy_collectors) for q in shortest)
        phys[5]+=c*threatened/len(shortest)
        if threatened<len(shortest):
            phys[6]+=c
    for pos,age in own[3].values():
        usable=min(turns,max(tree_life-age,0))
        phys[12]+=usable/(10.0*tree_life)
        # Earliest visible geometric approach then stay to seize; no claim it will occur.
        first_capture=min((distances[e][pos]+1 for e in enemy_planters),default=usable+1)
        phys[13]+=min(usable,max(first_capture-1,0))/(10.0*tree_life)
    return agg+[v*turns/50 for v in agg]+phys


def features(players, seat, remaining, size=15, tree_lifespan=50):
    if seat not in (0,1) or not isinstance(remaining,int) or remaining<0:
        raise ValueError("Invalid physical seat or remaining time")
    a=_side(players,seat,remaining,size,tree_lifespan)
    b=_side(players,1-seat,remaining,size,tree_lifespan)
    row=[x-y for x,y in zip(a,b)]
    if len(row)!=len(FULL) or not all(map(math.isfinite,row)):
        raise ValueError("Invalid source board")
    return row


def predict_residual(row,model):
    """Return predicted future net cash movement, in raw official score units."""
    if len(model["weights"])!=len(model["scale"]) or len(model["weights"])>len(row):
        raise ValueError("Wrong head dimensions")
    ans=1000*sum(w*x/s for w,x,s in zip(model["weights"],row,model["scale"]))
    if not math.isfinite(ans):
        raise ValueError("Nonfinite projection")
    return ans

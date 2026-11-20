"""Own return-action readout. Inputs are physical public rosters, never pending orders."""

import numpy as np

ACTIONS=("NONE","UP","RIGHT","DOWN","LEFT")
DELTA=((0,0),(-1,0),(0,1),(1,0),(0,-1))
CELLS=tuple(tuple(15*((p//15+y)%15)+(p%15+x)%15 for y,x in DELTA) for p in range(225))
def _distance(a,b):
    y=abs(a//15-b//15);x=abs(a%15-b%15)
    return min(x,15-x)+min(y,15-y)
TREE_RING=tuple(tuple(15*((p//15+dy)%15)+(p%15+dx)%15
                      for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy)
                for p in range(225))
D=tuple(tuple(_distance(a,b) for b in range(225)) for a in range(225))
NAMES=("stay","direct_bank","to_enemy_bank","home_distance","home_progress","can_bank_by_six",
       "last_chance_slack","lighter_enemy_on_destination","light_opponent_can_stay","light_opponent_moves",
       "second_light_approach","light_cargo_gap","min_enemy_distance","friendly_can_stay",
       "friendly_moves","friendly_empty_lower","enemy_planter","focal_value","enemy_tree_here",
       "enemy_trees_around","best_next_safety","safe_shortest_exits","second_base_distance",
       "cash_short_of_recruit","home_adjacent_enemies","remaining","other_carriers_near_home","visible_current_carbon","fractional_gather_then_bank","residual_slack_after_gather")


def features(players,seat,uid,step,remaining=None,carbon=None):
    """One row per NONE/UP/RIGHT/DOWN/LEFT in legal torus coordinates."""
    own,enemy=players[seat],players[1-seat]
    pos,mass,kind=own[2][uid]
    if kind!="COLLECTOR":
        raise ValueError("Return-ranking applies to a collector")
    homes=tuple(own[1].values());foes=tuple(enemy[1].values())
    if not homes:raise ValueError("No own base")
    remaining=299-step if remaining is None else remaining
    allies=[w for k,w in own[2].items() if k!=uid]
    collectors=[(p,c) for p,c,occ in enemy[2].values() if occ=="COLLECTOR"]
    light=[(p,c) for p,c in collectors if c<=mass]
    tree={row[0] for row in enemy[3].values()}
    def h(cell):return min(D[cell][b] for b in homes)
    def threats(cell):return sum(D[cell][p]<=1 for p,_ in light)
    spot_carbon=0. if carbon is None else float(carbon[pos])
    d0=h(pos);out=[]
    for i,z in enumerate(CELLS[pos]):
        route=h(z)
        next_cells=[q for q in CELLS[z][1:] if h(q)<route]
        potential=[0 if q in homes else threats(q) for q in next_cells]
        approaching=[(p,c) for p,c in light if D[z][p]<=1]
        h2=sorted(D[z][b] for b in homes)
        out.append([
            float(i==0),float(z in homes),float(z in foes),route/4,(d0-route),float(route<=5),
            min(max((remaining-1-route)/5,0),1),float(any(p==z and c<mass for p,c in collectors)),
            float(any(p==z for p,c in light)),min(len(approaching)/3,1),
            min(sum(D[z][p]==2 for p,c in light)/3,1),
            max((mass-c for p,c in approaching),default=0)/max(mass,1),
            min((D[z][p] for p,c in light),default=15)/15,
            float(any(w[0]==z for w in allies)), min(sum(D[z][w[0]]<=1 for w in allies)/3,1),
            float(any(w[2]=="COLLECTOR" and w[1]<=mass and D[z][w[0]]<=1 for w in allies)),
            float(any(w[2]=="PLANTER" and D[z][w[0]]<=1 for w in enemy[2].values())),
            min(mass,1000)/500,float(z in tree),sum(q in tree for q in TREE_RING[z])/8,
            1/(1+min(potential,default=0)),sum(v==0 for v in potential)/max(len(potential),1),
            h2[-1]/15,float(own[0]<30),min(sum(any(D[b][p]<=1 for b in homes) for p,c in light)/3,1),
            min(remaining,30)/30, min(sum(w[2]=="COLLECTOR" and w[1]>=50 and h(w[0])<=2 for w in allies)/3,1),
            min(spot_carbon,120)/100,float(i==0)*.3*spot_carbon/max(50,mass),float(i==0)*max(0,min(5,remaining-1-route))/5])
    return np.asarray(out,dtype=np.float64)


def score(rows,weights):
    """Predicted within-root relative six-step return fraction, not cash or a probability."""
    z=(np.asarray(rows)-np.asarray(weights["mean"]))/np.asarray(weights["scale"])
    v=np.tanh(z@np.asarray(weights["w0"])+np.asarray(weights["b0"]))@np.asarray(weights["w1"])
    return v-v.mean()

"""Return-route geometry based exclusively on the two public teams."""

NAMES = ('cargo', 'carriers', 'cargo_near', 'cargo_mid', 'cargo_far',
         'nearest_step_clear', 'nearest_all_steps_threatened', 'base_endpoint_threat',
         'nearest_step_alt', 'nearest_home_pressure', 'distant_rich_cargo', 'phase_x_cargo')
DISTANCE = tuple(tuple(min(abs(p//15-q//15), 15-abs(p//15-q//15))
                       + min(abs(p%15-q%15), 15-abs(p%15-q%15)) for q in range(225))
                 for p in range(225))
MOVES = tuple((((p//15-1)%15)*15+p%15, (p//15)*15+(p%15+1)%15,
               ((p//15+1)%15)*15+p%15, (p//15)*15+(p%15-1)%15) for p in range(225))


def raw(players, seat, step):
    own, foe = players[seat], players[1-seat]
    bases=list(own[1].values())
    reach=[[] for _ in range(225)]
    for p,cargo,role in foe[2].values():
        if role == 'COLLECTOR':
            for cell in (p,*MOVES[p]):
                reach[cell].append(cargo)
    low=[min(cargo,default=float('inf')) for cargo in reach]
    out=[0.0]*len(NAMES)
    for p,cargo,role in own[2].values():
        if role!='COLLECTOR' or cargo <= 0 or not bases:
            continue
        d=min(DISTANCE[p][b] for b in bases)
        shortest=[cell for cell in MOVES[p] if min(DISTANCE[cell][b] for b in bases)<d] if d else [p]
        viable=sum(low[cell]>cargo for cell in shortest)
        homes=[b for b in bases if DISTANCE[p][b]==d]
        out[0]+=cargo
        out[1]+=1
        out[2]+=cargo*(d<=2)
        out[3]+=cargo*(3<=d<=5)
        out[4]+=cargo*(d>=6)
        out[5]+=cargo*bool(viable)
        out[6]+=cargo*int(viable==0)
        out[7]+=cargo*int(min(low[b] for b in homes)<=cargo)
        out[8]+=cargo*int(viable>=2)
        out[9]+=cargo/max(d,1)
        out[10]+=max(0,cargo-200)*(d>=4)
        out[11]+=cargo*step/299
    return out

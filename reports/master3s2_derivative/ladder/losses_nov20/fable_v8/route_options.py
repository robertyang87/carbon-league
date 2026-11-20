"""List feasible approaches to each possible enemy base from one public pre-action roster."""
from functools import lru_cache
import gzip
import json
from pathlib import Path


def approaches(players, seat, enemy_id, size=15):
    def adjacent(cell):
        r,c=divmod(cell,size)
        return (((r-1)%size)*size+c, r*size+(c+1)%size, ((r+1)%size)*size+c, r*size+(c-1)%size)
    def distance(a,b):
        x=abs(a%size-b%size)
        y=abs(a//size-b//size)
        return min(x,size-x)+min(y,size-y)
    @lru_cache(None)
    def routes(a,b):
        if a==b:
            return 1
        d=distance(a,b)
        return sum(routes(n,b) for n in adjacent(a) if distance(n,b)<d)

    opponent=players[1-seat]
    start,heavy,kind=opponent[2][enemy_id]
    assert kind=="COLLECTOR" and heavy>0
    bases=set(opponent[1].values())
    result=[]
    for base in sorted(bases):
        d=distance(start,base)
        if not d:
            continue
        for approach in adjacent(base):
            if approach in bases or distance(start,approach)!=d-1:
                continue
            interceptors=[dict(id=wid,cell=state[0],cargo=state[1],distance=distance(state[0],approach))
                          for wid,state in players[seat][2].items() if state[2]=="COLLECTOR" and state[1]<heavy
                          and distance(state[0],approach)<=d-1]
            result.append(dict(base=base,base_distance=d,approach=approach,enemy_moves_to_approach=d-1,
                    routes_via_cell=routes(start,approach),total_shortest_routes=routes(start,base),
                    potential_interceptors=interceptors))
    return result


if __name__=="__main__":
    path=Path(__file__).resolve().parents[2]/"replays/5cef1a63ea3166ee1c05611fca1cb910-1.json.gz"
    observations=json.loads(gzip.decompress(path.read_bytes()))["steps"][116]
    # The wire format puts the roster and step shared by both seats in the first record.
    shared=observations[0]["observation"]
    assert observations[1]["observation"]["player"]==1 and shared["step"]==116
    print(json.dumps(approaches(shared["players"],1,"player-0-worker-12"),ensure_ascii=False,indent=2))

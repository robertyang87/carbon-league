"""Show whether a guard actually occupied a usable cut on an opponent's shortest return routes."""
from coverage import HERE,sr
from route import _TO,_D,_DIRECTION
def run(directory):
    records=[]
    for seat in (0,1):
        out=directory/f'seat{seat}';q=sr.load(out/'live_protocol.json');root=q['root'];src=sr.load(sr.ROOT/q['record']['source'])
        for arm in ('parent','approach'):
            sim=sr.FastCarbon(src['game_configuration']).reset(seed=q['seed'])
            for pair in src['actions'][:root['t']]:sim.step(*pair)
            tape=sr.load(out/f'live_{arm}_actions.json.gz');steps=[]
            for i,pair in enumerate(tape[:10]):
                ob=sim.obs(seat);own,enemy=ob['players'][seat],ob['players'][1-seat]
                worker=own[2].get(root['worker']);carrier=enemy[2].get(root['target']);entry=dict(i=i)
                if worker and carrier:
                    p=carrier[0];home=list(enemy[1].values());d=min(_D[p,b] for b in home)
                    guard=int(_TO[worker[0],_DIRECTION.index(pair[seat].get(root['worker']))])
                    actual=int(_TO[p,_DIRECTION.index(pair[1-seat].get(root['target']))])
                    down=[int(v) for v in _TO[p] if min(_D[v,b] for b in home)<d]
                    entry.update(guard=guard,enemy_landing=actual,enemy_start=p,load=carrier[1],home_distance=int(d),
                                 bankward=down,occupied_bankward_exits=int(guard in down),blocks_all_exits=bool(down and down==[guard]))
                sim.step(*pair);steps.append(entry)
            records.append(dict(seed=q['seed'],seat=seat,arm=arm,steps=steps))
    sr.retain(directory/'paths.json',records)
if __name__=='__main__':run(HERE)

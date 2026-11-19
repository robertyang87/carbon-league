"""Trace each actual seizure back to a defender's distance, including the frame before the enemy arrived."""
from measure import HERE,sr
from guard_tree import _D
from collections import Counter,deque

def audit(game):
    raw=sr.load(sr.ROOT/game["path"])
    cfg=raw["game_configuration"]
    sim=sr.FastCarbon(cfg).reset(seed=raw["seed"])
    seat=game["seat"]; history=deque(maxlen=11); lost=[]
    for t,pair in enumerate(raw["actions"]):
        own,opp=sim.obs(seat)["players"][::1 if seat==0 else -1]
        prior={uid: v[0] for uid,v in own[3].items()}
        history.append((t,own,opp,pair))
        sim.step(*pair)
        for tree,cell in prior.items():
            new=sim.trees.get(tree)
            if new is None or new.player==seat:continue
            enemy=next(k for k,w in opp[2].items() if w[0]==cell and w[2]=='PLANTER')
            frames=[]
            for at,p,e,cmd in history:
                if t-at not in (0,1,3,5,10):continue
                defenders=[(int(_D[v[0],cell]),k,cmd[seat].get(k),v[1]) for k,v in p[2].items() if v[2]=='COLLECTOR']
                frames.append(dict(t=at,enemy_before=e[2].get(enemy),enemy_move=cmd[1-seat].get(enemy),
                                   nearest=min(defenders,default=None),holder_before=tree in p[3]))
            lost.append(dict(t=t,tree=tree,cell=cell,enemy=enemy,frames=frames))
    return dict(source=game["path"],seat=seat,seed=game["seed"],lost=lost)
if __name__=='__main__':
    games=[audit(g) for g in sr.load(HERE/'paired.json.gz')]
    sr.retain(HERE/'seizures.json.gz',games)
    all_rows=[x for g in games for x in g['lost']]
    for lag in (0,1,3,5,10):
        dist=Counter()
        for row in all_rows:
            at=next((f for f in row['frames'] if row['t']-f['t']==lag),None)
            if at:dist[at['nearest'][0] if at['nearest'] else None]+=1
        print(lag,len(all_rows),sorted(dist.items(),key=str))

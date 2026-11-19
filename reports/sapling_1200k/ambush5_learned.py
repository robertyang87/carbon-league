"""Forecast loaded opponents' 3–5 step crossings from the frozen 32 training maps."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ambush5_probe import HERE, DIST, RUNS, run_game
from interception_fit import own_view, read_game
from ambush5_top5 import summarize


def signed(a,b):
    return ((b-a+7)%15)-7


def frame(origin,base):
    x,y=origin%15,origin//15
    dx,dy=signed(x,base%15),signed(y,base//15)
    swap=abs(dy)>abs(dx)
    return (swap,1 if dx>=0 else -1,1 if dy>=0 else -1)


def vector(origin,dest,axes):
    swap,sx,sy=axes
    vx=signed(origin%15,dest%15)*sx
    vy=signed(origin//15,dest//15)*sy
    return (vy,vx) if swap else (vx,vy)


def project(origin,offset,axes):
    swap,sx,sy=axes
    dx,dy=offset[::-1] if swap else offset
    return (origin%15+dx*sx)%15+15*((origin//15+dy*sy)%15)


def context(obs,previous,uid):
    me=obs['player']
    enemy=obs['players'][1-me]
    pos,cargo,_=enemy[2][uid]
    bases=sorted(enemy[1].values(),key=lambda b:(DIST[pos][b],b))
    home=bases[0]
    axes=frame(pos,home)
    old=previous.get(uid)
    last=tuple(-n for n in vector(pos,old[0],axes)) if old else (8,8)
    gathering=old is not None and cargo>old[1]+.4
    light=min((DIST[pos][q] for q,wc,k in obs['players'][me][2].values()
               if k=='COLLECTOR' and wc<cargo),default=30)
    length=DIST[pos][home]
    cb=int(cargo>=100)+int(cargo>=200)+int(cargo>=350)
    coarse=(last,cb,int(length>4))
    detailed=(coarse,gathering,min(light//2,3),int(length>7),len(bases)>1 and DIST[pos][bases[1]]-length<=1)
    return axes,coarse,detailed


class Forecast:
    def __init__(self):
        self.hist=[defaultdict(Counter) for _ in range(3)]
        self.min_probability=.35
        self.min_expected_cargo=45

    def add(self,obs,previous,uid,futures):
        pos=obs['players'][1-obs['player']][2][uid][0]
        axes,coarse,detailed=context(obs,previous,uid)
        for depth,end in futures:
            code=vector(pos,end,axes) if end is not None else None
            for hist,key in zip(self.hist,((),coarse,detailed)):
                hist[depth,key][code]+=1

    def offsets(self,depth,coarse,detailed):
        base=self.hist[0][depth,()]
        count=sum(base.values())
        pc={k:v/count for k,v in base.items()}
        for hist,key in zip(self.hist[1:],(coarse,detailed)):
            local=hist.get((depth,key),{})
            n=sum(local.values())
            if n:
                pc={k:(local.get(k,0)+16*v)/(n+16) for k,v in pc.items()}
        return [(offset,p) for offset,p in pc.items() if offset is not None and p>=self.min_probability]

    def crossings(self,obs,previous,uid):
        enemy=obs['players'][1-obs['player']]
        pos=enemy[2][uid][0]
        axes,coarse,detailed=context(obs,previous,uid)
        output={}
        for depth in (3,4,5):
            for offset,p in self.offsets(depth,coarse,detailed):
                dest=project(pos,offset,axes)
                if dest not in enemy[1].values():
                    output[depth,dest]=p
        return output


def fit(folder):
    selected=json.loads((HERE/'interception_fit.json').read_text())['game_ids']
    if set(selected['train']) & set(selected['val']):
        raise ValueError('overlapping game splits')
    manifest=json.loads((HERE/'top5_h2h_data.json').read_text())['games']
    index={r['id']:r for r in manifest}
    model=Forecast()
    for gid in selected['train']:
        item=index[gid]
        if item['split']!='train':
            raise ValueError('wrong training split')
        frames=read_game(folder,item)['steps']
        for t in range(0,len(frames)-5,2):
            for seat in (0,1):
                now=own_view(frames[t],seat)
                previous=own_view(frames[t-1],seat)['players'][1-seat][2] if t else {}
                enemy=now['players'][1-seat]
                for uid,(pos,cargo,kind) in enemy[2].items():
                    if kind!='COLLECTOR' or cargo<50 or not 2<=min(DIST[pos][b] for b in enemy[1].values())<=9:
                        continue
                    y=[]
                    active=True
                    for d in range(1,6):
                        future=frames[t+d][0]['observation']['players'][1-seat][2].get(uid)
                        active=active and future is not None and future[0] not in enemy[1].values() and future[1]>=50
                        if d>=3:
                            y.append((d,future[0] if active else None))
                    model.add(now,previous,uid,y)
    return model,selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replays',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--exploratory',action='store_true')
    args=ap.parse_args()
    model,selected=fit(args.replays)
    if args.exploratory:
        model.min_probability=.15
        model.min_expected_cargo=25
    games=[]
    for name in RUNS:
        run=HERE/name
        header=json.loads((run/'results.json').read_text())
        for row in header['games']:
            g=run_game(run,row,header['header'],model)
            games.append(g)
            print('Sapling',g['seed'],g['seat'],len(g['learned']),flush=True)
    forecast={'training_ids':selected['train'],'validation_ids':selected['val'],
              'fit_carrier_roots':sum(model.hist[0][3,()].values()),
              'policy':{'posterior_smoothing_pseudocount':16,'min_probability':model.min_probability,'min_expected_cargo':model.min_expected_cargo}}
    sapling={'all_roots':sum(g['eligible_roots'] for g in games),'summary':summarize([r for g in games for r in g['learned']]),'games':games}
    args.out.write_text(json.dumps(dict(forecast=forecast,sapling=sapling),separators=(',',':'),allow_nan=False)+'\n')
    # Same frozen predictor and exact same validation roots as the geometric benchmark.
    from ambush5_top5 import run
    run(args.replays,args.out.with_name(args.out.stem.replace('_results','_top5')+'.json'),model)


if __name__=='__main__':
    main()

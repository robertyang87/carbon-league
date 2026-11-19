"""Compare early defensive geometry in all 33 appendix games and the ten Sapling development games."""
from pathlib import Path
import gzip,json,sys
from collections import Counter
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'midgame_lane_dev'))
from coverage import sr
from route import _D,_TO,_DIRECTION
TOP=HERE.parents[1]/'all_top5'
sys.path.insert(0,str(TOP))
import fetch
def scan(seat,frames,cmds,payments):
    seen={};out=[]
    for t in range(100,min(201,len(cmds))):
        own,enemy=frames[t][seat],frames[t][1-seat];homes=list(enemy[1].values())
        assigned={k:int(_TO[w[0],_DIRECTION.index(cmds[t][seat].get(k))]) for k,w in own[2].items()}
        for uid,(p,load,kind) in enemy[2].items():
            if kind!='COLLECTOR' or load<100:continue
            d=min(_D[p,b] for b in homes)
            if not 2<=d<=5:continue
            previous=seen.get(uid,-20);seen[uid]=t
            if previous+8>=t:continue
            exits=[int(q) for q in _TO[p] if min(_D[q,b] for b in homes)<d]
            guards={k:w for k,w in own[2].items() if w[2]=='COLLECTOR' and w[1]<load}
            can=[[k for k,w in guards.items() if _D[w[0],z]<=1] for z in exits]
            # Check bipartite feasibility even when there are more than two nearest-base exits.
            def match(i,taken):
                return i==len(can) or any(match(i+1,taken|{k}) for k in can[i] if k not in taken)
            actual={assigned[k] for k in guards}
            enemy_move=cmds[t][1-seat].get(uid)
            out.append(dict(t=t,target=uid,load=load,d=int(d),exits=exits,coverable=match(0,set()),covered=all(z in actual for z in exits),
                            covered_count=sum(z in actual for z in exits),enemy_cmd=enemy_move,
                            target_receipts=[(j,v) for j in range(t,min(t+10,len(cmds))) for s,k,v in payments[j] if k==uid and s==1-seat]))
    return out
def main():
    d=TOP/'appendix_afc3951b'
    cash={r['id']:r['turns'] for r in sr.load(d/'turn_cash.json.gz')['games']}
    games=[]
    for c in sr.load(d/'manifest.json')['games']:
        env=sr.load(fetch.CACHE/c['path'])
        frames=[f[0]['observation']['players'] for f in env['steps']]
        cmds=[[a.get('action') or {} for a in f] for f in env['steps'][1:]]
        receipts=[t[3] for t in cash[c['id']]]
        for seat,name in enumerate(c['seat_names']):
            if name not in fetch.NAMES:continue
            rows=scan(seat,frames,cmds,receipts)
            games.append(dict(group=name,seed=c['seed'],id=c['id'],seat=seat,rows=rows))
    from turning_point_ledger import CashEvents
    for c in sr.load(HERE.parent/'midgame_lane_dev/coverage.json.gz'):
        raw=sr.load(sr.ROOT/c['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        sim.cash=CashEvents(sim.cash);frames=[];receipts=[]
        for pair in raw['actions']:
            frames.append(sim.obs(c['seat'])['players']);sim.cash.events.clear();sim.step(*pair)
            receipts.append([(e[1],e[3],e[2]) for e in sim.cash.events if e[0]=='own_delivery'])
        games.append(dict(group='Sapling parent',seed=c['seed'],seat=c['seat'],rows=scan(c['seat'],frames,raw['actions'],receipts)))
    sr.retain(HERE/'positioning.json.gz',games)
    for g in sorted({v['group'] for v in games}):
        rows=[r for v in games if v['group']==g for r in v['rows']]
        print(g,len(rows),sum(r['coverable'] for r in rows),sum(r['covered'] for r in rows),sum(bool(r['target_receipts']) for r in rows))
if __name__=='__main__':main()

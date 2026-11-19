from pathlib import Path
import sys
from bank import HERE, choose
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr

def main():
    cases=[]
    for folder in ('planter_capture_full_dev/games/candidate','capture_reach_dev/games/parent','intercept_response_dev/games/parent'):
        for path in sorted((HERE.parent/folder).glob('raw*.gz')):
            raw=sr.load(path)
            seat=int(path.name.split('_')[-1].split('.')[0])
            sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
            hits=[];losses=[]
            for t,pair in enumerate(raw['actions']):
                decision=choose(sim.obs(seat),pair[seat])
                if decision:hits.append(decision)
                sim.step(*pair)
                losses.extend([t,*event] for event in sim.last_collisions if event[1]==seat and event[4]==1-seat and event[2]>=80)
            cases.append(dict(source=str(path.relative_to(sr.ROOT)),sha256=sr.sha(path),seat=seat,seed=raw['seed'],hits=hits,losses=losses))
    sr.retain(HERE/'audit.json',cases)
    for c in cases: print(c['seed'],c['seat'],len(c['hits']),[r['t'] for r in c['hits']][-12:],len(c['losses']))
if __name__=='__main__':main()

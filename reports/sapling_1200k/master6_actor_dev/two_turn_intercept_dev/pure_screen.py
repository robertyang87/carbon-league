"""Check genuine forecast gains independently of the harvesting term before using more court time."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
from route import choose
def screen():
    rows=[]
    for seat,folder in enumerate(('planter_capture_full_dev/games/candidate','capture_reach_dev/games/parent')):
        path,=list((HERE.parent/folder).glob(f'raw_*_{seat}.json.gz'));raw=sr.load(path)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);selected=[]
        for t,pair in enumerate(raw['actions']):
            if 15<=t<=280:
                obs=sim.obs(seat)
                pure=choose(dict(obs,carbon=[0]*225),pair[seat])
                if pure:
                    pos=sim.workers[pure['worker']].pos
                    cost=.2*obs['carbon'][pos]*((pure['old']=='NONE')-(pure['alternative']=='NONE'))
                    if pure['gain']-cost>=5:
                        selected.append(dict(**pure,local_harvest_cost=cost,net_gain=pure['gain']-cost))
            sim.step(*pair)
        rows.append(dict(seat=seat,seed=raw['seed'],proposals=selected))
    sr.retain(HERE/'pure_screen.json',rows)
    print([(r['seat'],len(r['proposals']),r['proposals'][:1]) for r in rows])
if __name__=='__main__':screen()

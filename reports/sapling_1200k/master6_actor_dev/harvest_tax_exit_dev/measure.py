from pathlib import Path
import sys
from collections import Counter
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
sys.path.insert(0,str(HERE.parent/'intercept_response_dev'))
from forecast import _D,_TO,_DIRECTION
from routing import choose

def run():
    rows=[];counts=Counter()
    for g in sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);s=g['seat'];cfg=raw['game_configuration']
        sim=sr.FastCarbon(cfg).reset(seed=g['seed']); alt=sr.FastCarbon(cfg).reset(seed=g['seed'])
        for t,pair in enumerate(raw['actions']):
            change=choose(sim.obs(s),pair[s],cfg,_D,_TO,_DIRECTION,counts)
            if change:
                cmd=[dict(a) for a in pair];cmd[s][change['worker']]=change['alternative']
                alt.set_state(sim.get_state());alt.step(*cmd)
            sim.step(*pair)
            if change:
                worker=change['worker']
                def after(board):
                    w=board.workers.get(worker)
                    return dict(cash=list(board.cash),cargo=w.carbon if w else None)
                rows.append(dict(seed=g['seed'],seat=s,t=t,path=g['path'],pack=raw['pack_sha256'],
                            source_sha256=g['source_sha256'],choice=change,parent=after(sim),altered=after(alt)))
        assert list(sim.cash)==raw['official_score_by_seat']
    sr.retain(HERE/'opportunities.json.gz',rows)
    for r in rows[:6]:print(r['seed'],r['t'],r['seat'],r['choice']['nominal_saving'])
    sr.retain(HERE/'funnel.json',dict(counts));print('TOTAL',len(rows),counts)
if __name__=='__main__':run()

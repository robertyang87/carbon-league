"""For each missed witness, substitute one hunter action against the recorded response."""
import argparse
import gzip
import json
import sys
from pathlib import Path
from raid_timing import HERE,DEST,MOVE
from defense_run import atomic,compact
sys.path.insert(0,str(HERE.parents[2]/'code'))
from fastsim import FastCarbon

def main():
    opts=argparse.ArgumentParser(description=__doc__);opts.add_argument('--cache',type=Path,required=True);a=opts.parse_args()
    evidence=json.loads(gzip.decompress((HERE/'raid_timing.json.gz').read_bytes()))
    out={}
    commands=('NONE','UP','RIGHT','DOWN','LEFT')
    for category,examples in evidence['cases'].items():
        rows=[]
        for e in examples:
            game=json.loads(gzip.decompress((a.cache/'replays'/(e['id']+'.json.gz')).read_bytes()))
            frames=game['steps'];r=e['root'];t=r['step'];s=r['seat'];wid=f"player-{s}-worker-{r['hunter']}";vid=f"player-{1-s}-worker-{r['victim']}"
            for tr in e['trace']:
                i=tr['frame'];ps=frames[i][0]['observation']['players'];w=ps[s][2].get(wid);v=ps[1-s][2].get(vid)
                if w and v:
                    threat={p for _,(q,c,k) in ps[s][2].items() if k=='COLLECTOR' and c<=v[1] for p in DEST[q]}
                    tr['safe_target_cells']=[p for p in DEST[v[0]] if p not in threat and p not in ps[s][1].values()]
                    tr['shared_non_enemybase_cells']=[p for p in DEST[w[0]] if p in DEST[v[0]] and p not in ps[1-s][1].values()]
            if category=='missed':
                sim=FastCarbon(game['configuration']).load_observation(frames[0][0]['observation'])
                for i in range(t):
                    sim.step(*[agent['action'] or {} for agent in frames[i+1]])
                snap=sim.get_state();original=[agent['action'] or {} for agent in frames[t+1]]
                sim.step(*original)
                truth=frames[t+1][0]['observation'];actual=sim.full_observation()
                if actual['players']!=truth['players'] or actual['carbon']!=truth['full_carbon'] or actual['trees']!=truth['trees']:
                    raise ValueError('unmodified branch failed')
                target_dest=DEST[r['victim_cell']][MOVE[original[1-s].get(vid)]]
                op=commands[DEST[r['hunter_cell']].index(target_dest)]
                amended=[dict(x) for x in original]
                if op=='NONE':amended[s].pop(wid,None)
                else:amended[s][wid]=op
                sim.set_state(snap);sim.step(*amended)
                hit=[mass for victim,side,mass,credit,winner in sim.last_collisions if victim==vid and side==1-s and credit==s and winner==s]
                if not hit or wid not in sim.workers or sim.workers[wid].pos!=target_dest or target_dest in dict((p,b) for b,p in sim.bases[1-s]):
                    raise ValueError('single alternative did not receive cargo')
                e['fixed_current_opponent_branch']={'alternate':op,'landing':target_dest,'transferred_now':float(sum(hit)),
                                                   'attacker_immediately_banks':target_dest in dict((p,b) for b,p in sim.bases[s])}
            rows.append(e)
        out[category]=rows
    atomic(HERE/'raid_timing_examples.json.gz',gzip.compress(compact(out),mtime=0))
if __name__=='__main__':main()

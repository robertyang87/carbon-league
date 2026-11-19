"""Check the fixed Top5 estimates on visible states and actual submitted Master3s2 actions."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from tree_raid_replay import HERE, SOURCES, audit
from tree_raid_model import tree_raid_rows
import cnet2_feat as geometry


def main():
    weights = {stage:np.array(json.loads((HERE/name).read_text())['weights'])
               for stage,name in [('occupied','tree_raid_fit.json'),('approach','tree_raid_approach.json')]}
    counts=defaultdict(lambda: {'opportunities':0,'arrivals':0,'predicted_sum':0.,'squared_error_sum':0.})
    for source in SOURCES:
        folder=HERE/source
        for row in json.loads((folder/'results.json').read_text())['games']:
            raw=(folder/row['raw_file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['raw_sha256']:
                raise ValueError('raw identity')
            j=json.loads(gzip.decompress(raw))
            seed,seat=j['seed'],j['candidate_seat']
            random.seed(7919*seed+104729*seat+31)
            sim=audit.exam.FastCarbon(j['game_configuration']).reset(seed=seed)
            for t,actions in enumerate(j['actions']):
                obs=sim.obs(seat)
                if 50<=t<250 and obs['players'][1-seat][0]>=sim.config['seizeCost']:
                    foes=obs['players'][1-seat][2]
                    for stage,w in weights.items():
                        for uid,pos,_,features in tree_raid_rows(obs,geometry,sim.config,approaches=stage=='approach'):
                            move=geometry.MOVES.index(actions[1-seat].get(uid) or 'NONE')
                            hit=int(geometry.DEST[foes[uid][0],move]==pos)
                            p=float(1./(1+np.exp(-np.dot(features,w))))
                            group=stage+('_covered' if features[4] else '_uncovered')
                            rec=counts[group]
                            rec['opportunities']+=1; rec['arrivals']+=hit
                            rec['predicted_sum']+=p;rec['squared_error_sum']+=(p-hit)**2
                sim.step(*actions)
            if audit.prior.hashed(sim.get_state()) != row['final_state_sha256']:
                raise ValueError('end identity')
    for r in counts.values():
        r['p_forecast']=r.pop('predicted_sum')/r['opportunities']
        r['p_observed']=r['arrivals']/r['opportunities']
        r['brier']=r.pop('squared_error_sum')/r['opportunities']
    out=HERE/'tree_raid_opp_check.json'
    out.write_text(json.dumps(dict(sources=SOURCES,range=[50,249],cash='affordable immediately',counts=counts),indent=2)+'\n')
    print(dict(counts))
if __name__=='__main__':
    main()

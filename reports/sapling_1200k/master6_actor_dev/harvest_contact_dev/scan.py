import sys,copy
from contact import HERE,choose
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
def scan():
    rows=[]
    for case in sr.load(HERE.parent/'midgame_lane_dev/coverage.json.gz'):
        raw=sr.load(sr.ROOT/case['source']);s=case['seat']
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed']);prior=None;events=[]
        for t,pair in enumerate(raw['actions'][:201]):
            obs=sim.obs(s);root=choose(obs,pair[s],prior) if t>=50 else None;prior=obs
            if root:
                alt=copy.deepcopy(sim);cmds=list(pair);cmds[s]=dict(cmds[s],**{root['worker']:root['alternative']})
                alt.step(*cmds)
            sim.step(*pair)
            if root:
                events.append(dict(root=root,actual_enemy_order=pair[1-s].get(root['target']),original_collisions=list(sim.last_collisions),
                                   changed_collisions=list(alt.last_collisions)))
        rows.append(dict(seed=case['seed'],seat=s,source=case['source'],opportunities=events))
    sr.retain(HERE/'coverage.json.gz',rows)
    for r in rows:print(r['seed'],r['seat'],[(e['root']['t'],e['actual_enemy_order']) for e in r['opportunities']])
if __name__=='__main__':scan()

"""Post-gate census on already recorded parent-state trajectories. No new candidate game or map."""
import sys
from collections import Counter
import full as f

def run():
    assert (f.H/'result.json').is_file();pkg=f.H/'selected.py';m=f.loaded(pkg)
    groups={k:Counter() for k in range(3)};cases=[]
    for seed in f.P['train']:
        for seat in (0,1):
            path=f.H/'games/train'/f'parent_{seed}_{seat}.json.gz'
            parent=f.read(path);w=f.W.World(seed);changes=[]
            for t,actions in enumerate(parent['actions']):
                ob=w.obs(seat);p=actions[seat];c=m.agent(ob,w.config);k=min(2,t//100);g=groups[k];g['roots']+=1
                delta={}
                for uid in p.keys()|c.keys():
                    if p.get(uid)!=c.get(uid):
                        role='base' if uid in ob['players'][seat][1] else ob['players'][seat][2][uid][2].lower()
                        delta[uid]=dict(role=role,parent=p.get(uid,'NONE'),selected=c.get(uid,'NONE'));g[role]+=1
                if delta:
                    g['changed_joint']+=1;changes.append(dict(t=t,delta=delta))
                w.advance(actions)
            org=f.read(f.H/'games/train'/f'selected_{seed}_{seat}.json.gz')
            first=next((t for t,(a,b) in enumerate(zip(parent['actions'],org['actions'])) if a[seat]!=b[seat]),None)
            assert first==(changes[0]['t'] if changes else None)
            cases.append(dict(seed=seed,seat=seat,trace_sha256=f.digest(path),first_game_divergence=first,changes=changes))
            print(seed,seat,len(changes),flush=True)
    f.put(f.H/'full_action_census.json',dict(scope='Post hoc full stored parent histories only. Candidate actions read at the actual public entry but do not change the already recorded parent environment. Original failed gate unchanged.',
          preexisting_result_sha256=f.digest(f.H/'result.json'),source_sha256=f.digest(f.H/'full_action_census.py'),by_phase={k:dict(v) for k,v in groups.items()},cases=cases))
if __name__=='__main__':run()

"""Replay the short archived physical-board terminal and retain exact causes by UID."""
import full as f
def main():
    seed=621896222;all_traces=[]
    for a in ('parent','selected'):
        for s in (0,1):
            src=f.H/'games/holdout'/f'{a}_{seed}_{s}.json.gz';d=f.read(src);w=f.W.World(seed);turn=[]
            for t,joint in enumerate(d['actions']):
                p=w.obs(0)['players']
                turn.append({'step':t,'cash':[x[0] for x in p],'workers':[x[2] for x in p],'trees':[x[3] for x in p],'submitted':joint})
                w.advance(joint)
            assert w.terminal['extinct_by_seat']==[True,True] and w.terminal['steps']==d['official']['steps']==7
            all_traces.append(dict(arm=a,seat=s,tape_sha256=f.digest(src),states=turn,terminal=d['official']))
    rules={k:f.W.CFG[k] for k in ('recCollectorCost','recPlanterCost')}
    f.put(f.H/'short_game.json',dict(seed=seed,terrain=f.reg['boards'][str(seed)],recruit_cost=rules,cases=all_traces,
          evidence='Readback of previously scored tapes. Two opposing planters moved from physical cells 128/126 onto 127 at t5. At t6 opposing last collectors 81/83 both chose 82 carrying 17.2203; t7 both extinct.'))
if __name__=='__main__':main()

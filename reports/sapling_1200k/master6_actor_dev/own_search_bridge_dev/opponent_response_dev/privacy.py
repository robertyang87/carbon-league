"""Change every hidden carbon cell before inference, then restore the real replay before advancement."""
from pathlib import Path
import numpy as np
import train as run

def check():
    rec=run.load(run.HERE/'sources.json')
    wt=dict(np.load(run.HERE/'opponent.npz',allow_pickle=False))
    rows=[]
    for source in rec['records']:
        if source['split']!='rollout_hold': continue
        raw=run.load(run.ROOT/source['raw'])
        s=source['side']
        sim=run.bridge.FastCarbon(raw['game_configuration']).reset(seed=source['seed'])
        chains=[(run.F.Feat(s),run.F.Feat(1-s)) for _ in range(2)]
        alters=0
        frames=0
        hidden_cells=0
        for t,joint in enumerate(raw['actions'][:40]):
            original=sim.carbon.copy()
            honest=sim.obs(s)
            # Only the independent check ever reads the enemy's actual private observation.
            private_carbon_before=sim.obs(1-s)['carbon']
            visible=set()
            for pos,_,kind in honest['players'][s][2].values():
                visible.update([pos] if kind=='COLLECTOR' else sim._nbr4[pos])
            for cell in range(sim.n_cells):
                if cell not in visible:
                    sim.carbon[cell]=37.125 if original[cell]!=37.125 else 12.875
                    hidden_cells+=1
            altered=sim.obs(s)
            alters+=int(private_carbon_before!=sim.obs(1-s)['carbon'])
            sim.carbon=original
            assert honest==altered
            predicted=[]
            for (local,shadow),view in zip(chains,(honest,altered)):
                local.observe(view)
                x=shadow.observe(run.bridge.public_enemy(view,local.est))
                predicted.append((x,run.bridge.decode(shadow,x,wt)[0]))
            for key,a in predicted[0][0].items():
                assert np.array_equal(a,predicted[1][0][key]),key
            assert predicted[0][1]==predicted[1][1]
            frames+=1
            sim.step(*joint)
        rows.append(dict(raw_sha256=source['raw_sha256'],seed=source['seed'],seat=s,
                         public_observation_and_prediction_equal=frames,enemy_private_view_changed=alters,
                         hidden_cell_perturbations=hidden_cells))
    assert sum(x['enemy_private_view_changed'] for x in rows)>0
    run.bridge.save(run.HERE/'privacy.json',{'weights_sha256':run.sha(run.HERE/'opponent.npz'),'source_sha256':run.sha(Path(__file__)),'games':rows})
if __name__=='__main__':
    check()

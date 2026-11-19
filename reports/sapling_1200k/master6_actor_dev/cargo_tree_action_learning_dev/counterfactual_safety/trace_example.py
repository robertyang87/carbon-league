"""Explain the single held near-home case where all target orders match but control alone keeps cargo."""
from gather import HERE,EXP,read,write,policy,FastCarbon,arena
def main():
    card=next(c for c in read(EXP/'panel.json')['cards'] if c['seed']==621497120 and c['seat']==0)
    raw=read(arena.ROOT/card['path']);cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=card['seed'])
    files={arm:(EXP/'candidate.py' if arm=='parent' else HERE/(arm+'.py')) for arm in ('parent','control','candidate')}
    models={k:policy.load(p) for k,p in files.items()}
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(0)
        if t in (0,296):
            actions={k:m.agent(obs,cfg) for k,m in models.items()}
            if t==296:
                seat=0;uid='player-0-worker-53';pre=obs['players'][seat]
                changes={wid:{k:a.get(wid,'STAY') for k,a in actions.items()} for wid in pre[2] if len({a.get(wid,'STAY') for a in actions.values()})>1}
                st=sim.get_state();out={}
                for k,a in actions.items():
                    fork=FastCarbon(cfg);fork.set_state(st);fork.step(a,pair[1]);out[k]=dict(command=a.get(uid,'STAY'),alive=uid in fork.workers,held=fork.workers[uid].carbon if uid in fork.workers else 0)
                assert out['control']['alive'] and not out['candidate']['alive']
                write(HERE/'near_home_example.json',dict(raw_sha256=card['sha256'],seed=card['seed'],seat=0,t=t,uid=uid,public_target=pre[2][uid],
                    changed_worker_orders=changes,public_worker_context={wid:pre[2][wid] for wid in changes},opponent_order=pair[1],all_own_orders=actions,targets=out))
                return
        else:
            for m in models.values():m._FEAT.observe(obs)
        sim.step(*pair)
if __name__=='__main__':main()

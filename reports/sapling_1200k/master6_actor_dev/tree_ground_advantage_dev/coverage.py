from fit import HERE,ACTOR,sr
from decision import consider,update_memory
import overlay
overlay.consider=consider;overlay.update_memory=update_memory
def run():
    known=sr.load(ACTOR/'tactical_exchange_dev/paired.json.gz')
    fresh=sorted((ACTOR/'elite_role_mix_dev/conditional/fresh_paired_dev/games').glob('*/raw_*.json.gz'))
    sources=[sr.ROOT/g['path'] for g in known]+fresh;weights=sr.load(HERE/'weights.json')['coef']
    rows=[]
    for source in sources:
        raw=sr.load(source);s=raw['candidate_seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);picker=overlay.GroundSelection()
        exact=source in fresh and source.parent.name=='parent'
        if exact:mod=sr.live.module(HERE/'candidate.py','real_ground_candidate')
        record=None
        for t,cmd in enumerate(raw['actions']):
            obs=sim.obs(s);expected=picker.act(obs,sim.config,cmd[s],weights)
            if exact:
                assert mod.agent(obs,sim.config)==expected,(raw['seed'],s,t)
            if expected!=cmd[s]:
                assert record is None
                chosen=picker.seats[s]['selected'];uid=chosen['worker'];alt=sr.FastCarbon(sim.config);alt.set_state(sim.get_state());alt.cash=list(alt.cash)
                pair=[dict(a) for a in cmd];pair[s]=expected;alt.step(*pair)
                tree=next((v for v in alt.trees.values() if v.pos==chosen['cell'] and v.player==s),None)
                record=dict(**chosen,step_verified=tree is not None and uid in alt.workers)
                # Official engine verifies both the identity and the single changed joint step.
                env=sr.live.begin(raw)
                for old in raw['actions'][:t]:env.step(old)
                env.step(pair);assert sr.live.public(env,s)['players']==alt.obs(s)['players']
            sim.step(*cmd)
        rows.append(dict(source=str(source.relative_to(sr.ROOT)),seed=raw['seed'],seat=s,actor=raw['pack_sha256'],exact_entry=exact,first=record))
    sr.retain(HERE/'coverage.json',dict(parent=sr.sha(ACTOR/'actor_partner_dev/candidate.py'),candidate=sr.sha(HERE/'candidate.py'),games=rows))
    print('called',len(rows),'triggers',sum(r['first'] is not None for r in rows),'335',[(r['seat'],r['exact_entry'],r['first']) for r in rows if r['seed']==621758335])
if __name__=='__main__':run()

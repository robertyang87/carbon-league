"""Verify deployed routing and the real Sapling contribution on each completed full candidate tape."""
from measure import HERE,sr,_D,_TO,_DIRECTION,choose

def scan(s):
    seed=sr.load(HERE/'protocol.json')['seeds'][0]
    path=HERE/f'games/candidate/raw_{seed}_{s}.json.gz'
    raw=sr.load(path)
    assert raw['pack_sha256']==sr.sha(HERE/'candidate.py')
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=seed)
    alt=sr.FastCarbon(raw['game_configuration']).reset(seed=seed)
    net=sr.live.module(HERE.parent/'planter_capture_full_dev/candidate.py','toll_parent_replay')
    mix=net._mixed_actions;ablation=[];changes=[]
    def probe(x,aux):
        act=mix(x,aux)
        if t%17==0:
            weight=net._HYBRID['student_weight']
            try:
                net._HYBRID['student_weight']=0.
                masked=mix(x,aux)
            finally:net._HYBRID['student_weight']=weight
            full,other=map(net._FEAT.to_actions,(act,masked))
            def routing(cmd):
                z=choose(obs,cmd,sim.config,_D,_TO,_DIRECTION)
                if z:cmd[z['worker']]=z['alternative']
                return cmd
            full,other=map(routing,(full,other))
            diff=[k for k in obs['players'][s][2] if full.get(k)!=other.get(k)]
            ablation.append(dict(t=t,n=len(obs['players'][s][2]),different=diff))
        return act
    net._mixed_actions=probe
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(s);baseline=net.agent(obs,sim.config);candidate=dict(baseline)
        change=choose(obs,baseline,sim.config,_D,_TO,_DIRECTION)
        if change:
            candidate[change['worker']]=change['alternative']
            alt.set_state(sim.get_state());cf=list(pair);cf[s]=baseline;alt.step(*cf)
        assert candidate==pair[s],(s,t)
        sim.step(*pair)
        if change:
            uid=change['worker']; w=sim.workers.get(uid);v=alt.workers.get(uid)
            changes.append(dict(t=t,choice=change,candidate_cash=list(sim.cash),baseline_cash=list(alt.cash),
                                candidate_cargo=w.carbon if w else None,baseline_cargo=v.carbon if v else None))
    assert list(sim.cash)==raw['official_score_by_seat']
    sr.retain(HERE/f'audit_seat{s}.json',dict(source_sha256=sr.sha(path),edits=changes,ablation=ablation,scores=list(sim.cash)))
    print(s,len(changes),sum(len(a['different']) for a in ablation),list(sim.cash))
if __name__=='__main__':
    import sys
    scan(int(sys.argv[1]))

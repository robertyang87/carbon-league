from scan import sr,HERE
def verify():
    r=[]
    for seat in (0,1):
        d=HERE/f'seat{seat}';q=sr.load(d/'live_protocol.json');root=q['root'];raw=sr.load(sr.ROOT/q['record']['source'])
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
        net=sr.live.module(HERE/'candidate.py','harvest_pack')
        for t,cmd in enumerate(raw['actions'][:root['t']+1]):
            a=net.agent(sim.obs(seat),raw['game_configuration'])
            expected=dict(cmd[seat])
            if t==root['t']:expected[root['worker']]=root['alternative']
            assert a==expected,(seat,t)
            joint=list(cmd);joint[seat]=a;sim.step(*joint)
        a=net.agent(sim.obs(seat),raw['game_configuration'])
        assert a==sr.load(d/f"follow_{root['t']+1}.json")['executed']
        r.append(dict(seat=seat,through=root['t']+1,seed=q['seed'],candidate_sha256=sr.sha(HERE/'candidate.py')))
    sr.retain(HERE/'deployment.json',r)
if __name__=='__main__':verify()

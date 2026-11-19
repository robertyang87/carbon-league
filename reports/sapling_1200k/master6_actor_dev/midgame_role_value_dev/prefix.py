from data import sr,HERE
def check():
    rows=[]
    for seat in (0,1):
        out=HERE/f'seat{seat}';q=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed']);net=sr.live.module(HERE/'candidate.py','role_pack_check')
        for t,pair in enumerate(raw['actions'][:q['root_t']+1]):
            action=net.agent(sim.obs(seat),raw['game_configuration']);expected=dict(pair[seat])
            if t==q['root_t']:expected[q['root']['worker']]='RECCOLLECTOR'
            assert action==expected
            cmd=list(pair);cmd[seat]=action;sim.step(*cmd)
        rows.append(dict(seed=q['seed'],seat=seat,through=q['root_t'],pack=sr.sha(HERE/'candidate.py')))
    sr.retain(HERE/'deployment.json',rows)
if __name__=='__main__':check()

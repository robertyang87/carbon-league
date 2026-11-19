from response import HERE,sr
from measure import choose,_D,_TO,_DIRECTION
import sys
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
from ledger import check
def run():
    pairs=[];parity=[]
    for s,out in ((0,HERE/'seat0'),(1,HERE/'exact_parent/seat1')):
        q=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
        actor=sr.live.module(HERE/'candidate.py','toll_parity')
        count=0
        for t,pair in enumerate(raw['actions'][:q['root_t']+1]):
            obs=sim.obs(s);change=choose(obs,pair[s],sim.config,_D,_TO,_DIRECTION);want=dict(pair[s])
            if change:want[change['worker']]=change['alternative'];count+=1
            assert actor.agent(obs,sim.config)==want
            sim.step(*pair)
        assert change and change['worker']==q['root']['worker']
        parity.append(dict(seat=s,changes=count,steps=q['root_t']+1,pack=sr.sha(HERE/'candidate.py')))
        pairs.append([check(out,s,arm) for arm in ('parent','approach')])
    sr.retain(HERE/'exact_parent_terminal.json',pairs);sr.retain(HERE/'parity.json',parity)
    gains=[b['margin']-a['margin'] for a,b in pairs]
    saved=[b['first_five'][0]['after'][b['root']['worker']][1]-a['first_five'][0]['after'][a['root']['worker']][1] for a,b in pairs]
    banks=[sum(x[2] for x in b['deposits'] if x[1]==b['root']['worker']) for _,b in pairs]
    passed=min(saved)>0 and max(banks)>0 and sum(gains)>0 and min(gains)>=-300
    sr.retain(HERE/'gate.json',dict(margin_gains=gains,root_freight_saved=saved,focal_deposits=banks,passed=passed));return passed
if __name__=='__main__':print(run())

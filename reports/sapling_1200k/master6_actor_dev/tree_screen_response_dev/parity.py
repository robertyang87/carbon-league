"""Reproduce the changed deployed commands on the three complete original pre-action histories."""
from screen_trial import HERE,sr
import sys
sys.path.insert(0,str(HERE.parent/'tactical_exchange_dev'))
from approach import choose

def run():
    rows=[]
    for out in (HERE/'seat0',HERE/'first_trigger/seat0',HERE/'opposite_map/seat1'):
        q=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
        seat=q['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
        net=sr.live.module(HERE/'candidate.py','screen_public_parity')
        previous={};edits=[];missing=[]
        for t,pair in enumerate(raw['actions'][:q['root_t']+1]):
            obs=sim.obs(seat)
            choice=choose(obs,pair[seat],sim.config,previous)
            cmd=dict(pair[seat])
            if choice:
                cmd[choice['uid']]=choice['after'];edits.append(t)
            actual=net.agent(obs,sim.config)
            if cmd!=actual:missing.append(t)
            previous=obs['players'][1-seat][2]
            sim.step(*pair)
        assert q['root_t'] in edits
        rows.append(dict(seed=q['seed'],seat=seat,edits=len(edits),steps=edits,mismatch=len(missing),bad_steps=missing))
    sr.retain(HERE/'parity.json',rows)
    print(rows)
if __name__=='__main__':run()

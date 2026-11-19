"""Replay 10 preceding turns, including whether the victim could still reach a bank."""
from audit import sr,HERE
from bank import _D
def extract():
    trails=[]
    for case in sr.load(HERE/'audit.json'):
        for end,uid,seat,amount,_,_ in case['losses']:
            raw=sr.load(sr.ROOT/case['source'])
            sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
            history=[]
            for t,pair in enumerate(raw['actions']):
                ob=sim.obs(seat)
                if end-10<=t<=end:
                    own,enemy=ob['players'][seat],ob['players'][1-seat]
                    w=own[2].get(uid)
                    history.append(dict(t=t,worker=w,action=pair[seat].get(uid),home_distance=int(min(_D[w[0],b] for b in own[1].values())) if w else None,
                        cash=[v[0] for v in ob['players']],commands=pair,players=ob['players']))
                sim.step(*pair)
                if t==end:break
            trails.append(dict(seed=case['seed'],seat=seat,loss=amount,worker=uid,history=history))
    sr.retain(HERE/'loss_trails.json.gz',trails)
    for r in trails:print(r['seed'],[(p['t'],p['home_distance'],p['action']) for p in r['history']])
if __name__=='__main__':extract()

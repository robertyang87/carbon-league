"""Run the actual new package on every original prefix of the three changed same-parent games."""
from fit import HERE,ACTOR,sr
def run():
    rows=sr.load(HERE/'probe.json')
    parent=sr.sha(ACTOR/'planter_capture_full_dev/candidate.py')
    sources=list(dict.fromkeys(r['source'] for r in rows if r['score']['accept'] and r['pack']==parent))
    out=[]
    for src in sources:
        raw=sr.load(sr.ROOT/src);s=raw['candidate_seat'];cfg=raw['game_configuration'];sim=sr.FastCarbon(cfg).reset(seed=raw['seed'])
        net=sr.live.module(HERE/'candidate.py','payoff_screen_probe')
        edits={r['t']:r['root'] for r in rows if r['source']==src and r['score']['accept']}
        for t,pair in enumerate(raw['actions']):
            ob=sim.obs(s);actual=net.agent(ob,cfg)
            expected=dict(pair[s])
            if t in edits:expected[edits[t]['uid']]=edits[t]['after']
            assert expected==actual,(src,t,actual,expected)
            sim.step(*pair)
        out.append(dict(source=src,seed=raw['seed'],seat=s,steps=len(raw['actions']),changed=sorted(edits)))
    sr.retain(HERE/'parity.json',dict(parent=parent,pack=sr.sha(HERE/'candidate.py'),sources=out))
    print(out)
if __name__=='__main__':run()

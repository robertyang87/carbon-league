"""Replay deployed policies on identical public inputs and check the exact recorded commands."""
import sys
from match import HERE,ACTOR,PARENT,CAND,native

def compare(path,recorded,first):
    raw=native.read(path)
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"]);seat=raw["candidate_seat"]
    parent=native.module(PARENT,"freight_comparison_parent")
    cand=native.module(CAND,"freight_comparison_candidate")
    changes=[]
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(seat)
        if not first or t==0 or any(cand.carriers(obs["players"],1-seat)):
            a,b=parent.agent(obs,sim.config),cand.agent(obs,sim.config)
            assert (a if recorded=='parent' else b)==pair[seat]
            if a!=b:
                units={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
                changes.append(dict(step=t,units=units))
                if first:break
        else:
            parent._FEAT.observe(obs);cand._FEAT.observe(obs)
        sim.step(*pair)
    return dict(seed=raw['seed'],seat=seat,trajectory_sha256=native.sha(path),changes=changes)

if __name__=='__main__':
    before=sys.argv[1:]==['before']
    # Both historical arms used exactly this frozen parent package.
    sources=[ACTOR/'capture_reach_dev/games/parent',ACTOR/'planter_capture_full_dev/games/candidate'] if before else [HERE/'games/candidate']
    rows=[compare(p,'parent' if before else 'candidate',before) for source in sources for p in sorted(source.glob('raw_*.json.gz'))]
    assert len(rows)==(4 if before else 2)
    native.save(HERE/('preflight.json' if before else 'changes.json'),rows)

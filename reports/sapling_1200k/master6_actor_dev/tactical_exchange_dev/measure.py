"""Enumerate every changed-action root and compare with its naturally sampled simultaneous opponent action."""
from guard_tree import HERE, intervention
import sys
sys.path.insert(0,str(HERE.parent/"tree_response_training"))
import select_roots as sr


def verify(path, advance=False):
    raw=sr.load(path)
    cfg=raw["game_configuration"]
    seat=int(path.stem.split("_")[-1].split(".")[0])
    parent=sr.FastCarbon(cfg).reset(seed=raw["seed"])
    alt=sr.FastCarbon(cfg).reset(seed=raw["seed"])
    rows=[]
    from frontier import preempt
    prior={}
    for t,pair in enumerate(raw["actions"]):
        obs=parent.obs(seat)
        choice=preempt(obs,pair[seat],cfg,prior) if advance else intervention(obs,pair[seat],cfg)
        prior=obs["players"][1-seat][2]
        if choice:
            changed=[dict(c) for c in pair]
            changed[seat][choice["uid"]]=choice["after"]
            alt.set_state(parent.get_state())
            alt.step(*changed)
        parent.step(*pair)
        if choice:
            def read(sim):
                tree=sim.trees.get(choice["tree"])
                return dict(holder=None if tree is None else tree.player,
                            balance=[float(c) for c in sim.cash],
                            enemy_alive=choice["enemy"] in sim.workers,
                            collector_alive=choice["uid"] in sim.workers)
            rows.append(dict(t=t,choice=choice,enemy_command=pair[1-seat].get(choice["enemy"]),
                             parent=read(parent),changed=read(alt)))
    assert all(abs(a-b)<0.011 for a,b in zip(parent.cash,raw["official_score_by_seat"]))
    return dict(seed=raw["seed"],seat=seat,path=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path),steps=len(raw["actions"]),rows=rows)


if __name__=="__main__":
    prior=sr.load(HERE.parent/"harvest_contact_dev/coverage.json.gz")
    paths=[sr.ROOT/r["source"] for r in prior]
    for batch in ("midgame_role_value_dev","harvest_contact_dev"):
        paths.extend(sorted((HERE.parent/batch/"games").glob("*/raw_*.json.gz")))
    assert len(set(paths))==len(paths)
    advance=sys.argv[1:]==["advance"]
    results=[verify(p,advance) for p in paths]
    sr.retain(HERE/("advance.json.gz" if advance else "paired.json.gz"),results)
    print([(r["seed"],r["seat"],len(r["rows"])) for r in results])

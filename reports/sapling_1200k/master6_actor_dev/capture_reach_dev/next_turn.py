"""One-transition alternate approach against the simultaneous, actually issued enemy commands."""
from match import HERE,native

def audit(row):
    s=row["seat"]
    raw=native.read(next((HERE/"games/candidate").glob(f"raw_*_{s}.json.gz")))
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    parent=native.module(HERE.parent/"planter_capture_full_dev/candidate.py","threat_parent")
    edits=[e for e in row["edits"] if e.get("prior_enemy_tree") and e.get("prior")]
    by_step={e["t"] for e in edits}
    rows=[]
    for t,pair in enumerate(raw["actions"][:max(by_step,default=0)+1]):
        obs=sim.obs(s)
        if t==0 or t in by_step:
            old=parent.agent(obs,sim.config)
        else:
            parent._FEAT.observe(obs)
        if t in by_step:
            fork=native.exam.FastCarbon(native.exam.CFG).set_state(sim.get_state())
            joint=list(pair);joint[s]=old;fork.step(*joint)
            for e in [x for x in edits if x["t"]==t]:
                tid=e["prior_tree"];cell=e["prior_cell"];wid=e["worker"]
                tree=fork.trees.get(tid)
                def dist(a,b):
                    y,x=abs(a//15-b//15),abs(a%15-b%15)
                    return min(y,15-y)+min(x,15-x)
                nearest=min((dist(cell,w.pos) for w in fork.workers.values() if w.player!=s),default=99)
                available=(wid in fork.workers and fork.workers[wid].pos==cell and tree is not None and
                           tree.player==1-s and tree.age<fork.config["treeLifespan"] and fork.cash[s]>=20)
                rows.append(dict(step=t,unit=wid,tree=tid,available=available,nearest_after=nearest,
                                 uncontested=available and nearest>1))
        sim.step(*pair)
    return dict(seat=s,branches=rows)
if __name__=="__main__":
    out=[audit(r) for r in native.read(HERE/"outcomes.json")]
    native.save(HERE/"next_turn.json",out)
    for r in out:
        a=r["branches"]; print(r["seat"],len(a),sum(x["available"] for x in a),sum(x["uncontested"] for x in a))

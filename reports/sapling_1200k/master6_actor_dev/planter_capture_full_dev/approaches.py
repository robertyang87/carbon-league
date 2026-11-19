"""Trace whether a changed approach leaves a live, affordable and defensible chance to seize."""
from match import HERE, native


def distance(a,b):
    dy,dx=abs(a//15-b//15),abs(a%15-b%15)
    return min(dx,15-dx)+min(dy,15-dy)

def trace(r):
    s=r["seat"]
    path,=list((HERE/"games/candidate").glob(f"raw_*_{s}.json.gz"))
    raw=native.read(path)
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    rows=[dict(change=c,history=[]) for c in r["changes"] if c.get("kind")=="planter" and c.get("target_owner")==1-s]
    for t,pair in enumerate(raw["actions"]):
        for row in rows:
            c=row["change"]
            if not c["step"]-3<=t<=c["step"]+5:continue
            worker=sim.workers.get(c["unit"])
            tree=sim.trees.get(c["target"])
            enemies=[dict(id=wid,pos=w.pos,dist=distance(w.pos,c["dest"]),order=pair[1-s].get(wid))
                     for wid,w in sim.workers.items() if w.player!=s and distance(w.pos,c["dest"])<=2]
            if t == c["step"]+1 and worker and worker.pos == c["dest"] and tree and tree.player != s:
                branch = native.exam.FastCarbon(native.exam.CFG).set_state(sim.get_state())
                new_pair = [dict(commands) for commands in pair]
                new_pair[s].pop(c["unit"],None)
                branch.step(*new_pair)
                target = branch.trees.get(c["target"])
                row["single_turn_stay"] = dict(alive=c["unit"] in branch.workers,
                    captured=bool(target and target.player==s),enemy_actions=pair[1-s],
                    cash_after=branch.cash[s], tree_payment=branch.trees_dict[c["target"]][1] if target and target.player==s else 0)
            row["history"].append(dict(step=t,pos=worker.pos if worker else None,order=pair[s].get(c["unit"]),
                                      tree_owner=tree.player if tree else None,cash=sim.cash[s],nearby_enemies=enemies))
        sim.step(*pair)
    return rows

if __name__=="__main__":
    out=[row for r in native.read(HERE/"audit.json") for row in trace(r)]
    native.save(HERE/"approach_checks.json",out)
    for r in out:
        c=r["change"];print(c["step"],[(h["pos"],h["order"],h["tree_owner"],len(h["nearby_enemies"])) for h in r["history"][3:7]])

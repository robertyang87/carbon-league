"""Follow every actual extra stay to purchase and all future income on the resulting tree."""
from match import HERE, PARENT, CAND, native


def realize(path):
    raw = native.read(path)
    s = raw["candidate_seat"]
    sim = native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    parent = native.module(PARENT, "payback_parent")
    cand = native.module(CAND, "payback_candidate")
    launches, track, revisions = [], {}, []
    for t, pair in enumerate(raw["actions"]):
        obs = sim.obs(s)
        a, b = parent.agent(obs,sim.config), cand.agent(obs,sim.config)
        assert b == pair[s]
        changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        if changed:
            revisions.append([t,changed])
        stayed = [(k,sim.workers[k].pos) for k in changed if k in sim.workers and
                  not sim.workers[k].is_collector and not b.get(k) and a.get(k)]
        old = set(sim.trees)
        cfg = sim.config
        alive = sum(tree.age < cfg["treeLifespan"] for tree in sim.trees.values())
        price = cfg["plantCost"]
        if len(sim.player_trees[s]) > cfg["playerTreeProtectiveNumber"]:
            price += cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** alive
        sim.step(*pair)
        newborn = {v.pos:k for k,v in sim.trees.items() if k not in old and v.player==s}
        for wid, pos in stayed:
            tid = newborn.get(pos)
            item = dict(step=t,worker=wid,site=pos,tree=tid,plant_cost=price if tid else None,own_gross=0.,enemy_gross=0.)
            launches.append(item)
            if tid:
                track[tid] = item
        for tid, event in track.items():
            if tid in sim.trees:
                field = "own_gross" if sim.trees[tid].player == s else "enemy_gross"
                event[field] += sim.trees_dict[tid][1]
    return dict(seat=s, changed_steps=revisions, extra_stays=launches)

if __name__ == "__main__":
    records = [realize(f) for f in sorted((HERE/"games/candidate").glob("raw_*.json.gz"))]
    assert len(records)==2
    native.save(HERE/"realization.json",records)

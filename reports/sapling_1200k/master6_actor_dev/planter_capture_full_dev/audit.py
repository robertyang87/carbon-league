"""Replay both policy decisions on candidate states; follow actual title transfers and receipts."""
from match import HERE, CAND, PARENT, native


def analyze(path):
    raw = native.read(path)
    seat = raw["candidate_seat"]
    sim = native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    cand = native.module(CAND, "capture_candidate")
    parent = native.module(PARENT, "capture_parent")
    changes, captures = [], []
    active = {}
    for t, pair in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        actual = cand.agent(obs, sim.config)
        base = parent.agent(obs, sim.config)
        assert pair[seat] == actual
        before = {tid:tr.player for tid,tr in sim.trees.items()}
        on_tree = {tr.pos:tid for tid,tr in sim.trees.items()}
        for k in sorted(actual.keys() | base.keys()):
            if actual.get(k) == base.get(k):
                continue
            ev = dict(step=t,unit=k,parent=base.get(k),actual=actual.get(k))
            if k in sim.workers:
                unit = sim.workers[k]
                action = actual.get(k)
                dest = sim._move_nbr[action][unit.pos] if action else unit.pos
                ev.update(start=unit.pos,dest=dest,kind="collector" if unit.is_collector else "planter",
                          target=on_tree.get(dest),target_owner=before.get(on_tree.get(dest)))
            changes.append(ev)
        sim.step(*pair)
        for tid in list(active):
            if tid not in sim.trees or sim.trees[tid].player != seat:
                del active[tid]
        for tid,tr in sim.trees.items():
            if tr.player == seat and before.get(tid) == 1-seat:
                worker = sim.trees_dict[tid][0]
                event = dict(tree=tid,step=t,worker=worker,age=tr.age,site=tr.pos,
                             cost=sim.config["seizeCost"],gross=0.,held_steps=0)
                captures.append(event)
                active[tid] = event
            if tid in active:
                active[tid]["gross"] += sim.trees_dict[tid][1]
                active[tid]["held_steps"] += 1
    return dict(seat=seat,changes=changes,captures=captures)


if __name__ == "__main__":
    rows = [analyze(p) for p in sorted((HERE/"games/candidate").glob("raw_*.json.gz"))]
    assert len(rows) == 2
    native.save(HERE/"audit.json",rows)

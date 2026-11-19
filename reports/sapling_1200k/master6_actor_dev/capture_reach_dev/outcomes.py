"""Audit each altered command and all tree receipts on the candidate trajectory."""
from match import HERE, CAND, PARENT, native

def dest(sim,w,cmd):
    return sim._move_nbr[cmd][w.pos] if cmd else w.pos

def check(path):
    raw=native.read(path)
    seat=raw["candidate_seat"]
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    cand=native.module(CAND,"reach_outcome_candidate")
    parent=native.module(PARENT,"reach_outcome_parent")
    edits=[]
    flows=[]
    for t,pair in enumerate(raw["actions"]):
        obs=sim.obs(seat)
        actual=cand.agent(obs,sim.config)
        old=parent.agent(obs,sim.config)
        assert actual==pair[seat]
        titles={tid:tr.player for tid,tr in sim.trees.items()}
        cells={tr.pos:tid for tid,tr in sim.trees.items()}
        fork=None
        if old!=actual:
            fork=native.exam.FastCarbon(native.exam.CFG).set_state(sim.get_state())
            joint=list(pair)
            joint[seat]=old
            fork.step(*joint)
        for wid in sorted(actual.keys()|old.keys()):
            if actual.get(wid)==old.get(wid):
                continue
            worker=sim.workers.get(wid)
            item=dict(t=t,worker=wid,prior=old.get(wid),actual=actual.get(wid))
            if worker is not None:
                cell=dest(sim,worker,old.get(wid))
                tid=cells.get(cell)
                item.update(origin=worker.pos,prior_cell=cell,prior_tree=tid,
                            prior_enemy_tree=titles.get(tid)==1-seat,
                            alternate_alive=wid in fork.workers)
            edits.append(item)
        sim.step(*pair)
        for item in edits:
            if item["t"]==t:
                item["actual_alive"]=item["worker"] in sim.workers
            if item["t"]<=t<=item["t"]+5:
                worker=sim.workers.get(item["worker"])
                item.setdefault("actual_path",[]).append(worker.pos if worker else None)
        for tid,tr in sim.trees.items():
            if titles.get(tid)==1-seat and tr.player==seat:
                flows.append(dict(step=t,tree=tid,gross=0.0,turns=0,active=True))
        for flow in flows:
            if not flow["active"]: continue
            tree=sim.trees.get(flow["tree"])
            flow["active"]=tree is not None and tree.player==seat
            if flow["active"]:
                flow["turns"]+=1
                flow["gross"]+=sim.trees_dict[tree.id][1]
    return dict(seat=seat,edits=edits,captures=flows)

if __name__=="__main__":
    rows=[check(p) for p in sorted((HERE/"games/candidate").glob("raw_*.json.gz"))]
    assert len(rows)==2
    native.save(HERE/"outcomes.json",rows)

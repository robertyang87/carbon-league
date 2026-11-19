"""Attribute all realized tree payments to the original planter and current owner."""
from pathlib import Path
from match import HERE, native


def audit(path):
    raw = native.read(path)
    sim = native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    trees = {}
    for t, commands in enumerate(raw["actions"]):
        sim.step(*commands)
        for tid, tree in sim.trees.items():
            owner = tree.player
            item = trees.setdefault(tid, dict(origin=owner, cell=tree.pos, planted=t, spans=[]))
            spans = item["spans"]
            if not spans or spans[-1]["owner"] != owner:
                spans.append(dict(owner=owner, start=t, end=t, income=0.0))
            span = spans[-1]
            span["end"] = t
            span["income"] += sim.trees_dict[tid][1]
    seat = raw["candidate_seat"]
    sums = []
    row = native.read(path.with_name(path.name.replace("raw_", "row_").replace(".json.gz", ".json")))
    for side in [seat, 1-seat]:
        born = [v for v in trees.values() if v["origin"] == side]
        held = [s for v in trees.values() for s in v["spans"] if s["owner"] == side]
        income = sum(s["income"] for s in held)
        days = sum(s["end"] - s["start"] + 1 for s in held)
        ref = row["snapshots"][str(raw["steps"])]["candidate" if side==seat else "opponent"]
        assert abs(income-ref["cumulative_tree_absorption_income"]) < 1e-5
        assert len(born) == ref["new_trees_planted"]
        sums.append(dict(born=len(born), captures=len(held)-len(born), tree_turns=days,
                         per_tree_turn=income/days, income=income,
                         on_enemy_origin=sum(s["income"] for v in trees.values() if v["origin"]!=side
                                             for s in v["spans"] if s["owner"]==side)))
    return dict(arm=path.parent.name,seat=seat,raw_sha256=native.sha(path),trees=trees,summary=sums)


if __name__ == "__main__":
    rows = [audit(path) for path in sorted((HERE/"games").glob("*/raw_*.json.gz"))]
    assert len(rows) == 4
    native.save(HERE/"tree_ledger.json.gz", rows)
    for r in rows:
        print(r["arm"],r["seat"],r["summary"])

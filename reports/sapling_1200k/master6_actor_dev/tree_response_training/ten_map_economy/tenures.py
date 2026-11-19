"""Follow every cash-producing tree through ownership changes to official termination."""
from collections import Counter
import gzip
import json
from census import HERE, ROOT, TREE, FastCarbon, CashEvents, read, sha


def account(tape, seed, seat):
    sim = FastCarbon(tape["game_configuration"]).reset(seed=seed)
    sim.cash = CashEvents(sim.cash)
    active = {}
    finished = []

    def create(tree, step, origin, price, worker):
        return dict(tid=tree.id, pos=tree.pos, owner=tree.player, start=step,
                    age_at_entry=tree.age, origin=origin, worker=worker, price=price,
                    gross=0., cargo_tax=0., collision=0., preturns=0)

    for tree in sim.trees.values():
        active[tree.id] = create(tree, 0, "opening", 0., None)
    for t, joint in enumerate(tape["actions"]):
        for holding in active.values():
            holding["preturns"] += 1
        sim.cash.events.clear()
        sim.step(*joint)
        cash_events = sim.cash.events
        purchases = {(owner,extra[1]):(name, -amount, extra[0]) for name,owner,amount,*extra in cash_events
                     if name in ("plant","seize")}
        for tid in list(active):
            tree = sim.trees.get(tid)
            if tree is None or tree.player != active[tid]["owner"]:
                holding = active.pop(tid)
                holding["end"] = t+1
                holding["exit"] = "seized_by_opponent" if tree else "expired"
                finished.append(holding)
        for tid, tree in sim.trees.items():
            if tid not in active:
                origin, price, worker = purchases.pop((tree.player,tree.pos))
                active[tid] = create(tree,t+1,origin,price,worker)
        if purchases:
            raise ValueError("tree payment missing from state")
        for category, owner, value, *extra in cash_events:
            if category == "tree_absorption":
                holding = active[extra[0]]
                if holding["owner"] != owner:
                    raise ValueError("income assigned to wrong owner")
                holding["gross"] += value
                holding["cargo_tax"] += extra[2]
            elif category == "tree_collision_cash":
                # Cash from a collision is paid before potential reseizure/expiry.
                matching = [h for h in [*active.values(), *finished[-len(cash_events):]]
                            if h["tid"] == extra[2] and h["owner"] == owner
                            and h["start"] <= t < h.get("end",t+1)]
                if len(matching) != 1:
                    raise ValueError("collision tree holder ambiguous")
                matching[0]["collision"] += value
    for holding in active.values():
        holding.update(end=len(tape["actions"]), exit="terminal")
        finished.append(holding)
    for h in finished:
        if h["end"] - h["start"] != h["preturns"]:
            raise ValueError("tree tenure off by one")
        for name in ("gross","cargo_tax","collision","price"):
            h[name] = round(h[name],6)
    return finished


def main():
    census = read(HERE / "games.json.gz")
    rows = []
    for receipt in census["source_files"]:
        for g in read(ROOT / receipt)["games"]:
            tape = ROOT / g["raw_path"]
            if sha(tape) != g["raw_sha256"]:
                raise ValueError("source tape changed")
            holdings = account(read(tape),g["seed"],g["seat"])
            game = next(r for r in census["games"] if (r["seed"],r["seat"]) == (g["seed"],g["seat"]))
            for physical in (0,1):
                owned = [h for h in holdings if h["owner"] == physical]
                expected = sum(p["accounts"][physical].get("tree_absorption",0) for p in game["phases"])
                turns = sum(p["turn_exposure"][physical].get("tree_turns",0) for p in game["phases"])
                if abs(sum(h["gross"] for h in owned)-expected) > .001 or sum(h["preturns"] for h in owned)!=turns:
                    raise ValueError("individual holdings do not reconcile")
            rows.append({"seed":g["seed"],"seat":g["seat"],"raw_sha256":g["raw_sha256"],"holdings":holdings})
    out=gzip.compress(json.dumps(rows,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    (HERE / "tenures.json.gz").write_bytes(out)


if __name__ == "__main__":
    main()

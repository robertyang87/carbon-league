"""Compare both first-use wins with every loss of the exact incumbent on the same register."""
from collections import Counter
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
LEDGER = next(p for p in HERE.parents if (p / "turning_point_ledger.py").exists())
ROOT = LEDGER.parents[1]
sys.path.insert(0, str(LEDGER))
from turning_point_ledger import CashEvents, FastCarbon, compact, sha, write

def actions(game):
    files = game["artifacts"]
    source = files.get("raw", files.get("archive"))
    path = ROOT / source["path"]
    if sha(path.read_bytes()) != source["sha256"]:
        raise ValueError(f"source changed: {path}")
    if "raw" in files:
        record = json.loads(gzip.decompress(path.read_bytes()))
        tape = record["actions"]
    else:
        member = next(k for k in files if k.endswith("/turns.jsonl"))
        with zipfile.ZipFile(path) as archive:
            data = archive.read(member)
        if sha(data) != files[member]["sha256"]:
            raise ValueError("source action stream changed")
        tape = [[call["response"]["action"] for call in row["calls"]]
                for row in map(json.loads, data.splitlines())]
    if len(tape) != 299 or sha(compact(tape)) != game["actions_sha256"]:
        raise ValueError("different actions")
    return tape

def components(events, seat):
    out = Counter()
    for kind, side, amount, *subject in events:
        sign = 1 if seat == side else -1
        if kind == "tree_absorption":
            out["ground_rent"] += sign * (amount - subject[2])
            out["cargo_tax"] += sign * subject[2]
        else:
            out[kind] += sign * amount
    return out

def replay(game):
    sim = FastCarbon().reset(seed=game["seed"])
    if hashlib.sha256(bytes(map(int, sim.carbon))).hexdigest() != game["terrain"]:
        raise ValueError("wrong opening terrain")
    sim.cash = CashEvents(sim.cash)
    seat = game["seat"]
    margins = [sim.cash[seat] - sim.cash[1-seat]]
    rows = []
    phases = [Counter(), Counter(), Counter()]
    for t, joint in enumerate(actions(game)):
        obs = [sim.obs(s) for s in (0, 1)]
        sim.cash.events.clear()
        sim.step(*joint)
        margin = sim.cash[seat] - sim.cash[1-seat]
        diff = components(sim.cash.events, seat)
        if not isclose(sum(diff.values()), margin-margins[-1], abs_tol=1e-5):
            raise ValueError("cash categories differ from physical transition")
        phases[t//100].update(diff)
        rows.append(dict(t=t, public_before=obs, action_by_physical_seat=joint,
                         cash_after=list(sim.cash), events=list(sim.cash.events)))
        margins.append(margin)
    if list(sim.cash) != game["official_scores_by_physical_seat"]:
        raise ValueError("official terminal score differs")
    changes = [b-a for a,b in zip(margins,margins[1:])]
    pivots = {}
    for name, choose in (("gain", max), ("loss", min)):
        t = choose(range(len(changes)), key=changes.__getitem__)
        pivots[name] = dict(t=t, change=changes[t], prior_and_at=rows[max(0,t-10):t+1])
    last_nonahead = max(t for t,m in enumerate(margins) if m<=0)
    return dict(seed=game["seed"], seat=seat, result=game["result"], category=game["category"],
                terrain=game["terrain"], actions_sha256=game["actions_sha256"],
                score_by_seat=list(sim.cash), checkpoints={str(t):margins[t] for t in (0,50,100,150,200,250,275,299)},
                permanent_lead_from_state=last_nonahead+1 if margins[-1]>0 else None,
                phases=[dict(p) for p in phases], pivots=pivots)

def main():
    path = HERE.parent/"history.json"
    games = [replay(game) for game in json.loads(path.read_bytes())["games"]]
    fresh = [r for r in games if r["category"]=="first_registered_pair"]
    if len(fresh)!=20 or len(games)!=21:
        raise ValueError("unexpected incumbent ledger size")
    write(HERE/"paths.json.gz", dict(history_sha256=sha(path.read_bytes()), games=games), compressed=True)
    for row in fresh:
        checkpoints=row["checkpoints"]
        print(row["seed"],row["seat"],row["result"],
              *[round(checkpoints[str(t)],2) for t in (100,200,250,299)],
              *[round(row["phases"][2].get(k,0),2) for k in ("own_delivery","ground_rent")],
              row["permanent_lead_from_state"])

if __name__=="__main__":
    main()

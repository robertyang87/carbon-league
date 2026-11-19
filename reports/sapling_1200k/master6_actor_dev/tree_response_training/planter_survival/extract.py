"""Emit one paired record per eligible planter and validate each factual/alternative step twice."""
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

from model import MOVES, destination, features

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
BASE = TRIAL.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from turning_point_ledger import CashEvents, FastCarbon
from official_game_replay import ENGINE
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

INPUTS = {"train": TRIAL, "validation": TRIAL / "batches/intake_02"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    blob = path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix == ".gz" else blob)


def write(path, obj):
    plain = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    payload = gzip.compress(plain, mtime=0) if path.suffix == ".gz" else plain + b"\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def pair(row):
    def allowed(action):
        return row["actions"][action]["entry_legal"] and not row["actions"][action]["friendly_conflict"]
    if not allowed("NONE"):
        return None
    old = row["parent_command"]
    if old != "NONE":
        return old if allowed(old) else None
    key = f"{row['seed']}|{row['seat']}|{row['t']}|{row['uid']}|"
    alternatives = [a for a in MOVES if a != "NONE" and allowed(a)]
    return min(alternatives, key=lambda a: hashlib.sha256((key+a).encode()).digest(), default=None)


def step(sim, state, cfg, joint):
    # Full board and opponent's simultaneous commands are solely label-side.
    idgen._id_dict.clear()
    for s in (0,1):
        idgen._id_dict[f"player-{s}-worker-"] = sim.worker_counters[s]
        idgen._id_dict[f"player-{s}-tree-"] = sim.tree_counters[s]
    after = Board(state, cfg, joint).next().observation
    sim.cash = CashEvents(sim.cash)
    sim.step(*joint)
    observed = sim.full_observation()
    for key in ("players", "trees", "carbon"):
        if after[key] != observed[key]:
            raise ValueError(f"official Board vs FastCarbon after {state['step']}, field {key}")
    return after, sim.cash.events


def label(before, result, seat, uid, cell, order, lifespan):
    after, records = result
    died = uid not in after["players"][seat][2]
    after_tree = next((k for k,(p,a) in after["players"][seat][3].items() if p == cell),None)
    old = next(((s,k,a) for s,pl in enumerate(before["players"]) for k,(p,a) in pl[3].items() if p==cell),None)
    kind = None
    if after_tree and not died and order == "NONE" and after["trees"][after_tree][0] == uid:
        if old is None or old[2] >= lifespan:
            if old is not None and after_tree == old[1]:
                raise ValueError("same id on expired-then-replanted tree")
            if after["players"][seat][3][after_tree][1] != 1:
                raise ValueError("new tree has unexpected age")
            kind = "plant"
        elif old[0] != seat:
            if after_tree != old[1]:
                raise ValueError("captured tree id changed")
            kind = "seize"
    debits = [e for e in records if e[0] in ("plant","seize") and e[1]==seat and e[3:5]==[uid,cell]]
    if (kind is None and debits) or (kind and (len(debits) != 1 or debits[0][0] != kind)):
        raise ValueError(f"ownership transition and transaction disagree at {before['step']}")
    return died, kind, -float(debits[0][2]) if debits else 0.


def read_index(directory, spec):
    path = directory / "opportunities.jsonl.gz"
    if sha(path) != spec["opportunities_sha256"] or sha(directory/"source_receipt.json") != spec["receipt_sha256"]:
        raise ValueError("unregistered opportunity/source ledger")
    counts = defaultdict(Counter)
    records = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            seed, seat, t, uid = (row[k] for k in ("seed","seat","t","uid"))
            count = counts[seed,seat]
            count["source_roots"] += 1
            order = pair(row)
            hold = row["actions"]["NONE"]
            if order is None:
                if hold["friendly_conflict"]:
                    reason = "friendly_conflict_on_hold"
                elif not hold["entry_legal"]:
                    reason = "illegal_hold"
                else:
                    reason = "no_registered_legal_uncontested_move"
                count[reason] += 1
                continue
            records[seed,seat,t].append((uid, row["parent_command"], order, hold["can_pay_now"], hold["price"], row["position"]))
            count["eligible_roots"] += 1
    if sorted({s for s,_ in counts}) != sorted(spec["maps"]):
        raise ValueError("wrong split membership")
    return counts, records


def extract_game(game, index):
    tape = ROOT / game["raw_path"]
    if sha(tape) != game["raw_sha256"]:
        raise ValueError("native source checksum mismatch")
    raw = load(tape)
    cfg = raw["game_configuration"]
    seat, seed = game["seat"], game["seed"]
    sim = FastCarbon(cfg).reset(seed=seed)
    if hashlib.sha256(bytes(int(x) for x in sim.carbon)).hexdigest() != game["map_sha256"]:
        raise ValueError("incorrect generation seed or map")
    alt = FastCarbon(cfg)
    history = deque(maxlen=3)
    out = []
    for t,joint in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        state = sim.full_observation()
        saved = sim.get_state()
        factual = step(sim,state,cfg,joint)
        for uid, old, move, payable, quote, origin in index.get((seed,seat,t),[]):
            if obs["players"][seat][2][uid][0] != origin or (joint[seat].get(uid) or "NONE") != old:
                raise ValueError("registered point and actual source action disagree")
            row = dict(seed=seed,seat=seat,t=t,uid=uid,origin=origin,actual=old,payable=payable,physical_site=quote,actions=[])
            for order in ("NONE", move):
                if order == old:
                    outcome = factual
                else:
                    changed = [side.copy() for side in joint]
                    if order == "NONE":
                        changed[seat].pop(uid,None)
                    else:
                        changed[seat][uid] = order
                    alt.set_state(saved)
                    outcome = step(alt,state,cfg,changed)
                cell = destination(origin,order)
                vec,geo = features(obs,list(history),joint[seat],uid,order)
                died,kind,paid = label(state,outcome,seat,uid,cell,order,cfg["treeLifespan"])
                if not geo and died:
                    raise ValueError("death outside public one-step reach bound")
                row["actions"].append(dict(name=order,cell=cell,died=died,kind=kind,paid=paid,geometric=geo,x=vec))
            out.append(row)
        history.append(obs)
    if [round(x,2) for x in sim.cash] != game["scores"]:
        raise ValueError("factually replayed terminal scores differ")
    return out


def main():
    split = sys.argv[1]
    cfg = load(HERE/"protocol.json")["split"][split]
    folder = INPUTS[split]
    counts,index = read_index(folder,cfg)
    for game in load(folder/"source_receipt.json")["games"]:
        seed,seat = game["seed"],game["seat"]
        dest = HERE/f"{split}_{seed}_{seat}.json.gz"
        out = extract_game(game,index)
        if len(out) != counts[seed,seat]["eligible_roots"]:
            raise ValueError("missed or repeated root")
        write(dest,dict(seed=seed,seat=seat,tape_sha256=game["raw_sha256"],engine_sha256=sha(ENGINE/"zerosum_env/envs/carbon/helpers.py"),
                        source_counts=dict(counts[seed,seat]),steps=game["steps"],rows=out))
        print(split,seed,seat,len(out),sha(dest),flush=True)


if __name__ == "__main__":
    main()

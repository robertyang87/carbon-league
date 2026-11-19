"""Cash channels, simultaneous deposits and cumulative lead on every registered training source."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
BASE = TREE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from turning_point_ledger import FastCarbon, CashEvents

ACCOUNTS = ("own_delivery", "tree_absorption", "hire_collector", "hire_planter", "plant", "seize",
            "tree_collision_cash", "base_collision_cash", "opponent_worker_delivery_to_this_base", "rounding")


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def channel_diff(accounts, seat):
    return {name:round(accounts[seat][name] - accounts[1-seat][name], 4) for name in ACCOUNTS}


def game_row(path, cert):
    tape = read(path)
    s = cert["seat"]
    sim = FastCarbon(tape["game_configuration"]).reset(seed=cert["seed"])
    if sha(path) != cert["transcript_sha256"] or len(tape["actions"]) != cert["steps"]:
        raise ValueError("replay source identity differs")
    sim.cash = CashEvents(sim.cash)
    initial = tuple(sim.cash)
    phases = []
    cash_score = []
    all_accounts = [Counter(), Counter()]
    deposits = []
    for t, joint in enumerate(tape["actions"]):
        if t % 50 == 0:
            accounts = [Counter(), Counter()]
            exposure = [Counter(), Counter()]
        for tree in sim.trees.values():
            exposure[tree.player]["tree_turns"] += 1
        for worker in sim.workers.values():
            key = "collector_turns" if worker.is_collector else "planter_turns"
            exposure[worker.player][key] += 1
        sim.cash.events.clear()
        before = list(sim.cash)
        sim.step(*joint)
        net = [0., 0.]
        split_deposits = [[], []]
        for category, owner, amount, *detail in sim.cash.events:
            if category not in ACCOUNTS:
                raise ValueError(category)
            accounts[owner][category] += amount
            all_accounts[owner][category] += amount
            net[owner] += amount
            if category == "own_delivery":
                split_deposits[owner].append([detail[0], round(amount, 4)])
        if any(abs(sim.cash[i] - before[i] - net[i]) > .00001 for i in (0, 1)):
            raise ValueError("cash accounts do not reconcile")
        for owner, credits in enumerate(split_deposits):
            if len(credits) >= 2:
                deposits.append([t+1, owner, credits])
        margin = round(sim.cash[s] - sim.cash[1-s], 2)
        cash_score.append(margin)
        if (t+1) % 50 == 0 or t+1 == len(tape["actions"]):
            phases.append({"post":t+1, "cash_lead":margin, "signed_cash_components":channel_diff(accounts,s),
                           "accounts":[dict(row) for row in accounts], "turn_exposure":[dict(x) for x in exposure]})
    official = cert["official_reward_by_seat"]
    margin = official[s] - official[1-s]
    expected = initial[s] - initial[1-s] + sum(channel_diff(all_accounts,s).values())
    if (tape["official_score_by_seat"] != official or tape["last_post_cash"] != list(sim.cash)
            or abs(expected - cash_score[-1]) > .005):
        raise ValueError("final cash, official score, or sign differs")
    final_negative_suffix = next((t+1 for t in range(len(cash_score)) if max(cash_score[t:]) < 0), None)
    return {"seed":cert["seed"], "seat":s, "sha256":sha(path), "map_sha256":cert["map_sha256"],
            "official_margin":round(margin,2), "terminal_cash_margin":cash_score[-1],
            "official_minus_cash":round(margin-cash_score[-1],2), "steps":len(cash_score),
            "first_state_from_which_cash_lead_stays_negative":final_negative_suffix,
            "components":channel_diff(all_accounts,s),"phases":phases,"multi_delivery":deposits}


def main():
    locations = [TREE, *(TREE / f"batches/intake_{i:02}" for i in range(2,6))]
    rows = []
    for loc in locations:
        proven = read(loc / "source/official_replay.json")["runs"]["source"]["games"]
        ledger = read(loc / "source_receipt.json")
        if sha(loc / "source/official_replay.json") != ledger["official_sha256"]:
            raise ValueError("source proof identity differs")
        for cert in proven:
            g = next(x for x in ledger["games"] if (x["seed"],x["seat"]) == (cert["seed"],cert["seat"]))
            if g["raw_sha256"] != cert["transcript_sha256"] or g["map_sha256"] != cert["map_sha256"]:
                raise ValueError("source fingerprint differs")
            rows.append(game_row(ROOT / g["raw_path"], cert))
    maps = {r["map_sha256"] for r in rows}
    if len(rows) != 20 or len(maps) != 10 or any(sorted(r["seat"] for r in rows if r["map_sha256"] == m) != [0,1] for m in maps):
        raise ValueError("missing a game or physical seat")
    data = {"source_files":{str((loc / "source_receipt.json").relative_to(ROOT)):sha(loc/"source_receipt.json")
                            for loc in locations}, "games":rows}
    out = gzip.compress(json.dumps(data, separators=(",",":"), allow_nan=False).encode(), mtime=0)
    (HERE / "games.json.gz").write_bytes(out)
    print([round(sum(r["components"][k] for r in rows)/len(rows),2) for k in ACCOUNTS])


if __name__ == "__main__":
    main()

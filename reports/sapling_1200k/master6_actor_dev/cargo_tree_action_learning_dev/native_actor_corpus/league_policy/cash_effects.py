"""Attribute paired development score differences to both players' actual cash flow."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
LEDGER = next(p for p in HERE.parents if (p / "turning_point_ledger.py").is_file())
sys.path.insert(0, str(LEDGER))
from turning_point_ledger import CashEvents, FastCarbon, compact, sha, write


def replay(card):
    path = HERE / "evidence" / card["raw_file"]
    payload = path.read_bytes()
    if sha(payload) != card["raw_sha256"]:
        raise ValueError(f"Different action tape: {path}")
    g = json.loads(gzip.decompress(payload))
    seat = g["candidate_seat"]
    sim = FastCarbon().reset(seed=g["seed"])
    if sha(bytes(map(int, sim.carbon))) != g["initial_terrain_u8_sha256"]:
        raise ValueError("Different opening")
    initial = list(sim.cash)
    sim.cash = CashEvents(sim.cash)
    ledger = [[Counter(), Counter()] for _ in range(3)]
    for t, orders in enumerate(g["actions"]):
        balances = list(sim.cash)
        sim.cash.events.clear()
        sim.step(*orders)
        changes = [Counter(), Counter()]
        for kind, owner, signed, *entity in sim.cash.events:
            if kind == "tree_absorption":
                changes[owner]["ground_rent"] += signed - entity[2]
                changes[owner]["cargo_tax"] += entity[2]
            else:
                changes[owner][kind] += signed
        for s in (0, 1):
            if not math.isclose(sum(changes[s].values()), sim.cash[s] - balances[s], abs_tol=1e-5):
                raise ValueError("Cash-flow categories do not reconstruct physical balances")
            ledger[t // 100][s].update(changes[s])
    if list(sim.cash) != g["cash_by_seat"] or g["official_score_by_seat"] != card["official_score_by_seat"]:
        raise ValueError("Physical cash or official score differs")
    def account(side):
        result = Counter({"opening_cash": initial[side],
                          "terminal_scoring_adjustment": g["official_score_by_seat"][side] - sim.cash[side]})
        for period in ledger:
            result.update(period[side])
        if not math.isclose(sum(result.values()), g["official_score_by_seat"][side], abs_tol=1e-5):
            raise ValueError("Official score not reconstructed")
        return result
    return dict(seed=g["seed"], learner_seat=seat, actions_sha256=sha(compact(g["actions"])),
                own_account=account(seat), opponent_account=account(1-seat),
                physical_seat_phase_cash=ledger, steps=len(g["actions"]), official_by_seat=g["official_score_by_seat"])


def comparison(pair):
    child, parent = (replay(pair["arms"][key]) for key in ("candidate", "parent"))
    by_side = {}
    for field in ("own_account", "opponent_account"):
        delta = child[field].copy()
        delta.subtract(parent[field])
        by_side[field] = delta
    delta = by_side["own_account"].copy()
    delta.subtract(by_side["opponent_account"])
    observed = pair["arms"]["candidate"]["margin"] - pair["arms"]["parent"]["margin"]
    if not math.isclose(sum(delta.values()), observed, abs_tol=1e-5):
        raise ValueError("Paired signed components disagree with official margin difference")
    return dict(seed=pair["seed"], seat=pair["seat"], opponent=pair["opponent"],
                first_changed_step=pair["first_changed_free_command_step"], official_margin_change=observed,
                paired_margin_by_source=delta, change_by_side=by_side, candidate=child, parent=parent)


def main():
    source = HERE / "evidence/results.json"
    result = json.loads(source.read_bytes())
    pairs = [comparison(p) for p in result["pairs"]]
    if result["status"] != "complete" or len({(r["seed"],r["seat"]) for r in pairs}) != 12:
        raise ValueError("Incomplete development comparison")
    totals = Counter()
    for p in pairs:
        totals.update(p["paired_margin_by_source"])
    change = {k:v/len(pairs) for k,v in totals.items()}
    means = {key:mean(sum(p["change_by_side"][key].values()) for p in pairs)
             for key in ("own_account", "opponent_account")}
    write(HERE / "evidence/cash_effects.json.gz", dict(development_results_sha256=sha(source.read_bytes()),
          mean_margin_by_source=change, mean_score_change_by_side=means, pairs=pairs), compressed=True)
    print(json.dumps(dict(mean_margin_by_source=change, mean_score_change_by_side=means), sort_keys=True))


if __name__ == "__main__":
    main()

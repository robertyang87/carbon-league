"""Replay both development arms and account for the cash swings preceding each official result."""
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
economy_path = HERE.parent / "development" / "economy.py"
spec = importlib.util.spec_from_file_location("strategic_development_economy", economy_path)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ledger = prior.ledger


def audit(hosted, protocol_path, candidate="candidate", parent="parent"):
    plan = json.loads(protocol_path.read_bytes())
    source = hosted / "settlement.json"
    settlement = json.loads(source.read_bytes())
    require = ledger.require
    require(settlement["records_reconciled"], "Expected verified match receipts")
    require(settlement["protocol_sha256"] == ledger.sha(protocol_path.read_bytes()), "Protocol changed")
    require(set(plan["arms"]) == {candidate, parent}, "Expected two registered development arms")

    games = settlement["games"]
    cards = {(g["seed"], g["seat"], g["arm"]): g for g in games}
    pairs = {(seed, seat) for seed in plan["seeds"] for seat in (0, 1)}
    require(len(cards) == len(games) == 2 * len(pairs), "Missing or duplicate game")
    require(set(cards) == {(seed, seat, arm) for seed, seat in pairs for arm in plan["arms"]}, "Different game roster")
    tickets = settlement["archives"]
    require(len(tickets) == len(pairs) and {(t["seed"], t["seat"]) for t in tickets} == pairs,
            "Missing or duplicate map-seat archive")
    cfg = ledger.config()
    receipts, contrasts = [], []
    for ticket in tickets:
        seed, seat = ticket["seed"], ticket["seat"]
        path = hosted / (str(ticket["id"]) + ".zip")
        require(ledger.sha(path.read_bytes()) == ticket["sha256"], "Match archive changed")
        with ZipFile(path) as archive:
            games_by_arm = {arm: ledger.replay(cards[seed, seat, arm], archive, cfg) for arm in plan["arms"]}
        contrasts.append(ledger.comparison(games_by_arm[candidate], games_by_arm[parent]))
        for game in games_by_arm.values():
            referenced = set()
            game["pivots"] = prior.trace_pivots(game["cash_margin_by_state"], {"both": referenced})
            rows = game.pop("transitions")
            game["selected_turns"] = {t: rows[t] for t in sorted(referenced)}
            receipts.append(game)
    by_map = []
    channels = {
        "normal_delivery": ("own_delivery",),
        "tree_net": ("ground_rent", "cargo_tax", "plant", "seize"),
        "ground_rent": ("ground_rent",),
    }
    for seed in plan["seeds"]:
        group = [p for p in contrasts if p["seed"] == seed]
        numeric = all(p["numeric"] for p in group)
        components = {}
        for name, keys in channels.items():
            components[name] = mean(
                sum(p["learner_finances"].get(k, 0) - p["opponent_finances"].get(k, 0) for k in keys)
                for p in group
            ) if numeric else None
        by_map.append({
            "seed": seed, "both_seats_numeric": numeric,
            "candidate_minus_parent_official_margin": mean(p["official_margin_delta"] for p in group) if numeric else None,
            "candidate_minus_parent_net_cash": components,
        })
    return {
        "settlement_sha256": ledger.sha(source.read_bytes()), "protocol_sha256": settlement["protocol_sha256"],
        "maps": by_map, "tickets": contrasts, "games": receipts,
    }


if __name__ == "__main__":
    directory = HERE / "hosted"
    result = audit(directory, HERE / "protocol.json")
    target = directory / "economy.json.gz"
    data = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    encoded = gzip.compress(data.encode(), mtime=0)
    with NamedTemporaryFile(dir=directory, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(target)
    print(json.dumps(result["maps"], ensure_ascii=False, indent=2))

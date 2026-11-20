"""Reproduce the registered official cash margins and the earlier tree-income comparison."""
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import mean
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prior_economic_audit", HERE.parent/"development"/"economy.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ledger = prior.ledger
TREE_FIELDS = ("ground_rent", "plant", "seize", "cargo_tax")


def main():
    hosted = HERE/"hosted"
    source = hosted/"settlement.json"
    settlement = json.loads(source.read_bytes())
    protocol = json.loads((HERE/"protocol.json").read_bytes())
    if not settlement["records_reconciled"] or settlement["protocol_sha256"] != ledger.sha((HERE/"protocol.json").read_bytes()):
        raise ValueError("Unverified or different match settlement")
    cards = {(g["seed"], g["seat"], g["arm"]): g for g in settlement["games"]}
    cfg = ledger.config()
    rows, contrasts = [], []
    for ticket in settlement["archives"]:
        seed, seat = ticket["seed"], ticket["seat"]
        path = hosted/(str(ticket["id"])+".zip")
        if ledger.sha(path.read_bytes()) != ticket["sha256"]:
            raise ValueError("Archive changed")
        with zipfile.ZipFile(path) as archive:
            games = {arm: ledger.replay(cards[seed,seat,arm], archive, cfg) for arm in ("parent","bc")}
        contrast = ledger.comparison(games["bc"], games["parent"])
        if contrast["numeric"]:
            def tree(g):
                return sum(g.get(k,0) for k in TREE_FIELDS)
            tree_delta = tree(contrast["learner_finances"])-tree(contrast["opponent_finances"])
            contrast["net_tree_cash_delta"] = tree_delta
        contrasts.append(contrast)
        for game in games.values():
            used = set()
            game["pivots"] = prior.trace_pivots(game["cash_margin_by_state"], {"used":used})
            game["_used_turns"] = used
            game["selected_turns"] = prior.turn_lookup(game)
            rows.append(game)
    if len(rows) != 16 or len(contrasts) != 8:
        raise ValueError("Incomplete accounting")
    maps = []
    for seed in protocol["seeds"]:
        subset = [row for row in contrasts if row["seed"] == seed]
        if {r["seat"] for r in subset} != {0,1} or len(subset) != 2:
            raise ValueError("Unpaired map")
        numeric = all(r["numeric"] for r in subset)
        def net(row, name):
            return row["learner_finances"].get(name,0)-row["opponent_finances"].get(name,0)
        maps.append(dict(seed=seed, both_official_scores_numeric=numeric,
            net_ground_rent_change=mean(net(r,"ground_rent") for r in subset) if numeric else None,
            net_tree_cash_change=mean(r["net_tree_cash_delta"] for r in subset) if numeric else None,
            net_normal_delivery_change=mean(net(r,"own_delivery") for r in subset) if numeric else None,
            official_margin_change=mean(r["official_margin_delta"] for r in subset) if numeric else None))
    result=dict(settlement_sha256=ledger.sha(source.read_bytes()),code_sha256=ledger.sha(Path(__file__).read_bytes()),
                protocol_sha256=settlement["protocol_sha256"], tree_fields=TREE_FIELDS, map_contrasts=maps,
                ticket_contrasts=contrasts,games=rows)
    raw=json.dumps(result,allow_nan=False,separators=(",",":"),sort_keys=True).encode()
    target=hosted/"economy.json.gz"
    temp=target.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(raw,mtime=0))
    temp.replace(target)
    print(json.dumps(maps,indent=2),flush=True)


if __name__ == "__main__":
    main()

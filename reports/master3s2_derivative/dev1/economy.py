"""Descriptive post-match cash accounting from actions in the eight verified development games."""
import gzip
import importlib.util
import json
from pathlib import Path
from statistics import mean
import zipfile

import matrix as m

spec = importlib.util.spec_from_file_location("prior_development_economy",m.STRATEGY / "development/economy.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ledger = prior.ledger
GROUPS = {
    "opening": ("opening_cash",),
    "own_delivery": ("own_delivery",),
    "land_rent": ("ground_rent",),
    "cargo_tax": ("cargo_tax",),
    "collisions_and_other_deliveries": ("tree_collision_cash", "base_collision_cash", "opponent_worker_delivery_to_this_base"),
    "hiring": ("hire_collector", "hire_planter"),
    "planting_and_seizing": ("plant", "seize"),
    "settlement_and_rounding": ("terminal_scoring_adjustment", "rounding"),
}


def summarize():
    hosted = m.HERE / "hosted"
    reg, receipt = m.read(m.PLAN), m.read(hosted / "settlement.json")
    m.require(receipt["records_reconciled"] and receipt["protocol_sha256"] == m.sha(m.PLAN), "Expected fully verified schedule")
    m.require(sorted(k for fields in GROUPS.values() for k in fields) == sorted(ledger.KIND), "Incomplete cash decomposition")
    archives = {a["id"]: a for a in receipt["archives"]}
    required = {(t["seed"], t["seat"]) for t in reg["schedule"]}
    m.require({(g["seed"], g["seat"]) for g in receipt["games"]} == required and len(receipt["games"]) == len(required), "Different lineup")
    cfg=ledger.config()
    games=[]
    for g in receipt["games"]:
        path=hosted/g["archive_path"]
        m.require(m.sha(path)==archives[g["archive_id"]]["sha256"], "Archive mismatch")
        with zipfile.ZipFile(path) as z:
            book=ledger.replay(g,z,cfg)
        seat=g["seat"]
        by=book["signed_account_by_seat"]
        actual=book["official_margin"]
        group_difference={name:sum(by[seat].get(k,0)-by[1-seat].get(k,0) for k in keys) for name, keys in GROUPS.items()}
        if actual is not None:
            ledger.check_equal(sum(group_difference.values()),actual,"Source margins do not reconcile to terminal score")
        needed=set()
        book["turn_references"]=prior.trace_pivots(book["cash_margin_by_state"],{m.ARM:needed})
        rows=book.pop("transitions")
        book["selected_turns"]={t:rows[t] for t in sorted(needed)}
        book["cash_margin_components"]=group_difference
        games.append(book)
    maps=[]
    for seed in reg["seeds"]:
        pair=[b for b in games if b["seed"]==seed]
        m.require(len(pair)==2,"Incomplete map")
        complete=all(b["official_margin"] is not None for b in pair)
        maps.append(dict(seed=seed, official_mean_margin=mean(b["official_margin"] for b in pair) if complete else None,
                         cash_component_means={k:mean(b["cash_margin_components"][k] for b in pair) for k in GROUPS} if complete else None))
    return dict(schema="master3s2-derivative-development-cash/1", protocol_sha256=m.sha(m.PLAN),
                settlement_sha256=m.sha(hosted/"settlement.json"), games=games, maps=maps,
                description="Candidate minus intact original by physical seat. Source amounts are signed; the last group accounts for official terminal adjustments. Each selected swing links back 3, 5 and 10 turns. This is descriptive cash accounting.")


if __name__=="__main__":
    result=summarize()
    target=m.HERE/"hosted/economy.json.gz"
    encoded=json.dumps(result, ensure_ascii=False, separators=(",",":"), sort_keys=True,allow_nan=False).encode()
    temp=target.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(encoded,mtime=0))
    temp.replace(target)
    print(json.dumps(result["maps"], ensure_ascii=False,allow_nan=False))

"""Freeze the complete eligible registration cohort before seeing candidate predictions."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np
from fit_distribution import HERE, AUX, old, read, base
spec = importlib.util.spec_from_file_location("prior_clearance", AUX / "ladder_transfer" / "evaluate.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def build():
    p = read(HERE / "fresh_protocol.json")
    old_ids, old_seeds, old_terrain, files = ledger.clearance()
    transfer = AUX / "ladder_transfer" / "selection.json"
    files[str(transfer.relative_to(ledger.ROOT))] = old.sha(transfer)
    for c in read(transfer)["cards"]:
        old_ids.add(c["id"])
        old_seeds.add(c["seed"])
        old_terrain.add(c["terrain_sha256"])
    names = set(read(transfer)["full_prior_coverage"]["training_exact_opponent_names"])
    old_ids = set(old_ids)
    seen_id, seen_seed, seen_terrain = set(), set(), set()
    excluded, cards = [], []
    for tag in p["batches"]:
        f = ledger.LIB / tag / "manifest.json"
        t = f.with_name("validation_maps.jsonl.gz")
        files[str(f.relative_to(ledger.ROOT))] = old.sha(f)
        files[str(t.relative_to(ledger.ROOT))] = old.sha(t)
        manifest = read(f)
        terrain = {r["id"]: r for r in ledger.entries(t)}
        ranks = {v["id"]:v["rank"] for v in manifest["top5"]}
        m6 = [v["id"] for v in manifest["top5"] if v["name"] == "claude-AlphaCarbon-Master6"]
        if len(m6) != 1:
            raise ValueError("ambiguous Master6 entry")
        for g in manifest["games"]:
            ids = g["seat_strategy_ids"]
            seats = [s for s, v in enumerate(ids) if v in ranks and v != m6[0]]
            reason = None
            if m6[0] in ids or not seats:
                reason = "outside non-Master6 entire-match cohort"
            elif not g["clean_terminal"] or not g["frame_verified"] or not all(
                 isinstance(x, (float,int)) and np.isfinite(x) for x in g["score_by_seat"]):
                reason = "unclean terminal"
            if reason:
                excluded.append({"id":g["id"],"batch":tag,"reason":reason})
                continue
            seat = min(seats, key=lambda s:ranks[ids[s]])
            terrain_hash = terrain[g["id"]]["initial_terrain_u8_sha256"]
            if terrain[g["id"]]["seed"] != g["seed"]:
                raise ValueError("map declaration mismatch")
            if g["id"] in old_ids or g["seed"] in old_seeds or terrain_hash in old_terrain:
                excluded.append({"id":g["id"],"batch":tag,"reason":"registered prior exposure"})
                continue
            if g["id"] in seen_id or g["seed"] in seen_seed or terrain_hash in seen_terrain:
                raise ValueError("nonunique new map")
            seen_id.add(g["id"]); seen_seed.add(g["seed"]); seen_terrain.add(terrain_hash)
            opp = g["seat_names"][1-seat]
            delta = g["score_by_seat"][seat] - g["score_by_seat"][1-seat]
            cards.append({"gid":len(cards),"id":g["id"],"seed":g["seed"],"terrain_sha256":terrain_hash,
                         "sha256":g["sha256"],"seat":seat,"actor":g["seat_names"][seat],"opponent":opp,
                         "known_opponent":opp in names,"batch":tag,"split":"fresh","turns":g["transitions"],
                         "margin":delta,"wdl":"W" if delta>0 else "L" if delta<0 else "D"})
        fault = f.parent / "unscored" / "cards.json"
        if fault.exists():
            files[str(fault.relative_to(ledger.ROOT))] = old.sha(fault)
            for gid in read(fault):
                excluded.append({"id":gid,"batch":tag,"reason":"unscored separate appendix fault register"})
    result = {"cards":cards,"excluded":excluded,"archive_sha256":files,
              "fresh_protocol_sha256":old.sha(HERE/"fresh_protocol.json"),
              "selection_source_sha256":old.sha(Path(__file__)),
              "training_exact_opponent_registrations":sorted(names),
              "exclusions_before_selection":{"ids":len(old_ids),"seeds":len(old_seeds),"terrain":len(old_terrain)},
              "cohort_counts":{"actor":dict(Counter(c["actor"] for c in cards)),
                              "unseen_exact_opponent_maps":sum(not c["known_opponent"] for c in cards)}}
    tmp = HERE / ".fresh.tmp"
    base.write_json(tmp,result)
    tmp.replace(HERE / "fresh_panel.json")
    print("fresh",result["cohort_counts"],flush=True)


if __name__ == "__main__":
    build()

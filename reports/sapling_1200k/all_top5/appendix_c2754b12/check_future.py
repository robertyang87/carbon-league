"""Validate historical catalog coverage and every H10/H30 target for the new game."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(TOP))
from context import Corpus
from continue_appendix import previous_manifests
from economy_extract import atomic, visible_ledger
from future_economy import FutureEconomy
from future_economy_check import check_for_side


def catalog():
    indexed = {}
    indexes = []
    for manifest_path in previous_manifests(HERE):
        directory = manifest_path.parent
        prefix = "context" if directory == TOP else "context_" + directory.name
        index = TOP / f"{prefix}_index.tsv.gz"
        receipt = TOP / f"{prefix}_receipt.json"
        if not index.is_file():
            index, receipt = directory / "context_index.tsv.gz", directory / "context_receipt.json"
        raw = index.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        assert fingerprint == json.loads(receipt.read_bytes())["index_sha256"]
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode()), delimiter="\t"))
        expected = {r["id"]: r for r in json.loads(manifest_path.read_bytes())["games"]}
        assert len(rows) == len(expected) and set(r["id"] for r in rows) == set(expected)
        for row in rows:
            gid = row["id"]
            assert gid not in indexed and expected[gid]["sha256"] == row["replay_sha256"]
            indexed[gid] = int(row["transitions"])
        indexes.append({"file": str(index.relative_to(TOP)), "sha256": fingerprint, "games": len(rows)})
    return indexed, indexes


def verify_game(corpus, gid):
    game = corpus.open(gid)
    future = FutureEconomy(game)
    config = game._data["configuration"]
    ledger = visible_ledger(game._frames, config)[0]
    counts = {"10": {"full": 0, "censored": 0}, "30": {"full": 0, "censored": 0}}
    for t in range(game.turns):
        by_seat = [future.labels(t, s) for s in (0, 1)]
        for h in (10, 30):
            end = min(t+h, game.turns)
            complete = t+h <= game.turns
            a, b = [v[str(h)] for v in by_seat]
            actual = "target" if complete else "observed_to_terminal"
            empty = "observed_to_terminal" if complete else "target"
            for result in (a, b):
                assert result[empty] is None and result["full_horizon"] == complete
                assert result["observed_steps"] == end-t
                terminal = None if end < game.turns else "scheduled" if end == config["episodeSteps"]-1 else "early"
                assert result["terminal_at_observed_end"] == terminal
            assert a[actual]["own"] == b[actual]["opponent"] and a[actual]["opponent"] == b[actual]["own"]
            for s, side in enumerate(("own", "opponent")):
                check_for_side(game, t, end, s, a[actual][side], ledger[t:end])
            counts[str(h)]["full" if complete else "censored"] += 2
    return {"id": gid, "raw_sha256": corpus.rows[gid]["replay_sha256"], "transitions": game.turns,
            "physical_seat_windows": counts}


def main():
    indexed, indexes = catalog()
    corpus = Corpus(index=TOP / "context_appendix_c2754b12_index.tsv.gz")
    result = {"scope": "All historical catalogs through this appendix are checked against frozen score identities. Numeric economic targets are independently tested at every decision of the new game.",
              "historical_indexed_games": len(indexed),
              "historical_seat_decisions": 2 * sum(indexed.values()),
              "catalogs": indexes, "new_games": [verify_game(corpus, gid) for gid in sorted(corpus.rows)]}
    out = HERE / "future_economy_check.json"
    atomic(out, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    print(result["historical_indexed_games"], result["new_games"])


if __name__ == "__main__":
    main()

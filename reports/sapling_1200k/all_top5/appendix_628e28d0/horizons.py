"""Recheck future income from public transitions at the six turning points and censoring boundaries."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from context import Corpus
from economy_extract import visible_ledger
from future_economy import FutureEconomy
from future_economy_check import check_for_side


def main():
    index = HERE / "context_index.tsv.gz"
    receipt = json.loads((HERE / "context_receipt.json").read_bytes())
    assert hashlib.sha256(index.read_bytes()).hexdigest() == receipt["index_sha256"]
    corpus = Corpus(manifest=HERE / "manifest.json")
    indexed = Corpus(index=index)
    assert indexed.order == corpus.order and indexed.size == corpus.size
    for gid, row in corpus.rows.items():
        assert {k: str(v) for k, v in row.items()} == indexed.rows[gid]
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))["rows"]
    rows = []
    coverage = {h: {"full":0, "censored":0} for h in (10, 30)}
    for case in cases:
        game = corpus.open(case["id"])
        provider = FutureEconomy(game)
        start = case["economy"]["swing"]["a"]
        turns = sorted({0, start-10, start-5, start-3, start, game.turns-30, game.turns-10, game.turns-1})
        decisions = []
        for h in coverage:
            full = 2 * max(0, game.turns-h+1)
            coverage[h]["full"] += full
            coverage[h]["censored"] += 2*game.turns-full
        for t in turns:
            for h in coverage:
                end = min(game.turns, t+h)
                ledger = visible_ledger(game._frames[t:end+1], game._data["configuration"])[0]
                labels = [provider.labels(t, seat)[str(h)] for seat in (0,1)]
                assert all(label["observed_steps"] == end-t and label["full_horizon"] == (t+h<=game.turns) for label in labels)
                field = "target" if t+h <= game.turns else "observed_to_terminal"
                other = "observed_to_terminal" if t+h <= game.turns else "target"
                assert all(label[other] is None for label in labels)
                for seat in (0,1):
                    own = labels[seat][field]["own"]
                    assert own == labels[1-seat][field]["opponent"]
                    check_for_side(game, t, end, seat, own, ledger)
                decisions.append(dict(t=t, horizon=h, observed_steps=end-t, complete=(end==t+h),
                                      delivery=[label[field]["own"]["own_collector_delivery"] for label in labels],
                                      tree_absorption=[label[field]["own"]["tree_absorption"] for label in labels]))
        rows.append(dict(id=case["id"], verified_seat_windows=2*len(decisions), windows=decisions))
    out = dict(scope="Only six new official replays, physical seats 0 then 1. Exact and shortened horizons separate.",
               source_manifest_sha256=hashlib.sha256((HERE/"manifest.json").read_bytes()).hexdigest(),
               games=len(rows), decision_seats=corpus.size, available_seat_windows=coverage,
               audited_seat_windows=sum(row["verified_seat_windows"] for row in rows), rows=rows)
    (HERE/"future_economy_check.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()

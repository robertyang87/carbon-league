#!/usr/bin/env python3
"""corpus_batch1_verify -- spot-check final-score reproducibility.

Re-plays a spread of corpus-batch-1 games (>=10, covering all five
matchups) with the identical harness + seeding protocol and requires
the re-run's final cash to match the recorded trace EXACTLY (0.00),
and the full 301-frame cash series to match too (stronger than the
historic final-only check).  The re-run overwrites the trace file
with the identical game, so the check is non-destructive when it
passes.

Run after corpus_batch1_run.py has finished:
  python3 corpus_batch1_verify.py [n_per_tag]
"""

import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus_batch1_run as R  # noqa: E402

HERE = R.HERE
TRACES = os.path.join(HERE, "traces")


def pick_games(n_per_tag=2):
    manifest = json.load(open(R.MANIFEST))
    by_tag = {}
    for g in manifest:
        by_tag.setdefault(g["tag"], []).append(g)
    picked = []
    for tag, gs in sorted(by_tag.items()):
        gs = sorted(gs, key=lambda g: (g["seed"], g["side"]))
        # spread: first n_per_tag-1 by seed + the last one
        sel = gs[:max(n_per_tag - 1, 1)] + ([gs[-1]] if n_per_tag > 1
                                            and len(gs) > 1 else [])
        picked.extend(sel[:n_per_tag] if len(sel) > n_per_tag else sel)
    return picked


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    picked = pick_games(n)
    print(f"verifying {len(picked)} games")
    bad = 0
    for g in picked:
        path = os.path.join(TRACES, g["file"])
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            old = json.load(fh)
        old_cash = old["meta"]["final_cash"]
        old_series = [f["cash"] for f in old["frames"]]

        job = {"tag": g["tag"], "seed": g["seed"], "side": g["side"],
               "policy_file": None, "opp_file": None}
        # recover the two files from the matrix by tag
        for tag, pf, of, pn, on, _seeds, teachers in R.MATRIX:
            if tag == g["tag"]:
                job.update({"policy_file": pf, "opp_file": of,
                            "policy_name": pn, "opp_name": on,
                            "teachers": teachers})
                break
        r = R.play_record(job)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            new = json.load(fh)
        new_series = [f["cash"] for f in new["frames"]]
        d0 = abs(r["final_cash"][0] - old_cash[0])
        d1 = abs(r["final_cash"][1] - old_cash[1])
        series_ok = new_series == old_series
        ok = d0 == 0.0 and d1 == 0.0 and series_ok
        bad += 0 if ok else 1
        print(f"{g['tag']} seed{g['seed']} side{g['side']}: "
              f"recorded {old_cash} rerun {r['final_cash']} "
              f"diff {d0:.2f}/{d1:.2f} series={'OK' if series_ok else 'DIFF'} "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"{len(picked) - bad}/{len(picked)} exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

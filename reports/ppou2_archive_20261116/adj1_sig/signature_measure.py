#!/usr/bin/env python3
"""ppou2_adj1 tier-1 signature measurement: pool member / candidate vs Echo4
on the diagnostic fixed set, recording eval_v53's own ledger per game
(treesteps = sum over steps of live trees owned, cash = terminal cash,
cargo stranded, recruits) for both seats. Runs ON L20 with
cwd=/root/carbon/probe_m2/code. No pinned file is modified.

Usage: signature_measure.py <pack_file> <seed_lo> <seed_hi> <out_jsonl> <workers> [opp]
Games: for each seed in [lo, hi), ONE game with seat = seed % 2
(a=pack, b=echo4_int8) -- the diagnostic-field caliber; with 'opp' the pack
plays seat 1 - seed % 2 (opposite-seat pass; strategy-group ruling
2026-11-16: members and candidate both seats, 1000 seeds x 2).
Output: one JSON line per game with the full ledger_extra of both sides.
Chunk file is written to <out>.tmp and renamed on completion (resume-safe:
an existing final file is not overwritten; caller skips it).
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
import eval_v53

ECHO4 = "exp/echo4_int8.py"


def one(task):
    pack, seed, side = task
    res = eval_v53.play_game(("sig", pack, ECHO4, seed, side))
    le = res["ledger_extra"]
    return {"seed": seed, "side": side, "margin": res["margin"],
            "win": res["win"], "end_step": res["nsteps"],
            "a_errors": res.get("a_errors"), "b_errors": res.get("b_errors"),
            "pol": le["pol"], "opp": le["opp"]}


def main():
    pack, lo, hi, out_path, workers = (sys.argv[1], int(sys.argv[2]),
                                       int(sys.argv[3]), sys.argv[4],
                                       int(sys.argv[5]))
    opp = len(sys.argv) > 6 and sys.argv[6] == "opp"
    tasks = [(pack, s, (1 - s % 2) if opp else s % 2) for s in range(lo, hi)]
    tmp = out_path + ".tmp"
    with Pool(workers, maxtasksperchild=50) as pool, open(tmp, "w") as out:
        for rec in pool.imap_unordered(one, tasks, chunksize=4):
            out.write(json.dumps(rec) + "\n")
            out.flush()
    os.replace(tmp, out_path)
    print(json.dumps({"pack": pack, "seeds": [lo, hi], "games": len(tasks),
                      "out": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

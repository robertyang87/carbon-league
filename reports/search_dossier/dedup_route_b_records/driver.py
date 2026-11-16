#!/usr/bin/env python3
"""Resumable A/B driver: one game per subprocess, validated and merged
per leg; safe to kill and relaunch at any point -- completed games are
skipped. Progress in driver.log; per-game runner output in
driver_games.log. Exits when all three leg compares exist.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

LEGS = [
    ("h2", [("dedup", "diag/dedup_h2.py"), ("orig", "diag/orig_h2.py")]),
    ("h1", [("orig", "diag/orig_h1.py"), ("dedup", "diag/dedup_h1.py")]),
    ("ship", [("orig", "../echo4_search_int8.py"), ("dedup", "echo4_search_int8_dedup.py")]),
]
SEEDS = [(986200, 0), (986201, 1)]  # (seed, pack_seat) per game index


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open("driver.log", "a") as f:
        f.write(line + "\n")


def game_ok(path):
    try:
        d = json.load(open(path))
        g = d["games"][0]
        return g["steps"] == 300 and g["completed"] and len(d["games"]) == 1
    except Exception:
        return False


def run_game(leg, who, pack, gi):
    seed, seat = SEEDS[gi]
    part = f"part_{leg}_{who}_g{gi}.json"
    if game_ok(part):
        log(f"skip {part}")
        return True
    log(f"run {part} pack={pack} seed={seed} seat={seat}")
    with open("driver_games.log", "a") as lf:
        r = subprocess.run([sys.executable, "ab_runner.py", pack, "1", str(seed), part, str(seat)],
                           stdout=lf, stderr=subprocess.STDOUT)
    ok = game_ok(part)
    log(f"{'done' if ok else 'FAILED(rc=%s)' % r.returncode} {part}")
    return ok


def merge(leg, who):
    parts = [f"part_{leg}_{who}_g{gi}.json" for gi in (0, 1)]
    if not all(game_ok(p) for p in parts):
        return False
    d0 = json.load(open(parts[0]))
    d1 = json.load(open(parts[1]))
    out = dict(d0)
    out["games"] = d0["games"] + d1["games"]
    json.dump(out, open(f"{leg}_{who}_full.json", "w"))
    log(f"merged {leg}_{who}_full.json")
    return True


def main():
    # seed the already-landed dedup H2 game 0 (from the killed chain2 run)
    legacy = "dedup_h2_full.json"
    part00 = "part_h2_dedup_g0.json"
    if not os.path.exists(part00) and os.path.exists(legacy):
        d = json.load(open(legacy))
        if len(d["games"]) >= 1:
            d["games"] = d["games"][:1]
            json.dump(d, open(part00, "w"))
            log("seeded part_h2_dedup_g0.json from dedup_h2_full.json")
    for leg, packs in LEGS:
        for who, pack in packs:
            for gi in (0, 1):
                run_game(leg, who, pack, gi)
        if all(merge(leg, who) for who, _ in packs):
            cmp_out = f"cmp_{leg}.json"
            if not os.path.exists(cmp_out):
                names = {who: f"{leg}_{who}_full.json" for who, _ in packs}
                with open("driver_games.log", "a") as lf:
                    subprocess.run([sys.executable, "ab_compare.py",
                                    names["orig"], names["dedup"], cmp_out],
                                   stdout=lf, stderr=subprocess.STDOUT)
                log(f"compared -> {cmp_out}")
    log("DRIVER-ALL-DONE")


if __name__ == "__main__":
    main()

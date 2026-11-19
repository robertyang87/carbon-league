"""Recompute from archived records. No simulations and no network requests."""
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ARCHIVE = "d599ec679718dd4407a718aa6ad810e35e2d3983"
SOURCE_MD5 = "57f9a594cce4f2ec9e3573d8d9d26ca1"
FULL_WRAPPER = "8d209ddb46faafae151aae16b7e17d55"
NOS_WRAPPER = "f5d7b7ee27b78def437fa1d7ff9bc088"
IT50 = "c35305af7f8a7bc58afb9cfb4918518c"
HARNESS = "354bcd1bc2dc8bc188c813f53d28c893"
WORKERS = 62


def archived(path):
    return subprocess.check_output(
        ["git", "show", f"{ARCHIVE}:synced/h29/{path}"], cwd=ROOT,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )


def read_arm(tag, expected_opponent):
    rows = [json.loads(line) for line in archived(f"bigeval_live/{tag}.jsonl").splitlines()]
    assert len(rows) == 300
    assert {row["seed"] for row in rows} == set(range(20291101, 20291401))
    for row in rows:
        assert row["side"] == row["seed"] % 2 and not any(row[k] for k in ("err", "serr", "berr"))
        assert row["hh"] == HARNESS and row["bh"] == expected_opponent
        assert abs(row["sa"] - row["sb"] - row["margin"]) < 1e-8
        assert row["lx"]["pol"]["cash270"] > 0 and row["lx"]["opp"]["cash270"] > 0
    return rows


def wave_report(rows):
    result = []
    for start in range(0, len(rows), WORKERS):
        wave = rows[start:start + WORKERS]
        seeds = sorted(row["seed"] for row in wave)
        assert seeds == list(range(20291101 + start, 20291101 + start + len(wave)))
        result.append({"n": len(wave), "first_receipt": min(r["t"] for r in wave),
                       "last_receipt": max(r["t"] for r in wave),
                       "first_seed": seeds[0], "last_seed": seeds[-1]})
    return result


def seconds(clock):
    h, m, s = map(int, clock.split(":"))
    return 3600 * h + 60 * m + s


def paired(old, off):
    match = {r["seed"]: r for r in off}
    result = {}
    for column in ("margin", "sa", "sb", "odep", "otinc"):
        d = [r[column] - match[r["seed"]][column] for r in old]
        mean = statistics.mean(d)
        half = 1.96793 * statistics.stdev(d) / math.sqrt(len(d))
        result[column] = {"mean_full_minus_off": round(mean, 3),
                          "map_paired_t95": [round(mean-half, 3), round(mean+half, 3)]}
    return result


def new_results():
    arms = {"parent": [], "candidate": []}
    for batch in ("direct_original", "replicate_original"):
        base = HERE.parent / batch
        binding = json.loads((base / "binding.json").read_bytes())
        rows = json.loads((base / "scores.json").read_bytes())
        for row in rows:
            arm, seat, seed = row["arm"], row["seat"], row["seed"]
            clock_bytes = (base / f"games/{arm}/wall_{seed}_{seat}.json.gz").read_bytes()
            assert hashlib.sha256(clock_bytes).hexdigest() == row["wall_sha256"]
            clock = json.loads(gzip.decompress(clock_bytes))
            assert clock["identity"]["binding"][arm] == binding[arm]
            calls = clock["calls"]
            assert len(calls) == row["actual_opponent_calls"] == 299
            arms[arm].append((row, calls))
    result = {}
    for arm, records in arms.items():
        assert len({(r["seed"], r["seat"]) for r, _ in records}) == len(records) == 8
        calls = [call for _, game in records for call in game]
        result[arm] = {
            "games": len(records), "wins": sum(r["result"] == "win" for r, _ in records),
            "maps": sorted({r["seed"] for r, _ in records}),
            "own": round(statistics.mean(r["own"] for r, _ in records), 2),
            "opponent": round(statistics.mean(r["opponent"] for r, _ in records), 2),
            "margin": round(statistics.mean(r["margin"] for r, _ in records), 2),
            "opponent_calls": len(calls), "depth8": sum(c["max_depth"] == 8 for c in calls),
            "opponent_search_return_counts": {k:sum(c["kind"] == k for c in calls) for k in sorted({c["kind"] for c in calls})},
            "opponent_mean_public_ms": round(1000*statistics.mean(c["public_wall_seconds"] for c in calls), 3),
            "opponent_total_public_hours": round(sum(c["public_wall_seconds"] for c in calls)/3600, 4),
        }
    return result


def main():
    old = read_arm("abD_it50_3s2", FULL_WRAPPER)
    off = read_arm("abC_it50_3s2nosearch", NOS_WRAPPER)
    prior = read_arm("abB_echo4_3s2nosearch", NOS_WRAPPER)
    assert all(r["ah"] == IT50 for r in old + off)
    assert all(r["bst"] == {"other_errors": 0} for r in old)
    for path, expected in (("code/packs/ppou2_m_it50.py", IT50),
                           ("code/bigeval.py", HARNESS), ("code/exp/opp_m3s2.py", NOS_WRAPPER),
                           ("data/master3s2_source.py", SOURCE_MD5)):
        assert hashlib.md5((ROOT / path).read_bytes()).hexdigest() == expected
    # Use the preceding serial cell's last record. A D worker can return before the
    # parent serializes its completion, so D's *first* receipt is not a start bound.
    window = seconds(max(r["t"] for r in old)) - seconds(max(r["t"] for r in prior)) + 1
    total_worker_s = WORKERS * window
    candidate_ms_min = statistics.mean(r["ms"] - .005 for r in old)
    # At least 270 actions per game are directly observable (cash270 populated).
    # Even with varying game lengths subtract only 270 candidate calls per game.
    def remaining_ms(step_floor):
        return 1000 * total_worker_s / (len(old) * step_floor) - candidate_ms_min
    f0 = json.loads(archived("m3s2_timing_cert.json"))
    assert f0["identity"]["pack"]["child_md5s"] == [SOURCE_MD5]*20
    return {
        "archive": ARCHIVE, "registered_workers": WORKERS,
        "D": {"wins": sum(r["win"] for r in old), "games":len(old),
              "seat_wins": [sum(r["win"] for r in old if r["side"] == s) for s in (0, 1)],
              "mean": {k:round(statistics.mean(r[k] for r in old), 3) for k in ("sa","sb","margin","ms")},
              "waves": wave_report(old), "opponent_ledger_keys": sorted(old[0]["bst"])},
        "C_no_search": {"wins":sum(r["win"] for r in off), "waves":wave_report(off)},
        "paired_D_minus_C": paired(old, off),
        "D_clock_upper_bounds_assuming_registered_serial_B_then_D": {
            "previous_B_last_receipt": max(r["t"] for r in prior), "last_D_receipt":max(r["t"] for r in old),
            "window_s_including_rounding":window, "combined_worker_s_max": total_worker_s,
            "min_observed_steps_all_games":270, "B_ms_per_step_max_from_270_floor":round(remaining_ms(270),3),
            "B_ms_per_step_max_for_registered_299_steps":round(remaining_ms(299),3)},
        "F0_separate_direct_certificate": {"pack":f0["pack"], "opponent":f0["opp"],
            **{k:f0["aggregate"][k] for k in ("steps","step_wall_ms","peak_rss_mib_child_self")}},
        "direct_and_replication":new_results(),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))

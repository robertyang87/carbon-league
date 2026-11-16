"""S3 root-list merge verification (search group, independent of the script's
own merge_shards checks).

Usage:
  python3 merge_verify.py --shards trial_out out_c1_lane1 ... \
      --expect-seeds 986200-986499 --out MERGE_VERIFICATION.json
  python3 merge_verify.py --merged <merged_dir> --expect-seeds 986200-986499 \
      --out MERGE_VERIFICATION.json   # verify an already-merged directory

Checks: seed union exactly == expected range (no missing/extra/overlap),
root_id uniqueness, per-game root counts vs shard reports, class distribution,
settlement-decidable roots (all step 298, argmax values), extinction-terminated
games (stop_reason != time_limit_299) listed for the freeze record's
first-extinction cross-check note. Exits non-zero on any failure.
"""
import argparse
import hashlib
import json
import os
import sys


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shard(d):
    roots, reg, reports = [], [], []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name.startswith("roots_") and name.endswith(".json"):
            roots.extend(json.load(open(p, encoding="utf-8"))["roots"])
        elif name.startswith("register_") and name.endswith(".json"):
            reg.extend(json.load(open(p, encoding="utf-8"))["register"])
        elif name.startswith("report_") and name.endswith(".json"):
            reports.append(json.load(open(p, encoding="utf-8")))
    return roots, reg, reports


def parse_range(s):
    a, b = s.split("-")
    return set(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--merged", default=None)
    ap.add_argument("--expect-seeds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.merged:
        doc = json.load(open(os.path.join(args.merged,
                                          "S3_ROOT_LIST_FROZEN.json"),
                             encoding="utf-8"))
        roots = doc["roots"]
        reg = json.load(open(os.path.join(args.merged,
                                          "S3_GAME_REGISTER.json"),
                             encoding="utf-8"))["register"]
        reports = [json.load(open(os.path.join(
            args.merged, "S3_ENUMERATION_REPORT.json"), encoding="utf-8"))]
        sources = [args.merged]
    else:
        roots, reg, reports = [], [], []
        for d in args.shards:
            r, g, rep = load_shard(d)
            roots.extend(r)
            reg.extend(g)
            reports.extend(rep)
        sources = args.shards

    want = parse_range(args.expect_seeds)
    failures = []

    seeds = {986200 + g["game"] for g in reg} | \
        {r["seed"] for r in roots}
    if seeds != want:
        failures.append(f"seed union mismatch: missing "
                        f"{sorted(want - seeds)[:10]} (n={len(want - seeds)}), "
                        f"extra {sorted(seeds - want)[:10]} (n={len(seeds - want)})")

    ids = [r["root_id"] for r in roots]
    if len(ids) != len(set(ids)):
        failures.append(f"root_id not unique: {len(ids)} ids, "
                        f"{len(set(ids))} unique")

    games = sorted({g["game"] for g in reg})
    if len(games) != len(reg):
        failures.append("register has duplicate game numbers")
    if reg and sorted(g["game"] for g in reg) != list(range(min(games),
                                                            max(games) + 1)):
        failures.append("register games not contiguous")

    # per-game root counts vs register root_count
    from collections import Counter
    per_game = Counter(r["game"] for r in roots)
    for g in reg:
        if per_game.get(g["game"], 0) != g.get("root_count"):
            failures.append(f"game {g['game']}: register root_count "
                            f"{g.get('root_count')} != roots file "
                            f"{per_game.get(g['game'], 0)}")
            break

    classes = Counter(r.get("trigger_class") for r in roots)
    if set(classes) - {"class1"}:
        failures.append(f"non-class1 roots present: "
                        f"{dict(classes)}")

    dec = [r for r in roots if r.get("settlement_decidable")]
    if any(r["step"] != 298 for r in dec):
        failures.append("settlement-decidable root not at step 298")
    argmax_nonzero = sum(1 for r in dec if r.get("settlement_argmax") != 0)

    ext = [g for g in reg
           if g.get("stop_reason") not in (None, "time_limit_299")]
    ext_games = sorted(g["game"] for g in ext)

    # shard reports cross-check
    shard_total = sum(rep.get("total_roots", 0) for rep in reports)
    if reports and shard_total != len(roots):
        failures.append(f"shard reports total roots {shard_total} != "
                        f"merged {len(roots)}")

    result = {
        "sources": sources,
        "expect_seeds": args.expect_seeds,
        "games": len(games),
        "total_roots": len(roots),
        "unique_root_ids": len(set(ids)),
        "classes": dict(classes),
        "settlement_decidable": len(dec),
        "settlement_argmax_nonzero": argmax_nonzero,
        "zero_root_games": [g["game"] for g in reg
                            if g.get("root_count", 1) == 0],
        "extinction_games": ext_games,
        "first_extinction_crosscheck": (
            "no extinction game in this set" if not ext_games else
            f"first extinction game {ext_games[0]}: per-case leaf-F01 "
            f"cross-check result must be filed with the freeze package"),
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

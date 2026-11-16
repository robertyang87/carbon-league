#!/usr/bin/env python3
"""Merge per-game k0_duel shard JSONs into one full-run-equivalent result.

Usage: merge_duel_shards.py FULL_MANIFEST SHARD_JSON... OUT
Shard JSONs must be given in manifest game order. Rows are pooled in that
order (identical to a single full run over the same manifest), the summary is
recomputed with k0_duel's own cluster_stats / cluster_bootstrap_ci, and
roots_sha256 is recomputed from the FULL manifest via k0_duel.load_roots.
Writes the final JSON with k0_duel._atomic_dump (_sanitize applied).
"""
import json, os, sys
sys.path.insert(0, "/home/arm/junyang/carbon-repo/code")
os.chdir("/home/arm/junyang/carbon-repo/code")
import k0_duel

def main():
    manifest, *rest = sys.argv[1:]
    *shard_paths, out = rest
    assert len(shard_paths) >= 1
    rows = []
    first = None
    for p in shard_paths:
        d = json.load(open(p))
        assert d["kind"] == "k0_duel_result" and d["version"] == 1, p
        if first is None:
            first = d
        else:
            assert d["student"] == first["student"] and d["parent"] == first["parent"], p
            assert d["seed_cont"] == first["seed_cont"] and d["teacher_style"] == first["teacher_style"], p
        rows.extend(d["rows"])
    specs, roots_sha, _blobs = k0_duel.load_roots(manifest, 200, 299)
    # Each shard row carries its own shard roots_sha256; a full run writes
    # the full-manifest sha on every row. Normalize so the merged product
    # is structurally identical to a full run (values untouched).
    for r in rows:
        r["roots_sha256"] = roots_sha
    assert len(rows) == len(specs) == 9900, "rows=%d specs=%d" % (len(rows), len(specs))
    keys = {(r["game"], r["step"], r["me"]) for r in rows}
    want = {(s["game"], s["step"], s["me"]) for s in specs}
    assert keys == want and len(keys) == 9900, "row coverage mismatch"
    stats = k0_duel.cluster_stats(rows, "du")
    blo, bhi = k0_duel.cluster_bootstrap_ci(rows, first["seed_cont"], "du")
    result = {"kind": "k0_duel_result", "version": 1,
              "student": first["student"], "parent": first["parent"],
              "roots_sha256": roots_sha,
              "seed_cont": first["seed_cont"], "teacher_style": first["teacher_style"],
              "sources": first["sources"],
              "summary": {"delta_u": stats["game_mean"], "n_roots": stats["n_rows"],
                          "n_games": stats["n_games"], "cluster": stats,
                          "cluster_bootstrap_ci95": [blo, bhi]},
              "rows": rows,
              "window": {"step_min": 200, "step_max": 299}}
    k0_duel._atomic_dump(k0_duel._sanitize(result), out)
    s = result["summary"]
    print("MERGED %d rows / %d games delta_u=%s t_ci95=%s boot_ci95=%s" % (
        len(rows), s["n_games"], s["delta_u"], s["cluster"]["game_ci95_t"],
        s["cluster_bootstrap_ci95"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())

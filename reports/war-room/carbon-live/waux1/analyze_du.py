#!/usr/bin/env python3
"""waux1 exam verification + paired dU vs wr1 baseline.

Sources: sync/h29 branch files via gh_api (the archived record).
U = 0.5*W + 0.5*sigmoid(margin/1000); W: win 1 / draw 0.5 / loss 0 (margin sign).
Paired per seed within each field; pooled = 400 pairs. Bootstrap percentile
CI, 20000 resamples, fixed RNG seed.
"""
import base64, json, math, sys
import numpy as np

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, REF = "robertyang87", "carbon-league", "sync/h29"


def fetch_lines(path):
    f = gh_api.get_file(OWNER, REPO, path, ref=REF)
    raw = base64.b64decode(f["content"])
    rows = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["seed"]] = r
    return rows, f["sha"], len(raw)


def stats(rows):
    margins = np.array([rows[s]["margin"] for s in sorted(rows)], dtype=float)
    wins = int((margins > 0).sum())
    mean = float(margins.mean())
    se = float(margins.std(ddof=1) / math.sqrt(len(margins)))
    return wins, mean, se, float(margins.min())


def ms_max(rows, key):
    vals = [rows[s].get(key) for s in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def U(margin):
    w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))


def paired(a_rows, b_rows):
    seeds = sorted(set(a_rows) & set(b_rows))
    d = np.array([U(a_rows[s]["margin"]) - U(b_rows[s]["margin"]) for s in seeds])
    return seeds, d


def boot_ci(d, rng, n_boot=20000):
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    first = None
    fields = {}
    for field, atag, btag in [
        ("m2", "m9xj570waux1_m2", "m9xj570wr1_m2"),
        ("echo2", "m9xj570waux1_echo2", "m9xj570wr1_echo2"),
    ]:
        a, sha_a, nb_a = fetch_lines(f"synced/h29/bigeval_live/{atag}.jsonl")
        b, sha_b, nb_b = fetch_lines(f"synced/h29/bigeval_live/{btag}.jsonl")
        if first is None:
            first = a[sorted(a)[0]]
        aw, am, ase, aworst = stats(a)
        bw, bm, bse, bworst = stats(b)
        errs_a = sum(1 for s in a if a[s].get("err"))
        print(f"== {field} ==")
        print(f" waux1: n={len(a)} seeds {min(a)}-{max(a)} wins={aw} mean={am:.2f} se={ase:.2f} "
              f"worst={aworst:.0f} err_rows={errs_a} ms_max={ms_max(a, 'ms')} wms_max={ms_max(a, 'wms')} "
              f"sha={sha_a[:8]} bytes={nb_a}")
        print(f" wr1  : n={len(b)} seeds {min(b)}-{max(b)} wins={bw} mean={bm:.2f} se={bse:.2f} worst={bworst:.0f} sha={sha_b[:8]} bytes={nb_b}")
        seeds, d = paired(a, b)
        fields[field] = (seeds, d)
        print(f" paired n={len(seeds)} seed_match={len(seeds) == len(a) == len(b)}")
    print("line keys sample:", sorted(first.keys()))

    rng = np.random.default_rng(20261113)
    all_d = []
    for field, (seeds, d) in fields.items():
        lo, hi = boot_ci(d, rng)
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        print(f"dU {field}: n={len(d)} mean={d.mean():+.5f} se={se:.5f} CI95=[{lo:+.5f},{hi:+.5f}]")
        all_d.append(d)
    d = np.concatenate(all_d)
    lo, hi = boot_ci(d, rng)
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    mean = float(d.mean())
    print(f"dU POOLED: n={len(d)} mean={mean:+.5f} se={se:.5f} CI95=[{lo:+.5f},{hi:+.5f}]")
    ok = (mean >= 0.02) and (lo > 0)
    print(f"CRITERION: mean>=0.02 -> {mean >= 0.02}; CI_lo>0 -> {lo > 0}; VERDICT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

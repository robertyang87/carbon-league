#!/usr/bin/env python3
"""paired_du -- paired dU between ANY two archived bigeval per-game logs.

The section-17 verdicts pair a checkpoint against the standing baseline
(m3s2r1). That answers "is it better than the incumbent?", not "what did
the training segment change?" -- the second question needs the learner's
own earlier state as the reference. This tool pairs any two per-game
logs that were played on the same (seed, side) keys.

  U = 0.5*W + 0.5*sigmoid(margin/1000), W = 1 / 0.5 / 0 (win/draw/loss)
  dU = mean over seeds of [U_a - U_b]; CI = percentile bootstrap over
  seed clusters (section 20 R2), B resamples, fixed rng seed.

A source is a file path or REF:PATH (read with `git show`), e.g.
  tools/paired_du.py origin/sync/h31:synced/h31/bigeval_live/it95_echo2.jsonl \
                     origin/sync/h31:synced/h31/bigeval_live/snap_70_echo2.jsonl
Several A,B pairs may be pooled: --pool A1 B1 A2 B2 ...
Exit 2 on any err row, duplicate key or key-set mismatch (never a
silent partial pairing).
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys

import numpy as np


def terminal_u(m):
    w = 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))


def load(src):
    if os.path.exists(src):
        text = open(src).read()
    elif ":" in src:
        text = subprocess.run(["git", "show", src], check=True, capture_output=True, text=True).stdout
    else:
        sys.exit(f"paired_du: no such file or git object: {src}")
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("err") is not None:
            print(f"paired_du: {src}:{n} carries err={r['err']!r}", file=sys.stderr); sys.exit(2)
        k = (int(r["seed"]), int(r["side"]))
        if k in out:
            print(f"paired_du: {src}:{n} duplicate key {k}", file=sys.stderr); sys.exit(2)
        out[k] = float(r["margin"])
    return out


def clusters(a_src, b_src):
    a, b = load(a_src), load(b_src)
    if set(a) != set(b):
        print(f"paired_du: key sets differ ({len(set(a) ^ set(b))} unmatched (seed, side) keys): "
              f"{a_src} vs {b_src}", file=sys.stderr)
        sys.exit(2)
    by = collections.defaultdict(list)
    for k in sorted(a):
        by[k[0]].append(terminal_u(a[k]) - terminal_u(b[k]))
    same = sum(a[k] == b[k] for k in a)
    side = {n: {"games": len(d), "wins": int(sum(m > 0 for m in d.values())),
                "mean_margin": round(float(np.mean(list(d.values()))), 2),
                "mean_u": round(float(np.mean([terminal_u(m) for m in d.values()])), 4)}
            for n, d in (("a", a), ("b", b))}
    return np.array([np.mean(v) for _s, v in sorted(by.items())]), side, same


def summarize(c, B, seed):
    rng = np.random.default_rng(seed)
    bs = c[rng.integers(0, len(c), (B, len(c)))].mean(1)
    return {"dU": round(float(c.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "clusters": int(len(c)), "p_boot_le0": round(float((bs <= 0).mean()), 4),
            "pos": int((c > 0).sum()), "neg": int((c < 0).sum()), "zero": int((c == 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="A B (one pairing)")
    ap.add_argument("--pool", nargs="+", default=None, help="A1 B1 A2 B2 ...: per-pair results plus the pooled one")
    ap.add_argument("-B", type=int, default=20000)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    srcs = a.pool if a.pool else a.sources
    if len(srcs) < 2 or len(srcs) % 2 or (a.pool and a.sources):
        ap.error("give exactly A B, or --pool with an even number of sources")
    allc = []
    for i in range(0, len(srcs), 2):
        c, side, same = clusters(srcs[i], srcs[i + 1])
        allc.append(c)
        print(json.dumps({"a": srcs[i], "b": srcs[i + 1], **summarize(c, a.B, a.rng), "sides": side,
                          "identical_margin_games": same}))
    if len(allc) > 1:
        print(json.dumps({"pooled": len(allc), **summarize(np.concatenate(allc), a.B, a.rng)}))


if __name__ == "__main__":
    main()

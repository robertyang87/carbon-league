"""Descriptive reliability bins and CPU-only cost on the frozen four training maps."""

import argparse
import gzip
import json
from pathlib import Path
import os
import time
import numpy as np
import runtime
from study import HERE, roots, sha, source, write

BINS = np.array([0, 0.10, 0.25, 0.50, 0.75, 1.0])


def calibration(rows):
    report = {}
    for estimator in ("baseline", "risk"):
        p = []
        y = []
        for r in rows:
            if r["load"] < 200:
                continue
            for k, base in enumerate(r["baseline"]):
                if base > 0:
                    p.append(r[estimator][k])
                    y.append(r["labels"][k])
        predictions = np.array(p)
        outcomes = np.array(y)
        groups = np.digitize(predictions, BINS[1:-1])
        detail = []
        for i in range(len(BINS) - 1):
            keep = groups == i
            pred = predictions[keep]
            observed = outcomes[keep]
            detail.append(
                dict(
                    lo=BINS[i],
                    hi=BINS[i + 1],
                    n=len(pred),
                    mean_score=float(pred.mean()) if len(pred) else None,
                    observed_rate=float(observed.mean()) if len(pred) else None,
                )
            )
        ece = sum(
            b["n"] * abs(b["mean_score"] - b["observed_rate"]) for b in detail if b["n"]
        ) / len(outcomes)
        report[estimator] = dict(pooled_ece=ece, bins=detail)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    hold = json.loads(gzip.decompress((out / "held_orders.json.gz").read_bytes()))
    write(
        out / "reliability.json",
        dict(
            artifact_sha256=sha((out / "held_orders.json.gz").read_bytes()),
            note="Descriptive bins fixed for this follow-up after the registered test. Pooled actions, correlated inside games. Intervals include left boundary; last includes 1.",
            metrics=calibration(hold),
        ),
    )
    head = json.loads((out / "weights.json").read_bytes())
    w = np.asarray(head["weights"], dtype=np.float32)
    durations = []
    counts = []
    for c in json.loads((HERE / "protocol.json").read_bytes())["maps"]["train"]:
        g = source(c)
        p = None
        for t in range(299):
            previous = p
            p = g["steps"][t][0]["observation"]["players"]
            if t not in range(7, 294, 13):
                continue
            for _ in range(3):
                start = time.perf_counter_ns()
                s = runtime.prepare(p, c["seat"], t, previous)
                n = 0
                for uid, info in s["our"].items():
                    if info[2] == "COLLECTOR":
                        runtime.forecast(s, uid, w, head["recruit_probability"])
                        n += 1
                durations.append((time.perf_counter_ns() - start) / 1e6)
                counts.append(n)
    write(
        out / "local_cost.json",
        dict(
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            weight_sha256=sha((out / "weights.json").read_bytes()),
            unique_roots=len(durations) // 3,
            repetitions=3,
            all_current_collectors_per_call=True,
            total_scored_units=sum(counts),
            milliseconds={
                k: float(np.quantile(durations, q))
                for k, q in [("median", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1)]
            },
            note="Local resident standalone readout only. No original actor/search/IPC/init; this is not a certified platform timing.",
        ),
    )


if __name__ == "__main__":
    main()

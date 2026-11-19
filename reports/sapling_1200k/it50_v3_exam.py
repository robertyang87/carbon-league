"""Frozen dual-policy direct diagnostic. One worker, original Master3s2, both seats."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cash_v3_exam as scoring

ROOT = scoring.exam.ROOT
PRIOR = scoring.exam.prior
PACK = ROOT / "data/weights/sapling_1200k/hybrid_it50_v3.py"
SHA = "b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
MAP = 620098101
CPU = 4


def write_seat(seat, out):
    original = PRIOR.shared.module
    candidate = []

    def track_module(path, name):
        mod = original(path, name)
        if Path(path).resolve() == PACK.resolve():
            candidate.append(mod)
        return mod

    PRIOR.shared.module = track_module
    result = scoring.game(("it50_75_cash_v3_25", str(PACK), MAP, seat, CPU))
    assert len(candidate) == 1
    result["hybrid_config"] = candidate[0]._HYBRID
    result["action_attribution_on_shared_prefix"] = candidate[0]._COUNTS
    PRIOR.shared.atomic_json(out, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seat", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.out.exists():
        p.error("fresh output required")
    if PRIOR.digest(PACK) != SHA or PRIOR.digest(PRIOR.SOURCE, "md5") != PRIOR.EXPECTED_MD5:
        raise ValueError("unexpected candidate or opponent identity")
    if args.seat is not None:
        return write_seat(args.seat, args.out)
    result = {"candidate": "it50 75% + Sapling cash_v3 25% shared-prefix raw legal logits, guard off, style 2",
              "pack_sha256": SHA, "pack_bytes": PACK.stat().st_size,
              "examiner": "original complete Master3s2, TEMP=1 internal SEED=None; bounded wall-clock search",
              "examiner_md5": PRIOR.digest(PRIOR.SOURCE, "md5"), "map": MAP, "seats": [0, 1],
              "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
              "status": "running", "utc_started": PRIOR.utc(), "games": []}
    PRIOR.shared.atomic_json(args.out, result)
    with tempfile.TemporaryDirectory() as tmp:
        for seat in (0, 1):
            one = Path(tmp) / f"{seat}.json"
            subprocess.run([sys.executable, __file__, "--seat", str(seat), "--out", str(one)], check=True)
            row = json.loads(one.read_text())
            result["games"].append(row)
            PRIOR.shared.atomic_json(args.out, result)
            print(seat, row["margin"], row["result"], row["actual_transitions"], flush=True)
    result["status"] = "complete"
    result["utc_finished"] = PRIOR.utc()
    PRIOR.shared.atomic_json(args.out, result)


if __name__ == "__main__":
    main()

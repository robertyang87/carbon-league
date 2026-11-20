"""Download verification uses the same per-arm replay, official results and timing as the hosted runner."""
import argparse
import importlib.util
import json
from pathlib import Path
import matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--matrix-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    path = matrix.BASE / "settle.py"
    spec = importlib.util.spec_from_file_location("two_map_official_auditor", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.m = matrix
    audit.CONTRASTS = (("candidate", "parent"),)
    result = audit.merge(args.run, args.commit, args.archives, args.out, args.matrix_sha)
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    return int(not result["records_reconciled"])


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply the signature for the full-backbone policy to the established fixed-replay limit test."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "learned_carbon_observation/ladder_timing"))
import time_replay as engine
from make_reference import PACKAGE_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-sha", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    engine.PLAN = HERE / "protocol.json"
    engine.PLAN_SHA = args.registration_sha
    raw, reference, history = engine.check_inputs()
    files = engine.live.PLAN["files"]
    if files["candidate"]["sha256"] != PACKAGE_SHA or files["variant_adapter"]["sha256"] != engine.live.sha(Path(__file__)):
        raise ValueError("variant identity mismatch")
    if args.check:
        print("Verified public replay and candidate", history, PACKAGE_SHA)
        return 0
    if not args.image or not args.out:
        parser.error("provide --check or both --out and --image")
    code = asyncio.run(engine.run(args, raw, reference, history))
    result = json.loads((args.out / "receipt.json").read_bytes())
    return int(bool(code) or result.get("command_match_counts") != [299, 299])


if __name__ == "__main__":
    sys.exit(main())

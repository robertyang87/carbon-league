"""Apply the pinned decoder registration to the original, immutable two-seat timing engine."""
import argparse
import asyncio
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"ladder_timing"))
import time_replay as engine

PACKAGE_SHA="61dae17267eb6f581efba533750134ccc62c6d5316526fce9a0e5db1996d6ead"


def configure(registration_sha):
    engine.PLAN=HERE/"protocol.json"
    engine.PLAN_SHA=registration_sha
    raw,reference,history=engine.check_inputs()
    assert engine.live.PLAN["files"]["candidate"]["sha256"]==PACKAGE_SHA
    assert engine.live.PLAN["files"]["variant_adapter"]["sha256"]==engine.live.sha(Path(__file__))
    return raw,reference,history


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--registration-sha",required=True)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    p.add_argument("--image")
    args=p.parse_args()
    raw,ref,history=configure(args.registration_sha)
    if args.check:
        print("Verified decoder package, reference actions and public replay",PACKAGE_SHA,history)
    elif args.out and args.image:
        sys.exit(asyncio.run(engine.run(args,raw,ref,history)))
    else:
        p.error("provide --check or both --out and --image")

"""Regenerate 598 public commands on the original signed fixed tape, with one fresh process per seat."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CARGO = HERE.parents[1]
PRIOR = CARGO / "learned_carbon_observation/decoder/ladder_timing"
sys.path.insert(0, str(CARGO / "ladder_live"))
import live

PACKAGE_SHA = "6c54899562382552b95bc1299d83119088ef1d1206b93135e33769815ed69c9f"
SEED = 621471120


def collect(cpu):
    os.sched_setaffinity(0, {cpu})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    live.configure(PRIOR / "protocol.json", HERE / ".unused-claim", HERE / ".unused-lock")
    item = live.PLAN["files"]["replay"]
    raw_path = live.ROOT / item["path"]
    pack = HERE.parent / "full.py"
    if live.sha(pack) != PACKAGE_SHA or live.sha(raw_path) != item["sha256"]:
        raise ValueError("student or archive identity differs")
    protocol = json.loads((HERE.parent / "protocol.json").read_bytes())
    if SEED not in protocol["quarantine"]:
        raise ValueError("timing terrain not quarantined from fitting")
    raw = json.loads(gzip.decompress(raw_path.read_bytes()))
    if raw["seed"] != SEED or raw["initial_terrain_u8_sha256"] != live.PLAN["physical_map_sha256"] or len(raw["actions"]) != raw["steps"] or raw["steps"] != 299:
        raise ValueError("historical trajectory failed identity/length checks")
    worker = CARGO / "ladder_timing/worker.py"
    output = []
    for seat in (0, 1):
        env, config = live.official()
        with subprocess.Popen([sys.executable, "-u", str(worker), str(pack), str(cpu), "0"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            ready = json.loads(process.stdout.readline())
            if ready["sha256"] != PACKAGE_SHA or ready["affinity"] != [cpu] or ready["numpy"] != "1.26.4":
                raise ValueError("local persistent worker identity mismatch")
            rows = []
            for step, joint in enumerate(raw["actions"]):
                obs = live.observe(env, seat, 5.)
                if obs["step"] != step or obs["player"] != seat or env.done:
                    raise ValueError("archived trajectory out of order")
                process.stdin.write(live.encoded({"observation": obs, "configuration": config}))
                process.stdin.flush()
                reply = json.loads(process.stdout.readline())
                call = {"status": "ok", "response": reply}
                live.check_action(call, obs)
                if call["status"] != "ok":
                    raise ValueError(call)
                rows.append({"step": step, "action": reply["action"]})
                env.step(joint)
            process.stdin.close()
            if process.wait() != 0 or not env.done or [s.reward for s in env.states] != raw["official_score_by_seat"]:
                raise ValueError("worker or archive endpoint mismatch")
        output.append({"seat": seat, "timing": {"ready": {"pack_sha256": PACKAGE_SHA}, "rows": rows}})
    return {"student": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    compressed = gzip.compress(live.encoded(collect(args.cpu)), mtime=0)
    with args.out.open("xb") as handle:
        handle.write(compressed)
    print(hashlib.sha256(compressed).hexdigest())

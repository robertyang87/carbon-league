"""One conditional H24 timing-window comparison, driven by frozen hosted receipts."""
import argparse
import ast
import base64
import hashlib
import zlib
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "fresh_original"
sys.path.insert(0, str(PRIOR))
import play
from verify_pair import verify_game

live = play.live
ARMS = ("parent", "student", "window")


def verify_registration(matrix):
    base = play.check(play.load(PRIOR/"protocol.json"))
    for artifact in matrix["files"].values():
        path = live.ROOT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or live.sha(path) != artifact["sha256"]:
            raise ValueError(f"Artifact changed: {path}")
    pack = play.load(HERE.parent/"pack.json")
    serving = play.load(HERE.parent/"qualification.json")
    files = matrix["files"]
    if pack["all_tensor_provenance"] != {key:files[role]["sha256"] for key,role in
            (("parent","parent"),("all_steps","student"),("parent_weights","parent_weights"),
             ("trained_weights","window_weights"))}:
        raise ValueError("Window weights or source package mismatch")
    tensors = {r[0] for r in pack["embedded_subset"]}
    if tensors != {"U.weight","U.bias","V.weight","V.bias"} or pack["included_tensor_bytes"] != 14464:
        raise ValueError("Wrong parameter subset")
    import numpy as np
    with np.load(live.ROOT / files["parent_weights"]["path"], allow_pickle=False) as old, np.load(
            live.ROOT / files["window_weights"]["path"], allow_pickle=False) as new:
        if set(new.files) != set(old.files) or {k for k in new.files if not np.array_equal(new[k],old[k])} != tensors:
            raise ValueError("Additional checkpoint parameters changed")
        trained = b"".join(new[r[0]].astype("<f4").tobytes() for r in pack["embedded_subset"])
    body = ast.parse((live.ROOT / files["window"]["path"]).read_bytes())
    payload, = [ast.literal_eval(node.value) for node in body.body if isinstance(node,ast.Assign)
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id == "_WINDOW_BYTES"]
    if zlib.decompress(base64.b64decode(payload,validate=True)) != trained or hashlib.sha256(trained).hexdigest() != pack["window_quantized_sha256"]:
        raise ValueError("Window package does not embed the four trained FP32 tensors")
    n = sum(r["n"] for r in serving["rows"])
    if (pack["candidate"] != files["window"]["sha256"] or serving["sha"] != pack["candidate"]
            or not serving["local_precheck_passed"] or serving["slow"] != 0
            or serving["inside"] != 128 or n != 1196 or serving["effective_inside"] != 67
            or any(r["wrong_in"] or r["wrong_out"] for r in serving["rows"])):
        raise ValueError("Window deployment qualification failed")
    for arm in ARMS:
        if files[arm]["bytes"] > base["platform"]["source_limit_bytes"]:
            raise ValueError(f"Submission oversize: {arm}")
    if play.load(HERE/"clearance.json")["physical_boards"] != matrix["physical_boards"]:
        raise ValueError("Physical terrain clearance changed")
    return base


def activation(matrix, artifacts):
    prereg = play.load(PRIOR/"protocol.json")
    registered_run = str(matrix["prior_run"])
    outcomes = []
    for seed in prereg["seeds"]:
        for seat in (0, 1):
            folder = artifacts/f"h24-{seed}-{seat}-{registered_run}"
            both = [verify_game(folder/arm, seed, seat, arm, PRIOR/"protocol.json") for arm in ("parent","student")]
            if (len({g["image_id"] for g in both}) != 1 or
                  any(str(g["workflow_run"]) != registered_run or not g["counted"] or
                      g["status"] not in ("complete","forfeit") for g in both)):
                raise ValueError(f"Incomplete or foreign prior pair: {seed}/{seat}")
            tested = both[1]
            outcomes.append({"seed":seed, "seat":seat, "status":tested["status"],
                             "win":tested["winner_seat"] == seat})
    return {"prior_run":registered_run, "eligible":any(not r["win"] for r in outcomes),
            "student_outcomes":outcomes}


def run_session(matrix, base, args):
    schedule = [t for t in matrix["schedule"] if (t["seed"], t["seat"]) == (args.seed,args.seat)]
    if len(schedule) != 1 or set(schedule[0]["order"]) != set(ARMS) or len(schedule[0]["order"]) != 3:
        raise ValueError("Unregistered job")
    gate = play.load(args.activation_receipt)
    if (not gate["eligible"] or str(gate["prior_run"]) != str(matrix["prior_run"])
            or len(gate["student_outcomes"]) != 4):
        raise ValueError("Prior official results did not activate this comparison")
    group = sorted(os.sched_getaffinity(0))[:2]
    if len(group) != 2:
        raise ValueError("Two shared CPU cores required")
    image = subprocess.check_output(["docker","image","inspect","--format","{{.Id}}",args.image],text=True).strip()
    args.out.mkdir(exist_ok=True, parents=True)
    status = []
    for arm in schedule[0]["order"]:
        out = args.out/arm
        out.mkdir(exist_ok=False)
        effective = play.game_plan(matrix,base,arm,args.seed,args.seat)
        effective["matrix_sha256"] = live.sha(HERE/"protocol.json")
        effective["activation_receipt_sha256"] = live.sha(args.activation_receipt)
        live.save(out/"registration.json",effective)
        live.configure(out/"registration.json",out/"attempt.json",args.out/".match.lock")
        try:
            status.append(asyncio.run(live.run(argparse.Namespace(image=image, out=out, play=True,
                                                 cpus=",".join(map(str,group))))))
        except Exception as exc:
            live.save(out/"orchestration_error.json", {"arm":arm, "error":repr(exc)})
            status.append(2)
    return max(status)


def verify_session(out, seed, seat):
    records, errors = {}, {}
    for arm in ARMS:
        try:
            records[arm] = verify_game(out/arm,seed,seat,arm,HERE/"protocol.json")
            game = play.load(out/arm/"registration.json")
            if game["activation_receipt_sha256"] != live.sha(out/"activation.json"):
                raise ValueError("Different prior activation receipt")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            errors[arm] = f"{type(exc).__name__}: {exc}"
    for key in ("image_id", "workflow_run"):
        if len({r[key] for r in records.values()}) > 1:
            errors["session"] = f"Different {key}"
    return {"seed":seed, "seat":seat, "games":records, "verification_errors":errors}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("check","activate","play","verify"))
    cli.add_argument("--out",type=Path)
    cli.add_argument("--prior-artifacts",type=Path)
    cli.add_argument("--activation-receipt",type=Path)
    cli.add_argument("--seed",type=int)
    cli.add_argument("--seat",type=int,choices=(0,1))
    cli.add_argument("--image",default="sapling-window:registered")
    args = cli.parse_args()
    m = play.load(HERE/"protocol.json")
    base = verify_registration(m)
    if args.mode == "check":
        return 0
    if args.mode == "activate":
        result = activation(m,args.prior_artifacts)
        live.save(args.activation_receipt, result)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"],"a") as fd:
                print("play=" + str(result["eligible"]).lower(),file=fd)
        print(json.dumps(result))
        return 0
    if args.mode == "verify":
        result = verify_session(args.out,args.seed,args.seat)
        print(json.dumps(result,indent=2,allow_nan=False))
        return int(bool(result["verification_errors"]))
    return run_session(m,base,args)


if __name__ == "__main__":
    sys.exit(main())

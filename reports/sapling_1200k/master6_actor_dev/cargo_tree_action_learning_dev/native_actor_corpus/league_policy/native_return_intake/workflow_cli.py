"""Bind the intake sparse checkout and reuse the frozen Actions exposure verifier."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get("GITHUB_WORKSPACE", next(p for p in HERE.parents if (p/".git").exists())))
PREVIOUS=HERE.parent/"natural_original/fresh_repeat"
HELPER=PREVIOUS/"workflow_cli.py"
HELPER_SHA="061c4e2e42329dd2f424edddedcbe2e20c578d465c0041fcdef3319c62023794"
STEP="Play two independent original return draws"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(digest):
    protocol=HERE/"protocol.json"
    if not re.fullmatch(r"[0-9a-f]{64}",digest) or sha(protocol)!=digest:
        raise ValueError("Dispatch digest differs from the signed intake protocol")
    files=json.loads(protocol.read_bytes())["files"]
    expected={f"previous_archive_{i}" for i in range(4)}
    archives={k:v["path"] for k,v in files.items() if v["path"].lower().endswith(".zip")}
    pattern=re.escape((PREVIOUS/"hosted").relative_to(ROOT).as_posix())+r"/[0-9]+\.zip"
    if set(archives)!=expected or len(set(archives.values()))!=4 or any(not re.fullmatch(pattern,p) for p in archives.values()):
        raise ValueError("Expected four signed fresh-repeat receipt archives")
    total=sum(int(subprocess.check_output(["git","cat-file","-s","HEAD:"+p],cwd=ROOT)) for p in archives.values())
    if total>2_000_000:raise ValueError("Unexpected total archive size")
    paths={}
    for f in files.values():
        p,h=f["path"],f["sha256"]
        path=PurePosixPath(p)
        if (path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+",p)
            or not re.fullmatch("[0-9a-f]{64}",h) or paths.setdefault(p,h)!=h):
            raise ValueError("Invalid or conflicting frozen dependency")
    subprocess.run(["git","sparse-checkout","add","--stdin"],cwd=ROOT,
                   input=("\n".join("/"+p for p in sorted(paths))+"\n").encode(),check=True)
    for p,h in paths.items():
        if sha(ROOT/p)!=h:raise ValueError(f"Changed intake dependency {p}")

def job(seed,seat):return f"native-intake-{int(seed)}-{int(seat)}"

def exposure_helpers(seed,seat):
    if sha(HELPER)!=HELPER_SHA:raise ValueError("Unexpected historical exposure verifier")
    spec=importlib.util.spec_from_file_location("frozen_action_guard",HELPER)
    verifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    # The isolated module's guard and receipt generator share the same physical ticket identity.
    verifier.ROOT=ROOT
    verifier.WORKFLOW="sapling-native-return-intake.yml"
    verifier.STEP=STEP
    verifier.ARMS=("draw_a","draw_b")
    verifier.prefix=job
    api_pages=verifier.pages
    def checked_pages(endpoint,key):
        records=api_pages(endpoint,key)
        if key=="jobs":
            for entry in records:
                if entry.get("name")!=job(seed,seat) or entry.get("status")=="queued":continue
                steps=entry.get("steps")
                if (not isinstance(steps,list) or not any(s.get("name")==STEP for s in steps)):
                    raise ValueError("Prior named intake ticket has no auditable play-step record")
        return records
    verifier.pages=checked_pages
    return verifier

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","guard","exposure"))
    parser.add_argument("--digest")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--seat",type=int,choices=(0,1))
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.mode=="prepare":prepare(args.digest)
    elif args.mode=="guard":exposure_helpers(args.seed,args.seat).guard(args.seed,args.seat)
    else:exposure_helpers(args.seed,args.seat).exposure(args.out,args.seed,args.seat)

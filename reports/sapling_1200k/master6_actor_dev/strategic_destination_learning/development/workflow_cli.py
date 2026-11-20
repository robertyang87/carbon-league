"""Sparse-materialize frozen inputs, and retain one exposure ticket per physical map and seat."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/".git").exists())
NATURAL=HERE.parents[1]/"cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original"
SUITE="sapling-strategic-destination-development.yml"
ARMS=("parent","bc","joint")
EXPOSURE_STEP="Play three frozen strategic agents"


def prepare(digest):
    raw=(HERE/"protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError("Incorrect requested registration")
    frozen=json.loads(raw);files={}
    for entry in frozen["files"].values():
        path,expected=entry["path"],entry["sha256"]
        if path.startswith("/") or ".." in Path(path).parts or not re.fullmatch(r"[0-9A-Za-z_./-]+",path):
            raise ValueError("Invalid sparse input")
        if not re.fullmatch(r"[a-f0-9]{64}",expected) or files.setdefault(path,expected)!=expected:
            raise ValueError("Invalid frozen digest")
    base=NATURAL/"fresh_repeat/workflow_cli.py"
    if not base.relative_to(ROOT).as_posix() in files: raise ValueError("No registered previous exposure checker")
    subprocess.run(["git","-C",str(ROOT),"sparse-checkout","add","--stdin"],check=True,
                   input=("\n".join("/"+p for p in sorted(files))+"\n").encode())
    for path,digest in files.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest: raise ValueError("Frozen input changed: "+path)


def prior():
    p=NATURAL/"fresh_repeat/workflow_cli.py"
    s=importlib.util.spec_from_file_location("original_exposure_guard",p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.ROOT=ROOT;m.WORKFLOW=SUITE;m.STEP=EXPOSURE_STEP;m.ARMS=ARMS
    m.prefix=lambda seed,seat:f"strategic-dev-{seed}-{seat}"
    return m


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","guard","exposure"))
    p.add_argument("--digest");p.add_argument("--out",type=Path);p.add_argument("--seed",type=int);p.add_argument("--seat",type=int)
    a=p.parse_args()
    if a.mode=="prepare":prepare(a.digest)
    elif a.mode=="guard":prior().guard(a.seed,a.seat)
    else:prior().exposure(a.out,a.seed,a.seat)

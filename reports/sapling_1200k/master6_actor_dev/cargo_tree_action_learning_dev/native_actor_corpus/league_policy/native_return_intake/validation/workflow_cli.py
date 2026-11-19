"""Verify the signed validation closure and reject previously exposed validation tickets."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/".git").exists())
WF="sapling-native-return-validation.yml"
HELPER=HERE.parents[1]/"natural_original/fresh_repeat/workflow_cli.py"
SEEDS=(621758470,621758471)


def prepare(expected):
    p=HERE/"protocol.json"
    if not p.is_file() or not (HERE/"release.json").is_file():
        raise ValueError("Qualified training release not bound to held-out maps")
    if not re.fullmatch("[0-9a-f]{64}",expected or "") or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise ValueError("Unregistered holdout digest")
    d=json.loads(p.read_bytes())
    if d["seeds"]!=list(SEEDS) or d["expected_games"]!=8:raise ValueError("Wrong frozen map cohort")
    files={}
    for f in d["files"].values():
        path,digest=f["path"],f["sha256"]
        if (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or not re.fullmatch("[0-9a-zA-Z_./-]+",path)
                or not re.fullmatch("[0-9a-f]{64}",digest) or files.setdefault(path,digest)!=digest):
            raise ValueError("Invalid or conflicting sparse input")
    subprocess.run(["git","-C",str(ROOT),"sparse-checkout","add","--stdin"],
                   input=("\n".join("/"+path for path in sorted(files))+"\n").encode(),check=True)
    for path,digest in files.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError("Frozen input changed: "+path)


GAME_STEP="Play the frozen eight-game holdout pair"
ARMS=("sampled_parent","sampled_child")


def prior_check(runs,current):
    own=[x for x in runs if str(x["id"])==str(current)]
    if len(own)!=1 or own[0]["event"]!="workflow_dispatch" or own[0]["head_branch"]!="main":
        raise ValueError("Current workflow identity missing")
    older=[r for r in runs if r["event"]=="workflow_dispatch" and r["head_branch"]=="main"
           and r["run_number"]<own[0]["run_number"]]
    if any(r["status"]!="completed" for r in older):
        raise ValueError("Earlier workflow still has an unresolved ticket")
    return older


def pages(endpoint,key):
    data=json.loads(subprocess.check_output(["gh","api","--paginate","--slurp",endpoint]))
    if not data or any(not isinstance(p.get(key),list) for p in data):
        raise ValueError("Incomplete Actions inventory")
    rows=[r for p in data for r in p[key]]
    if len(rows)!=data[0]["total_count"] or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Incomplete or duplicate API inventory")
    return rows


def zero_exposure(proof,prior_id,attempt,seed,seat):
    if (proof["seed"],proof["seat"],str(proof["run_id"]),str(proof["attempt"]))!=(seed,seat,str(prior_id),str(attempt)):
        return False
    arms=proof["arms"]
    return (proof["confirmed_no_exposure"] is True and set(arms)==set(ARMS)
            and all(a.get("attempt") is False and type(a.get("initializations")) is int
                    and a["initializations"]==0 and a.get("action_calls")==[0,0] for a in arms.values()))


def ensure_setup_only(prior,jobs,receipt):
    """Allow recovery only if every prior ticket was skipped before play or has affirmative zero exposure."""
    names={f"native-validation-{s}-{p}":(s,p) for s in SEEDS for p in (0,1)}
    if not jobs or not any(j["name"]=="matrix" for j in jobs):
        raise ValueError("No authoritative prior planning job")
    planners=[j for j in jobs if j["name"]=="matrix"]
    tickets=[j for j in jobs if j["name"] in names]
    if len({j["name"] for j in tickets}) != 4 and any(j.get("conclusion")=="success" for j in planners):
        raise ValueError("Prior planning succeeded with incomplete ticket inventory")
    for j in jobs:
        name=j["name"]
        if j["status"]!="completed":raise ValueError("Prior job is not terminal")
        if name=="matrix":continue
        if name not in names:
            # The Actions API may include the single unevaluated matrix job when its dependency failed.
            if name in ("pair","native-validation--") and j.get("conclusion")=="skipped":continue
            raise ValueError("Unrecognized prior job: "+name)
        steps=j.get("steps")
        if not isinstance(steps,list):raise ValueError("Prior ticket has no authoritative steps")
        matches=[s for s in steps if s["name"]==GAME_STEP]
        if len(matches)!=1:
            if not matches and j.get("conclusion")=="skipped":continue
            raise ValueError("Could not identify prior play step")
        game=matches[0]
        if game.get("conclusion")=="skipped":continue
        if not game.get("started_at"):raise ValueError("Prior play step has uncertain exposure")
        seed,seat=names[name]
        attempt=j.get("run_attempt",prior["run_attempt"])
        if not zero_exposure(receipt(prior["id"],name,attempt),prior["id"],attempt,seed,seat):
            raise ValueError(f"Prior attempt, action or unknown exposure for {seed}/{seat}")


def guard():
    env=os.environ
    if env["GITHUB_RUN_ATTEMPT"]!="1":raise ValueError("A recovery dispatch must obtain an independent Actions run ID")
    repo=env["GITHUB_REPOSITORY"]
    api=f"/repos/{repo}/actions/"
    older=prior_check(pages(api+f"workflows/{WF}/runs?branch=main&event=workflow_dispatch&per_page=100","workflow_runs"),env["GITHUB_RUN_ID"])
    def receipt(run,name,attempt):
        label=f"exposure-{name}-{run}-{attempt}"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["gh","run","download",str(run),"--repo",repo,"--name",label,"--dir",tmp],check=True)
            return json.loads((Path(tmp)/"exposure.json").read_bytes())
    for prior in older:
        ensure_setup_only(prior,pages(api+f"runs/{prior['id']}/jobs?filter=all&per_page=100","jobs"),receipt)


def exposure(out,seed,seat):
    if seed not in SEEDS or seat not in (0,1):raise ValueError("Unknown validation map")
    if hashlib.sha256(HELPER.read_bytes()).hexdigest()!="061c4e2e42329dd2f424edddedcbe2e20c578d465c0041fcdef3319c62023794":
        raise ValueError("Different exposure audit implementation")
    spec=importlib.util.spec_from_file_location("held_exposure",HELPER)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ARMS=("sampled_parent","sampled_child")
    m.exposure(out,seed,seat)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","guard","exposure"))
    p.add_argument("--digest");p.add_argument("--seed",type=int);p.add_argument("--seat",type=int,choices=(0,1));p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.mode=="prepare":prepare(a.digest)
    elif a.mode=="guard":guard()
    else:exposure(a.out,a.seed,a.seat)

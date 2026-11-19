"""Materialize a frozen sparse tree and audit previous Actions exposure by physical map and seat."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

ROOT = Path(os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[8]))
HERE = Path(__file__).resolve().parent
CARGO = HERE.parents[3]
WORKFLOW = "sapling-natural-fresh-repeat.yml"
STEP = "Play registered whole-game pair"
ARMS = ("sampled_parent", "sampled_child")

def run(args):
    return subprocess.check_output(args, cwd=ROOT)

def prepare(digest):
    data = (HERE / "protocol.json").read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}",digest) or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Dispatch SHA differs from immutable protocol")
    plan = json.loads(data)
    def files(x):
        if isinstance(x,dict):
            if isinstance(x.get("path"),str) and isinstance(x.get("sha256"),str):
                yield x["path"],x["sha256"]
            else:
                for v in x.values():yield from files(v)
        elif isinstance(x,list):
            for v in x:yield from files(v)
    items=list(files(plan["files"]))
    if not items:raise ValueError("No registered provenance inputs")
    sampler = (CARGO/"onpolicy_native/training_policy.py").relative_to(ROOT).as_posix()
    paths=set()
    refs={}
    archives = {plan["files"][f"previous_archive_{i}"]["path"] for i in range(4)}
    allowed = (HERE.parent/"hosted").relative_to(ROOT).as_posix()
    if len(archives)!=4 or not all(re.fullmatch(re.escape(allowed)+r"/[0-9]+\.zip",p) for p in archives):
        raise ValueError("Missing or unexpected historic evidence archive")
    total=sum(int(run(["git","cat-file","-s","HEAD:"+p])) for p in archives)
    if total>2_000_000:raise ValueError("Unexpected historical archive size")
    for p,h in items:
        if PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts or not re.fullmatch(r"[A-Za-z0-9_./-]+",p):
            raise ValueError(f"Unexpected nonrepository path {p}")
        if not re.fullmatch("[0-9a-f]{64}", h) or refs.setdefault(p,h)!=h:
            raise ValueError(f"Conflicting digest for {p}")
        if p.lower().endswith(".zip") and p not in archives:
            raise ValueError(f"Historic zip is outside the fresh-match checkout: {p}")
        if p!=sampler:paths.add("/"+p)
    if sampler in refs:
        paths.update("/"+(CARGO/q).relative_to(ROOT).as_posix() for q in ("onpolicy_native/build.py","onpolicy_native/sampler.py","candidate.py"))
    subprocess.run(["git","sparse-checkout","add","--stdin"],input=("\n".join(sorted(paths))+"\n").encode(),cwd=ROOT,check=True)
    if sampler in refs:subprocess.run([sys.executable,str(CARGO/"onpolicy_native/build.py")],check=True)
    for p,h in refs.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError(f"Wrong registered dependency bytes: {p}")

def pages(api,key):
    data=json.loads(run(["gh","api","--paginate","--slurp",api]))
    if not isinstance(data,list) or not data or any(not isinstance(p.get(key),list) for p in data):
        raise ValueError("No reliable Actions inventory to establish zero prior exposure")
    return [r for p in data for r in p[key]]

def prefix(seed,seat):return f"fresh-natural-{int(seed)}-{int(seat)}"

def guard(seed,seat):
    env=os.environ
    repo=env["GITHUB_REPOSITORY"]
    api=f"/repos/{repo}/actions/"
    job=prefix(seed,seat)
    for r in pages(api+f"workflows/{WORKFLOW}/runs?branch=main&event=workflow_dispatch&per_page=100","workflow_runs"):
        if str(r["id"])==env["GITHUB_RUN_ID"]:continue
        for j in pages(api+f"runs/{r['id']}/jobs?filter=all&per_page=100","jobs"):
            if j["name"]!=job:continue
            steps=j.get("steps")
            if not isinstance(steps,list):raise ValueError("Prior ticket has no inspectable steps")
            exposed=any(s["name"]==STEP and s.get("started_at") and s.get("conclusion")!="skipped" for s in steps)
            if not exposed:continue
            artifact=f"exposure-{job}-{r['id']}-{j['run_attempt']}"
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["gh","run","download",str(r["id"]),"--repo",repo,"--name",artifact,"--dir",tmp],check=True)
                x=json.loads((Path(tmp)/"exposure.json").read_bytes())
                if (x["seed"],x["seat"],x["run_id"],x["attempt"])!=(seed,seat,str(r["id"]),str(j["run_attempt"])):
                    raise ValueError("Unbound recovery receipt")
                if x.get("confirmed_no_exposure") is not True:raise ValueError(f"Prior game or unknown exposure on {seed}/{seat} in run {r['id']}")

def exposure(out,seed,seat):
    data={}
    for a in ARMS:
        p=out/a/"receipt.json"
        receipt=json.loads(p.read_bytes()) if p.is_file() else {}
        status=dict(attempt=(out/a/"attempt.json").exists(),initializations=receipt.get("strategy_initializations"),action_calls=receipt.get("action_calls"))
        data[a]=status
    permitted=all((not v["attempt"]) and type(v["initializations"]) is int and v["initializations"]==0 and v["action_calls"]==[0,0] for v in data.values())
    target=out/"exposure.json"
    target.write_text(json.dumps(dict(seed=seed,seat=seat,run_id=os.environ["GITHUB_RUN_ID"],attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        protocol_sha256=os.environ["DIGEST"],arms=data,confirmed_no_exposure=permitted),sort_keys=True)+"\n")

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("prepare","guard","exposure"))
    cli.add_argument("--digest");cli.add_argument("--seed",type=int);cli.add_argument("--seat",type=int,choices=(0,1));cli.add_argument("--out",type=Path)
    a=cli.parse_args()
    if a.mode=="prepare":prepare(a.digest)
    elif a.mode=="guard":guard(a.seed,a.seat)
    else:exposure(a.out,a.seed,a.seat)

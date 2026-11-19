"""Re-evaluate both policies after a defender gains reach of a threatened tree."""
from pathlib import Path
import fcntl
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"capture_pressure_branches"))
import compare as runner

sr = runner.sr
runner.HERE = HERE
runner.PRIOR = HERE.parent/"planter_capture_full_dev"

def prepare():
    r = sr.load(HERE.parent/"tactical_exchange_dev/selected.json")
    path = sr.ROOT/r["source"]
    raw = sr.load(path)
    receipt = sr.load(path.with_name(path.name.replace("raw_","receipt_").replace(".json.gz",".json")))
    assert sr.sha(path) == r["source_sha256"] == receipt["game"]["raw_sha256"]
    assert raw["pack_sha256"] == sr.sha(runner.PRIOR/"candidate.py")
    case = r["choice"]
    assert raw["actions"][r["t"]][r["seat"]][case["uid"]] == case["before"]
    out = HERE/"seat0"
    out.mkdir(exist_ok=True)
    sr.retain(out/"official_replay.json",{"runs":{out.name:{"games":[receipt["official"]]}}})
    sr.retain(out/"live_protocol.json",dict(seed=r["seed"],seat=r["seat"],root_t=r["t"],
             family="tree_screen",source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=raw["pack_sha256"],
             record=dict(seed=r["seed"],seat=r["seat"],source=r["source"],source_sha256=sr.sha(path)),
             root=dict(t=r["t"],worker=case["uid"],target=case["enemy"],tree=case["tree"],
                       old=case["before"],alternative=case["after"])))
    sr.live.prepare(out)

if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare()
    elif len(sys.argv)>1:runner.play(0,sys.argv[1])
    else:
        with (runner.TREE/".continuation.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            tasks=[subprocess.Popen(["taskset","-c",str(i),sys.executable,__file__,a])
                   for i,a in enumerate(("parent","approach"))]
            codes=[job.wait() for job in tasks]
            if any(codes): raise RuntimeError(codes)

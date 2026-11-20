"""One fresh process per pack and backend; request timing includes JSON serialization and parsing."""
import argparse
import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
REPLAY=D/"ladder/replays/798a279b8c4cc1de4d5ab84de34d61ce-1.json.gz"


def worker(path, no_torch):
    if no_torch:
        # Mirror a baseline image with no optional Torch installation. The parent
        # catches ImportError and continues with its NumPy policy when available.
        sys.modules["torch"]=None
    import types
    tape=json.loads(gzip.decompress(REPLAY.read_bytes()))
    module=types.ModuleType("cold_player")
    start=time.perf_counter()
    src=path.read_bytes()
    exec(compile(src,"entry.py","exec"),module.__dict__)
    cold=time.perf_counter()-start
    shared=tape["steps"][0][0]["observation"]
    request={"observation":{k:v for k,v in shared.items() if k!="full_carbon"},
             "configuration":{k:v for k,v in tape["configuration"].items() if k!="randomSeed"}}
    start=time.perf_counter()
    request=json.loads(json.dumps(request))
    issued=json.loads(json.dumps(module.agent(**request)))
    return dict(package=path.name,sha256=hashlib.sha256(src).hexdigest(),search_available=bool(module._SEARCH_OK),
                torch_masked=no_torch,cold_s=cold,step0_s=time.perf_counter()-start,issued=issued,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path);parser.add_argument("--no-torch",action="store_true")
    parser.add_argument("--cpu",type=int,default=4)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[v]="1"
    if args.worker:
        print(json.dumps(worker(args.worker,args.no_torch)));return
    rows=[]
    for pack in (D/"cold_start/candidate.py", HERE/"candidate.py"):
        for no_torch in (False,True):
            cmd=[sys.executable,__file__,"--worker",str(pack),"--cpu",str(args.cpu)]
            if no_torch:cmd.append("--no-torch")
            text=subprocess.check_output(cmd,text=True)
            rows.append(json.loads(text.strip().splitlines()[-1]))
    (HERE/"cold_receipt.json").write_text(json.dumps({"scope":"Local isolated processes. Affinity only. No adjudication or cgroup limits.","tests":rows},indent=2)+"\n")


if __name__=="__main__":main()

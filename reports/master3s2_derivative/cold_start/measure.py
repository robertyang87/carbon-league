"""Alternating cold child comparisons with full public IPC from parent to parsed response."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PATHS={"v1":HERE.parent/"opponent_policy/candidate.py","v2":HERE/"candidate.py"}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(tag,mode,fixture,trial,cpu):
    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","CARBON_DEBUG":"1"}
    started=time.perf_counter()
    proc=subprocess.Popen(["taskset","-c",str(cpu),sys.executable,str(HERE/"worker.py"),str(PATHS[tag]),mode],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    try:
        def read():
            line=proc.stdout.readline()
            if not line:raise RuntimeError(proc.stderr.read()[-3000:])
            return json.loads(line)
        def send(d):
            t=time.perf_counter()
            proc.stdin.write(json.dumps(d,separators=(",",":"))+"\n");proc.stdin.flush()
            reply=read()
            return reply,time.perf_counter()-t
        ready=read()
        init=time.perf_counter()-started
        assert ready["source"]==sha(PATHS[tag])
        hist=fixture["history"]
        stages=list(range(0,12)) if trial%2 else [0]
        stages.append(len(hist)-1)
        turns=[]
        for t in stages:
            obs=hist[t]["observation"]
            if t==stages[-1]:
                send(dict(op="restore",seat=fixture["seat"],history=hist[:t],sample_seed=fixture["seed"]^2761))
            res,wall=send(dict(op="act",observation=obs,configuration=fixture["config"],sample_seed=fixture["seed"]^2761))
            assert isinstance(res["action"],dict) and res["step"]==t
            res["external_s"]=wall;turns.append(res)
            send(dict(op="correct",players=obs["players"],wire=hist[t]["wire"]))
        proc.stdin.write('{"op":"quit"}\n');proc.stdin.flush();proc.stdin.close()
        assert proc.wait(timeout=10)==0
        return dict(version=tag,mode=mode,fixture=fixture["terrain"],seat=fixture["seat"],trial=trial,
                    cold_process_to_parsed_ready_s=init,ready=ready,turns=turns)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()

def main(cpu):
    fixtures=json.loads(gzip.decompress((HERE/"requests.json.gz").read_bytes()))
    out=HERE/"startup.json.gz"
    if out.exists():raise ValueError("Cannot overwrite frozen measurements")
    records=[]
    for j in range(6):
        mode="plain" if j<4 else "profile"
        choices=("v1","v2") if j%2==0 else ("v2","v1")
        for tag in choices:
            rec=run(tag,mode,fixtures[j%2],j,cpu)
            records.append(rec)
            print(j,tag,mode,round(rec["cold_process_to_parsed_ready_s"],3),max(round(x["external_s"],3) for x in rec["turns"]),flush=True)
    data=dict(cpu=cpu,v1=sha(PATHS["v1"]),v2=sha(PATHS["v2"]),worker=sha(HERE/"worker.py"),
              runner=sha(Path(__file__)),requests=sha(HERE/"requests.json.gz"),rows=records,
              quota="Local diagnostic. cgroup readback from each process, CPU affinity identical, parent package/fs cache can be warm. Fixed per-pair random samples; returns use true 0.58 adaptive wall. Full-request elapsed includes serialization and parsing; at midgame the played prefix was restored outside the measured agent request.")
    out.write_bytes(gzip.compress((json.dumps(data,sort_keys=True,allow_nan=False,indent=2)+"\n").encode(),mtime=0))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpu",type=int,required=True);main(ap.parse_args().cpu)

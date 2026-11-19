"""Resume one bounded experiment after the eight pre-existing games and the shared native court are free."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from register import HERE,read,seal,sha

def update(step,**fields):
    path=HERE/"status.json";new=path.with_suffix(".partial")
    new.write_text(json.dumps({"state":step,"at_unix":time.time(),"supervisor_pid":os.getpid(),**fields})+"\n")
    new.replace(path)

def execute():
    state=read(HERE/"mechanism.json.gz")
    assert state["roots_sha256"]==sha(HERE/"roots.json") and state["gate"]=="PASS"
    # Timing telemetry was requested before the first native arm. Reject a stale data collector.
    assert (HERE/"responses/timing_fixture.json").is_file(),"opaque worker must supply timed-fixture receipt before native use"
    for split in ("train","held"):
        if split=="held" and read(HERE/"train_gate.json")["decision"]!="PASS":
            update("stopped_at_training_gate",formal_games=0);return 0
        vals=HERE/"responses"/(split+"_labels.json")
        if not vals.is_file():
            while True:
                update("waiting_or_running_"+split,active_budget_seconds=2100)
                rc=subprocess.run([sys.executable,str(HERE/"responses/native.py"),"run-approved",split]).returncode
                if rc==75:time.sleep(30);continue
                if rc==3:update("stopped_at_active_budget",stage=split);return 3
                if rc:raise RuntimeError(f"{split} native failed with exit {rc}")
                if not vals.is_file():raise ValueError("native completed without full label set")
                break
        subprocess.run([sys.executable,str(HERE/"fit_gate.py"),split],check=True)
    update("completed_"+read(HERE/"held_gate.json")["decision"],formal_games=0);return 0

def main():
    with (HERE/".supervisor.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        try:return execute()
        except Exception as exc:
            update("error",detail=str(exc));raise
if __name__=="__main__":sys.exit(main())

"""Detach-safe single-owner orchestration; all persistent inputs and outputs live next to this file."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from core import HERE,read,save,sha
import budget

STATE=HERE/"state.json"
RECEIPT=HERE/"completion.json"
ETA=4*3600


def phase(state, name, **meta):
    state.update(phase=name,pid=os.getpid(),updated_unix=time.time(),**meta)
    save(STATE,state,False)
    print(json.dumps({"phase":name,**meta},ensure_ascii=False),flush=True)


def execute(cmd,cpu,lock,extra=None):
    env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CARBON_DEBUG":"1",
         "PYTHONDONTWRITEBYTECODE":"1",**(extra or {})}
    return subprocess.run(["taskset","-c",str(cpu),sys.executable,"-u",*cmd],cwd=HERE,env=env,pass_fds=(lock.fileno(),)).returncode


def command(cmd,lock):
    ret=execute(cmd,2,lock)
    if ret:raise RuntimeError(f"{cmd} exited {ret}")


def perform(stage,state,lock):
    path=HERE/"native"/f"{stage}_complete.json"
    while not path.exists():
        b=budget.remaining()
        if b<=2:
            phase(state,"active_budget_exhausted",remaining_active_seconds=b)
            return False
        if time.time()>state["started_unix"]+ETA:
            phase(state,"queue_deadline_reached",remaining_active_seconds=b)
            return False
        phase(state,f"{stage}_native",remaining_active_seconds=round(b,2))
        ret=execute(["native/run.py",stage],1,lock,
                    {"STRATIFIED_REMAINING_SECONDS":str(b),"STRATIFIED_PHASE":stage})
        if ret==75:
            phase(state,f"{stage}_queued_or_paused",remaining_active_seconds=round(budget.remaining(),2))
            time.sleep(12)
        elif ret!=0:
            raise RuntimeError(f"native/{stage} returned {ret}")
    command(["native/run.py","labels",stage],lock)
    return True


def finish(state,decision):
    result={"decision":decision,"protocol_sha256":sha(HERE/"protocol.json"),"cards_sha256":sha(HERE/"cards.json"),
            "artifacts":{str(p.relative_to(HERE)):sha(p) for p in (HERE/"weights.json",HERE/"fit_verdict.json",HERE/"package.json",
            HERE/"held_verdict.json", HERE/"native/train_labels.json",HERE/"native/held_labels.json",HERE/"native/time_ledger.json")
            if p.exists()}}
    save(RECEIPT,result)
    phase(state,"complete",decision=decision,remaining_active_seconds=round(budget.remaining(),2))


def run(lock):
    state=read(STATE) if STATE.exists() else {"started_unix":time.time()}
    if RECEIPT.exists():
        phase(state,"complete",decision=read(RECEIPT)["decision"])
        return
    phase(state,"checking_inputs")
    if not read(HERE/"support.json")["eligible"]:
        finish(state,"HOLD_INSUFFICIENT_OPPORTUNITIES")
        return
    command(["native/run.py","prepare"],lock)
    if not perform("train",state,lock):
        finish(state,"STOP_NATIVE_BUDGET_OR_QUEUE")
        return
    phase(state,"fit_and_qualify")
    command(["learn.py","fit"],lock)
    command(["build.py"],lock)
    command(["package_probe.py","--learned"],lock)
    if read(HERE/"fit_verdict.json")["decision"]!="PASS":
        finish(state,"HOLD_NO_CROSS_MAP_TRAINING_GAIN")
        return
    command(["learn.py","select"],lock)
    command(["held_deploy_validation.py"],lock)
    if not read(HERE/"held_deploy_validation.json")["pass"]:
        finish(state,"HOLD_ENTRYPOINT_DISAGREES")
        return
    if not perform("held",state,lock):
        finish(state,"STOP_NATIVE_BUDGET_OR_QUEUE")
        return
    command(["learn.py","held"],lock)
    if read(HERE/"held_verdict.json")["decision"]!="PASS":
        finish(state,"HOLD_HELD_ROOT_GATE")
        return
    phase(state,"fresh_swapped_seat_development")
    pilot_deadline=state.setdefault("pilot_started_unix",time.time())+7200
    while not (HERE/"source/pilot_results.json").exists():
        left=pilot_deadline-time.time()
        if left<=0:
            finish(state,"STOP_FULL_PILOT_BUDGET")
            return
        phase(state,"fresh_swapped_seat_development",pilot_wall_seconds_remaining=round(left,2))
        code=execute(["source/pilot.py","--run-if-passed","--max-seconds",str(max(1,int(left)))],2,lock)
        if code==75:
            time.sleep(12)
        elif code==2:
            finish(state,"STOP_FULL_PILOT_IDENTITY_OR_VERIFICATION")
            return
        elif code!=0:
            raise RuntimeError(f"pilot process {code}")
        elif not (HERE/"source/pilot_results.json").exists():
            raise ValueError("pilot exited without the complete registered board receipt")
    finish(state,"DEV_PILOT_RECORDED")


def main():
    with (HERE/".job.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("An established supervisor or its worker already owns the queue.")
            return
        try:run(lock)
        except Exception as exc:
            state=read(STATE) if STATE.exists() else {"started_unix":time.time()}
            phase(state,"error_recoverable",error=f"{type(exc).__name__}: {exc}")
            raise

if __name__=="__main__":main()

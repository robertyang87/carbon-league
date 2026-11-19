"""Active-wall ledger. Call reserve only while holding the real, nonblocking native arena lock."""
import contextlib
import os
import time
from pathlib import Path
from core import HERE,read,save

DEFAULT=HERE/"native"/"time_ledger.json"


def snapshot(path,limit):
    d=read(path) if path.exists() else {"limit":float(limit),"charged":0.,"active":None,"history":[]}
    if d["limit"]!=limit:
        raise ValueError("budget changed across stages or restart")
    return d


def remaining(limit=2100.,path=DEFAULT,clock=time.time):
    d=snapshot(path,limit)
    opened=d["active"]
    extra=max(0.,clock()-opened["start"]) if opened else 0.
    return max(0.,limit-d["charged"]-extra)


@contextlib.contextmanager
def reserve(limit=2100.,path=DEFAULT,clock=time.time):
    # Only the real arena owner can enter; this makes any preceding open lease an interrupted one.
    # An external SIGKILL leaves a durable open interval. Conservatively charge elapsed time
    # through recovery, capped by the unused budget. Thus a stale process can never extend the quota.
    d=snapshot(path,limit)
    start=clock()
    if d["active"]:
        old=d["active"]
        expense=min(limit-d["charged"],max(0.,start-old["start"]))
        d["charged"]+=expense
        d["history"].append({**old,"end":start,"seconds":expense,"closure":"recovered_conservatively"})
    amount=max(0.,limit-d["charged"])
    d["active"]={"start":start,"pid":os.getpid(),"phase":os.environ.get("STRATIFIED_PHASE","test")}
    save(path,d,immutable=False)
    try:
        yield amount
    finally:
        end=clock()
        cost=min(amount,max(0.,end-start))
        d["charged"]+=cost
        d["history"].append({**d["active"],"end":end,"seconds":cost,"closure":"closed"})
        d["active"]=None
        save(path,d,immutable=False)

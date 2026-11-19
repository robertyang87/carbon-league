"""Durable active wall shared across all defense phases, entered only under the real arena lock."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

PATH=Path(__file__).resolve().parent/"time_ledger.json"

def _read(limit):
    if not 0<limit<=2100:raise ValueError("defense requires an active allowance at most 2100 s")
    data=json.loads(PATH.read_bytes()) if PATH.exists() else {"limit":limit,"charged":0.,"active":None,"intervals":[]}
    if data["limit"]!=limit:raise ValueError("registered native allowance changed")
    return data

def _publish(d):
    tmp=PATH.with_name(f".{PATH.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(d,allow_nan=False,separators=(",",":"))+"\n");tmp.replace(PATH)

def remaining(limit=2100):
    d=_read(limit);opened=d["active"]
    extra=max(0,time.time()-opened["opened"]) if opened else 0
    return max(0,limit-d["charged"]-extra)

@contextmanager
def reserve(limit=2100):
    d=_read(limit)
    def close(now,why):
        x=d["active"];cost=min(limit-d["charged"],max(0,now-x["opened"]))
        d["charged"]+=cost;d["intervals"].append({**x,"end":now,"charged":cost,"closed":why});d["active"]=None
    start=time.time()
    if d["active"]:close(start,"previous owner interrupted; charged until recovery")
    d["active"]={"opened":start,"pid":os.getpid(),"phase":os.getenv("STRATIFIED_PHASE",os.getenv("CARGO_PHASE","native"))}
    _publish(d)
    try:yield max(0,limit-d["charged"])
    finally:close(time.time(),"completed interval");_publish(d)

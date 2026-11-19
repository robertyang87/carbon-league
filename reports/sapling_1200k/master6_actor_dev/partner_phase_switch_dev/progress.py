"""Read-only status. Pair a margin only after both authentic arms exist; count partial work separately."""
import gzip,json,sys,time
from pathlib import Path
from collections import Counter
from register import HERE,read

def main():
    registry=read(HERE/"roots.json")["roots"];counts=Counter();pairs=[];jobs=[]
    for r in registry:
        streams=("primary","repeat") if r["repeat"] else ("primary",)
        for stream in streams:
            parent=HERE/"responses"/"cases"/stream/r["id"]
            dual={}
            for arm in ("it50","it20"):
                f=parent/(arm+".json.gz")
                if f.exists():
                    item=read(f)
                    assert item["binding"]["id"]==r["id"] and item["statuses"]==["DONE","DONE"]
                    dual[arm]=item["official_margin"];counts["signed_arms"]+=1;counts["signed_steps"]+=len(item["history"])
                elif (part:=parent/(arm+".partial.json.gz")).exists():
                    try: item=read(part)
                    except FileNotFoundError:continue
                    jobs.append({"id":r["id"],"arm":arm,"stream":stream,"past_root_original_calls":len(item["history"]),"start":r["t"]})
            if len(dual)==2:pairs.append({"id":r["id"],"stream":stream,"split":r["split"],"regime":r["regime"],"delta":dual["it20"]-dual["it50"]})
    meter=HERE/"responses/active_time.json"
    amount=0.
    if meter.exists():
        state=read(meter);amount=state["charged"]
        if state["active"]: amount+=max(0.,time.time()-state["active"]["start"])
    print(json.dumps({"state":read(HERE/"status.json")["state"],"charged_active_seconds":round(amount,2),"signed":dict(counts),
                      "pairs":pairs,"partial":jobs},separators=(",",":")))
if __name__=="__main__":main()

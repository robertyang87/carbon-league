"""Summarize immutable finished arms and pin the public timing receipt without live process IDs."""
from collections import Counter
import gzip
import json
from statistics import mean,median
from core import HERE,read,sha,save

def main():
    labels=read(HERE/"native/train_labels.json")
    rows=[r for r in labels["cases"] if not r["diagnostic"]]
    times=[]
    for r in labels["cases"]:
        for arm in ("base","alt"):
            z=read(HERE/"native"/r["id"]/(arm+".json.gz"))
            assert z["statuses"]==["DONE","DONE"] and len(z["history"])==z["completed_original_calls"]
            times.extend(h["wall_seconds"] for h in z["history"])
    costs=read(HERE/"native/time_ledger.json")
    assert costs["active"] is None and costs["charged"]==sum(h["seconds"] for h in costs["history"])
    out={"verified_train_labels_sha256":sha(HERE/"native/train_labels.json"),"primary_pairs":len(rows),
         "prespecified_second_stream_pairs":len(labels["cases"])-len(rows),"full_original_branches":len(labels["cases"])*2,
         "original_followup_calls":len(times),"original_followup_wall_seconds":sum(times),"mean":mean(times),"median":median(times),
         "total_locked_pair_wall_seconds":costs["charged"],"total_budget":costs["limit"],
         "no_pooled_replication":True,"training_primary_by_kind":{}}
    for kind in ("mine_hold","bank_route","evade","ore_route"):
        cases=[r for r in rows if r["proposal"]["kind"]==kind]
        out["training_primary_by_kind"][kind]={"n":len(cases),"positive_margin":sum(c["delta"]["margin"]>0 for c in cases),
              "means":{k:mean(c["delta"][k] for c in cases) for k in ("margin","principal","own_delivery","enemy_delivery")}}
    save(HERE/"verified_costs_and_returns.json",out)
    print(out["original_followup_calls"],out["total_locked_pair_wall_seconds"])

if __name__=="__main__":main()

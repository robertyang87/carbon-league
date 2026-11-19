"""Aggregate genuine original search meters without treating paired random streams as independent maps."""
from collections import Counter,defaultdict
import math
from register import HERE,sha,seal,read

def summarize(cards):
    wall=[];cpu=[];depth=Counter();rolls=Counter();cands=Counter();counts=Counter();reasons=Counter()
    for row in cards:
        for name in ("searched","aborted","single","changed"):counts[name]+=row[name]
        if row["depth"] is None:reasons[row.get("null_reason") or "no_depth"]+=1
        else:depth[str(row["depth"])]+=1
        if row["rolls"] is not None:rolls[str(row["rolls"])]+=1
        if row["cands"] is not None:cands[str(row["cands"])]+=1
        wall.append(row["wall_time_ns"]/1e9);cpu.append(row["process_time_ns"]/1e9)
    def stats(v):
        a=sorted(v)
        return {"sum_seconds":sum(v),"mean_seconds":sum(v)/len(v),"p95_observed_seconds":a[math.ceil(.95*len(a))-1],"max_seconds":a[-1]}
    return {"original_followup_steps":len(wall),"wall":stats(wall),"process_cpu":stats(cpu),
            "depth":dict(depth),"rolls":dict(rolls),"candidate_counts":dict(cands),"counters":dict(counts),"no_depth":dict(reasons)}
def main(stage):
    file=HERE/"responses"/(stage+"_labels.json");labels=read(file);everything=defaultdict(list);each=[]
    for rec in labels["pairs"]:
        for arm in ("it50","it20"):
            meta=rec[arm];p=HERE/"responses"/meta["raw"]
            assert sha(p)==meta["sha256"]
            d=read(p);timed=[r["original_search_meter"] for r in d["history"]]
            assert len(timed)==meta["original_followup_calls"] and all(row["wall_time_ns"]>0 and row["process_time_ns"]>0 for row in timed)
            each.append({"id":rec["id"],"stream":rec["stream"],"arm":arm,**summarize(timed)})
            everything[arm].extend(timed)
    seal(HERE/(stage+"_timing.json"),{"label_sha256":sha(file),"joint_original_root_is_archived_not_counted":True,
          "scope":"Both arms ran independently with the same cpu affinity, BLAS and original agent clock. Process and wall time run just outside master.agent. Warm state initialization, actor computation, checkpoint writes, restore work and lock waits are separate; the study budget also includes the former while holding the court.",
          "by_arm":{a:summarize(v) for a,v in everything.items()},"by_case":each})
if __name__=="__main__":
    import sys
    main(sys.argv[1])

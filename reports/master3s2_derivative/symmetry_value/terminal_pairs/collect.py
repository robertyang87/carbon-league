"""Verify repeat attribution on every registered late root and run the first qualified pairs."""
import gzip
import json
import os
from branches import HERE, REGISTRATION, freight, run_pair, sha, token, write


def collect():
    os.environ["CARBON_DEBUG"]="1"
    freight.check(os.sched_getaffinity(0)=={2},"Use isolated logical CPU 2 for these paired suffixes")
    report_raw=(HERE.parent/"search_roots.json.gz").read_bytes()
    search=json.loads(gzip.decompress(report_raw))
    freight.check(search["protocol_sha256"]==REGISTRATION["search_protocol_sha256"] and
                  search["parent"]==REGISTRATION["base"] and
                  search["candidate"]=="69b6730dcc990404bae939b5978126c580a4731b4c0559b7e6c499f387e7597b","Search identity")
    lookup={(r["seed"],r["seat"],r["step"]):r for r in search["roots"]}
    eligible=REGISTRATION["eligibility"]
    freight.check(len(lookup)==32 and len(eligible)==24,"Different preregistered scope")
    outcomes=[];selected=[]
    for ticket in eligible:
        key=tuple(ticket[k] for k in ("seed","seat","step"))
        row=lookup[key]
        freight.check(all(ticket[k]==row[k] for k in ticket),"Root identity disagreement")
        a,b,c,d=[row["arms"][k] for k in ("P1","P2","C1","C2")]
        menu=all(z["menu"]==a["menu"] for z in (b,c,d))
        depth=all(all(z[k]==a[k] for k in ("depth_by_arm","depth","rollouts")) for z in (b,c,d))
        other=all(all(z[k]==a[k] for k in ("before_scores","act0","opponent_samples","before_order")) for z in (b,c,d))
        repeated=a["issued"]==b["issued"] and c["issued"]==d["issued"]
        committed=all(z["rescore"].get("stage")=="committed" and z["issued"]!=z["before_order"] for z in (c,d))
        changed=a["issued"]!=c["issued"]
        good=menu and depth and other and repeated and committed and changed
        freight.check(good==row["assessment"]["attributable_changed"],"Independent attribution disagreement")
        admitted=good and search["root_mechanism_pass"] and not search["admitted_over_058"] and len(selected)<4
        outcomes.append(dict(root=key,same_menu=menu,same_depth=depth,same_original_path_and_value=other,
                             repeated=repeated,both_committed_and_changed=committed,physical_difference=changed,admitted=admitted))
        if admitted:
            selected.append((ticket,dict(cost=row["starting_cost"],menu_sha256=token(a["menu"]),
                                         parent=a["issued"],rotated=c["issued"])))
    write("selection.json",dict(registration=sha((HERE/"protocol.json").read_bytes()),
                                source_sha256=sha(report_raw),both_comparators_global_pass=search["root_mechanism_pass"],
                                holdout=0,late_roots=outcomes,selected=[t for t,a in selected]))
    pairs=[run_pair(ticket,admission,idx) for idx,(ticket,admission) in enumerate(selected)]
    write("results.json",dict(late_work=len(outcomes),qualified=len(selected),pairs=pairs))


if __name__=="__main__":
    collect()

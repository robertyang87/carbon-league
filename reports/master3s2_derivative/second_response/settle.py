"""Recover second-pass admission from actual sampled opponent commands, then recompute policy choices."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent


def settle(raw):
    data=json.loads(gzip.decompress(raw))
    rows=[]
    for entry in data["roots"]:
        control, trial=(entry["arms"][str(k)] for k in (1,2))
        full=trial["rounds"]==2
        second="1" in trial.get("opponent_orders",{})
        first=trial.get("first_finalists",[])
        if full:
            assert second
            status="completed"
        elif second:
            assert not trial["second_values"]
            status="started_and_discarded"
        elif trial["depth"]==8 and len(first)>1:
            status="admission_denied"
        else:
            status="first_pass_ineligible"
        row=dict(seed=entry["seed"],seat=entry["seat"],step=entry["step"],status=status,
                 matched_first_pass=entry["first_stage_exact"],across_calls_changed=entry["across_calls_changed"],
                 returned_decision_changed_by_second=False)
        if full:
            default,challenger=trial["finalists"]
            assert default==0 and len(first)>1
            first_vals, second_vals=trial["first_values"],trial["second_values"]
            assert challenger==max((c for c in first if c),key=lambda c:first_vals[str(c)])
            a=first_vals[str(challenger)]-first_vals["0"]
            b=second_vals[str(challenger)]-second_vals["0"]
            expected=challenger if a+b>0 else default
            assert trial["menu"][expected]==trial["issued"]
            if row["matched_first_pass"]:
                assert trial["menu"][challenger if a>0 else default] == control["issued"]
            row.update(challenger=challenger,first_advantage=a,second_advantage=b,paired_mean_advantage=(a+b)/2,
                       opponent_root_changed=trial["opponent_orders"]["0"]!=trial["opponent_orders"]["1"],
                       returned_decision_changed_by_second=(a>0)!=((a+b)>0))
        rows.append(row)
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(),counts=Counter(r["status"] for r in rows),
                within_call_changed=sum(r["returned_decision_changed_by_second"] for r in rows),
                cross_call_changed=sum(r["across_calls_changed"] for r in rows),
                first_stage_exact=sum(r["matched_first_pass"] for r in rows),roots=rows)


if __name__=="__main__":
    result=settle((HERE/"roots.json.gz").read_bytes())
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

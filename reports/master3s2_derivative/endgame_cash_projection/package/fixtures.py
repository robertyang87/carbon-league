"""Check fallback and tie behavior on the deployed selector using its real archived simulated rosters."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
sys.path.insert(0,str(D/"opponent_policy"))
from original_policy import module,sha


def main():
    import torch
    torch.set_num_interop_threads(1)
    raw=(HERE/"candidate.py").read_bytes()
    model=module(raw);torch.set_num_threads(1)
    with gzip.open(D/"action_rank_data/train/621591808_0_279_root.json.gz","rt") as f:
        r=json.load(f)
    options=r["arms"];ids=[a["candidate_index"] for a in options]
    root_step=r["ticket"]["step"];seat=r["ticket"]["seat"]
    examples={}
    def fresh():
        b=[];means={}
        for a in options:
            cid=a["candidate_index"];leaf=a["leaves"][0]
            p=leaf["players"]
            b.append([cid,0,dict(players=copy.deepcopy(p),step=leaf["leaf_step"]),None,[copy.deepcopy(p) for _ in range(8)]])
            means[cid]=a["original_value"]
        return dict(incumbent=ids[0],alive=ids,mean=means,depth_by_id={i:8 for i in ids},
                    boards=b,orders=r["menu"],depth=8,rolls=1,cut=False,step=root_step,seat=seat,cfg={},start=time.perf_counter())
    def verify(name,reason,second=False,mutate=None,**patch):
        data=fresh();data.update(patch)
        if mutate:mutate(data)
        actual=model._cash_pick(**data)
        if actual[0]!=ids[int(second)] or actual[1]!=reason:
            raise RuntimeError(f"{name}: unexpected choice or reason {actual}")
        examples[name]=dict(returned=actual[0],why=actual[1])
    verify("positive_on_actual_rosters","compared",True)
    verify("real_h8_cash_tie_keeps_native","compared",mutate=lambda d:d["boards"][1][2].update(players=d["boards"][0][2]["players"]))
    verify("incomplete","not_applicable",cut=True)
    verify("only_h6","not_applicable",depth=6)
    verify("only_one","not_applicable",alive=ids[:1])
    verify("outside_coverage","not_applicable",step=280)
    verify("unregistered_multiroll","not_applicable",rolls=2)
    verify("different_rules","different_rules",cfg=dict(episodeSteps=200))
    verify("insufficient_time","time_before",start=time.perf_counter()-1)
    verify("missing_equal_depth","native_selection",depth_by_id={ids[0]:8,ids[1]:7})
    verify("native_return_not_highest","native_selection",mutate=lambda d:d["mean"].update({ids[0]:-100.}))
    verify("missing_roll","missing_scenario",mutate=lambda d:d["boards"].pop())
    verify("native_end_terminal","terminal_or_missing_history",mutate=lambda d:d["boards"][0][2].update(step=299))
    def earlier_death(d,both):
        p=d["boards"][0][4][0]
        for s in ((0,1) if both else (0,)):
            p[s]=[29,p[s][1],{},{}]
    verify("survived_later_after_true_extinction","official_extinction",mutate=lambda d:earlier_death(d,False))
    verify("double_extinction","official_extinction",mutate=lambda d:earlier_death(d,True))
    verify("malformed_features","fallback_error",mutate=lambda d:d["boards"][1][2].update(players=[]))
    original=model._cash_residual
    try:
        model._cash_residual=lambda x,model:float("nan")
        verify("nonfinite_falls_back","nonfinite")
    finally:
        model._cash_residual=original
    clock=model._time
    class SlowHead:
        def __init__(self):self.count=0
        def perf_counter(self):
            self.count+=1
            return .001 if self.count==1 else 1.
    try:
        model._time=SlowHead()
        verify("deadline_passed_while_comparing","time_after",start=0.)
    finally:
        model._time=clock
    receipt=dict(source_sha256=sha(raw),distinct_checks=len(examples),checks=examples)
    (HERE/"fixtures.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(len(examples),"checks")


if __name__=="__main__":
    main()

"""Verify a finished split by replaying every archived order to its official terminal score."""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
import sys

from collect import HERE, PLAN, PLAN_BYTES, freight, save_json, sha
from audit import replay


def verify(split):
    cards=json.loads((HERE.parent/"opponent_policy/protocol.json").read_bytes())["games"]
    folder=HERE/split
    counts=Counter()
    records=[]
    for ticket in PLAN["tickets"]:
        if ticket["split"]!=split:
            continue
        stem=f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}"
        def path(tail):
            return folder/f"{stem}_{tail}"
        pair=json.loads(path("pair.json").read_bytes())
        source=next(c for c in cards if c["sha256"]==ticket["source_sha256"])
        raw=path("root.json.gz").read_bytes()
        root=json.loads(gzip.decompress(raw))
        freight.check(pair["ticket"]==root["ticket"]==ticket and pair["protocol_sha256"]==sha(PLAN_BYTES)
                      and pair["root_sha256"]==sha(raw),"Root/label identities disagree")
        public=root["public_root"]
        freight.check(public["step"]==ticket["step"] and public["player"]==ticket["seat"]
                      and "full_carbon" not in public,"Wrong or private root")
        if split=="held":
            forecast=HERE.parent/"terminal_value_pairs"/f"held_forecast_{ticket['ordinal']}.json"
            freight.check(pair["blind_forecast_sha256"]==sha(forecast.read_bytes()),"Invalid blind-input seal")
        summary=dict(seed=ticket["seed"],seat=ticket["seat"],step=ticket["step"],depth=root["depth"],
                     total_wall=pair["total_wall"],total_cpu=pair["total_cpu"],preparation_wall=root["preparation_wall"],
                     status=pair["status"])
        counts["roots"]+=1
        if pair["status"]=="skipped":
            freight.check(pair["reason"]==root["reason"] and not path("first.json.gz").exists()
                          and not path("second.json.gz").exists(),"Excluded root contains a label")
            summary["reason"]=pair["reason"]
        else:
            freight.check(pair["status"]=="complete" and root["depth"]==8 and root["rolls"]==1
                          and root["arms"][0]["order"]!=root["arms"][1]["order"] and
                          pair["leaf_sha256"]==sha(path("leaf.npz").read_bytes()),"Unbound features or menus")
            all_evidence=[]
            for name in ("first","second"):
                arm=pair["arms"][name]
                evidence=replay(path(name+".json.gz"),source,ticket)
                freight.check(evidence["sha256"]==arm["evidence_sha256"] and evidence["official_scores"]==arm["scores"],
                              "Compact receipt disagrees with independent referee")
                counts[f"wins_{name}"]+=arm["win"]
                counts["joint_transitions"]+=evidence["transitions"]
                all_evidence.append(evidence)
            targets=[max(-2.5,min(2.5,r["official_margin"]/3000.))+.5*((r["official_margin"]>0)-(r["official_margin"]<0)) for r in all_evidence]
            delta=targets[0]-targets[1]
            freight.check(abs(delta-pair["target"])<1e-10,"Incorrect exact terminal return")
            summary.update(arms=all_evidence,target=delta,
                           first_minus_second_official_margin=all_evidence[0]["official_margin"]-all_evidence[1]["official_margin"])
            counts["complete_pairs"]+=1
            counts["first_margin_higher"]+=summary["first_minus_second_official_margin"]>0
            counts["equal_official_margins"]+=summary["first_minus_second_official_margin"]==0
        records.append(summary)
    counts["recorded_seat_steps"]=2*counts["joint_transitions"]
    counts["unique_real_suffix_public_calls"]=counts["recorded_seat_steps"]-2*counts["complete_pairs"]
    counts["additional_failed_root_calls"]=counts["roots"]-counts["complete_pairs"]
    counts["warm_prefix_calls"]=6*counts["roots"]
    return dict(protocol_sha256=sha(PLAN_BYTES),summary=dict(counts),tickets=records,
                total_cpu=sum(r["total_cpu"] for r in records),total_wall=sum(r["total_wall"] for r in records))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("split",choices=("train","held"))
    split=p.parse_args().split
    result=verify(split)
    print(save_json(HERE/(split+"_replay_check.json"),result),result["summary"])

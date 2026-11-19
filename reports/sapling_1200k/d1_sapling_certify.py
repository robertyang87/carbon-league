"""Replay official rewards and independently check D1's actual selected prefixes with the two original policies."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

from d1_sapling_build import PARENT
from official_game_replay import audit
from tree_audited_game import exam, prior
from tree_win_gate import validate_game, verify_raw
from turning_point_ledger import compact, sha, write


def certify(folder):
    source = json.loads((folder/"results.json").read_bytes())
    official = audit(folder)
    module = prior.shared.module(PARENT, "d1_reference_hybrid")
    report = []
    for row in source["games"]:
        validate_game(row)
        verify_raw(folder, row, source["header"]["pack_sha256"], source["header"]["weight_sha256"])
        raw=json.loads(gzip.decompress((folder/row["raw_file"]).read_bytes()))
        ident=row["candidate_search_audit"]
        payload=(folder/ident["file"]).read_bytes()
        if sha(payload)!=ident["sha256"]:
            raise ValueError("search audit checksum")
        trace=json.loads(gzip.decompress(payload))
        seed, seat=row["seed"],row["candidate_seat"]
        if (trace["raw_sha256"]!=row["raw_sha256"] or trace["pack_sha256"]!=source["header"]["pack_sha256"] or
                (trace["seed"],trace["seat"])!=(seed,seat) or len(trace["calls"])!=len(raw["actions"])):
            raise ValueError("search audit belongs to another trajectory")
        sim=exam.FastCarbon(raw["game_configuration"]).reset(seed=seed)
        changes=contributions=complete=0
        checked_slots=0
        for t,(commands, live) in enumerate(zip(raw["actions"],trace["calls"])):
            observed=sim.obs(seat)
            baseline=module.agent(observed,sim.config)
            if t==0:
                shadow=module.sys.modules["cnet2_feat"].Feat(seat)
            x=shadow.observe(observed)
            played=commands[seat]
            if live["root"]!=t or sha(compact(played))!=live["executed_sha256"] or live["base"]!=baseline:
                raise ValueError(f"recorded A0/played action mismatch {seed}/{seat}/{t}")
            if (played!=baseline)!=live["search_changed"]:
                raise ValueError("reported search change is wrong")
            labels=shadow.labels(played)
            base=shadow.labels(baseline)
            active=np.flatnonzero(x["tok_mask"][:len(labels)])
            modified=[slot for slot in active if labels[slot]!=base[slot]]
            first=modified[0] if modified else -1
            expert,student=[n.act(x,shadow.aux,teacher=2,forced=labels)[1] for n in module._NETS]
            mixed_choices=np.argmax(0.75*expert+0.25*student,axis=1)
            expert_choices=expert.argmax(axis=1)
            actual_student=[]
            for slot in active:
                if slot==first:
                    continue
                if labels[slot]!=mixed_choices[slot]:
                    raise ValueError(f"unforced slot differs from independent joint policy {seed}/{seat}/{t}/{slot}")
                checked_slots+=1
                if slot>first and labels[slot]!=expert_choices[slot]:
                    actual_student.append([shadow.own_uid[slot],int(labels[slot]),int(expert_choices[slot])])
            if live["unforced_sapling_decisions"]!=actual_student:
                raise ValueError(f"live student attribution differs from the two independent policies {seed}/{seat}/{t}")
            changes+=bool(modified)
            complete+=live["completed_h1"]
            contributions+=len(actual_student)
            sim.step(*commands)
        if (changes!=ident["search_changes"] or complete!=ident["completed_h1"] or contributions!=ident["unforced_sapling_decisions"]
                or checked_slots<=0 or contributions<=0):
            raise ValueError("end-of-game attribution mismatch")
        report.append(dict(seed=seed,seat=seat,result=row["result"],margin=round(row["margin"],2),
                           transitions=len(trace["calls"]), independently_checked_unforced_slots=checked_slots,
                           original_it50_argmax_changed_by_student=contributions, actual_search_changes=changes,
                           completed_h1=complete, master_depth_histogram=row["search"]["depth"],
                           candidate_call_seconds=row["call_seconds_by_seat"][seat]))
    return dict(official_replay=official,actual_prefix_and_source_checks=report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    write(a.out,certify(a.folder))

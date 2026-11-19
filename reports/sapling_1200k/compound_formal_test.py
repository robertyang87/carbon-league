"""Exercise compound admission and score boundaries on recorded development actions; draw no final maps."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile

from compound_formal import ACTORS, D1, PARENT, REPORT, ROOT, STUDENT, verify_actions
from income_target import FORMAL_METRIC, TARGET
from tree_win_gate import RESERVED_RANGE, digest, validate_game, validate_examiner_configuration, prove_candidate, prove_single_pack, verify_raw, verdict
from turning_point_ledger import compact, write


def tests():
    observed = {}
    def rejected(key, func):
        try:
            func()
        except FileNotFoundError as exc:
            observed[key] = f"missing file: {Path(exc.filename).name}"
        except (ValueError, KeyError) as exc:
            observed[key] = str(exc)
        else:
            raise AssertionError(key + " should reject")
    from tree_audited_game import exam, prior
    source = prior.shared.module(prior.SOURCE, "unmodified_master_config_test")
    cfg = {k:getattr(source,k) for k in exam.SEARCH_KEYS}
    validate_examiner_configuration(cfg)
    observed["source_config"] = {"source_md5": prior.digest(prior.SOURCE, "md5"), "fields":len(cfg)}
    a, r = ROOT / ACTORS, ROOT / REPORT
    weights = a/"cash_v3.npz"
    observed["builds"] = {key:prove_candidate(weights,a/key)[1] for key in (
        "cash_v3_agent.py","hybrid_it50_v3.py","d1_sapling.py")}
    observed["builds"]["soft_it50_v5_guard_agent.py"]=prove_candidate(
        a/"soft_it50_v5.npz",a/"soft_it50_v5_guard_agent.py")[1]
    inventory=r/"latest5_inventory.json"
    inventory_data=json.loads(inventory.read_bytes())
    seeds=sorted({g["seed"] for g in inventory_data["games"]})
    if len(inventory_data["games"])!=26 or len(seeds)!=26 or any(type(s) is not int for s in seeds):
        raise ValueError("the frozen 26-map ladder fixture differs")
    overlaps=[s for s in seeds if RESERVED_RANGE[0]<=s<RESERVED_RANGE[1]]
    if overlaps:
        raise ValueError("previously studied map found in the reserved range: "+str(overlaps))
    observed["ladder_26_map_isolation"]=dict(inventory_sha256=digest(inventory),
        source="reports/sapling_1200k/latest5_inventory.json",used_distinct_seeds=seeds,
        reserved_range=list(RESERVED_RANGE),overlap=overlaps)
    rejected("original_single_actor_function_rejects_hybrid",lambda:prove_single_pack(weights,a/"hybrid_it50_v3.py"))
    rejected("reject_self_declared_manifest",lambda:prove_candidate(r/"d1_sapling_dev/build_manifest.json",a/"d1_sapling.py"))
    rejected("reject_other_real_npz",lambda:prove_candidate(a/"soft_it50_v5.npz",a/"d1_sapling.py"))
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        single_dir=r/"tree_exam_soft_it50_v5"
        single=json.loads((single_dir/"results.json").read_bytes())
        sr=copy.deepcopy(single["games"][0])
        srh=single["header"]
        for tag,fields in (("empty_depth",dict(depth={})), ("impossible_depth",dict(depth={"99":sr["search"]["searched"]})),
                           ("missing_times",dict(time=[]))):
            broken=dict(sr,search=dict(sr["search"],**fields))
            rejected("reject_examiner_"+tag,lambda:validate_game(broken))
        srraw=json.loads(gzip.decompress((single_dir/sr["raw_file"]).read_bytes()))
        for name,changed in (("budget",dict(srraw["configuration"],_S_ABORT=0.005)),
                             ("temperature",dict(srraw["configuration"],_TEMP=0.0)),
                             ("missing_field",{k:v for k,v in srraw["configuration"].items() if k!="_S_STAGES"})):
            altered=dict(srraw,configuration=changed)
            fn=folder/sr["raw_file"];fn.write_bytes(gzip.compress(compact(altered),mtime=0));sr["raw_sha256"]=digest(fn)
            rejected("reject_changed_examiner_"+name,
                     lambda:verify_raw(folder,sr,srh["pack_sha256"],srh["weight_sha256"],formal=True))
        for tag,issues in (("present",["unclassified search call"]),("missing",None)):
            altered=dict(srraw,audit_messages=issues)
            if issues is None: del altered["audit_messages"]
            fn.write_bytes(gzip.compress(compact(altered),mtime=0));sr["raw_sha256"]=digest(fn)
            rejected("reject_raw_observer_warning_"+tag,
                     lambda:verify_raw(folder,sr,srh["pack_sha256"],srh["weight_sha256"],formal=True))
        bad_pack=folder/"changed.py"
        bad_pack.write_bytes((a/"hybrid_it50_v3.py").read_bytes()+b"\n")
        rejected("reject_unknown_compound_bytes",lambda:prove_candidate(weights,bad_pack))
        observed["real_actions_in_formal_envelope"]={}
        for arm,pack in (("delivery_parent_dev",PARENT),("d1_sapling_dev",D1)):
            archive=r/arm
            record=json.loads((archive/"results.json").read_bytes())
            # The genuine dev trace used a manifest digest. Only the identity envelope in this isolated test is
            # rewritten to the verified training NPZ, then both linked checksums are updated. No action changes.
            game=copy.deepcopy(record["games"][1])
            raw=json.loads(gzip.decompress((archive/game["raw_file"]).read_bytes()))
            assert raw["weight_identity"]==record["header"]["weight_sha256"] and raw["pack_sha256"]==pack
            raw["weight_identity"]=STUDENT
            tape=folder/game["raw_file"]
            tape.write_bytes(gzip.compress(compact(raw),mtime=0))
            game["raw_sha256"]=digest(tape)
            if pack==D1:
                key=game["candidate_search_audit"]
                trace=json.loads(gzip.decompress((archive/key["file"]).read_bytes()))
                trace["raw_sha256"]=game["raw_sha256"]
                path=folder/key["file"]
                path.write_bytes(gzip.compress(compact(trace),mtime=0))
                key["sha256"]=digest(path)
            rejected(arm+"_reject_other_weight_identity",
                     lambda:verify_raw(folder,game,pack,record["header"]["weight_sha256"],formal=True))
            account=verify_raw(folder,game,pack,STUDENT,formal=True,income=True,record_compound=True)
            expected=game["compound_policy_audit"]
            assert expected["student_changes"]>0 and expected["verified_roots"]==299 and account["official_margin"]==game["margin"]
            observed["real_actions_in_formal_envelope"][arm]=expected
            if pack==D1:
                assert expected["d1_changes"]>0 and expected["d1_rounds"]>0
                # Bind the tampered evidence to a new SHA: the content check must still reject.
                altered=copy.deepcopy(trace)
                t=next(i for i,ev in enumerate(altered["calls"]) if ev["search_changed"])
                altered["calls"][t]["completed_h1"]=0
                path.write_bytes(gzip.compress(compact(altered),mtime=0));key["sha256"]=digest(path)
                rejected("reject_changed_on_unfinished_D1_round",lambda:verify_actions(folder,game,raw,D1))
                path.unlink()
                rejected("reject_missing_live_D1_trace",lambda:verify_actions(folder,game,raw,D1))
            else:
                altered=copy.deepcopy(raw)
                bid=next(iter(altered["actions"][0][game["candidate_seat"]]))
                if altered["actions"][0][game["candidate_seat"]][bid]=="RECCOLLECTOR":
                    altered["actions"][0][game["candidate_seat"]][bid]="RECPLANTER"
                else:
                    altered["actions"][0][game["candidate_seat"]][bid]="RECCOLLECTOR"
                rejected("reject_nonhybrid_actual_action",lambda:verify_actions(folder,game,altered,pack))
                game["compound_policy_audit"]={**expected,"student_changes":0}
                rejected("reject_forged_student_attribution",lambda:verify_raw(folder,game,pack,STUDENT,formal=True))
    # Standalone verdict tests. Scores are artificial; these are 100 numbered mock maps, outside the reserved range.
    sample=game
    plan=dict(schema="sapling-90-and-income/3",income_metric=FORMAL_METRIC,income_minimum=TARGET,
              pack_sha256=D1,maps=[{"seed":s} for s in range(100)])
    def scores(wins,total):
        cents=[total//wins if i<wins else 0 for i in range(200)]
        cents[0]+=total-sum(cents)
        rows=[]
        proof=observed["real_actions_in_formal_envelope"]["d1_sapling_dev"]
        for i,c in enumerate(cents):
            seat=i%2
            final=[0.,0.]
            final[seat]=c/100
            rows.append(dict(sample, seed=i//2,candidate_seat=seat, margin=c/100,
                         official_score_by_seat=final,result="win" if c else "draw",
                         income_account={FORMAL_METRIC:c/100},compound_policy_audit=proof))
        return rows
    scenarios=[(180,10_000_000,True),(179,10_000_000,False),(180,9_999_999,False)]
    observed["joint_boundary"]=[]
    for wins,total,expected in scenarios:
        got=verdict(plan,scores(wins,total))
        assert got["observed_joint_target_met"] is expected
        observed["joint_boundary"].append(dict(wins=wins,mean=got["mean_official_margin"],passed=expected))
    rows=scores(180,10_000_000)
    rejected("reject_downgraded_compound_plan",lambda:verdict(dict(plan,schema="sapling-one-shot-90/2"),rows))
    rejected("reject_incomplete_schedule",lambda:verdict(plan,rows[:-1]))
    rejected("reject_duplicate_seat",lambda:verdict(plan,rows[:-1]+rows[-2:-1]))
    rejected("reject_no_student_contributions",lambda:verdict(plan,[dict(g,compound_policy_audit={
        **g["compound_policy_audit"],"student_changes":0}) for g in rows]))
    return dict(scope="Tests only. Hash-confirmed source rebuilds, original development actions in a scratch identity envelope, and synthetic numerical boundary cases. No final map was allocated or played.",checks=observed)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    write(parser.parse_args().out,tests())

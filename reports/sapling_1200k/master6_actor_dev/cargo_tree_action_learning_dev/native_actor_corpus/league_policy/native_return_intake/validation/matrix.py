"""Bind the qualified new return-trained 851c descendant, then run four paired original-opponent tickets."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parent
LEAGUE=INTAKE.parent
NATURAL=LEAGUE/"natural_original"
PREVIOUS=NATURAL/"fresh_repeat"
spec=importlib.util.spec_from_file_location("held_original_shared_matrix", PREVIOUS/"matrix.py")
shared=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
ROOT=shared.ROOT
PLAN=HERE/"protocol.json"
RELEASE=HERE/"release.json"
SEEDS=(621758470,621758471)
PARENT,CHILD=ARMS=("sampled_parent","sampled_child")
LEARNING="d348f38843eb899886b6643527dc57ba40d8c551ce8e2afd7f2640f63c56e75a"
INTAKE_PROTOCOL="f720c89c4d221f43f3cec81a73a3ce765c4d70c14dae44ee76c289a04ad346fc"
BEHAVIOR="851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"
EXAMINER="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
sha,read,file,require=shared.sha,shared.read,shared.evidence_file,shared.require


def schedule():
    return [dict(seed=seed,seat=seat,order=list(ARMS if (i+seat)%2==0 else reversed(ARMS)))
            for i,seed in enumerate(SEEDS) for seat in (0,1)]


def same_serving_code(a,b):
    permitted=(b"_W64",b"_QMETA",b"_WMETA")
    def code(path):
        result=[];seen=[]
        for line in path.read_bytes().splitlines(keepends=True):
            field=next((k for k in permitted if line.startswith(k+b" = ")),None)
            if field:
                seen.append(field);result.append(field+b"\n")
            else:result.append(line)
        require(sorted(seen)==sorted(permitted),"Unexpected editable package fields")
        return hashlib.sha256(b"".join(result)).digest()
    return code(a)==code(b)


def select_trial(f,learning):
    require(f["status"]=="training_mechanism_pass" and f["learning_sha256"]==LEARNING and f["parent_sha256"]==BEHAVIOR,
            "No eligible return-trained candidate")
    require((f["total_trajectories"],f["paired_map_seats"],f["physical_maps"],f["planned"])==(24,12,6,288)
            and 0<f["supported"]<=f["scheduled_reached"]<=288 and math.isfinite(f["gradient_l2"]) and
            f["gradient_l2"]>learning["epsilon_gradient_l2"],"Incomplete or nonfinite actual gradient")
    require(0<=f["parent_behavior_replay"]["max_signed_logit_difference_error"]<=.003 and
            0<=f["parent_behavior_replay"]["max_probability_error"]<=.00075,"Actual parent likelihood parity failed")
    require([t["step"] for t in f["trials"]]==learning["steps"]==[.1,.05,.025,.0125],"Missing or reordered trial")
    eligible=[]
    for t in f["trials"]:
        kl,delta=t["conditional_kl_supported"],t["realized_weighted_logp_delta_per_scheduled"]
        require(math.isfinite(kl) and math.isfinite(delta) and 0<=t["max_projection_gap_error"]<=.003,
                "Nonfinite or inconsistent projected odds")
        by_seat=t["ordinary_free_by_seat"]
        require(len(by_seat)==2,"Missing physical seat")
        passes=0<=kl<=.005 and delta>0 and all(v.get("changed_order_or_probability",0)>0 for v in by_seat)
        require(t["qualifies"] is passes,"Trial qualification differs from preregistered criteria")
        if passes:eligible.append(t)
    require(eligible,"No step passes the registered behavior gate")
    chosen=eligible[0]
    require((chosen["step"],chosen["pack_sha256"],chosen["weights_sha256"])==
            (f["selected_step"],f["selected_package_sha256"],f["selected_weights_sha256"]),"Wrong selected eligible step")
    return chosen


def trained(candidate_sha,fit_sha):
    require(sha(INTAKE/"learning.json")==LEARNING and sha(INTAKE/"protocol.json")==INTAKE_PROTOCOL,
            "Different training protocol")
    require(sha(INTAKE/"fit.json")==fit_sha and sha(INTAKE/"candidate.py")==candidate_sha and candidate_sha!=BEHAVIOR,
            "Candidate and fit do not match reviewed digests")
    f=read(INTAKE/"fit.json");chosen=select_trial(f,read(INTAKE/"learning.json"))
    require(chosen["pack_sha256"]==candidate_sha and chosen["weights_sha256"]==sha(INTAKE/"candidate.npz") and
            chosen["pack_bytes"]==(INTAKE/"candidate.py").stat().st_size<=10_000_000,"Wrong or oversized deployed package")
    require(sha(NATURAL/"candidate_sampled.py")==BEHAVIOR and same_serving_code(NATURAL/"candidate_sampled.py",INTAKE/"candidate.py"),
            "Candidate changed the natural policy or serving logic outside student weights")
    manifest=INTAKE/"hosted/training_manifest.json";settled=INTAKE/"hosted/settlement.json"
    require(f["archive_manifest_sha256"]==sha(manifest) and f["decision_audit_sha256"]==sha(INTAKE/"fit_decisions.json.gz")
            and f["learn_script_sha256"]==sha(INTAKE/"learning.py"),"Training results detached from actual receipts")
    m,r=read(manifest),read(settled)
    require(m["settlement_sha256"]==sha(settled) and m["protocol_sha256"]==INTAKE_PROTOCOL and
            m["training_eligible"] is True and r["records_reconciled"] is True and r["training_eligible"] is True and
            not r["verification_errors"] and not r["missing_tickets"] and len(m["episodes"])==len(r["games"])==24 and
            m["policy_sha256"]==BEHAVIOR and m["examiner_sha256"]==EXAMINER,"Missing verified original-opponent training data")
    return chosen


def bound_files():
    intake=read(INTAKE/"protocol.json")
    base={key:intake["files"][key] for key in ("opponent","environment","official_helpers","rules","live_registration",
         "live","worker","dockerfile","runner","verifier","guide","fit","original_settle","support_check","exporter",
         "collector","initial_configuration","reference_source","exposed_matrix","verified_matrix")}
    items={PARENT:NATURAL/"candidate_sampled.py",PARENT+"_weights":LEAGUE/"candidate.npz",CHILD:INTAKE/"candidate.py",
           CHILD+"_weights":INTAKE/"candidate.npz","source_training":INTAKE/"learning.json","fit_result":INTAKE/"fit.json",
           "fit_decisions":INTAKE/"fit_decisions.json.gz","learner":INTAKE/"learning.py","intake_plan":INTAKE/"protocol.json",
           "training_manifest":INTAKE/"hosted/training_manifest.json","training_settlement":INTAKE/"hosted/settlement.json",
           "validation_clearance":HERE/"clearance.json","validation_clearance_code":HERE/"clearance.py","validation_release":RELEASE,
           "validation_matrix":HERE/"matrix.py","validation_merge":HERE/"merge.py","validation_menu":HERE/"menu.py",
           "validation_helper":HERE/"workflow_cli.py","source_exposure_helper":PREVIOUS/"workflow_cli.py","validation_workflow":ROOT/".github/workflows/sapling-native-return-validation.yml"}
    base.update({k:file(v) for k,v in items.items()})
    for k,v in base.items():require(v==file(ROOT/v["path"]),"Changed registered input: "+k)
    require(base["opponent"]["sha256"]==base["reference_source"]["sha256"]==EXAMINER and base[PARENT]["sha256"]==BEHAVIOR
            and base[PARENT+"_weights"]["sha256"]==read(INTAKE/"learning.json")["editable_source_npz"],"Different control or original")
    return base


def frozen():
    release=read(RELEASE)
    chosen=trained(release["candidate_sha256"],release["fit_sha256"])
    require(release["weights_sha256"]==chosen["weights_sha256"] and release["training_manifest_sha256"]==sha(INTAKE/"hosted/training_manifest.json")
            and re.fullmatch("[0-9a-f]{40}",release["training_commit"]),"Release evidence changed")
    clearance=read(HERE/"clearance.json");boards=clearance["by_seed"]
    require(clearance["seeds"]==list(SEEDS) and set(boards)=={str(i) for i in SEEDS} and len(set(boards.values()))==2 and
            not clearance["seed_intersections"] and not clearance["physical_intersections"],"Wrong or previously exposed holdout")
    train=read(INTAKE/"protocol.json");require(not(set(SEEDS)&set(train["seeds"]) or
            set(boards.values())&set(train["physical_boards"].values())),"Training and validation map overlap")
    return dict(schema="sapling-native-return-holdout/1",purpose="Development on two unused physical maps; independent final exam remains reserved.",
        platform=train["platform"],arms=list(ARMS),seeds=list(SEEDS),physical_boards=boards,expected_games=8,schedule=schedule(),
        source_fit=release,selected_step=chosen["step"],endpoint="One full frozen parent–candidate pair at each map and seat, all eight attempts. Report full official win/draw/loss including timeouts, extinction rewards and per-seat means, paired official margin deltas by each map and seat, plus their two map means. Missing official margins stay missing. All wins are descriptive development outcomes.",
        execution="Original public temperature one, unseeded 0.58 s wall-adaptive search, both natural learners unseeded. Run paired arms cold on the same host/CPU set in serial order AB/BA/BA/AB without repeating exposed tickets. Recovery after a setup failure requires independently confirmed zero prior exposure across all four tickets. Include public I/O, step zero, game-level slow steps and original search depth.",
        files=bound_files())


shared.HERE=HERE;shared.PLAN=PLAN;shared.SEEDS=SEEDS;shared.ARMS=ARMS
shared.schedule=schedule;shared.frozen=frozen
check=shared.check
verify_block=shared.verify_block


def bind(rev,candidate_sha,fit_sha):
    require(re.fullmatch("[0-9a-f]{40}",rev) and all(re.fullmatch("[0-9a-f]{64}",h) for h in (candidate_sha,fit_sha)),
            "Specify exact reviewed main commit, candidate SHA-256 and fit SHA-256")
    subprocess.run(["git","-C",str(ROOT),"merge-base","--is-ancestor",rev,"origin/main"],check=True)
    chosen=trained(candidate_sha,fit_sha)
    inputs=[INTAKE/q for q in ("candidate.py","candidate.npz","learning.json","learning.py","protocol.json","fit.json",
            "fit_decisions.json.gz","hosted/training_manifest.json","hosted/settlement.json")]
    for path in inputs:
        stored=subprocess.check_output(["git","-C",str(ROOT),"show",rev+":"+path.relative_to(ROOT).as_posix()])
        require(hashlib.sha256(stored).hexdigest()==sha(path),"Local file differs from training commit: "+path.name)
    d=dict(training_commit=rev,candidate_sha256=candidate_sha,fit_sha256=fit_sha,weights_sha256=chosen["weights_sha256"],
           training_manifest_sha256=sha(INTAKE/"hosted/training_manifest.json"))
    payload=shared.old.canonical(d);dest=RELEASE
    if dest.exists(): require(dest.read_text()==payload,"A different validation release is already bound")
    else:
        with dest.open("x") as f:f.write(payload)
    sys.argv=[sys.argv[0],"freeze"]
    return shared.main()


if __name__=="__main__":
    if sys.argv[1:2]==["bind"]:
        q=argparse.ArgumentParser();q.add_argument("mode");q.add_argument("--training-commit",required=True)
        q.add_argument("--candidate-sha",required=True);q.add_argument("--fit-sha",required=True)
        a=q.parse_args();sys.exit(bind(a.training_commit,a.candidate_sha,a.fit_sha))
    sys.exit(shared.main())

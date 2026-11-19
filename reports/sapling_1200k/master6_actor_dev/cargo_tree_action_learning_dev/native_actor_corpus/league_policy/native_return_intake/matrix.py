"""Register paired independent full-game return samples from the actual native Master3s2."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
LEAGUE=HERE.parent
NATURAL=LEAGUE/"natural_original"
PREVIOUS=NATURAL/"fresh_repeat"

# Each import has its own private configuration. The frozen, already exercised
# runner writes per-arm cold receipts and serial host execution timestamps.
spec=importlib.util.spec_from_file_location("native_intake_shared_matrix", PREVIOUS/"matrix.py")
shared=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
old=shared.old
ROOT=shared.ROOT
PLAN=HERE/"protocol.json"
SEEDS=tuple(range(621758460,621758466))
ARMS=("draw_a","draw_b")
SHA="851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"
GREEDY="510c527a6f3e88f718e71e4b456e6e065abaeb1474587dc76fd11a5744d7c3d4"
WEIGHTS="bc916190f2f7484055f70a0020f502df137127d8a15b09a5addbc612c0ed849c"
PREVIOUS_PROTOCOL="2199436fd29ed9412ca21f221eaee72a85184d891b222bc670bed62bd1f417a4"
PREVIOUS_SETTLEMENT="fe727a57eaab4ebd0cf18fdac124dadc10d0358d587008726224386c31adc603"
PREVIOUS_COMMIT="d83ab1039680a0df70c372f3b300b0bfd559f139"
PREVIOUS_RUN=44625133102
LEARNING="d348f38843eb899886b6643527dc57ba40d8c551ce8e2afd7f2640f63c56e75a"
read,sha,require=shared.read,shared.sha,shared.require


def schedule():
    # Balance the two labels' positions within map and physical seat. Both copies have identical weights.
    return [dict(seed=s,seat=p,order=list(ARMS if (i+p)%2==0 else reversed(ARMS)))
            for i,s in enumerate(SEEDS) for p in (0,1)]


def previous_sources():
    plan_path=PREVIOUS/"protocol.json"; settlement_path=PREVIOUS/"hosted/settlement.json"
    require(sha(plan_path)==PREVIOUS_PROTOCOL and sha(settlement_path)==PREVIOUS_SETTLEMENT,"Wrong previous audit")
    plan,result=read(plan_path),read(settlement_path)
    require(result["workflow_run"]==PREVIOUS_RUN and result["commit"]==PREVIOUS_COMMIT and
            result["records_reconciled"] and not result["verification_errors"] and not result["missing_tickets"] and
            result["plan_sha256"]==PREVIOUS_PROTOCOL,"Previous whole-game evidence incomplete")
    original_order={(s,p,a) for s in plan["seeds"] for p in (0,1) for a in plan["arms"]}
    require(len(result["games"])==8 and {(g["seed"],g["seat"],g["arm"]) for g in result["games"]}==original_order,
            "Wrong previous input game set")
    arms=result["results"]["arms"]
    require(arms["sampled_child"]["all"]["wins"]==arms["sampled_parent"]["all"]["wins"]==0 and
            abs(result["results"]["child_minus_parent_mean_margin"]-678.7575)<1.e-7, "Wrong previous outcome")
    proofs={}
    for record in result["archives"]:
        p=settlement_path.parent/(str(record["id"])+".zip")
        require(p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],"Previous signed artifact differs")
        proofs[record["seed"],record["seat"]]=p
    require(len(proofs)==4,"Missing previous paired artifact")
    for g in result["games"]:
        trace=g["action_support"]; a=g["arm"]
        require(g["status"]=="complete" and g["counted"] and trace["matched_all_steps"] and
                trace["deployment_sha256"]==plan["files"][a]["sha256"] and
                g["timing_audit"]["executor_verified_for_scored_attempt"] and
                g["official_transitions"]==trace["official_transitions"]==trace["n_steps"] and
                g["official_scores_by_seat"]==trace["official_scores_by_seat"],"Previous independent replay incomplete")
        with zipfile.ZipFile(proofs[g["seed"],g["seat"]]) as z:
            for name,key in (("turns.jsonl","tape_sha256"),("receipt.json","receipt_sha256"),("registration.json","registration_sha256")):
                require(hashlib.sha256(z.read(a+"/"+name)).hexdigest()==trace[key],"Previous official evidence not bound")
    return plan, [proofs[k] for k in sorted(proofs)]


def frozen():
    previous,archives=previous_sources()
    require(sha(HERE/"clearance.json")=="d745b0dd3dc28d85a7e3422e636f55bb953a234ca7374f821ebc287d7c0849d6","Different six-map clearance")
    clearance=read(HERE/"clearance.json")
    boards=clearance["by_seed"]
    require(clearance["seeds"]==list(SEEDS) and set(boards)=={str(s) for s in SEEDS} and
            len(set(boards.values()))==len(SEEDS) and not clearance["seed_intersections"] and
            not clearance["physical_intersections"], "Training census is not six physically disjoint new maps")
    require(sha(HERE/"learning.json")==LEARNING, "Learning preregistration changed")
    files={k:previous["files"][k] for k in ("opponent","environment","official_helpers","rules",
           "live_registration","live","worker","dockerfile","runner","verifier","guide","exposed_matrix",
           "original_settle","support_check","exporter","collector","fit","initial_configuration","reference_source")}
    for arm in ARMS:
        files[arm]=previous["files"]["sampled_child"]
        files[arm+"_weights"]=previous["files"]["sampled_child_weights"]
    files.update({k:shared.evidence_file(path) for k,path in {
        "base_greedy":LEAGUE/"candidate.py","policy_audit":NATURAL/"candidate_audit.json",
        "natural_entropy_audit":NATURAL/"candidate_natural.json","verified_matrix":PREVIOUS/"matrix.py",
        "prior_sparse_helper":PREVIOUS/"workflow_cli.py","prior_protocol":PREVIOUS/"protocol.json",
        "prior_settlement":PREVIOUS/"hosted/settlement.json","learning_protocol":HERE/"learning.json",
        "clearance":HERE/"clearance.json","clearance_code":HERE/"clearance.py","intake_matrix":HERE/"matrix.py",
        "workflow_helper":HERE/"workflow_cli.py",
        "workflow":ROOT/".github/workflows/sapling-native-return-intake.yml"}.items()})
    for i,path in enumerate(archives): files[f"previous_archive_{i}"]=shared.evidence_file(path)
    for k,entry in files.items():require(shared.evidence_file(ROOT/entry["path"])==entry,"Signed dependency mismatch: "+k)
    require(files[ARMS[0]]==files[ARMS[1]] and files[ARMS[0]]["sha256"]==SHA and
            files["draw_a_weights"]==files["draw_b_weights"] and files["draw_a_weights"]["sha256"]==WEIGHTS and
            files["base_greedy"]["sha256"]==GREEDY and files["opponent"]["sha256"]==files["reference_source"]["sha256"]==old.PARTICIPANTS["opponent"][1],
            "Starting native actor, network parameters or examiner differ")
    entropy=read(NATURAL/"candidate_natural.json");audit=read(NATURAL/"candidate_audit.json")
    require(audit["deployment"]["source_sha256"]==GREEDY and audit["deployment"]["sha256"]==entropy["package_sha256"]==SHA
            and audit["deployment"]["training_weights_sha256"]==WEIGHTS and
            len({r["first_rng_state_sha256"] for r in entropy["training_roots"]})==2, "No bound fresh RNG evidence")
    return dict(schema="sapling-native-original-return-intake/1",
        purpose="24 full original-opponent trajectories for a single policy-network update of Sapling. All six development-derived training maps become exposed. No formal examination, source-opponent substitutes, new ranking or deployment claim.",
        arms=list(ARMS),seeds=list(SEEDS),physical_boards=boards,expected_games=24,schedule=schedule(),platform=previous["platform"],
        source=dict(workflow_run=PREVIOUS_RUN,commit=PREVIOUS_COMMIT,protocol_sha256=PREVIOUS_PROTOCOL,settlement_sha256=PREVIOUS_SETTLEMENT),
        actor=dict(sampling_package_sha256=SHA,greedy_source_sha256=GREEDY,weights_sha256=WEIGHTS,
                   role="Both labels denote the same frozen Sapling package. Sample at 12,36,...,276 only if the complete 2-command support exists; draw from its own exact logits. All other commands follow the public greedy/guard path."),
        pairing="Six seeds x both physical seats x two independent cold trajectories. Same registered original/actor, image, CPU set and job host within a pair; counterbalanced AB/BA by map index+physical seat. Independently cold-load both contestants and make real public requests from step 0. Private entropy is not injected, observed or reconstructed. Distinct container IDs and source/cold-start receipts establish isolation. Equal openings, complete trajectories, chosen actions or end scores are legitimate and must be retained.",
        estimator="For official numeric margin M_i (including official extinction), W_i=1/0.5/0 for win/draw/loss, U_i=0.5*W_i+0.5*sigmoid(M_i/1000). A_i=U_i-U_j, where j is the other independent draw on exactly the same map and seat. Use both trajectories symmetrically in the summed conditional log-probability gradient, equal per-trajectory weight, with no win/choice filters. Do not credit a single move with a causal margin. Details and learning controls are frozen in learning.json.",
        evidence="Retain all 24 official outcomes, timing failures, full public request/response clocks, official action tapes, original search counts/depth/budget, original termination or extinction, initial board and seat, cold container identities, qualified limits. Offline recover every prescheduled opportunity, complete physical support and prefix, actual issued command, exact conditional probabilities/chosen logp and pre-observation digest; all other steps must match singleton support.",
        incomplete="Run once; preserve every attempted trajectory. Missing official numeric score, absent pair, timing forfeit or unresolved certification invalidates the numerical training batch. Publish status and raw evidence, no fabricated U, substitute trial or replacement/automatic batch. Evaluations versus the original on fresh validation maps remain separate.",
        files=files)


# Configure an isolated copy of the shipped paired executor while preserving its published source and historical evidence.
shared.HERE=HERE;shared.PLAN=PLAN;shared.SEEDS=SEEDS;shared.ARMS=ARMS
shared.schedule=schedule;shared.frozen=frozen
basic_verify=shared.verify_block


def verify_block(directory,seed,seat):
    directory=Path(directory);checked=basic_verify(directory,seed,seat)
    receipts={}
    for arm in ARMS:
        f=directory/arm/"receipt.json"
        if f.is_file():receipts[arm]=read(f)
    if len(receipts)==2 and all(r["status"] in ("complete","forfeit","bilateral_failure") for r in receipts.values()):
        containers=[]
        for arm in ARMS:
            r=receipts[arm]; init=r.get("initialization",[]);c=r.get("containers",[])
            if len(c)!=2 or len(init)!=2:
                checked["verification_errors"]["independent_game_instances"]="Missing physical containers or cold load"
                continue
            containers.extend(v["id"] for v in c)
            game=checked["games"].get(arm)
            if game:
                actor=init[seat];opponent=init[1-seat]
                game["independent_cold_game"]=dict(candidate_container=c[seat]["id"],opponent_container=c[1-seat]["id"],
                        own_load_status=actor["status"],original_load_status=opponent["status"],
                        own_source_sha256=actor.get("response",{}).get("sha256"),natural_source_sha256=SHA,
                        examiner_source_sha256=opponent.get("response",{}).get("sha256"),
                        own_container_cgroup_inode=r["probes"][seat]["cgroup_inode"],
                        examiner_container_cgroup_inode=r["probes"][1-seat]["cgroup_inode"])
        if len(containers)!=4 or len(set(containers))!=4:
            checked["verification_errors"]["independent_game_instances"]="Source or examiner used repeated physical containers across the two trials"
    return checked


shared.verify_block=verify_block
check=shared.check

if __name__=="__main__":
    sys.exit(shared.main())

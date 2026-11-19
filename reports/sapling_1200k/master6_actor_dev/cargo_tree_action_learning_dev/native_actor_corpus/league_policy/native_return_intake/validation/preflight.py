"""Exercise qualification, zero-exposure recovery and no-weights behavior without running or opening a holdout game."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import matrix
import workflow_cli as wf


def refuses(n,fn):
    try:fn()
    except (FileNotFoundError,ValueError):return
    raise AssertionError(n)


def main():
    learning=matrix.read(matrix.INTAKE/"learning.json")
    trials=[]
    for i,size in enumerate(learning["steps"]):
        trials.append(dict(step=size,qualifies=i!=0,pack_sha256=str(i+1)*64,weights_sha256=str(i+5)*64,
                conditional_kl_supported=.006 if i==0 else .002,realized_weighted_logp_delta_per_scheduled=.008,
                max_projection_gap_error=1e-6,ordinary_free_by_seat=[{"changed_order_or_probability":2}]*2))
    chosen=trials[1]
    fit=dict(status="training_mechanism_pass",learning_sha256=matrix.LEARNING,parent_sha256=matrix.BEHAVIOR,
             total_trajectories=24,paired_map_seats=12,physical_maps=6,planned=288,supported=230,scheduled_reached=278,
             gradient_l2=.1,parent_behavior_replay=dict(max_signed_logit_difference_error=1e-6,max_probability_error=2e-7),
             trials=trials,selected_step=chosen["step"],selected_package_sha256=chosen["pack_sha256"],
             selected_weights_sha256=chosen["weights_sha256"])
    assert matrix.select_trial(fit,learning)==chosen
    for name,edit in (("missing whole trajectory",lambda x:x.update(total_trajectories=23)),
            ("earlier failed candidate substituted",lambda x:x.update(selected_step=.1)),
            ("nonfinite KL",lambda x:x["trials"][0].update(conditional_kl_supported=float("nan"))),
            ("false eligible flag on failed KL",lambda x:x["trials"][0].update(qualifies=True)),
            ("zero projected improvement",lambda x:x["trials"][1].update(realized_weighted_logp_delta_per_scheduled=0)),
            ("no right-seat response",lambda x:x["trials"][1].update(ordinary_free_by_seat=[{"changed_order_or_probability":2},{}])),
            ("wrong starting actor",lambda x:x.update(parent_sha256=chosen["pack_sha256"])),
            ("unsuccessful training",lambda x:x.update(status="no_eligible_registered_step"))):
        changed=deepcopy(fit);edit(changed)
        refuses(name,lambda:matrix.select_trial(changed,learning))
    current=dict(event="workflow_dispatch",head_branch="main",id=111,run_number=1,status="completed",run_attempt=1)
    assert not wf.prior_check([current],111)
    later={**current,"id":222,"run_number":2}
    assert wf.prior_check([current,later],222)==[current]
    refuses("earlier still active",lambda:wf.prior_check([{**current,"status":"in_progress"},later],222))
    refuses("missing inventory",lambda:wf.prior_check([],111))
    planner=dict(name="matrix",status="completed",conclusion="failure")
    def impossible(*_):
        raise AssertionError("No download for a skipped play step")
    wf.ensure_setup_only(current,[planner],impossible)
    tickets=[dict(name=f"native-validation-{seed}-{seat}",status="completed",conclusion="failure",
                  steps=[dict(name=wf.GAME_STEP,conclusion="skipped")]) for seed in wf.SEEDS for seat in (0,1)]
    planner["conclusion"]="success"
    wf.ensure_setup_only(current,[planner,*tickets],impossible)
    refuses("ticket inventory omitted following successful planner",lambda:wf.ensure_setup_only(current,[planner,*tickets[:3]],impossible))
    tickets[0]["steps"]=[dict(name=wf.GAME_STEP,conclusion="failure",started_at="2026-11-19T00:00:00Z")]
    proof=dict(run_id="111",attempt="1",seed=wf.SEEDS[0],seat=0,confirmed_no_exposure=True,
               arms={a:dict(attempt=False,initializations=0,action_calls=[0,0]) for a in wf.ARMS})
    wf.ensure_setup_only(current,[planner,*tickets],lambda *_:proof)
    for arm,field,value in [(wf.ARMS[0],"attempt",True),(wf.ARMS[1],"initializations",1),(wf.ARMS[1],"action_calls",[1,1])]:
        changed=deepcopy(proof);changed["arms"][arm][field]=value
        refuses("historically exposed",lambda:wf.ensure_setup_only(current,[planner,*tickets],lambda *_:changed))
    refuses("wrong exposure receipt",lambda:wf.ensure_setup_only(current,[planner,*tickets],lambda *_:dict(proof,seat=1)))
    refuses("unknown exposure",lambda:wf.ensure_setup_only(current,[planner,*tickets],lambda *_:dict(proof,confirmed_no_exposure=False)))
    schedule=matrix.schedule()
    assert [(r["seed"],r["seat"],r["order"]) for r in schedule]==[
        (s,p,list(matrix.ARMS if (i+p)%2==0 else reversed(matrix.ARMS)))
        for i,s in enumerate((621758470,621758471)) for p in (0,1)]
    assert len({(r["seed"],r["seat"],a) for r in schedule for a in r["order"]})==8
    with TemporaryDirectory() as tmp:
        old=wf.HERE
        try:
            wf.HERE=Path(tmp)
            refuses("missing training release before sparse setup",lambda:wf.prepare("0"*64))
        finally:wf.HERE=old
    deployed=matrix.NATURAL/"candidate_sampled.py"
    assert matrix.same_serving_code(deployed,deployed)
    print("Eight fixed tickets in four serial pairs; accepted reference fixture, rejected eight bad fits and exposed or unknown tickets, allowed proved setup-only recovery, rejected missing inventory and missing release. No games invoked.")


if __name__=="__main__":main()

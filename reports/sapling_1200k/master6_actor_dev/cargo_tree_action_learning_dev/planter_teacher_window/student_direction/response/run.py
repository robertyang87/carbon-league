"""Apply the already validated original-reply branch kernel to a new frozen direction-only learner."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CARGO = HERE.parents[2]
ENGINE = CARGO / "territory_sequence/response/run.py"
PLAN = HERE / "protocol.json"
REGISTERED = "354962947923d719cf29fb5aca844d4de715c1f47d9b812bdc69ead4f5912b53"

# The original run_arm and accounting read these six settings; the original opponent,
# its recorded warm costs, and both scoring implementations are reused verbatim.
spec = importlib.util.spec_from_file_location("direction_original_response", ENGINE)
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)
kernel.HERE = HERE
kernel.SEQUENCE = STUDY
kernel.STUDENT = STUDY / "candidate.py"
kernel.PLAN = PLAN
kernel.REGISTERED = REGISTERED
kernel.BEGIN, kernel.END = 128, 144
R = kernel.R


def preflight():
    if not __debug__:
        raise RuntimeError("run with assertions enabled")
    plan = R.load(PLAN)
    assert R.sha(PLAN) == REGISTERED
    assert R.sha(ENGINE) == plan["upstream_runner_sha256"]
    assert R.sha(kernel.PARENT) == plan["parent_sha256"]
    source = subprocess.check_output(["git","-C",str(R.ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    assert hashlib.sha256(source).hexdigest() == plan["examiner_sha256"] == R.sha(R.live.SOURCE)
    fit = R.load(STUDY/"protocol.json")
    assert fit["parent"] == plan["parent_sha256"]
    panel = CARGO/"territory_sequence/sources.json"
    teachers = CARGO/"planter_teacher_window/protocol.json"
    assert R.sha(panel) == plan["sequence_source_list_sha256"]
    assert R.sha(teachers) == plan["teacher_protocol_sha256"]
    forbidden = {r["terrain"] for r in R.load(panel)}|{r["terrain"] for r in R.load(teachers)["cases"]}
    checks = []
    for case in plan["cases"]:
        assert case["terrain"] not in forbidden and case["seed"] not in {621758326,621758339,*range(621758343,621758354)}
        raw = kernel.archive(case,plan)
        env = R.live.begin(raw)
        for both in raw["actions"][:kernel.BEGIN]:
            env.step(both)
        assert not env.done
        assert all(R.live.public(env,i)["step"]==kernel.BEGIN for i in (0,1))
        for both in raw["actions"][kernel.BEGIN:]:
            env.step(both)
        official = [row.reward for row in env.states]
        assert official==raw["official_score_by_seat"] and [x.status for x in env.states]==["DONE","DONE"]
        physical_end=[x[0] for x in env.states[0].observation.players]
        ledger=kernel.accounting(raw,raw["actions"][kernel.BEGIN:],physical_end)
        checks.append(dict(seed=case["seed"],seat=case["seat"],terrain=case["terrain"],
                           archived_official=official,post_root_cash_ledger_closed=True,
                           registered_turns=kernel.END-kernel.BEGIN,full_tape_steps=len(ledger["steps"])))
    return dict(protocol=REGISTERED, kernel=R.sha(ENGINE), runner=R.sha(Path(__file__)), checks=checks)


def seal(candidate_sha):
    """Replay held public observations, bounded to one changed planter direction per terrain."""
    plan=R.load(PLAN)
    assert R.sha(kernel.STUDENT)==candidate_sha and candidate_sha!=R.sha(kernel.PARENT)
    assert R.load(HERE/"preflight.json")==preflight()
    panel=R.load(CARGO/"territory_sequence/sources.json")
    paths={"parent":kernel.PARENT, "student":kernel.STUDENT}
    models={k:R.live.module(p,"public_direction_"+k) for k,p in paths.items()}
    targets={"UP","RIGHT","DOWN","LEFT"}
    changed=[]
    terrain_seen=set()
    games=0
    observations=0
    for c in panel:
        if c["split"]!="test" or c["source"]!="Master3s2" or c["terrain"] in terrain_seen:
            continue
        games+=1
        raw=R.load(R.ROOT/c["path"])
        assert R.sha(R.ROOT/c["path"])==c["sha256"]
        env=R.live.begin(raw)
        s=c["seat"]
        for t,joint in enumerate(raw["actions"][:193]):
            obs=R.live.public(env,s)
            if t==0 or (64<=t<=192 and t%4==0):
                acts={k:R.live.clean(m.agent(obs,raw["game_configuration"])) for k,m in models.items()}
                observations+=1
                if t:
                    ids=[u for u,worker in obs["players"][s][2].items() if worker[2]=="PLANTER"
                         and acts["parent"].get(u) in targets and acts["student"].get(u) in targets
                         and acts["parent"][u]!=acts["student"][u]]
                    if ids:
                        changed.append(dict(terrain=c["terrain"],seed=c["seed"],source_sha=c["sha256"],source_path=c["path"],
                            seat=s,step=t,workers=ids,commands=acts))
                        terrain_seen.add(c["terrain"])
                        break
            else:
                for m in models.values():
                    m._FEAT.observe(obs)
            env.step(joint)
        if len(terrain_seen)>=2:
            break
    return dict(study=REGISTERED,parent=R.sha(kernel.PARENT),student=candidate_sha,observations=observations,
        games_scanned=games,method="normal public entry over first physical held original tapes until at least two genuine move-to-move planter changes on distinct terrains",
        terrain_examples=changed,verified=len(terrain_seen)>=2)


def pair(i, binding):
    plan=R.load(PLAN)
    case=plan["cases"][i]
    warm_file=HERE/f"warm_{case['seed']}.json"
    if not warm_file.exists():
        with ProcessPoolExecutor(1) as pool:
            cost=pool.submit(R.native_warm).result()
        R.write(warm_file,dict(binding=binding,cost=cost))
    w=R.load(warm_file)
    assert w["binding"]==binding
    with ProcessPoolExecutor(2) as pool:
        a,b=pool.map(kernel.run_arm,[(case,"parent",i,w["cost"],binding),(case,"student",1-i,w["cost"],binding)])
    assert a["rng"]==b["rng"]
    return summarize(case,a,b)


def summarize(case,a,b):
    s=case["seat"]
    raw=kernel.archive(case,R.load(PLAN))
    arms={}
    for arm,z in [("parent",a),("student",b)]:
        env=R.live.begin(raw)
        for both in raw["actions"][:kernel.BEGIN]:
            env.step(both)
        actual_changes=[]
        for j,both in enumerate(z["joint_tape"]):
            t=kernel.BEGIN+j
            if t<kernel.END:
                obs=R.live.public(env,s)
                row=z["comparisons"][j]
                assert row["t"]==t and both[s]==row[arm]
                orders=[]
                for uid,(x,y) in row["changed"].items():
                    w=obs["players"][s][2].get(uid)
                    orders.append(dict(worker=uid,parent=x,student=y,role="BASE" if w is None else w[2]))
                if orders:
                    actual_changes.append(dict(t=t,directions=orders))
            env.step(both)
        official=[st.reward for st in env.states]
        assert all(x.status=="DONE" for x in env.states) and official==z["official_scores"]
        calls=z["original_calls"]
        assert len(calls)==len(z["joint_tape"]) and [r["step"] for r in calls]==list(range(kernel.BEGIN,kernel.BEGIN+len(calls)))
        arms[arm]=dict(score=official,margin=official[s]-official[1-s],student_parent_disagreements_on_this_trajectory=actual_changes,
            branch_wall=z["branch_wall_seconds"],original_calls=len(calls),original_cpu=sum(r["process_cpu_seconds"] for r in calls),
            original_wall=sum(r["public_wall_seconds"] for r in calls),clock=z["clock"],cpu=z["identity"]["cpu"])
    def first(side):
        return next((kernel.BEGIN+i for i,(x,y) in enumerate(zip(a["joint_tape"],b["joint_tape"])) if x[side]!=y[side]),None)
    changes=arms["student"]["student_parent_disagreements_on_this_trajectory"]
    by_window={}
    for key in ("window","through_199","whole"):
        av,bv=a["cash"][key],b["cash"][key]
        kinds=set().union(*av,*bv)
        by_window[key]={k:(bv[s].get(k,0)-bv[1-s].get(k,0))-(av[s].get(k,0)-av[1-s].get(k,0)) for k in sorted(kinds)}
    return dict(seed=case["seed"],terrain=case["terrain"],seat=s,arms=arms,official_delta=arms["student"]["margin"]-arms["parent"]["margin"],
        window_changed_student_turns=len(changes),first_executed_intervention=changes[0]["t"] if changes else None,
        first_between_arm_own_difference=first(s),first_between_arm_opponent_difference=first(1-s),
        net_component_deltas=by_window,trajectories={name:R.sha(HERE/f"{case['seed']}_{s}_{name}.json.gz") for name in arms})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","seal","run","rescore"))
    parser.add_argument("--candidate-sha")
    args=parser.parse_args()
    if args.stage=="preflight":
        R.write(HERE/"preflight.json",preflight())
    elif args.stage=="seal":
        proof=seal(args.candidate_sha)
        R.write(HERE/"public_changes.json",proof)
        print(proof["verified"],len(proof["terrain_examples"]),flush=True)
    else:
        proof=R.load(HERE/"public_changes.json")
        assert proof["verified"] and R.sha(kernel.STUDENT)==proof["student"]==args.candidate_sha
        assert len({x["terrain"] for x in proof["terrain_examples"]})>=2
        manifest=dict(protocol=REGISTERED,parent=R.sha(kernel.PARENT),student=R.sha(kernel.STUDENT),
                      original=R.sha(R.live.SOURCE),held_proof=R.sha(HERE/"public_changes.json"),engine=R.sha(ENGINE),
                      runner=R.sha(Path(__file__)),fit_plan=R.sha(STUDY/"protocol.json"))
        assert R.load(HERE/"preflight.json")==preflight()
        if args.stage=="run":
            with R.LOCK.open("a") as h:
                fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                rows=[pair(i,manifest) for i in range(2)]
        else:
            rows=[]
            for c in R.load(PLAN)["cases"]:
                z=[R.load(HERE/f"{c['seed']}_{c['seat']}_{arm}.json.gz") for arm in ("parent","student")]
                assert all(data["identity"]["binding"]==manifest for data in z)
                rows.append(summarize(c,*z))
        R.write(HERE/"results.json",dict(binding=manifest,rows=rows,scope="Two historical midgame interventions lasting sixteen steps with genuine independent original opponent replies, not new full matches."))
        print([(r["seed"],r["official_delta"]) for r in rows],flush=True)

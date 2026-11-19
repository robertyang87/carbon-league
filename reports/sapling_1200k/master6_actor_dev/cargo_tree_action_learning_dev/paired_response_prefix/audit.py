"""Reconcile paired historical Carbon action tapes with the official interpreter. No agent runs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ACTOR = BASE.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR.parent / "root_recovery_branches"))
from branches import begin, clean


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def physical(env):
    # Both seats' complete official board and independently masked views, rewards and statuses.
    # Agent actions are independently compared below. Runner timing is outside the game state.
    return [{"observation": {k: v for k, v in s.observation.items() if k != "remainingOverageTime"},
             "reward": s.reward, "status": s.status} for s in env.states]


def clock(calls, start, n):
    require([v["step"] for v in calls] == list(range(start, start + n)), "clock missing a live step")
    zeros = [v["step"] for v in calls if v["kind"] == "depth_zero_return_base"]
    return {"calls": n, "first_call": start, "zero_depth": len(zeros),
            "first_zero": next(iter(zeros), None), "last_zero": zeros[-1] if zeros else None,
            "cpu_seconds": round(sum(v["process_cpu_seconds"] for v in calls), 4),
            "wall_seconds": round(sum(v["public_wall_seconds"] for v in calls), 4)}


def branch(path, arm, start, last, seat, root_proposal=None, registry_sha=None, source_sha=None):
    doc = read(path)
    if registry_sha is not None:
        if "identity" in doc:
            identity=doc["identity"]
            require(identity["archive"]==source_sha and identity["binding"]["protocol"]==registry_sha
                    and identity["name"]==arm and doc["physical_seat"]==seat, "identity/registry mismatch")
        elif "decisions" in doc:
            require(doc["binding"]["archive"]==source_sha and doc["binding"]["protocol"]==registry_sha
                    and doc["binding"]["label"]==arm and doc["seat"]==seat, "teacher/registry mismatch")
        else:
            require(doc["binding"]["roots"]==registry_sha and doc["binding"]["arm"]==arm, "root/arm mismatch")
    tape = doc.get("joint_tape", doc.get("tape"))
    calls = next(doc[k] for k in ("original_calls", "enemy_calls", "calls") if k in doc)
    scores = next(doc[k] for k in ("official_scores", "scores", "score") if k in doc)
    require(doc["statuses"] == ["DONE", "DONE"], f"incomplete branch {path}")
    shadows = {}
    if root_proposal is not None:
        require(clean(tape[0][seat]) == clean(root_proposal[arm]), "registered root action mismatch")
        shadows[start] = clean(root_proposal["parent"])
    else:
        records = doc.get("comparisons", doc.get("decisions"))
        require([v["t"] for v in records] == list(range(start, min(last+1, start+len(tape)))),
                f"missing or duplicated intervention observations in {path}")
        for v in records:
            t = v["t"]
            expected = v["actual"] if "actual" in v else v[arm]
            require(clean(tape[t-start][seat]) == clean(expected), f"shadow mismatch {path}:{t}")
            if arm == "parent":
                require(clean(expected) == clean(v["parent"]), "baseline did not execute its own decision")
            shadows[t] = clean(v["parent"])
    return dict(tape=tape, scores=scores, shadow=shadows, clock=clock(calls, start, len(tape)),
                files={str(path.relative_to(ROOT)): sha(path)})


def window_cases():
    for name, rel in (("H24","territory_sequence/response"),
                      ("direction","planter_teacher_window/student_direction/response"),
                      ("teacher","planter_teacher_window"),("onpolicy","onpolicy_native/response")):
        folder = BASE / rel
        path = folder / ("roots.json" if name == "onpolicy" else "protocol.json")
        book = read(path)
        for entry in book["cases"]:
            seed, seat = entry["seed"], entry["seat"]
            base = f"{seed}_{seat}"
            if name == "onpolicy":
                start = last = entry["root"]["t"]
                source, source_sha = ROOT / entry["source"], entry["source_sha256"]
                expected = entry["root"]["proposals"]
                base = entry["id"]
            else:
                expected = None
                if name == "teacher":
                    start, last = book["first"], book["last_inclusive"]
                    source, source_sha = ROOT / entry["path"], entry["sha256"]
                else:
                    start, last = ((book["first_joint_step"], book["last_joint_step_inclusive"])
                                   if name == "direction" else
                                   (book["intervention"]["first_joint_t"], book["intervention"]["last_joint_t_inclusive"]))
                    source, source_sha = ACTOR / entry["actor_relative_path"], entry["source_sha256"]
            alternate = "teacher" if name == "teacher" else "student"
            yield dict(group=name, case=base, seed=seed, seat=seat, start=start, terrain=entry["terrain"],
                       source=source, source_sha=source_sha, registry={str(path.relative_to(ROOT)):sha(path)},
                       root_enemy_mode="independently_called", fixed_enemy=None,
                       parent=branch(folder/f"{base}_parent.json.gz","parent",start,last,seat,expected,sha(path),source_sha),
                       alternate=branch(folder/f"{base}_{alternate}.json.gz",alternate,start,last,seat,expected,sha(path),source_sha))


def earlier_window_cases():
    folder=ACTOR/"own_search_bridge_dev/teacher_window_dev"
    path=folder/"protocol.json"
    for r in read(path)["cases"]:
        seat=r["seat"]
        arms={}
        source=ROOT/r["source"]
        require(sha(source)==r["source_sha"],"teacher source changed")
        fixed=clean(read(source)["actions"][150][1-seat])
        for arm in ("parent","teacher"):
            f=folder/f"{r['seed']}_{seat}_{arm}.json.gz"
            d=read(f)
            require(d["protocol"]==sha(path) and d["seat"]==seat and d["arm"]==arm,"whole team teacher identity")
            tape,checks=d["tape"],d["checks"]
            require([v["t"] for v in checks]==list(range(150,158)), "incomplete whole team teacher window")
            require(clean(tape[0][1-seat])==fixed,"recorded root enemy mismatch")
            shadows={}
            for j,v in enumerate(checks):
                require(clean(tape[j][seat])==clean(v["submitted"]), "submitted teacher action differs from tape")
                if arm=="parent":
                    require(clean(v["submitted"])==clean(v["baseline"]), "baseline mismatch")
                shadows[v["t"]]=clean(v["baseline"])
            calls=[{"step":v["step"],"kind":"depth_zero_return_base" if v["zero_depth"] else "other",
                    "process_cpu_seconds":v["cpu"],"public_wall_seconds":v["wall"]} for v in d["enemy"]]
            arms[arm]=dict(tape=tape, scores=d["scores"], shadow=shadows, clock=clock(calls,151,len(tape)-1),
                           files={str(f.relative_to(ROOT)):sha(f)})
        yield dict(group="whole_team_teacher",case=f"{r['seed']}_{seat}",seed=r["seed"],seat=seat,start=150,
                   terrain=r["terrain"],source=source,source_sha=r["source_sha"],registry={str(path.relative_to(ROOT)):sha(path)},
                   root_enemy_mode="frozen_recorded_order",fixed_enemy=fixed,parent=arms["parent"],alternate=arms["teacher"])
    folder=ACTOR/"master3s2_policy_distill_dev/native_response"
    path=folder/"roots.json"
    for r in read(path)["cases"]:
        seat, start, source_sha=r["seat"],r["root"]["t"],r["source_sha256"]
        def distill_arm(name):
            return branch(folder/f"{r['id']}_{name}.json.gz",name,start,start,seat,r["root"]["proposals"],sha(path),source_sha)
        yield dict(group="source_distill",case=r["id"],seed=r["seed"],seat=seat,start=start,terrain=r["terrain"],
                   source=ROOT/r["source"],source_sha=source_sha,registry={str(path.relative_to(ROOT)):sha(path)},
                   root_enemy_mode="independently_called", fixed_enemy=None,parent=distill_arm("parent"),
                   alternate=distill_arm("student"))


def signed_cases():
    index = ACTOR / "joint_response_critic_dev/signed_roots.json.gz"
    rent = BASE / "ground_rent_native_evidence/roots.json.gz"
    rent_ids = {r["case"] for r in read(rent)}
    registry = {str(p.relative_to(ROOT)):sha(p) for p in (index, rent)}
    for r in read(index)["rows"]:
        seat, start = r["seat"], r["t"]
        arms = {}
        for arm in ("parent", "alternative"):
            spec = r["arms"][arm]
            path, receipt = ROOT / spec["action_file"], ROOT / spec["receipt"]
            require(sha(path)==spec["action_sha256"] and sha(receipt)==spec["receipt_sha256"], "signed arm identity")
            d = read(path)
            if isinstance(d,list):
                tape = d
            elif "joint0" in d:
                tape = [d["joint0"], *[h["joint"] for h in d["history"]]]
            elif "orders" in d:
                tape = d["orders"]
            else:
                a = d["arms"]["v1" if arm=="parent" else "candidate"]
                tape = [a["first_joint"],a["last_joint"]]
            require(len(tape)-1 == spec["calls"] and start+len(tape)==spec["terminal_step"], "signed call scope")
            require(clean(tape[0][seat])==clean(r[arm]) and clean(tape[0][1-seat])==clean(r["root_opponent"]),
                    "signed first joint action")
            arms[arm] = dict(tape=tape,scores=r[f"{arm}_scores"],shadow={start:clean(r["parent"])},
                             clock={"calls":spec["calls"],"first_call": start+1, "zero_depth":None,
                                    "coverage":"signed registry carries call count; per-call native depths not harmonized"},
                             files={str(path.relative_to(ROOT)):sha(path),str(receipt.relative_to(ROOT)):sha(receipt)})
        yield dict(group="signed_root", case=r["case"], seed=r["seed"], seat=seat, start=start,
                   terrain=r["terrain"], source=ROOT/r["source"], source_sha=r["source_sha256"],
                   root_enemy_mode="frozen_recorded_order", fixed_enemy=clean(r["root_opponent"]), registry=registry,
                   signed_family=r["family"], single_planter_rent_subset=r["case"] in rent_ids,
                   parent=arms["parent"], alternate=arms["alternative"])


def audit(c):
    raw = read(c["source"])
    require(sha(c["source"])==c["source_sha"] and raw["seed"]==c["seed"] and
            raw["initial_terrain_u8_sha256"]==c["terrain"] and raw["candidate_seat"]==c["seat"], "source identity")
    start, seat = c["start"], c["seat"]
    require(0 <= start < len(raw["actions"]), "unavailable source prefix")
    if c["fixed_enemy"] is not None:
        require(clean(raw["actions"][start][1-seat])==c["fixed_enemy"], "fixed root enemy not historical order")
    arms = [c["parent"], c["alternate"]]
    # The reference interpreter's worker and tree identifiers are process-global.
    # Reconstruct each branch from initialization through terminal before creating the other.
    tapes = [a["tape"] for a in arms]
    exposure = next((t for t,p in sorted(arms[1]["shadow"].items()) if
                     p != clean(tapes[1][t-start][seat])),None)
    signatures, scores = [], []
    for tape,arm in zip(tapes,arms):
        env=begin(raw)
        for prior in raw["actions"][:start]:
            require(not env.done, "source ends inside prefix")
            env.step(prior)
        require(env.states[0].observation.step==start, "wrong registration step")
        states=[]
        for joint in tape:
            require(not env.done, "archived orders outlast official game")
            states.append(digest(physical(env)))
            env.step(joint)
        states.append(digest(physical(env)))
        signatures.append(states)
        sc=[p.reward for p in env.states]
        require(env.done and all(p.status=="DONE" for p in env.states) and sc==arm["scores"], "official terminal mismatch")
        scores.append(sc)
    require(signatures[0][0]==signatures[1][0], "official replay differs at registered root")
    state = next((start+i for i,(p,b) in enumerate(zip(*signatures)) if p!=b),None)
    own=enemy=None
    root=None
    for i,orders in enumerate(zip(*tapes)):
        t=start+i
        p,b=[[clean(a) for a in joint] for joint in orders]
        different_own,different_enemy=p[seat]!=b[seat],p[1-seat]!=b[1-seat]
        if t==exposure:
            views=[states[i] for states in signatures]
            root={"both_live":True,"state_sha256":views,
                  "entire_prior_joint_prefix_equal":own is None and enemy is None,
                  "no_previous_official_state_difference":state is None or state>=t,
                  "state_equal":views[0]==views[1],"current_enemy_equal":not different_enemy,
                  "physical_own_different":different_own,
                  "parent_order_matches_alternate_shadow":p[seat]==arms[1]["shadow"][t],
                  "current_joint_sha256":[digest(p),digest(b)]}
        if different_own and own is None: own=t
        if different_enemy and enemy is None: enemy=t
    checks = ("both_live","entire_prior_joint_prefix_equal","no_previous_official_state_difference",
              "state_equal","current_enemy_equal","physical_own_different","parent_order_matches_alternate_shadow")
    passed = bool(root and all(root[k] for k in checks))
    if exposure is None: verdict="no_executed_alternate_intervention"
    elif root is None: verdict="parent_branch_already_ended"
    elif not root["entire_prior_joint_prefix_equal"] or not root["no_previous_official_state_difference"]:
        verdict="prior_branches_diverged"
    elif not root["current_enemy_equal"]: verdict="simultaneous_different_enemy"
    elif passed: verdict="matched_prefix_and_root_enemy"
    else: verdict="root_controls_disagree"
    margins=[s[seat]-s[1-seat] for s in scores]
    return dict(group=c["group"],case=c["case"],seed=c["seed"],seat=seat,start=start,
                signed_family=c.get("signed_family"),single_planter_rent_subset=c.get("single_planter_rent_subset",False),
                terrain=c["terrain"],source=str(c["source"].relative_to(ROOT)),source_sha256=c["source_sha"],
                registries=c["registry"],arm_inputs=[a["files"] for a in arms],root_enemy_mode=c["root_enemy_mode"],
                prior_archived_joint_sha256=digest(raw["actions"][:start]),first_actual_intervention=exposure,
                first_between_arm_own_order=own,first_between_arm_enemy_order=enemy,first_official_state_difference=state,
                first_arm_terminal_steps=[start+len(tape) for tape in tapes],root_at_exposure=root,
                prefix_verdict=verdict,free_from_preexisting_divergence_and_same_root_enemy=passed,
                official_scores_physical=scores,official_margins=[round(v,2) for v in margins],
                delta_official_margin=round(margins[1]-margins[0],2),original_clock=[a["clock"] for a in arms])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group",choices=["all","windows","earlier","signed"],default="all")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,
                        help="repo-relative paths plus SHA256s for one or more new pairs in the comparisons/original_calls format")
    args=parser.parse_args()
    rows=[]
    groups={"windows":[window_cases], "earlier":[earlier_window_cases], "signed":[signed_cases],
            "all":[window_cases,earlier_window_cases,signed_cases]}
    calls=groups[args.group]
    if args.manifest:
        require(args.group=="all","--manifest selects its own group")
        def custom():
            for r in read(args.manifest)["cases"]:
                p=ROOT/r["parent"]
                b=ROOT/r["alternate"]
                require(sha(p)==r["parent_sha256"] and sha(b)==r["alternate_sha256"],"custom arm digest")
                yield dict(group="custom",case=r["case"],seed=r["seed"],seat=r["seat"],start=r["start"],
                           terrain=r["terrain"],source=ROOT/r["source"],source_sha=r["source_sha256"],
                           registry={args.manifest.name:sha(args.manifest)},root_enemy_mode="independently_called",
                           fixed_enemy=None,parent=branch(p,"parent",r["start"],r["last"],r["seat"]),
                           alternate=branch(b,"student",r["start"],r["last"],r["seat"]))
        calls=[custom]
    for factory in calls:
        for case in factory():
            rows.append(audit(case))
    doc={"schema":1,"arm_order":["parent","alternate"],"source_scope":args.group,"rows":rows,
         "verdict_counts":dict(Counter(r["prefix_verdict"] for r in rows))}
    output=json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    tmp=args.output.with_suffix(args.output.suffix+".part")
    tmp.write_bytes(gzip.compress(output,mtime=0) if args.output.suffix==".gz" else output+b"\n")
    tmp.replace(args.output)
    print(len(rows),doc["verdict_counts"])


if __name__=="__main__":
    main()

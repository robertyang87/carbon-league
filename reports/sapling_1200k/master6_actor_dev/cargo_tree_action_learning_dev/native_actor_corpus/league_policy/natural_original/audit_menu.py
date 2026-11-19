"""Verify actual returned public orders against their conditional distribution after a hosted game.

Usage: python audit_menu.py GAME_DIR --arm candidate --out PATH
Use parent for the sampled control or greedy_parent for the deterministic control.
No second game, random-choice matching, or search by the opponent is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from export_behavior import FOLDER, HERE, ROOT, STEPS, PARENT_SHA

sys.path.insert(0, str(ROOT / FOLDER))
import collect

PACKS = {
    "candidate": (HERE/"candidate_sampled.py", "851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"),
    "parent": (HERE/"parent_sampled.py", "93d70be43a8cc358c05e4ac3d0870bded1e720aa539208ed0550523fc718e789"),
    "greedy_parent": (collect.PARENT, PARENT_SHA),
}


def check(value, message):
    if not value:
        raise ValueError(message)


class MenuDraw:
    """Inspect the two joint commands exactly at the deployed RNG call, before selection."""
    def __init__(self):
        self.menu = None

    def random(self):
        local = sys._getframe(1).f_locals
        check(self.menu is None, "Multiple draws from one scheduled root")
        slot, best, second = (int(local[k]) for k in ("slot", "best", "second"))
        probs = list(map(float, local["probs"]))
        trace = local["trace"][slot]
        legal = trace[2].tolist()
        opts = [dict(local["base_cmd"]), dict(local["alt_cmd"])]
        check(len(probs)==2 and all(0<p<1 for p in probs) and abs(sum(probs)-1.) < 1e-12,
              "Invalid conditional binary probabilities")
        check(best != second and legal[best] and legal[second] and opts[0]!=opts[1],
              "Duplicated or illegal conditional support")
        self.menu = dict(options=opts, probabilities=probs, slot=slot, actions=[best,second],
                         native_logits=local["logits"].tolist(), legal=legal, route=bool(local["route"]),
                         prefix=local["baseline"][:slot].tolist())
        # This offline return enumerates the default. The hosted draw is identified
        # exclusively by its actually emitted order matching exactly one member.
        return 0.


def per_order(actor, arm, observation, cfg, issued):
    step = observation["step"]
    sampled = arm != "greedy_parent"
    draw = MenuDraw()
    if sampled and step != 0:
        actor.mod._behavior_rng = draw
    ordinary = actor.act(observation,cfg)
    if draw.menu is not None:
        check(sampled and step in STEPS, f"An ordinary step exposed a menu at {step}")
        menu=draw.menu
        check(ordinary==menu["options"][0], f"Native default differs at {step}")
        matches=[i for i,cmd in enumerate(menu["options"]) if issued==cmd]
        check(len(matches)==1, f"Sent order is outside the deployed support at step {step}")
        pick=matches[0]
        return dict(step=step, supported=True, **menu, choice=pick, issued=issued,
                    logp=math.log(menu["probabilities"][pick]))
    check(ordinary==issued, f"Singleton-support order mismatch at step {step}")
    if sampled and step in STEPS:
        return dict(step=step, supported=False, options=[ordinary], probabilities=[1.],
                    choice=0, issued=issued, logp=0.)
    return None


def open_live(registration):
    path=collect.FIT/"ladder_live/live.py"
    spec=importlib.util.spec_from_file_location("league_menu_official_replay",path)
    live=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    # Only the configuration and replay observation builders are used.
    live.configure(registration, registration.parent/"attempt.json", registration.parent/"unused.lock")
    return live


def verify_menu(run_dir, arm, persist=False):
    folder=Path(run_dir)
    reg_file, receipt_file, turns_file=(folder/f for f in ("registration.json","receipt.json","turns.jsonl"))
    reg=json.loads(reg_file.read_bytes()); receipt=json.loads(receipt_file.read_bytes())
    reg_sha=collect.sha_file(reg_file); receipt_sha=collect.sha_file(receipt_file)
    pack, signed=PACKS[arm]
    seat,seed=reg["candidate_seat"],reg["seed"]
    entry=reg["files"]["candidate"]
    check(seat in (0,1) and reg["opponent_seat"]==1-seat, "Invalid seats")
    check(entry["sha256"]==collect.sha_file(ROOT/entry["path"])==collect.sha_file(pack)==signed,
          f"Wrong {arm} registered package")
    check(receipt["protocol_sha256"]==reg_sha and seat==receipt["candidate_seat"] and seed==receipt["seed"],
          "Receipt/registration mismatch")
    ledger_sha=collect.sha_file(turns_file) if turns_file.exists() else None
    check(receipt.get("turns_sha256")==ledger_sha, "Hosted turn ledger digest mismatch")
    doc=dict(schema="sapling-natural-action-support/1",arm=arm,seed=seed,seat=seat,
             match_status=receipt["status"],deployment_sha256=signed,tape_sha256=ledger_sha,
             registration_sha256=reg_sha,receipt_sha256=receipt_sha,official_scores_by_seat=None,
             n_steps=0,matched_all_steps=False,scheduled=0,supported=0,unsupported=0,
             alternative_count=0,ordinary=0,unchecked_returns=[],per_event=[],official_transitions=0)
    live=open_live(reg_file)
    if turns_file.exists():
        env,cfg=live.official()
        actor=collect.Player(pack)
        bank=[5.,5.]
        lines=turns_file.read_bytes().splitlines()
        for i,line in enumerate(lines):
            row=json.loads(line)
            check(not env.done and row["step"]==i==env.states[0].observation.step, f"Invalid turn {i}")
            both=[live.observe(env,s,bank[s]) for s in (0,1)]
            expected_sha=[hashlib.sha256(live.encoded(o)).hexdigest() for o in both]
            check(expected_sha==row["public_observation_sha256"], f"Invalid public observation at {i}")
            calls=row["calls"]
            call=calls[seat]
            if call["status"]=="ok":
                check(call["response"]["step"]==i, f"Wrong response step at {i}")
                ev=per_order(actor,arm,both[seat],cfg,call["response"]["action"])
                if ev is None:
                    doc["ordinary"]+=1
                else:
                    ev["public_observation_sha256"]=expected_sha[seat]
                    doc["scheduled"]+=1; doc["supported"]+=int(ev["supported"])
                    doc["unsupported"]+=int(not ev["supported"]); doc["alternative_count"]+=int(ev["choice"]==1)
                    doc["per_event"].append(ev)
                doc["n_steps"]+=1
            else:
                doc["unchecked_returns"].append(dict(step=i,status=call["status"]))
            bank=list(row["overage_remaining_seconds"])
            if all(c["status"]=="ok" for c in calls):
                env.step([c["response"]["action"] for c in calls])
                doc["official_transitions"]+=1
            elif i+1 != len(lines):
                raise ValueError("Ledger continues after failed request")
        if receipt["status"]=="complete":
            scores=[s.reward for s in env.states]
            check(env.done and [s.status for s in env.states]==["DONE","DONE"] and
                  scores==receipt["official_scores_by_seat"],"Official terminal scores differ")
            doc["official_scores_by_seat"]=scores
            doc["matched_all_steps"]=not doc["unchecked_returns"] and doc["n_steps"]==doc["official_transitions"]
    check(doc["official_transitions"]==receipt["official_transitions"],"Transition count differs")
    check(receipt["status"]!="complete" or doc["matched_all_steps"],"Complete match lacks verified orders")
    doc["status"]="complete" if doc["matched_all_steps"] else (
        "verified_returned_prefix" if doc["n_steps"] else "no_valid_returned_order")
    if persist:
        dest=folder/"menu_audit.json" if persist is True else Path(persist)
        collect.write_json(dest,doc)
    return doc


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game",type=Path);parser.add_argument("--arm",choices=PACKS,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=verify_menu(args.game,args.arm,persist=args.out or False)
    print(json.dumps({k:v for k,v in result.items() if k!="per_event"},sort_keys=True))

"""Frozen base-only recruit interventions on two exposed parent tapes, original responsive examiner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CONDITIONAL = HERE.parent / "conditional"
sys.path.insert(0, str(CONDITIONAL))
import responses as r

PLAN = HERE / "response_protocol.json"


def bind():
    r.PLAN = PLAN
    r.P = r.load(PLAN)
    r.HERE = HERE
    r.PACK["candidate"] = HERE / "candidate.py"


def preregister():
    old = r.load(CONDITIONAL / "match_protocol.json")
    offline = r.load(HERE / "result.json")
    if offline["candidate_sha256"] != r.sha(HERE / "candidate.py") or offline["counts"]["changed_base"] <= 0:
        raise ValueError("missing offline intervention")
    r.write(PLAN, dict(parent=offline["parent_sha256"], candidate=offline["candidate_sha256"], original=old["original"],
        exposures=old["exposures"], offline_sha256=r.sha(HERE/"result.json"), shared_harness_sha256=r.sha(CONDITIONAL/"responses.py"),
        experiment_sha256=r.sha(Path(__file__)), updated_gates=False,
        root_rule="On each of these two fixed tapes: latest time step in [1,269] where base recruitment commands differ, all worker commands match parent, and the accepted hires at the changed bases differ on same recorded opponent command. Remain on this tape and seat; absent exposure fails gate.",
        response="Replay both public histories. At root take each candidate's actual joint command against the recorded single opponent action. Subsequently the original Master3s2 and corresponding agent both act to real terminal. Share initial measured warm cost and seeded RNG per root; rotate cores by seat, retain native clock/search.",
        gate="On BOTH exposed maps strictly improve candidate-relative terminal margin, and nonnegative change to own gross operating revenue (own deliveries + tree absorption) and to tree-income margin relative to the opponent. Both branches must have paired initial RNG and warm cost, normal completion, no error or hard timeout, and the root accepted hires must differ. Any missing/failed condition withholds fresh games. No reroll.",
        interpretation="Two outcome-conditioned native parent tapes: mechanism screen only. No fresh seeds authorized internally."))


def accepted(sim, joint, seat):
    child = r.FastCarbon(sim.config)
    child.set_state(sim.get_state())
    child.cash = r.CashEvents(child.cash)
    before = int(child.worker_counters[seat])
    child.step(*joint)
    hires = sorted([[e[3],e[0],e[2]] for e in child.cash.events if e[1] == seat and e[0].startswith("hire_")])
    if len(hires) != int(child.worker_counters[seat])-before:
        raise ValueError("root accepted hire ledger mismatch")
    return hires


def choose(case):
    raw = r.inputs(case)
    models = {k:r.live.module(p,"select_same_"+k) for k,p in r.PACK.items()}
    cfg = raw["game_configuration"]
    sim = r.FastCarbon(cfg).reset(seed=raw["seed"])
    if list(sim.carbon) != raw["initial_full_carbon"]:
        raise ValueError("map identity mismatch")
    seat = raw["candidate_seat"]
    last = dict(**case,t=None)
    changes = []
    for t,joint in enumerate(raw["actions"][:270]):
        view = sim.obs(seat)
        orders = {k:r.live.clean(m.agent(view,cfg)) for k,m in models.items()}
        if r.live.clean(joint[seat]) != orders["parent"]:
            raise ValueError(f"frozen parent command mismatch at t={t}")
        a,b = orders["parent"],orders["candidate"]
        diff = sorted(u for u in (a.keys()|b.keys()) if a.get(u)!=b.get(u))
        if diff and t:
            bases = view["players"][seat][1]
            if all(u in bases for u in diff):
                treatments = {}
                for k,actions in orders.items():
                    action_pair = list(joint)
                    action_pair[seat] = actions
                    treatments[k] = accepted(sim,action_pair,seat)
                if treatments["parent"] != treatments["candidate"]:
                    last = dict(**case,t=t,**orders,changed_bases=diff,accepted_root= treatments,opponent=joint[1-seat])
                    changes.append(t)
        if not t and diff: raise ValueError("opening differs")
        sim.step(*joint)
    return dict(**last,all_eligible_ticks=changes)


def run():
    bind()
    if r.sha(CONDITIONAL/"responses.py") != r.P["shared_harness_sha256"] or r.sha(Path(__file__))!=r.P["experiment_sha256"]:
        raise ValueError("source drift")
    cases=r.load(HERE/"response_roots.json")["cases"]
    if any(c["t"] is None for c in cases):
        r.write(HERE/"response_gate.json",dict(protocol=r.sha(PLAN),eligible=False,reason="missing actual recruitment exposure",cases=cases))
        return
    results=[]
    with r.LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for c in cases:
            warm=HERE/f"response_warm_{c['seed']}_{c['seat']}.json"
            if not warm.exists():
                with ProcessPoolExecutor(1) as ex: cost=ex.submit(r.native_warm).result()
                r.write(warm,dict(protocol=r.sha(PLAN),examiner=r.P["original"],cost=cost))
            receipt=r.load(warm)
            if receipt["protocol"]!=r.sha(PLAN) or receipt["examiner"]!=r.P["original"]: raise ValueError("warm identity mismatch")
            cost=receipt["cost"]; seat=c["seat"]
            with ProcessPoolExecutor(2) as ex:
                old,new=ex.map(r.arm,[(c,"parent",seat,cost),(c,"candidate",1-seat,cost)])
            initialized=old["initial_rng"]==new["initial_rng"] and old["default_warm_cost"]==new["default_warm_cost"]==cost
            def gross(arm,s):
                events=arm["cash"]["events"][s]
                return events.get("tree_absorption",0.)+events.get("own_delivery",0.)
            dmargin=new["margin"]-old["margin"]
            dtree=new["cash"]["tree_income_margin"]-old["cash"]["tree_income_margin"]
            grossown=gross(new,seat)-gross(old,seat); grossopp=gross(new,1-seat)-gross(old,1-seat)
            changed=sum(a[1-seat]!=b[1-seat] for a,b in zip(old["tape"][1:],new["tape"][1:]))
            hard_overs=[sum(call["public_wall_seconds"]>3 for call in z["calls"]) for z in (old,new)]
            results.append(dict(seed=c["seed"],seat=seat,t=c["t"],accepted_root=c["accepted_root"],paired_initialization=initialized,
                parent_margin=old["margin"],candidate_margin=new["margin"],delta_margin=dmargin,
                delta_own_gross_revenue=grossown,delta_opponent_gross_revenue=grossopp,delta_tree_margin=dtree,
                delta_delivery_margin=new["cash"]["delivery_margin"]-old["cash"]["delivery_margin"],
                original_response_diff_turns=changed,hard_timeout_counts=hard_overs,calls=[len(z["calls"]) for z in (old,new)],
                cash_by_seat={z["arm"]:z["cash"]["events"] for z in (old,new)}, branch_ok=bool(initialized and not any(hard_overs) and dmargin>0 and grossown>=0 and dtree>=0)))
            print(c["seed"],results[-1]["delta_margin"],grossown,dtree,flush=True)
    r.write(HERE/"response_gate.json",dict(protocol=r.sha(PLAN),roots_sha256=r.sha(HERE/"response_roots.json"),eligible=all(c["branch_ok"] for c in results),results=results))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("freeze","select","run"));args=p.parse_args()
    if args.mode=="freeze":preregister()
    elif args.mode=="select":
        bind()
        if r.sha(Path(__file__))!=r.P["experiment_sha256"]:raise ValueError("experiment hash mismatch")
        cases=[choose(c) for c in r.P["exposures"]]
        r.write(HERE/"response_roots.json",dict(protocol=r.sha(PLAN),cases=cases))
        print([(c["seed"],c["seat"],c["t"],c["all_eligible_ticks"]) for c in cases],flush=True)
    else:run()

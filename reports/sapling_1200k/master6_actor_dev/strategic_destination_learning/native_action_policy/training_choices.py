"""Check which guarded alternatives really change an immediate physical outcome on training terrain."""
import argparse
from collections import Counter
import copy
import importlib.util
import random
from pathlib import Path
import sys

import numpy as np
from train import HERE, STUDY, entry, read, save, sha, check
from check import expected

ACTOR=STUDY.parent
SAP=ACTOR.parent
src=ACTOR/"cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/menu_coverage/coverage.py"
spec=importlib.util.spec_from_file_location("signed_economy_ledger",src)
ledger=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)
METRICS=("warehouse_deposit","plants","seizures","own_root_load_on_dead","opponent_cargo_looted")


def audit(work):
    cases=[]
    with np.load(HERE/"head.npz", allow_pickle=False) as archive:
        head = {k: archive[k] for k in archive}
    sources=read(STUDY/"sources.json")
    for card in sources["cards"]:
        if card["split"]!="train":
            continue
        archive=STUDY.parents[3]/card["path"]
        check(sha(archive)==card["sha256"],"historical source changed")
        tape=read(archive)
        seat=card["expert_seat"]
        menu=read(work/(card["sha256"]+".json.gz"))
        rr={r["t"]:r for r in menu["rows"] if r["possible"]}
        random.seed(card["seed"])
        np.random.seed(card["seed"])
        env=entry.official_rules(entry.ROOT)("carbon",configuration={**tape["game_configuration"],"randomSeed":card["seed"]})
        from zerosum_env.envs.carbon import idgen
        sim=ledger.FastCarbon(env.configuration).reset(seed=card["seed"])
        ledger.check_board(sim,env)
        for t,joint in enumerate(tape["actions"]):
            if t in rr:
                r=rr[t]
                root=sim.get_state()
                before=dict(idgen._id_dict)
                orders=[r["base_order"],*r["orders"]]
                unique={tuple(sorted(o.items())):o for o in orders}
                distribution=expected(r,head)
                events=[]
                for order in unique.values():
                    choices=list(joint)
                    choices[seat]=order
                    branch=copy.copy(env)
                    branch.states=copy.deepcopy(env.states)
                    branch.steps=env.steps[:-1]+[branch.states]
                    branch.logs=[]
                    branch.error=env.error.copy()
                    branch.step(choices,[])
                    reference=ledger.FastCarbon(env.configuration).set_state(root)
                    accounting=ledger.account(reference,choices,root)
                    ledger.check_board(reference,branch)
                    idgen._id_dict.clear()
                    idgen._id_dict.update(before)
                    me=accounting["by_seat"][seat]
                    rival=accounting["by_seat"][1-seat]
                    mass,prior=distribution[tuple(sorted(order.items()))]
                    events.append(dict(order=order,probability=float(mass),prior_probability=float(prior),metrics={k:me[k] for k in METRICS},
                                       rival_deposit=rival["warehouse_deposit"]))
                changed=[k for k in METRICS if len({round(e["metrics"][k],5) for e in events})>1]
                if len({round(e["rival_deposit"],5) for e in events})>1:
                    changed.append("rival_deposit")
                cases.append(dict(terrain=card["terrain"],seed=card["seed"],seat=seat,step=t,role=r["role"],
                                  public_opponent=card["opponent"],source=card["sha256"],root_opponent=joint[1-seat],
                                  changed=changed,events=events))
            sim.step(*joint)
            env.step(joint,[])
        check(env.done and [s.reward for s in env.states]==tape["official_score_by_seat"],"end disagrees")
        ledger.check_board(sim,env)
    save(HERE/"training_choices.json.gz",cases)
    groups={}
    for terrain in sorted({r["terrain"] for r in cases}):
        group=[r for r in cases if r["terrain"]==terrain]
        groups[terrain]=dict(seed=group[0]["seed"],roots=len(group),counts=Counter(k for r in group for k in r["changed"]))
    save(HERE/"training_coverage.json",dict(source=sha(STUDY/"sources.json"),package=sha(HERE/"candidate.py"),
         complete_roots=len(cases),unique_physical_commands=sum(len(r["events"]) for r in cases),
         metrics=Counter(k for r in cases for k in r["changed"]),by_terrain=groups,rows_sha=sha(HERE/"training_choices.json.gz"),
         qualification="Student policy proposals on historical original-expert seats. Opponent root order belongs to the other historical source package. Single-step official receipts only; no original terminal response or fitness estimate."))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--work",required=True,type=Path)
    audit(parser.parse_args().work)

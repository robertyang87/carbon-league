"""Run four persistent public agents only on logged BC boards; no alternative actions advance play."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
REPL=HERE.parent
STRATEGY=REPL.parent
M6=STRATEGY.parent
CARGO=M6/"cargo_tree_action_learning_dev"
sys.path[:0]=[str(STRATEGY/"delivery_interception"),str(CARGO/"native_actor_corpus/league_policy")]
import analyze
import collect
from zerosum_env import make

NAMES=("bc","parent","normalized_zero","literal_zero")
PAIRS=(("bc","parent"),("bc","normalized_zero"),("bc","literal_zero"),("normalized_zero","literal_zero"))


def sha(x):
    return hashlib.sha256(x).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def money(obs,s):
    return obs["players"][s][0]


def role(obs,seat,uid):
    _,bases,workers,_=obs["players"][seat]
    if uid in bases:
        return "BASE"
    return workers[uid][2]


def subject_key(uid):
    return int(uid.rsplit("-",1)[-1])


def run(seat,plan,settle):
    seed=plan["seed"]
    ticket=next(r for r in settle["archives"] if (r["seed"],r["seat"])==(seed,seat))
    game=next(g for g in settle["games"] if (g["seed"],g["seat"],g["arm"])==(seed,seat,"bc"))
    zip_path=REPL/"hosted"/f'{ticket["id"]}.zip'
    check(sha(zip_path.read_bytes())==ticket["sha256"],"ticket digest")
    with zipfile.ZipFile(zip_path) as z:
        rec=z.read("bc/receipt.json"); reg=json.loads(z.read("bc/registration.json")); tape=z.read("bc/turns.jsonl")
    check(sha(rec)==game["receipt_sha256"] and sha(tape)==json.loads(rec)["turns_sha256"],"receipt or commands changed")
    check(reg["files"]["candidate"]["sha256"]==plan["bc_sha256"] and reg["candidate_seat"]==seat and
          reg["seed"]==seed and game["official_transitions"]==299 and game["status"]=="complete","game identity")
    settings=analyze.read(CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz")["game_configuration"]
    cfg={**settings,"agentTimeout":20,"actTimeout":1,"runTimeout":1500}
    pubcfg={k:v for k,v in cfg.items() if k!="randomSeed"}
    np.random.seed(seed)
    env=make("carbon",configuration={**cfg,"randomSeed":seed})
    sim=analyze.FastCarbon(cfg).reset(seed=seed)
    check(sha(bytes(np.array(sim.carbon,np.uint8)))==game["terrain"]==sha(bytes(env.states[0].observation.full_carbon)),"map differs")
    sim.cash=analyze.CashEvents(sim.cash)
    variants={k:collect.Player(CARGO/"candidate.py" if k=="parent" else STRATEGY/"bc.py") for k in NAMES}
    check(len({id(v.modules["cnet2_np"]) for v in variants.values()})==len(variants),"shared policy module")
    for k,scores in (("normalized_zero",(1.,0.)),("literal_zero",(.75,0.))):
        variants[k].mod._HYBRID.update(expert_weight=scores[0],student_weight=scores[1])
    banks=[5.,5.]
    counts={f"{a}/{b}":{"different_roots":0,"different_uids":Counter(),"changed_slots":Counter(),
                            "actual_bc_cash_events_by_changed_unit":Counter(),"actual_bc_cash_by_changed_unit":Counter()}
            for a,b in PAIRS}
    rows=[]
    mismatch=[]
    total_slots=Counter()
    key_step=plan["time_windows"][str(seat)]
    for index,line in enumerate(tape.splitlines()):
        tick=json.loads(line)
        full=env.states[0].observation
        public=dict(step=full.step,player=seat,carbon=env.states[seat].observation.carbon,players=full.players,
                    trees=full.trees,remainingOverageTime=banks[seat])
        digest=sha((json.dumps(public,separators=(",",":"),allow_nan=False)+"\n").encode())
        check(tick["step"]==full.step==index and all(v["status"]=="ok" for v in tick["calls"]) and
              digest==tick["public_observation_sha256"][seat],"wrong applied observation")
        orders={k:p.act(public,pubcfg) for k,p in variants.items()}
        actual=[c["response"]["action"] for c in tick["calls"]]
        if orders["bc"]!=actual[seat]:
            mismatch.append(index)
        _,base,workers,_=public["players"][seat]
        total_slots["BASE"]+=len(base)
        total_slots.update(v[2] for v in workers.values())
        changed={}
        combined=set()
        for a,b in PAIRS:
            diff={u for u in orders[a].keys()|orders[b].keys() if orders[a].get(u,"NONE")!=orders[b].get(u,"NONE")}
            changed[a,b]=diff
            combined.update(diff)
        cash_before=[money(public,s) for s in (0,1)]
        sim.cash.events.clear()
        sim.step(*actual)
        env.step(actual)
        check(list(sim.cash)==[p[0] for p in env.states[0].observation.players],"cash mismatch")
        cs=sim.cash.events
        details=[]
        for uid in sorted(combined,key=lambda u:(role(public,seat,u),subject_key(u))):
            details.append(dict(uid=uid,role=role(public,seat,uid),
                a=[orders[k].get(uid,"NONE") for k in NAMES],
                actual_ledger=[e[:5] for e in cs if len(e)>3 and e[3]==uid and e[0]!="tree_absorption"]))
        for (a,b),diff in changed.items():
            count=counts[f"{a}/{b}"]
            count["different_roots"]+=int(bool(diff))
            count["different_uids"].update(diff)
            count["changed_slots"].update(role(public,seat,u) for u in diff)
            for e in cs:
                if len(e)>3 and e[1]==seat and e[3] in diff and e[0]!="tree_absorption":
                    count["actual_bc_cash_events_by_changed_unit"][e[0]]+=1
                    count["actual_bc_cash_by_changed_unit"][e[0]]+=e[2]
        entry=dict(step=index,public_sha256=digest,different=details)
        if key_step-10<=index<=key_step:
            # All active BC entities plus relevant actual arrivals; ALL arms compared on this shared public input.
            entry.update(orders=orders,official_actions=actual,cash_before=cash_before,cash_after=list(sim.cash),
                         actual_ledger=[e[:5] for e in cs if e[0]!="tree_absorption"],
                         worker_states=[{uid:list(v) for uid,v in p[2].items()} for p in public["players"]])
        rows.append(entry)
        banks=tick["overage_remaining_seconds"]
    check(len(rows)==299 and env.done and [x.reward for x in env.states]==game["official_scores_by_seat"],"official terminal differs")
    return dict(seat=seat,terrain=game["terrain"],ticket=ticket["sha256"],turns=sha(tape),parent_sha256=plan["parent_sha256"],
                bc_sha256=plan["bc_sha256"],official_scores=game["official_scores_by_seat"],
                check=dict(roots=len(rows),signed_public_inputs=len(rows),public_bc_mismatches=mismatch),
                total_active_unit_slots=dict(total_slots),comparisons=counts,rows=rows,
                official_candidate_timing=game["timing_by_seat"][seat],inprocess_comparator_own_timing={k:v.diagnostics() for k,v in variants.items()})


def main():
    plan=json.loads((HERE/"protocol.json").read_bytes())
    settlement_path=REPL/"hosted/settlement.json"
    check(sha(settlement_path.read_bytes())==plan["source_settlement_sha256"] and
          sha((STRATEGY/"bc.py").read_bytes())==plan["bc_sha256"] and
          sha((CARGO/"candidate.py").read_bytes())==plan["parent_sha256"],"frozen source differs")
    settlement=json.loads(settlement_path.read_bytes())
    cards=[]
    for seat in (0,1):
        card=run(seat,plan,settlement)
        rows=card.pop("rows")
        raw=json.dumps(rows,separators=(",",":"),allow_nan=False).encode()
        output=HERE/f"seat{seat}.json.gz"
        temp=output.with_suffix(".partial")
        temp.write_bytes(gzip.compress(raw,mtime=0))
        temp.replace(output)
        card["rows_sha256"]=sha(output.read_bytes())
        cards.append(card)
        print(seat,card["check"],{k:v["different_roots"] for k,v in card["comparisons"].items()},flush=True)
    result=dict(code_sha256=sha(Path(__file__).read_bytes()),protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
                actual_engine=sha(Path(analyze.FastCarbon.step.__code__.co_filename).read_bytes()),
                ledger_sha256=sha(Path(analyze.CashEvents.__setitem__.__code__.co_filename).read_bytes()),
                key_events_sha256=sha((HERE/"key_events.json").read_bytes()),seat_cards=cards,command_names=NAMES,
                ablation_reference=sha((STRATEGY/"partner_entry_audit/replay.py").read_bytes()),
                timing_scope="actual matched hosted requests separated from single-thread local in-process comparator timings")
    (HERE/"result.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")


if __name__=="__main__":
    main()

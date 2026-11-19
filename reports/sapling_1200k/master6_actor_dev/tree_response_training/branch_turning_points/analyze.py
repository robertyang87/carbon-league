"""Locate the earliest maximal one-step absolute paired cash-margin change on eight completed training roots."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
INTAKE = OUT.parent
BASE = INTAKE.parents[1]
sys.path[:0] = [str(BASE), str(BASE / "root_recovery_branches")]
from all_top5.context import visibility
from branches import begin, clean, public
from turning_point_ledger import CashEvents, FastCarbon

ARMS = ("pay", "defer")
LAGS = (10, 5, 3, 0)
PARENT = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
MASTER = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def read(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6)


def cents(amount):
    return int(round(amount * 100))


def own_margin(cash,seat):
    return cents(cash[seat]) - cents(cash[1-seat])


def fast_trace(raw, joint):
    sim = FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    sim.cash = CashEvents(sim.cash)
    turns = []
    for t, commands in enumerate(joint):
        before = list(sim.cash)
        sim.cash.events.clear()
        sim.step(*commands)
        events = list(sim.cash.events)
        after = list(sim.cash)
        for s in (0,1):
            need(close(before[s] + sum(e[2] for e in events if e[1] == s), after[s]), f"unreconciled cash at {t}/{s}")
        turns.append((before, after, events))
    return turns


def attribution(events, seat):
    by_seat = [Counter(),Counter()]
    for kind, target, amount, *_ in events:
        by_seat[target][kind] += amount
    kinds = by_seat[0].keys() | by_seat[1].keys()
    return {k: by_seat[seat][k] - by_seat[1-seat][k] for k in kinds}


def official_frames(raw, joint, trace, branch, samples):
    """Each selected marker uses pre-order public observations. Current commands and cash changes are separate."""
    env = begin(raw)
    out = {}
    for t, (cmds, (before, after, events)) in enumerate(zip(joint, trace, strict=True)):
        need(not env.done, f"unexpected official terminal before {t}")
        if t in samples:
            obs = public(env,0)
            players = obs["players"]
            views = []
            for seat in (0,1):
                local = public(env,seat)
                mask = visibility(players[seat][2],raw["game_configuration"]["size"])
                views.append({"physical_seat":seat,"visible_cells": [i for i,val in enumerate(mask) if val],
                              "carbon_on_visible_cells":[local["carbon"][i] for i,v in enumerate(mask) if v],
                              "remainingOverageTime":local["remainingOverageTime"]})
            need(obs["step"] == t, "off-by-one root")
            out[t] = {"t":t, "public_before_order":{"players":players,"trees":obs["trees"],"carbon_by_observer":views},
                      "actual_joint_orders":cmds, "cash_before_by_seat":before,"cash_after_by_seat":after,
                      "cash_events":events}
        env.step(cmds)
        need([cents(p[0]) for p in env.states[0].observation.players] == list(map(cents,after)),
             f"official/simulator cash differs on transition {t}")
    need(env.done and [x.status for x in env.states] == ["DONE","DONE"] and
         [x.reward for x in env.states] == branch["scores_by_physical_seat"], "official terminal differs")
    return out


def analyze_case(reg, label):
    folder = INTAKE / reg["case_directory"]
    source_file = INTAKE / "source" / Path(reg["source"]).name
    seed,seat,t0 = reg["seed"],reg["seat"],reg["step"]
    need(sha(source_file) == reg["source_sha256"] and read(folder/"training_label.json") == label and
         sha(folder/"root.json.gz") == label["root_sha256"] == reg["root_sha256"], "training source/root/label identity")
    raw,root = read(source_file),read(folder/"root.json.gz")
    root_commands = raw["actions"][t0]
    states = {}
    for arm in ARMS:
        br = label[arm]
        native = br["native_arm"]
        result = read(folder/f"live_{native}.json")
        p = folder / f"live_{native}_actions.json.gz"
        need(sha(p) == br["actual_action_tape_sha256"] == result["action_sha256"] and
             result["original_trajectory_sha256"] == reg["source_sha256"] and
             result["source_sha256"] == MASTER and result["pack_sha256"] == PARENT and
             sha(folder/"live_shared_init.json") == br["shared_stream_sha256"] == result["shared_init_sha256"] and
             result["score"] == br["scores_by_physical_seat"] and result["root"]["t"] == t0,
             "response pair identity, stream or action binding")
        tail = read(p)
        expected = root["v1" if native == "parent" else "alternative"]
        need(clean(tail[0][seat]) == expected and clean(tail[0][1-seat]) == clean(root_commands[1-seat]),
             "changed root joint action")
        joint = raw["actions"][:t0] + tail
        need(len(joint) == br["official_terminal_t"] and br["examiner_calls"] == len(tail)-1,
             "missing native call or terminal action")
        states[arm] = (joint, fast_trace(raw,joint))
    p,d = (states[a][1] for a in ARMS)
    n = min(len(p),len(d))
    # Currency is rounded each step; select in integer cents, so floating representation cannot break an exact tie.
    gaps = [own_margin(p[i][1],seat)-own_margin(p[i][0],seat)-own_margin(d[i][1],seat)+own_margin(d[i][0],seat)
            for i in range(t0,n)]
    peak = t0 + max(range(len(gaps)), key=lambda i:abs(gaps[i]))
    need(len(p) == len(d) and len(p) == raw["game_configuration"]["episodeSteps"]-1,
         "this first-intake diagnostic requires both normal horizon terminals")
    need(sum(gaps) == cents(label["target_delta_official_margin"]), "paired curve does not telescope to official score")
    which = [peak-i for i in LAGS if peak-i >= 0]
    checkpoints = {}
    channels = {}
    for a in ARMS:
        joint,trace = states[a]
        checkpoints[a] = official_frames(raw,joint,trace,label[a],which)
        channels[a] = attribution(trace[peak][2],seat)
    c = {k:round(channels["pay"].get(k,0)-channels["defer"].get(k,0),6) for k in sorted(channels["pay"].keys()|channels["defer"].keys())}
    need(close(sum(c.values()),gaps[peak-t0]/100), "unbalanced peak attribution")
    pay_gap, defer_gap = own_margin(p[peak][1],seat)-own_margin(p[peak][0],seat), own_margin(d[peak][1],seat)-own_margin(d[peak][0],seat)
    return {"case":reg["case_directory"],"seed":seed,"seat":seat,"root_t":t0,"worker":reg["uid"],"cell":reg["cell"],
            "paid":label["pay_intent_filled"],"debit":label["pay"]["root_debit"],"tree_first_tenure_net":label["pay"]["first_tenure_net"],
            "official_terminal_pay":label["pay"]["scores_by_physical_seat"],"official_terminal_defer":label["defer"]["scores_by_physical_seat"],
            "root_source_sha256":reg["source_sha256"],"root_sha256":reg["root_sha256"],
            "action_sha256":{a:label[a]["actual_action_tape_sha256"] for a in ARMS},
            "comparable_joint_turns":len(gaps),"terminal_delta_margin":round(sum(gaps)/100,2),"peak_turn":peak,
            "pay_step_margin":pay_gap/100,"defer_step_margin":defer_gap/100,"peak_paired_change":gaps[peak-t0]/100,
            "ties":sum(abs(q)==abs(gaps[peak-t0]) for q in gaps),"paired_gap_before":sum(gaps[:peak-t0])/100,
            "paired_gap_after":sum(gaps[:peak-t0+1])/100,"peak_channels":c,"contexts":checkpoints}


def main():
    selection,labels=read(INTAKE/"selection.json"),read(INTAKE/"training_labels.json")
    need(labels["selection_sha256"]==sha(INTAKE/"selection.json") and labels["protocol_sha256"]==sha(INTAKE/"protocol.json")
         and selection["source_receipt_sha256"]==sha(INTAKE/"source_receipt.json"),"unbound labels")
    by_root={r["root_sha256"]:r for r in labels["pairs"]}
    need(len(by_root)==len(selection["cases"])==8,"unexpected original intake scope")
    rows = [analyze_case(r,by_root[r["root_sha256"]]) for r in selection["cases"]]
    doc = dict(schema="sapling-offline-tree-response-cash-turns/1",training_labels_sha256=sha(INTAKE/"training_labels.json"),
               protocol_sha256=labels["protocol_sha256"],selection_sha256=labels["selection_sha256"],
               replay_script_sha256=sha(Path(__file__)),labeled_joint_turns=sum(r["comparable_joint_turns"] for r in rows),
               arms=16,records=rows)
    payload=json.dumps(doc,ensure_ascii=False,separators=(",",":"),sort_keys=True,allow_nan=False).encode()
    (OUT/"turns.json.gz").write_bytes(gzip.compress(payload,mtime=0))
    for r in rows:
        print(r["seed"],r["seat"],r["root_t"],r["terminal_delta_margin"],r["peak_turn"],r["peak_paired_change"],flush=True)


if __name__=="__main__":
    main()

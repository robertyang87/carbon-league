"""Replay physical learner commands and audit expert receipts without sampling a teacher policy."""
from collections import Counter
import gzip
import hashlib
import json
import math
import pathlib
import random

import numpy as np
import sys
import tempfile
import os

HERE = pathlib.Path(__file__).resolve().parent
STRATEGIC = HERE.parents[1]
ROOT = STRATEGIC.parents[3]
sys.path.insert(0, str(ROOT / "code/env_src/carbon_challenge-master"))
from zerosum_env import make

PLAN = "83b7a9257615d1b34b18802bb158aeeafe5838c899805fba774d035fe20a50e4"
DELTAS = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1), "NONE":(0,0)}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",",":"), allow_nan=False).encode()


def objsha(obj):
    return sha(encoded(obj))


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def expected_destinations(observation, orders):
    bases, workers = observation["players"][observation["player"]][1:3]
    check(isinstance(orders, dict) and orders.keys() <= (bases.keys() | workers.keys()), "foreign/dead command")
    for uid, action in orders.items():
        check(action in (("RECCOLLECTOR","RECPLANTER","NONE") if uid in bases else DELTAS), "action outside unit ABI")
    destinations = {}
    for uid, (index, *_rest) in workers.items():
        y,x = divmod(index,15)
        dy,dx = DELTAS[orders.get(uid,"NONE")]
        destinations[uid] = (y+dy)%15*15 + (x+dx)%15
    return destinations


def old_labels(pool, tickets):
    base = STRATEGIC.parent / "own_search_bridge_dev/teacher_window_dev/student"
    previous = load(base/"protocol.json")
    train = {x["sha256"]:x for x in pool["cards"] if x["split"] == "train"}
    new = {(x["source_sha256"], x["seat"], t) for x in tickets for t in range(32)}
    exact = []
    same_tape = 0
    new_hits = 0
    used = []
    for card in previous["cards"]:
        now = train.get(card["sha"])
        if now is None:
            continue
        check((card["seed"],card["seat"],card["terrain"]) == (now["seed"],now["candidate_seat"],now["terrain"]),
              "duplicate source SHA with contradictory physical identity")
        path = base / f"card_{card['id']:02d}.json.gz"
        evidence = load(path)
        check(evidence["source"] == now["sha256"],"historical teacher receipt binding")
        used.append({"path": str(path.relative_to(ROOT)), "sha256":sha(path.read_bytes())})
        for row in evidence["records"]:
            same_tape += 1
            new_hits += (card["sha"],card["seat"],row["t"]) in new
            if row["t"] in now["registered_times"]:
                exact.append(dict(seed=card["seed"], seat=card["seat"], step=row["t"],
                                  K=row["clock"]["last"][0], H=row["clock"]["last"][1],
                                  order_differs=row["teacher"]!=row["executed"]))
    check(same_tape == 32 and len(exact) == 4 and new_hits == 0,"historical sample join")
    return dict(prior_protocol_sha256=sha((base/"protocol.json").read_bytes()),
                receipts=used, exact_on_previous_strategic_roots=exact, same_tape_same_seat_window_roots=same_tape,
                overlaps_this_intake=new_hits, previous_window=[150,157],
                sampling="Historical original own-seat windows used predetermined private RNG streams and a warm search cost. This intake starts each original cold with natural sampling.",
                related_exclusions=["planter_teacher_window: selected planter commands on two other maps, t128–143",
                "postsearch/exact: recorded original playing the opposing seat",
                "joint_board_search/native_teacher: prescribed search candidate menu",
                "strong_teacher_dev: Apex V1 policy","build_sapling_cash_dagger and build_sapling_tree_dagger: Echo4 teacher"])


def inspect(card, record, protocol, tracker_class, belief_class):
    path = HERE / "traces" / f"{card['seed']}_{card['seat']}.json.gz"
    oldpath = ROOT / card["source_path"]
    tape = load(oldpath)
    trace = load(path)
    seed,seat = card["seed"],card["seat"]
    check(sha(path.read_bytes()) == record["sha256"] and sha(oldpath.read_bytes()) == trace["source_sha256"] == card["source_sha256"],
          "trace/native tape digest")
    check((trace["seed"],trace["seat"],trace["terrain"],trace["split"]) ==
          (seed,seat,card["terrain"],card["split"]),"root identity")
    check(trace["source_path"] == card["source_path"] and trace["teacher_sha256"] == protocol["teacher_sha256"] and
          trace["behavior_package_sha256"] == tape["pack_sha256"] == card["learner_pack_sha256"] and
          tape["candidate_seat"] == seat and trace["collector_sha256"] == sha((HERE/"correct.py").read_bytes()) and
          trace["protocol_sha256"] == PLAN and objsha(tape["actions"]) == card["body_actions_sha256"],"role/binary/action identity")
    check(trace["game_configuration"] == tape["game_configuration"] == card["game_configuration"],"game rules")
    check(trace["affinity"] == [0,1] and trace["torch_threads"] == 2 and math.isfinite(trace["cold_init_wall"]) and trace["cold_init_wall"]>0,"clock provenance")
    claim = load(path.with_suffix("").with_suffix(".claim.json"))
    check(claim == dict(protocol=PLAN, seed=seed, seat=seat, pid=trace["pid"]),"persistent process claim")
    random.seed(seed)
    np.random.seed(seed)
    env = make("carbon",configuration={**card["game_configuration"],"randomSeed":seed})
    check(sha(bytes(env.states[0].observation.full_carbon)) == card["terrain"],"physical layout")
    tracker = tracker_class(belief_class,seat)
    counts, depths, candidates = Counter(), Counter(), Counter()
    walls=[];cpus=[];prev_rng=None;prev_exec=None
    check([row["t"] for row in trace["rows"]] == list(range(32)),"missing signed root step")
    for row in trace["rows"]:
        t=row["t"]
        check(not env.done,"official engine reached terminal before root")
        visible=env.states[seat].observation
        common=env.states[0].observation
        view=dict(player=seat, step=common.step, players=common.players, carbon=visible.carbon,
                  trees=common.trees, remainingOverageTime=visible.remainingOverageTime)
        check(t == view["step"] and encoded(row["observation"]) == encoded(view) and objsha(view) == row["public_sha256"],"exact public view or signed bytes differ")
        actual=tape["actions"][t]
        check(actual[seat] == row["learner_order"] and actual[1-seat] == row["opponent_order"],"physical action tape mismatch")
        independent=expected_destinations(view,actual[seat])
        suggested=expected_destinations(view,row["teacher_order"])
        if prev_exec is not None:
            check(tracker.last_dest == prev_exec,"tracker failed across a real action boundary")
        tracker.step(view["players"],t,view["carbon"])
        tracker.set_actions(view["players"],row["teacher_order"])
        state=row["teacher_state"]
        check(tracker.last_dest == suggested and objsha(suggested) == state["before_intended_destinations_sha256"],"teacher order destinations")
        tracker.set_actions(view["players"],actual[seat])
        repaired=sum(independent[k]!=suggested[k] for k in independent)
        check(tracker.last_dest == independent and objsha(independent) == state["executed_destinations_sha256"] and
              repaired == state["repaired_workers"] and state["past_executed_destinations_observed"] is True,"history repair receipt")
        prev_exec=dict(independent)
        fingerprint=state["rng_stream_sha256"]
        check(len(bytes.fromhex(fingerprint))==32 and 0<=state["rng_index"]<=624,"RNG evidence")
        ident=(fingerprint,state["rng_index"])
        if prev_rng is not None and prev_rng[0]==ident[0]:
            check(prev_rng[1]<ident[1],"native RNG failed to advance")
        prev_rng=ident
        clock=row["teacher_clock"]
        k,h,rolls=clock["K"],clock["H"],clock["rolls"]
        for name in ("searched","single","aborted","changed"):
            check(clock[name] in (0,1),"unexpected native clock counter")
        check(math.isfinite(clock["wall"]) and math.isfinite(clock["cpu"]) and clock["wall"]>0 and clock["cpu"]>0,"nonpositive measured time")
        if clock["single"]:
            check((k,h,rolls,clock["searched"],clock["changed"],clock["aborted"],clock["native_seconds"])==(1,0,0,0,0,0,None),"single-candidate clock")
        else:
            check(k>=2 and h>=0 and rolls>=0 and bool(h)==bool(clock["searched"]) and
                  (clock["searched"] or clock["aborted"]) and math.isfinite(clock["native_seconds"]) and clock["native_seconds"]>=0,"multi-candidate clock")
        counts.update(dict(roots=1,multi=k>=2,positive_depth=h>0,single=k==1,aborted=clock["aborted"],
                           assigned_worker_intents=len(independent),repaired_workers=repaired,repaired_rows=repaired>0,
                           distinct_teacher_commands=row["teacher_order"]!=actual[seat]))
        candidates[k]+=1;depths[h]+=1;walls.append(clock["wall"]);cpus.append(clock["cpu"])
        env.step(actual,[])
    eligible=bool(counts["multi"]) and counts["positive_depth"]/counts["multi"]>=.8
    check((record["multi"],record["positive"],record["repaired_rows"],record["different_orders"],record["eligible"])==
          (counts["multi"],counts["positive_depth"],counts["repaired_rows"],counts["distinct_teacher_commands"],eligible),"collector totals")
    return dict(seed=seed,seat=seat,split=card["split"],terrain=card["terrain"],trace_sha256=record["sha256"],source_sha256=card["source_sha256"],
                first_private_rng_sha256=trace["rows"][0]["teacher_state"]["rng_stream_sha256"],
                counts=dict(counts),wall_seconds=sum(walls),cpu_seconds=sum(cpus),maximum_wall=max(walls),
                cold_init_wall=trace["cold_init_wall"],K_histogram=dict(candidates),H_histogram=dict(depths),eligible=eligible)


def main():
    p=load(HERE/"protocol.json");pool=load(ROOT/p["training_pool_path"]);collection=load(HERE/"collection.json")
    check(sha((HERE/"protocol.json").read_bytes())==PLAN and sha((ROOT/p["training_pool_path"]).read_bytes())==p["training_pool_sha256"],"registration")
    source=(ROOT/p["teacher_source_path"]).read_bytes()
    check(sha(source)==p["teacher_sha256"],"original source")
    lines=source.decode().splitlines();ns={}
    exec(compile("\n".join(lines[:502]+lines[556:560]),"<native-belief-only>","exec"),ns)
    maps=sorted((m for m in pool["maps"] if m["split"]=="train"),key=lambda m:(m["terrain"],m["seed"]))
    expected={(m["seed"],s,("fit" if i<4 else "held")) for i,m in enumerate(maps[:6]) for s in (0,1)}
    tickets=p["tickets"]; records={(r["seed"],r["seat"]):r for r in collection["tickets"]}
    check(len(maps)==24 and len(tickets)==len(records)==12 and {(c["seed"],c["seat"],c["split"]) for c in tickets}==expected,"map split")
    pool_cards={(c["seed"],c["candidate_seat"]):c for c in pool["cards"] if c["split"]=="train"}
    check(all(pool_cards[c["seed"],c["seat"]]["sha256"]==c["source_sha256"] for c in tickets),"source pool identity")
    results=[inspect(c,records[c["seed"],c["seat"]],p,ns["BeliefTracker"],ns["_Bel"]) for c in tickets]
    check(len({r["first_private_rng_sha256"] for r in results})==len(results),"private original process RNG stream reused")
    total=Counter()
    for r in results:
        total.update(r["counts"])
    ok=all(r["eligible"] for r in results)
    check(collection["training_eligible"]==ok and collection["roots"]==total["roots"]==384,"all roots/gate")
    output=dict(schema="sapling-learner-seat-independent-official-replay/1",protocol_sha256=PLAN,
                verifier_sha256=sha(pathlib.Path(__file__).read_bytes()),collection_sha256=sha((HERE/"collection.json").read_bytes()),
                previous_corpus=old_labels(pool,tickets),source_sha256=p["teacher_sha256"],counted_games=0,
                verification="All 384 signed public payloads (including clock and fog) are compared to official game observations. Both action dictionaries are compared to the raw learner tape; its real orders alone advance the official environment. Original tracker methods are independently replayed and both sets of intended destinations are rederived. Search time, its private RNG and live-state assertions are audited as historical process receipts; no teacher policy sampling is performed.",
                rows=results,counts=dict(total),fit_maps=4,held_maps=2,eligible=ok)
    with tempfile.NamedTemporaryFile(dir=HERE,delete=False,mode="wb") as f:
        f.write(json.dumps(output,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n")
    os.replace(f.name,HERE/"verification.json")
    print(total,ok)


if __name__=="__main__":
    main()

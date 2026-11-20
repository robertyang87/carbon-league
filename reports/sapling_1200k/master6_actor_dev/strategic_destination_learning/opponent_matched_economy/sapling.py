"""Yield the three closed Sapling batches and their per-step signed accounts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import zipfile

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parent
BATCHES = ("development", "replication", "elite_direct_policy/development")
NATIVE_SHA = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
spec = importlib.util.spec_from_file_location("matched_official_support", STRATEGY / "replication/closing_response/verify.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
ledger = shared.ledger


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, text):
    if not condition:
        raise ValueError(text)


def transitions(card, tape):
    # Inputs are the archived simultaneous orders. Neither policy executes here.
    seed = card["seed"]
    cfg = ledger.config()
    random.seed(seed)
    ledger.np.random.seed(seed)
    env = ledger.make("carbon", configuration={**cfg, "randomSeed": seed})
    sim = ledger.FastCarbon(cfg).reset(seed=seed)
    sim.cash = ledger.CashEvents(sim.cash)
    check(sha(bytes(map(int,sim.carbon))) == card["terrain"], "Terrain mismatch")
    bank = [5.,5.]
    for t,row in enumerate(tape):
        check(row["step"] == t and not env.done and shared.observations(env,bank)==row["public_observation_sha256"],
              "Public input or terminal mismatch")
        shared.check_physical(env,sim)
        before = env.states[0].observation.players
        calls = row["calls"]
        check(all(x["status"]=="ok" for x in calls),"Unscored transition")
        orders = [c["response"]["action"] for c in calls]
        sim.cash.events.clear()
        sim.step(*orders)
        env.step(orders)
        after = env.states[0].observation.players
        flow = ledger.cash_flow(sim.cash.events)
        bank = row["overage_remaining_seconds"]
        yield t,before,orders,flow,after
    shared.check_physical(env,sim)
    check(env.done and [s.reward for s in env.states] == card["official_scores_by_seat"],"Official final differs")


def games():
    for group,size in zip(BATCHES, (12,16,16),strict=True):
        hosted = STRATEGY/group/"hosted"
        settlement = json.loads((hosted/"settlement.json").read_bytes())
        cards = {(g["seed"],g["seat"],g["arm"]):g for g in settlement["games"]}
        check(len(cards)==size and settlement["records_reconciled"],"Incomplete Sapling batch")
        seen = set()
        for block in settlement["archives"]:
            path = hosted/(str(block["id"])+".zip")
            check(sha(path.read_bytes())==block["sha256"],"Bad hosted archive")
            with zipfile.ZipFile(path) as z:
                for name in sorted(n for n in z.namelist() if n.endswith("/receipt.json")):
                    arm=name.split("/")[0]
                    key=(block["seed"],block["seat"],arm)
                    g=cards[key]
                    rec=z.read(name)
                    check(sha(rec)==g["receipt_sha256"] and key not in seen,"Duplicate/changed receipt")
                    seen.add(key)
                    rec=json.loads(rec)
                    reg=json.loads(z.read(arm+"/registration.json"))
                    check(reg["files"]["opponent"]["sha256"]==NATIVE_SHA and reg["candidate_seat"]==g["seat"] and
                          reg["physical_map_sha256"]==g["terrain"] and rec["status"]=="complete" and
                          rec["official_scores_by_seat"]==g["official_scores_by_seat"],"Registered agent or terminal identity changed")
                    turns=z.read(arm+"/turns.jsonl")
                    tape=[json.loads(line) for line in turns.splitlines()]
                    check(sha(turns)==rec["turns_sha256"] and len(tape)==g["official_transitions"],"Bad turn archive")
                    meta = dict(id=f"{group}:{key[0]}:{key[1]}:{arm}",terrain=g["terrain"],actor=arm,
                                seat=g["seat"],official_scores_by_seat=g["official_scores_by_seat"],
                                cohort="Sapling",batch=group,seed=g["seed"],candidate_sha256=reg["files"]["candidate"]["sha256"],
                                opponent_sha256=NATIVE_SHA,archive_sha256=block["sha256"],turns_sha256=rec["turns_sha256"],
                                opponent_clock=g["examiner_search"],archive_id=block["id"])
                    yield meta,transitions(g,tape)
        check(seen==set(cards),"Skipped Sapling games")

"""Evaluate a frozen two-action economic policy with the actual historical simultaneous enemy order."""
from collections import Counter, deque
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

from pydantic_core import from_json

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "planter_threat_transfer_dev"
sys.path.insert(0, str(FROZEN))
import ladder_pairs as roots
from registry import A, REPLAYS, TOP, read, save, sha

SPLITS = ("new_opponent", "known_new_map")
WINDOWS = ((45, 125), (150, 235))


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def effective(order):
    return {uid: v for uid, v in order.items() if v and v != "NONE"}


def purchase_index(source):
    archive = TOP / source["source"]
    individual = REPLAYS.parent / "plant_work" / (source["id"] + ".gz")
    if individual.exists():
        record = read(individual)
        check(record["id"] == source["id"] and record["replay_sha256"] == source["raw_sha256"], "wrong local purchase ledger")
    else:
        all_cases = read(archive.parent / "cases.json.gz")
        match, = [r for r in all_cases["rows"] if r["id"] == source["id"]]
        check(match["source_sha256"] == source["raw_sha256"], "wrong appendix cash source")
        record = match["planting"]
    check(record["seed"] == source["seed"], "wrong recorded cash seed")
    purchases = {(x[0], x[1], x[2], x[4]): x for x in record["purchases"]}
    check(len(purchases) == len(record["purchases"]), "duplicate actual focal purchase")
    return purchases


def observed_return(purchases, t, s, uid, destination, command, matched):
    # A moving order can lead to a purchase by the same planter at that cell the very next turn.
    # The following decisions come from the actual replay; they are not either simulated branch's new policy.
    if not matched:
        return None
    target = t if command == "NONE" else t + 1
    record = purchases.get((target, s, uid, destination))
    if record is None:
        return {"purchased": False, "net_first_tenure": 0.0, "target_turn": target}
    gross = record[11] + record[13]
    return {"purchased": True, "target_turn": target, "tree_id": record[3], "expense": record[6],
            "gross_first_tenure": gross, "net_first_tenure": gross - record[6], "held_steps": record[10],
            "end_reason": record[9], "observed_following_orders": True}


def resolve(source, policy, trained, economic):
    f = REPLAYS / (source["id"] + ".json.gz")
    check(sha(f) == source["raw_sha256"], "wrong historical game")
    game = from_json(gzip.decompress(f.read_bytes()))
    seat = source["seat"]
    frames, cfg = game["steps"], game["configuration"]
    public_cfg = {k: v for k, v in cfg.items() if k.lower() not in {"randomseed", "seed", "id", "uuid"}}
    initial = frames[0][0]["observation"]
    terrain = hashlib.sha256(bytes(int(v) for v in initial["full_carbon"])).hexdigest()
    check(source["terrain"] == terrain, "wrong initial terrain")
    sim = roots.lots.FastCarbon(cfg).load_observation(initial)
    history = deque(maxlen=3)
    used = set()
    outputs = []
    purchases = purchase_index(source)
    for t in range(min(len(frames) - 1, WINDOWS[-1][1] + 1)):
        obs = sim.obs(seat)
        truth = frames[t][0]["observation"]
        check(obs["players"] == truth["players"] and obs["trees"] == truth["trees"] and
              obs["carbon"] == frames[t][seat]["observation"]["carbon"] and "full_carbon" not in obs,
              "historical public observation differs")
        actual = [v["action"] or {} for v in frames[t+1]]
        w = next((i for i, (lo, hi) in enumerate(WINDOWS) if lo <= t <= hi), None)
        if t == 0 or (w is not None and w not in used):
            old_n = len(policy._PAY_TRACE) if t else 0
            stay = policy.agent(obs, cfg)
            if len(policy._PAY_TRACE) > old_n:
                tr = policy._PAY_TRACE[-1]
                uid, cell = tr["uid"], tr["cell"]
                move = tr["base"]
                different = {k for k in stay.keys() | move.keys() if effective(stay).get(k) != effective(move).get(k)}
                check(w not in used and tr["window"] == w and tr["t"] == t and
                      different == ({uid} if tr["changed"] else set()) and uid not in effective(stay)
                      and len(history) == 3, "frozen root or nonfocal actions changed")
                used.add(w)
                if different:
                    pred = economic.decide(trained, obs, list(history), stay, move, uid, public_cfg)
                else:
                    # A selected a candidate, but its public action equals the parent's stay. Do not invent a move.
                    same = economic.scores(trained, obs, list(history), stay, uid, public_cfg)
                    pred = {"pick": "move", "stay": same, "move": same}
                check(pred["pick"] in ("stay","move") and all(
                    all(math.isfinite(float(pred[a][k])) for k in ("p_death", "net_tree", "utility"))
                    and 0 <= pred[a]["p_death"] <= 1 for a in ("stay","move")), "invalid economic output")
                branches = {}
                for a, orders in (("stay", stay), ("move", move)):
                    joint = actual.copy()
                    joint[seat] = orders
                    outcome = roots.apply_branch(sim, seat, uid, joint)
                    order = effective(orders).get(uid, "NONE")
                    matches = effective(orders) == effective(actual[seat])
                    dest = sim._move_nbr[order][cell] if order in sim._move_nbr else cell
                    realised = observed_return(purchases, t, seat, uid, dest, order, matches)
                    if realised is not None and order == "NONE":
                        check(realised["purchased"] == outcome["paid"] and
                              abs(realised.get("expense", 0) - outcome["expense"]) < .005,
                              "official one-step and actual ledger disagree on purchase")
                    branches[a] = {**outcome, "command": order, "cash_change": [
                        outcome["post_cash"][i]-obs["players"][i][0] for i in range(2)],
                        "matches_observed_whole_own_order": matches, "observed_future_only": realised,
                        "own_order": orders}
                if not tr["changed"]:
                    check(branches["stay"] == branches["move"], "equal actions with divergent official outcome")
                outputs.append({"window":w,"t":t,"uid":uid,"cell":cell,"kind":tr["kind"],"quoted":tr["quoted_cost"],
                                "changed_root":tr["changed"],"history_times":[v["step"] for v in history],
                                "public_input_sha256":digest([obs,list(history),public_cfg]), "enemy_order":actual[1-seat],
                                "observed_own_order":actual[seat],"pick":pred["pick"],
                                "predictions":{a:{k:float(pred[a][k]) for k in ("p_death","net_tree","utility")}
                                               for a in ("stay","move")},"branches":branches})
        elif len(used)<2:
            policy._FEAT.observe(obs)
        if len(used)==2:
            break
        history.append(obs)
        sim.step(*actual)
    return {"source":source,"selected":outputs,"missed_windows":[i for i in range(2) if i not in used]}


def summary(games):
    results = {}
    for split in SPLITS:
        maps = [r for r in games if r["source"]["split"] == split]
        check(len(maps) == 24, "wrong held split size")
        counts = Counter()
        diff = {"stay": [], "move": []}
        for m in maps:
            selected = m["selected"]
            counts["maps_without_any_root"] += int(not selected)
            counts["missed_windows"] += len(m["missed_windows"])
            counts["selected"] += len(selected)
            for a in diff:
                differences = [int(r["branches"][r["pick"]]["survived"]) - int(r["branches"][a]["survived"]) for r in selected]
                # Maps with no eligible root have zero changes under all three policies.
                diff[a].append(statistics.mean(differences) if differences else 0.)
            for r in selected:
                counts["distinct_arms"] += int(r["changed_root"])
                for name, a in (("A_stay","stay"),("parent","move"),("learned",r["pick"])):
                    v = r["branches"][a]
                    counts[name+"_survived"] += int(v["survived"])
                    counts[name+"_purchases"] += int(v["paid"])
                    counts[name+"_purchase_expense"] += v["expense"]
                    counts[name+"_historical_full_order_matches"] += int(v["matches_observed_whole_own_order"])
                counts["actual_learned_departures_from_A"] += int(r["pick"]=="move" and r["changed_root"])
        results[split] = {"maps":len(maps),"windows":len(maps)*2,"counts":dict(counts),
                          "map_equal_survival_difference_vs_A":statistics.mean(diff["stay"]),
                          "map_equal_survival_difference_vs_parent":statistics.mean(diff["move"]),
                          "by_map":[{"id":m["source"]["id"],"vs_A":a,"vs_parent":p}
                                    for m,a,p in zip(maps,diff["stay"],diff["move"],strict=True)]}
    return results


def main():
    import economic
    plan = read(HERE/"plan.json")
    selected = [r for r in plan["maps"] if r["split"] in SPLITS]
    check(len(selected)==len({r["terrain"] for r in selected})==48, "wrong held maps")
    model = read(HERE/"model.json")
    policy = roots.old.pack()
    identity = {"plan_sha256":sha(HERE/"plan.json"),"runner_sha256":sha(Path(__file__)),
                "weights_sha256":sha(HERE/"model.json"),"economic_code_sha256":sha(Path(economic.__file__)),
                "frozen_A_sha256":sha(A/"candidate.py"),"official_branch_runner_sha256":sha(Path(roots.__file__))}
    folder=HERE/"ladder_eval_games";folder.mkdir(exist_ok=True)
    games=[]
    for r in selected:
        p = folder / (r["id"]+".json.gz")
        if not p.exists():
            save(p, {"identity":identity, "game":resolve(r,policy,model,economic)})
        item = read(p)
        check(item["identity"]==identity and item["game"]["source"]==r, "old root result from other experiment")
        games.append(item["game"])
    scores=summary(games)
    output={"identity":identity,"splits":scores,"passed_paired_survival_condition":
            scores["new_opponent"]["map_equal_survival_difference_vs_A"]>=0,
            "evidence":[{"path":str(p.relative_to(HERE)),"sha256":sha(p)} for p in sorted(folder.glob("*.json.gz"))],
            "full_games_started":0,"cash_scope":"Only named exact root whole-joint matches have a recorded later actual first spell. Unmatched arms have no future label. Same-root counterfactuals compare only paired official first-step outcomes."}
    save(HERE/"ladder_eval.json",output)
    print(json.dumps({k:{**v,"by_map":"see ledger"} for k,v in scores.items()}),flush=True)


if __name__=="__main__":
    main()

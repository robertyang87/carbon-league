"""Recompute the spent-map closing suffixes from physical orders. No policy or search is loaded."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import mean
import zipfile

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_economy", D.parent / "economy_check.py")
replay_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_lib)
ledger = replay_lib.ledger
from zerosum_env.envs.carbon import idgen

SHA = "1f770df8ed4cb841edb22a7a382ead37d53021bb365a92cec968196e975f0cd2"
UID = "player-0-worker-47"
ARMS = ("bc_order", "parent_order")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha((json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n").encode())


def read_gz(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def physical_key(env):
    o = env.states[0].observation
    return digest([o.step, o.full_carbon, o.players, o.trees, dict(idgen._id_dict)])


def observations(env, bank):
    shared = env.states[0].observation
    return [digest(dict(step=shared.step, player=s, carbon=env.states[s].observation.carbon,
                        players=shared.players, trees=shared.trees, remainingOverageTime=bank[s])) for s in (0,1)]


def read_sources():
    need(sha((D/"protocol.json").read_bytes()) == SHA, "changed diagnostic protocol")
    p = json.loads((D/"protocol.json").read_bytes())
    path = D.parent/"hosted/14588219048.zip"
    need(sha(path.read_bytes()) == p["zip_sha256"], "different hosted game")
    with zipfile.ZipFile(path) as z:
        raw = z.read("bc/turns.jsonl")
        receipt = json.loads(z.read("bc/receipt.json"))
        reg = json.loads(z.read("bc/registration.json"))
    need(reg["seed"] == p["seed"] and reg["physical_map_sha256"] == p["terrain"] and
         reg["candidate_seat"] == p["seat"] == 0 and receipt["status"] == "complete" and
         sha(raw) == receipt["turns_sha256"] == p["turns_sha256"], "different archived match")
    tape = [json.loads(line) for line in raw.splitlines()]
    need(len(tape) == receipt["official_transitions"] == p["terminal_step"], "short reference")
    anchor = p["anchor_orders"]
    ref = [c["response"]["action"] for c in tape[p["root_step"]]["calls"]]
    need(ref == [anchor["bc_order"], anchor["enemy"]], "root is not the hosted pair")
    changed = {uid for uid in anchor["bc_order"].keys() | anchor["parent_order"].keys()
               if anchor["bc_order"].get(uid) != anchor["parent_order"].get(uid)}
    need(changed == {UID, "player-0-worker-48"}, "unexpected joint intervention")
    return p, tape, receipt


def check_physical(env, sim):
    o = env.states[0].observation
    v = sim.obs(0)
    need(v["players"] == o.players and v["trees"] == o.trees and sim.step_count == o.step,
         f"simulator entity mismatch at {o.step}")
    need(all(math.isclose(x,y,rel_tol=0,abs_tol=1.e-6) for x,y in zip(sim.carbon, o.full_carbon, strict=True)),
         f"simulator carbon mismatch at {o.step}")


def step_orders(p, tape, rec, ticket=None):
    cfg = ledger.config()
    seed, root = p["seed"], p["root_step"]
    random.seed(seed)
    ledger.np.random.seed(seed)
    env = ledger.make("carbon", configuration={**cfg,"randomSeed":seed})
    sim = ledger.FastCarbon(cfg).reset(seed=seed)
    sim.cash = ledger.CashEvents(sim.cash)
    need(sha(bytes(env.states[0].observation.full_carbon)) == sha(bytes(map(int, sim.carbon))) == p["terrain"],
         "initial terrain disagreement")
    bank = [5.,5.]
    accounts = [Counter(),Counter()]
    events, at_root, root_hash, root_cash, banks = [], None, None, None, []
    for t in range(p["terminal_step"]):
        need(not env.done and env.states[0].observation.step == t, "premature terminal or wrong turn")
        check_physical(env, sim)
        actual, marker = tape[t], None
        if ticket is not None and t >= root:
            marker = ticket["rows"][t-root]
            orders = marker["orders"]
            need(marker["step"] == t and observations(env,bank) == marker["public_hashes"] and
                 physical_key(env) == marker["board_before"] and bank == marker["bank_before"],
                 f"source observation/board/clock {t}")
            o = env.states[0].observation
            need(o.players[0][2].get(UID) == marker["worker47_before"] and list(sim.cash) == marker["cash_before"],
                 f"pre-action cash/worker {t}")
            clock = [marker["student_clock"], marker["enemy_clock"]]
            for c in clock:
                need(math.isfinite(c["wall"]) and c["wall"] >= 0, f"invalid clock {t}")
            if t == root:
                need(orders == [p["anchor_orders"][ticket["arm"]],p["anchor_orders"]["enemy"]],
                     "wrong entire root intervention/shared reply")
                ref = actual["calls"][1]
                need(clock[1]["wall"] == ref["elapsed"] and clock[1]["cpu"] == ref["response"]["public_cpu_seconds"]
                     and clock[1]["K"] == ref["response"]["search_last"]["candidates"]
                     and clock[1]["H"] == ref["response"]["search_last"]["depth"], "wrong hosted source clock")
            else:
                c = clock[1]
                need(c["K"] >= 1 and c["H"] >= 0 and c["rolls"] >= 0 and c["cpu"] >= 0 and
                     sum(c.get(s,0) for s in ("single","searched","aborted")) > 0, f"missing live original clock {t}")
            nextbank = [bank[s]-max(0.,clock[s]["wall"]-1.) for s in (0,1)]
            need(nextbank == marker["bank_after"], f"clock accounting {t}")
        else:
            need(observations(env,bank) == actual["public_observation_sha256"] and actual["step"] == t and
                 all(c["status"] == "ok" for c in actual["calls"]), f"archived public inputs {t}")
            orders = [c["response"]["action"] for c in actual["calls"]]
            nextbank = actual["overage_remaining_seconds"]
        if t == root:
            root_hash = physical_key(env)
            root_cash = list(sim.cash)
            at_root = observations(env,bank)
            need(at_root == p["root_public_sha256"], "wrong root public observation")
        before = list(sim.cash)
        sim.cash.events = []
        sim.step(*orders)
        env.step(orders)
        check_physical(env, sim)
        if marker:
            need(physical_key(env) == marker["board_after"] and env.states[0].observation.players[0][2].get(UID) == marker["worker47_after"] and
                 list(sim.cash) == marker["cash_after"], f"reported post-action state {t}")
        flow = ledger.cash_flow(sim.cash.events)
        for s in (0,1):
            ledger.check_equal(sum(flow[s].values()), sim.cash[s]-before[s], f"signed cash flow {t}/{s}")
            if t >= root:
                accounts[s].update(flow[s])
        if t >= root:
            deposits = [e for e in sim.cash.events if e[0] in ("own_delivery","opponent_worker_delivery_to_this_base")]
            events.append(dict(step=t, orders=orders, before=before, after=list(sim.cash),
                               component_cash_by_seat=flow, deliveries=deposits,
                               worker47_after=env.states[0].observation.players[0][2].get(UID)))
            if marker:
                banks.append([marker["student_clock"]["wall"],marker["enemy_clock"]["wall"]])
        bank = nextbank
    scores = [s.reward for s in env.states]
    expected = rec["official_scores_by_seat"] if ticket is None else ticket["official_scores_by_seat"]
    need(scores == expected and env.done and sim.step_count == p["terminal_step"], "terminal result")
    margin = scores[0]-scores[1]
    if ticket is not None:
        need(ticket["root_public_sha256"] == at_root and ticket["root_board_sha256"] == root_hash and
             ticket["margin"] == margin, "source root/result mismatch")
    for s in (0,1):
        accounts[s]["terminal_scoring_adjustment"] = scores[s]-sim.cash[s]
        ledger.check_equal(root_cash[s]+sum(accounts[s].values()), scores[s], "terminal cash")
    return dict(root_board_sha256=root_hash, root_cash=root_cash, scores=scores, margin=margin, accounts=accounts,
                worker47_deliveries=[[r["step"],e[2],e[4]] for r in events for e in r["deliveries"] if e[3] == UID],
                suffix=events, clock_slow_by_seat=[sum(b[s] > 1 for b in banks) for s in (0,1)] if banks else None)


def verify():
    p, tape, rec = read_sources()
    historic = step_orders(p, tape, rec)
    idx = json.loads((D/"results.json").read_bytes())
    need(idx["protocol_sha256"] == SHA and idx["runner_sha256"] == sha((D/"run.py").read_bytes()) and
         idx["prefix_sha256"] == sha((D/"prefix.json.gz").read_bytes()) and idx["whole_games"] == 0, "different run")
    prefix = read_gz(D/"prefix.json.gz")
    need(prefix["protocol"] == SHA and prefix["orig_sha"] == p["original_sha256"] and len(prefix["rows"]) == p["root_step"]+1,
         "bad prefix recovery")
    for t,row in enumerate(prefix["rows"]):
        actual=tape[t]
        need(row["step"]==t and row["public_hashes"]==actual["public_observation_sha256"]
             and row["used_archived_enemy_order"]==actual["calls"][1]["response"]["action"] and
             row["corrected"]==(row["shadow_enemy_order"]!=row["used_archived_enemy_order"]), "invalid prefix receipt")
    rows = []
    need([(x["pair"],x["arm"]) for x in idx["tickets"]] ==
         [(z["pair"],arm) for z in p["pairs"] for arm in z["order"]], "incomplete or extra tickets")
    for pair in p["pairs"]:
        games, signatures, first = {}, [], {}
        for arm in pair["order"]:
            path = D/"tickets"/f'{pair["pair"]}_{arm}.json.gz'
            tkt = read_gz(path)
            ref = next(x for x in idx["tickets"] if x["pair"]==pair["pair"] and x["arm"]==arm)
            need(sha(path.read_bytes()) == ref["sha256"] and tkt["margin"]==ref["margin"] and tkt["arm"]==arm
                 and tkt["pair"]==pair["pair"] and tkt["stream_seed"]==pair["stream_seed"] and tkt["protocol"]==SHA
                 and tkt["seed"]==p["seed"] and tkt["terrain"]==p["terrain"] and tkt["seat"]==p["seat"]
                 and len(tkt["rows"])==p["terminal_step"]-p["root_step"], "unbound source or invalid ticket")
            signatures.append((tkt["root_student_sha256"],tkt["root_private_sha256"],tkt["root_board_sha256"]))
            g = step_orders(p,tape,rec,tkt)
            need(g["root_board_sha256"] == historic["root_board_sha256"], "historical board mismatch")
            g["source_sha256"] = ref["sha256"]
            games[arm], first[arm] = g, tkt["rows"]
        need(len(set(signatures))==1,"different initial private/learner states within pair")
        bc, par = [games[k] for k in ARMS]
        components = ledger.minus(ledger.minus(bc["accounts"][0],par["accounts"][0]),
                                  ledger.minus(bc["accounts"][1],par["accounts"][1]))
        delta = bc["margin"]-par["margin"]
        ledger.check_equal(sum(components.values()), delta, "paired economic decomposition")
        first_change = next((a["step"] for a,b in zip(first[ARMS[0]],first[ARMS[1]],strict=True)
                             if a["orders"][1]!=b["orders"][1]),None)
        rows.append(dict(pair=pair["pair"], games=games, bc_minus_parent=delta,
                         margin_components=components, first_enemy_order_difference=first_change))
    return dict(protocol_sha256=SHA, verifier_sha256=sha(Path(__file__).read_bytes()),
                result_index_sha256=sha((D/"results.json").read_bytes()), root_map_count=1,whole_games=0,
                archived_real_game_suffix=historic, paired_runs=rows, mean_paired_margin_change=mean(x["bc_minus_parent"] for x in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-only",action="store_true")
    args = parser.parse_args()
    if args.archived_only:
        p,t,r = read_sources()
        result = step_orders(p,t,r)
        print(json.dumps({k:result[k] for k in ("scores","margin","worker47_deliveries","root_board_sha256")}))
    else:
        result = verify()
        out = D/"verified.json.gz"
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(result,allow_nan=False,sort_keys=True,separators=(",",":")).encode(),mtime=0))
        tmp.replace(out)
        print(json.dumps({k:result[k] for k in ("mean_paired_margin_change","whole_games")}))

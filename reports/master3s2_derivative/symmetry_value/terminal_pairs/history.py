"""Restore the two physical public views and the official referee of an archived training match."""
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parents[1]
ROOT = DERIVATIVE.parents[1]
sys.path.insert(0, str(DERIVATIVE / "freight_response"))
import run as freight

REGISTRATION = json.loads((HERE / "protocol.json").read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token(value):
    return sha(json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode())


def file_record(card):
    source = ROOT / card["path"]
    data = source.read_bytes()
    freight.check(sha(data) == card["source_sha256"], "Changed historic tape")
    return json.loads(gzip.decompress(data))


def source_cards():
    source = (DERIVATIVE / "opponent_policy/protocol.json").read_bytes()
    freight.check(sha(source) == REGISTRATION["source_ledger"], "Different training source inventory")
    rows = json.loads(source)["games"]
    result = {}
    for ticket in REGISTRATION["eligibility"]:
        matches = [c for c in rows if c["seed"] == ticket["seed"] and c["expert_seat"] == ticket["seat"]
                   and c["terrain"] == ticket["terrain"] and c["sha256"] == ticket["source_sha256"]
                   and c["path"] == ticket["path"] and c["split"] == "train"]
        freight.check(len(matches) == 1, "Source is not one of the signed fit maps")
        result[ticket["source_sha256"]] = matches[0]
    return result


def setup(ticket, card):
    tape = file_record(ticket)
    acts = tape["actions"]
    freight.check(token(acts) == card["actions_sha256"] and len(acts) == card["transitions"], "Incorrect recorded orders")
    cfg = tape["game_configuration"]
    random.seed(ticket["seed"])
    freight.np.random.seed(ticket["seed"])
    env = freight.make("carbon",configuration={**cfg, "randomSeed":ticket["seed"]})
    sim = freight.econ.FastCarbon(cfg).reset(seed=ticket["seed"])
    terrain = sha(bytes(env.states[0].observation.full_carbon))
    freight.check(terrain == ticket["terrain"] == sha(bytes(map(int,sim.carbon))), "Different starting terrain")
    sim.cash = freight.econ.CashEvents(sim.cash)
    return tape,acts,cfg,env,sim


def views(env):
    bank = [c.observation.get("remainingOverageTime",60) for c in env.states]
    return [freight.public(env,s,bank) for s in (0,1)]


def advance(env,sim,orders):
    sim.cash.events.clear()
    sim.step(*orders)
    env.step(orders,[])
    cash = [p[0] for p in env.states[0].observation.players]
    freight.check(list(sim.cash) == cash, "Independent engine cash disagreement")
    return cash


def registered_sources():
    cards = source_cards()
    rootset = defaultdict(set)
    for row in REGISTRATION["eligibility"]:
        rootset[row["source_sha256"]].add(row["step"])
    result = []
    for key,card in cards.items():
        ticket = next(r for r in REGISTRATION["eligibility"] if r["source_sha256"] == key)
        tape,actions,_,env,sim = setup(ticket,card)
        balances,roots = [],{}
        for i in range(len(actions)+1):
            obs = views(env)
            balances.append(list(sim.cash))
            if i in rootset[key]:
                roots[str(i)] = dict(public_sha256_by_seat=[freight.digest(x) for x in obs],
                                     sorted_public_sha256_by_seat=[token(x) for x in obs],
                                     incoming_cash=balances[-1], prior_orders_sha256=token(actions[:i]))
            if i<len(actions):
                advance(env,sim,actions[i])
        freight.check(env.done and token(balances)==card["official_cash_path_sha256"] and
                      [s.reward for s in env.states]==tape["official_score_by_seat"],"Archive failed official replay")
        result.append(dict(seed=ticket["seed"],seat=ticket["seat"],terrain=ticket["terrain"],source_sha256=key,
                           history_actions_sha256=card["actions_sha256"],
                           official_cash_path_sha256=card["official_cash_path_sha256"],roots=roots))
    return dict(registration=sha((HERE/"protocol.json").read_bytes()),sources=result)


if __name__=="__main__":
    out=HERE/"restored_sources.json"
    raw=(json.dumps(registered_sources(),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    freight.check(not out.exists() or out.read_bytes()==raw,"Prior source-hash receipt differs")
    out.write_bytes(raw)
    print(sha(raw))

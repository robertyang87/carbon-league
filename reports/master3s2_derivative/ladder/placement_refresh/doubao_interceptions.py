"""One-step branches around every opposing deposit in the archived worst ten-step window."""
import copy
import json
import gzip
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "code/env_src/carbon_challenge-master"))
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

MATCH = "4c3bc230e7cc588cf3dc70dbf1dfa8ae"
MOVES = dict(NONE=(0,0), UP=(0,-1), RIGHT=(1,0), DOWN=(0,1), LEFT=(-1,0))


def cells(cell, move, size):
    x, y = cell % size, cell // size
    dx, dy = MOVES[move]
    return (y+dy) % size * size + (x+dx) % size


def run():
    ledger = json.load(gzip.open(HERE/(MATCH+"-1.ledger.json.gz")))
    env = json.load(gzip.open(HERE/"replays"/(MATCH+"-1.json.gz")))
    size = env["configuration"]["size"]
    deliveries = [d for d in ledger["worst_windows"]["10"]["returns"] if d["seat"] == 1]
    frames = env["steps"]
    idgen.reset()
    checked = 0
    candidates = []
    opportunities = []
    for t in range(299):
        root = copy.deepcopy(frames[t][0]["observation"])
        root["carbon"] = root["full_carbon"]
        teams = root["players"]
        orders = [e.get("action") or {} for e in frames[t+1]]
        targets = [e for e in deliveries if e["t"]-10 <= t < e["t"]]
        counters = dict(idgen._id_dict)
        def advance(commands):
            idgen._id_dict.clear()
            idgen._id_dict.update(counters)
            return Board(copy.deepcopy(root), env["configuration"], commands).next().observation

        for d in targets:
            enemy = d["worker"]
            if enemy not in teams[1][2]:
                continue
            to = cells(teams[1][2][enemy][0], orders[1].get(enemy, "NONE"), size)
            contacts = 0
            for uid, (cell, cargo, role) in teams[0][2].items():
                if role != "COLLECTOR":
                    continue
                for action in MOVES:
                    if cells(cell,action,size) != to or orders[0].get(uid,"NONE") == action:
                        continue
                    contacts += 1
                    alt = [dict(o) for o in orders]
                    if action == "NONE":
                        alt[0].pop(uid,None)
                    else:
                        alt[0][uid] = action
                    post = advance(alt)["players"]
                    observed = frames[t+1][0]["observation"]["players"]
                    delta = [(post[s][0]-observed[s][0]) for s in (0,1)]
                    candidates.append(dict(step=t, arrival=d["t"], value=d["amount"], uid=uid, enemy=enemy,
                        cargo=cargo, enemy_cargo=teams[1][2][enemy][1], source_cell=cell, same_cell=to,
                        base_owner=next((s for s,p in enumerate(teams) if to in p[1].values()),None),
                        original=orders[0].get(uid,"NONE"), alternative=action,
                        own_after=post[0][2].get(uid), enemy_after=post[1][2].get(enemy),
                        actual_own_after=observed[0][2].get(uid), actual_enemy_after=observed[1][2].get(enemy),
                        money_change=[round(v,2) for v in delta], money_margin_change=round(delta[0]-delta[1],2)))
            opportunities.append(dict(arrival=d["t"],enemy=enemy,step=t,physical_contact_alternatives=contacts))
        control = advance(orders)
        actual = frames[t+1][0]["observation"]
        assert control["players"] == actual["players"] and control["trees"] == actual["trees"] and control["carbon"] == actual["full_carbon"]
        checked += 1
    result = dict(match=MATCH, verified_original_steps=checked, observed_window=[218,228], deliveries=deliveries,
                  note="Arrival events selected retrospectively to locate concrete missed interceptions. All same-cell alternatives from friendly collectors on each of preceding 10 actual transitions enumerated with the other orders fixed. Branches last one official step. No estimate of reactive bank or full-game value.",
                  opportunities=opportunities, alternatives=candidates)
    (HERE/"doubao_intercepts.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(len(opportunities),len(candidates))
    for c in candidates:
        print(c["step"],c["enemy"],round(c["enemy_cargo"],2),c["uid"],c["alternative"],c["own_after"],c["enemy_after"],c["base_owner"],c["money_change"])


if __name__ == "__main__":
    run()

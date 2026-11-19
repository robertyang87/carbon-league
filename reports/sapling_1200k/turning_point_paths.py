"""Reconstruct named workers and trees before illustrative cash turns.

The official interpreter and FastCarbon consume exactly the recorded joint commands.
Public boards are checked at every transition. Turn t describes a pre-action state;
the score and position after its commands belong to state t+1.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import official_game_replay as official
import tree_audited_game as arena
import cnet2_feat as feat

HERE = Path(__file__).resolve().parent
CASES = [
    dict(case="双基地同回合兑现", run="tree_exam_soft_it50_v5", seed=620106812, seat=1,
         end=44, workers=["player-0-worker-0", "player-0-worker-5", "player-0-worker-9",
                          "player-0-worker-11", "player-1-worker-5"],
         trees=["player-1-tree-0", "player-1-tree-1"]),
    dict(case="买工、采收、返航后领先翻转", run="tree_exam_soft_it50_v5_guard",
         seed=620106811, seat=0, end=48,
         workers=["player-1-worker-6", "player-0-worker-0", "player-0-worker-1"], trees=[]),
    dict(case="树吸收现金越过领先线", run="tree_exam_soft_it50_deep214",
         seed=620106811, seat=1, end=38, workers=["player-0-worker-2", "player-0-worker-3",
                         "player-1-worker-0", "player-1-worker-1"],
         trees=["player-0-tree-2", "player-0-tree-3"]),
    dict(case="最后一工同载相撞", run="tree_exam_soft_it50_v5", seed=620106811, seat=1,
         end=37, workers=["player-1-worker-1", "player-1-worker-2", "player-1-worker-3",
                         "player-0-worker-10"], trees=[]),
]


def cash_gap(state, seat):
    return round(state["cash"][seat] - state["cash"][1-seat], 2)


def destination(position, action):
    return int(feat.DEST[position, feat.MOVES.index(action or "NONE")])


def registration(state, item, cfg):
    role = item["type"]
    price = "recCollector" if role == "COLLECTOR" else "recPlanter"
    count = sum(w["type"] == role for w in state["workers"].values())
    return cfg[price+"Cost"] + count*cfg[price+"IncreaseCost"]


def track_worker(state, commands, uid, cfg):
    w = state["workers"].get(uid)
    if not w:
        return None
    side, pos = w["player"], w["pos"]
    act = commands[side].get(uid) or "NONE"
    dest = destination(pos, act)
    cargo = w["carbon"] * (1-cfg["moveCost"] if dest != pos else 1)
    threats = []
    foes = []
    for enemy, other in state["workers"].items():
        if side == other["player"] or dest not in feat.DEST[other["pos"]]:
            continue
        other_cargo = other["carbon"] * (1-cfg["moveCost"] if other["pos"] != dest else 1)
        lethal = w["type"] == "PLANTER" or (other["type"] == "COLLECTOR" and other_cargo <= cargo)
        if lethal:
            threats.append(enemy)
        if dest == destination(other["pos"], commands[1-side].get(enemy)):
            foes.append(enemy)
    home = tuple(cell for _,cell in state["bases"][side])
    return dict(position=pos, cargo=round(w["carbon"],4), role=w["type"], command=act,
                destination=dest, home_distance=min(int(feat.DIST[pos,cell]) for cell in home),
                destination_home_distance=min(int(feat.DIST[dest,cell]) for cell in home),
                existing_foes_with_individual_kill_reach=threats,
                actual_existing_foes_ordered_to_same_cell=foes)


def trace(case):
    run = HERE / case["run"]
    scorecard = json.loads((run / "results.json").read_bytes())
    row = next(row for row in scorecard["games"] if
               (row["seed"],row["candidate_seat"]) == (case["seed"],case["seat"]))
    raw_bytes = (run / row["raw_file"]).read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != row["raw_sha256"]:
        raise ValueError("changed actions archive")
    raw = json.loads(gzip.decompress(raw_bytes))
    seed, seat, end = case["seed"], case["seat"], case["end"]
    cfg = dict(raw["game_configuration"], randomSeed=seed)
    official.np.random.seed(seed)
    env = official.make("carbon", configuration=cfg)
    sim = arena.exam.FastCarbon(cfg).reset(seed=seed)
    diffs, births, inspected = [], {}, {}
    alternatives = []
    branch_boards = {}
    selected_states = {end-10,end-5,end-3,end-1}
    receipts = []
    tree_books = {tid:{"ownership_events":[], "gross_income_by_seat":[0.,0.]} for tid in case["trees"]}
    for t, actions in enumerate(raw["actions"]):
        state = sim.get_state()
        if t in selected_states:
            inspected[t] = dict(pre_action_step=t, cash_by_seat=list(sim.cash), cash_gap=cash_gap(state,seat),
                worker_counts_by_seat=[len(sim.player_workers[p]) for p in (0,1)],
                tree_counts_by_seat=[len(sim.player_trees[p]) for p in (0,1)],
                focal_workers={uid:track_worker(state,actions,uid,cfg) for uid in case["workers"]},
                focal_trees={tid:state["trees"].get(tid) for tid in case["trees"]})
        if case["case"] == "最后一工同载相撞" and t == end-1:
            uid = "player-1-worker-1"
            for move in feat.MOVES:
                altered = [dict(command) for command in actions]
                if move == "NONE":
                    altered[seat].pop(uid,None)
                else:
                    altered[seat][uid] = move
                shadow = arena.exam.FastCarbon(cfg)
                shadow.set_state(state)
                shadow.step(*altered)
                after = shadow.get_state()
                branch_boards[move] = shadow._players_obs()
                alternatives.append(dict(move=move, actual=move == actions[seat].get(uid,"NONE"),
                    cost=0, worker_survived=uid in shadow.workers, cash=shadow.cash[seat],
                    public_state_hash=arena.prior.hashed(shadow._players_obs()),
                    extinct=bool(arena.exam._extinct(after,seat)), worker=after["workers"].get(uid)))

        sim.step(*actions)
        env.step(actions)
        after = sim.get_state()
        # The original engine exposes workers, tree ages, owners, cash and received tree income.
        board = env.states[0].observation
        if board["players"] != sim._players_obs() or board["trees"] != after["trees_dict"]:
            raise ValueError(f"original board differs at t={t+1}")
        new = set(after["workers"]) - set(state["workers"])
        born = {uid:dict(position=after["workers"][uid]["pos"],role=after["workers"][uid]["type"],
                         cost=registration(state,after["workers"][uid],cfg)) for uid in sorted(new)}
        for uid in born.keys() & set(case["workers"]):
            births[uid] = dict(settled_step=t+1, **born[uid])
        tree_income = [sum(after["trees_dict"][tid][1] for tid,tr in after["trees"].items()
                           if tr["player"] == p) for p in (0,1)]
        swaps = {tid:[state["trees"][tid]["player"],tree["player"]] for tid,tree in after["trees"].items()
                 if tid in state["trees"] and tree["player"] != state["trees"][tid]["player"]}
        focus_income = {tid:round(after["trees_dict"][tid][1],5) for tid in case["trees"] if tid in after["trees"]}
        for tid, book in tree_books.items():
            if t >= end or tid not in after["trees"]:
                continue
            now = after["trees"][tid]
            owner = now["player"]
            old = state["trees"].get(tid)
            if old is None or old["player"] != owner:
                mature = sum(v["age"] < cfg["treeLifespan"] for v in state["trees"].values())
                owned = sum(v["player"] == owner for v in state["trees"].values())
                cost = cfg["seizeCost"] if old else cfg["plantCost"] + (
                    cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"] ** mature
                    if owned > cfg["playerTreeProtectiveNumber"] else 0.)
                ids = [u for u,w in state["workers"].items() if w["pos"] == now["pos"]
                       and w["type"] == "PLANTER" and w["player"] == owner and not actions[owner].get(u)]
                if len(ids) != 1:
                    raise ValueError(f"no unique paying planter for {tid} at {t}")
                book["ownership_events"].append(dict(settled_step=t+1, cell=now["pos"],
                    owner=owner, actor=ids[0], cost=cost, kind="seizure" if old else "plant"))
            book["gross_income_by_seat"][owner] += float(after["trees_dict"][tid][1])
        if end-10 <= t < end:
            worker_path = {}
            for uid in case["workers"]:
                worker = state["workers"].get(uid)
                if worker is None:
                    continue
                move = actions[worker["player"]].get(uid) or "NONE"
                dest = destination(worker["pos"],move)
                homes = state["bases"][worker["player"]]
                worker_path[uid] = [worker["pos"], round(worker["carbon"],4), move, dest,
                                    min(int(feat.DIST[worker["pos"],cell]) for _,cell in homes)]
            receipts.append(dict(pre_action_step=t,post_step=t+1,cash_gap_after=cash_gap(after,seat),
                focal_worker_path=worker_path,
                tree_income_by_seat=[round(v,4) for v in tree_income],focal_trees_income=focus_income,
                recruited=born, deaths=sorted(set(state["workers"]) - set(after["workers"])),
                tree_owner_changes=swaps, collision_outcomes=sim.last_collisions))
        if t == end-1:
            ending=dict(post_step=end,cash_by_seat=after["cash"],cash_gap=cash_gap(after,seat),
                workers_alive_by_seat=[len(sim.player_workers[p]) for p in (0,1)],
                focal_workers={uid:after["workers"].get(uid) for uid in case["workers"]},
                focal_trees={tid:after["trees"].get(tid) for tid in case["trees"]})
        diffs.append(cash_gap(after,seat))
    if not env.done or [s.reward for s in env.states] != row["official_score_by_seat"]:
        raise ValueError("official terminal result mismatch")
    if arena.prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError(f"terminal simulator state mismatch {case}")
    # Original interpreter carries an ID generator; evaluate the independent
    # branches after the factual game has completely finished.
    for item in alternatives:
        t = end - 1
        altered = [dict(command) for command in raw["actions"][t]]
        uid = "player-1-worker-1"
        if item["move"] == "NONE":
            altered[seat].pop(uid, None)
        else:
            altered[seat][uid] = item["move"]
        official.np.random.seed(seed)
        counter = official.make("carbon", configuration=cfg)
        for command in raw["actions"][:t]:
            counter.step(command)
        counter.step(altered)
        observed = counter.states[0].observation["players"]
        if observed != branch_boards[item["move"]]:
            raise ValueError("official one-step alternative differs")
        item["official_done"] = counter.done
    return dict(**case,raw_sha256=row["raw_sha256"],terminal_state_sha256=row["final_state_sha256"],
        official_boards_verified=len(diffs),terminal_score_by_seat=row["official_score_by_seat"],
        terminal_margin=row["margin"],peak_cash_gap=max([0,*diffs]),
        final_positive_state=max([0]+[i+1 for i,m in enumerate(diffs) if m>0]),
        focal_recruitments=births,focal_tree_receipts_to_case_end=tree_books,pre_states=inspected,ten_transitions=receipts,at_end=ending,
        single_step_counterfactual=alternatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = dict(unit="case illustrations, drawn from previously used development maps",
        position="official row-major cell ID, wraparound 15x15; home distance in cardinal moves",
        focal_worker_path_fields=["pre_cell","pre_cargo","actual_command","intended_destination","pre_home_distance"],
        threat="unilateral lethal reach of existing enemy worker, includes ties; listed separately from observed arrivals; excludes new recruitment",
        cases=[trace(c) for c in CASES])
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
if __name__ == "__main__":
    main()

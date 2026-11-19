"""Replay the three native arms through official Board; retain both actors' turning-point evidence."""
import argparse
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ACTOR = DEV.parents[1]
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ACTOR / "symmetry_dev/economy"))
import verify as official
sys.path.insert(0, str(BASE / "all_top5"))
from context import visibility
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon

ARMS = ("parent", "control", "candidate")
CHANGES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1), "NONE": (0, 0)}
CATEGORIES = {
    "deposits": ("own_delivery",),
    "trees": ("tree_absorption",),
    "hires": ("hire_collector", "hire_planter"),
    "canopy_cost": ("plant", "seize"),
    "other": ("tree_collision_cash", "base_collision_cash", "opponent_worker_delivery_to_this_base", "rounding"),
}
PLAN = "553f899c47df00482d3fba1f1e41d09ef3860026034deff3db2005c425a084fc"
read, sha, check = official.read, official.sha, official.check


def publish(path, value):
    if isinstance(value, str):
        raw = value.encode()
    else:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if path.suffix == ".gz":
            raw = gzip.compress(raw, mtime=0)
    if path.exists():
        check(path.read_bytes() == raw, f"published artifact changed: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=HERE, delete=False) as stream:
        stream.write(raw)
    os.replace(stream.name, path)


@lru_cache(maxsize=None)
def dest(pos, command, size):
    row, col = divmod(pos, size)
    dr, dc = CHANGES[command]
    return (row + dr) % size * size + (col + dc) % size


@lru_cache(maxsize=None)
def reach(pos, size):
    return frozenset(dest(pos, a, size) for a in CHANGES)


@contextmanager
def casualties():
    """Record at the official delete site, including post-move cargo and recruits lost on their first turn."""
    original = official.helpers.Board._delete_worker
    deaths = []
    def observe(board, worker):
        caller = sys._getframe(1)
        check(caller.f_code is official.helpers.Board.next.__code__, "changed official deletion route")
        frame = caller.f_locals
        site = caller.f_lineno
        line = source_lines.get(site, "")
        check(line in ("board._delete_worker(worker)",), "changed delete site")
        if site == collision_line:
            winner = frame["winner"]
            cell = worker.cell
            kind = "worker_collision"
            if winner is not None:
                sink = ["worker", winner.player_id, winner.id]
            elif cell.tree_id is not None:
                sink = ["tree_cash", cell.tree.player_id, cell.tree_id]
            elif cell.recrtCenter_id is not None:
                sink = ["base_cash", cell.recrtCenter.player_id, cell.recrtCenter_id]
            else:
                sink = ["cell", None, None]
        else:
            check(site == invasion_line, "unknown native deletion")
            kind, sink = "empty_at_opponent_base", ["none", None, None]
        deaths.append({"worker": worker.id, "owner": worker.player_id, "role": str(worker.occupation),
                       "cell": worker.position.to_index(board.configuration.size), "cargo": worker.carbon,
                       "reason": kind, "transfer_to": sink})
        return original(board, worker)
    import inspect
    body, start = inspect.getsourcelines(official.helpers.Board.next)
    source_lines = {start+i: x.strip() for i,x in enumerate(body)}
    collision_line, invasion_line = [i for i, s in source_lines.items() if s == "board._delete_worker(worker)"]
    official.helpers.Board._delete_worker = observe
    try:
        yield deaths
    finally:
        official.helpers.Board._delete_worker = original


def public_state(state, size):
    # Non-visible cells are omitted even if their true amount is zero; the index and amount identify visible zeros.
    visible = []
    for side in state["players"]:
        mask = visibility(side[2], size)
        visible.append([[i,state["carbon"][i]] for i,x in enumerate(mask) if x])
    return {"players": state["players"], "tree_ledger": state["trees"], "visible_carbon_by_physical_seat": visible}


def threats(state, joint, size):
    """Count a laden carrier once per pre-state, irrespective of how many light rivals can reach it."""
    threats_by_seat = [[], []]
    for owner, assets in enumerate(state["players"]):
        enemies = state["players"][1-owner][2]
        for uid, (pos, cargo, kind) in assets[2].items():
            if kind != "COLLECTOR" or cargo <= 0:
                continue
            possible = reach(pos, size)
            # Standard registered engine has moveCost=0; caller checks before using pre-action loads.
            raiders = {eid: reach(ep, size) for eid, (ep, ec, ek) in enemies.items()
                       if ek == "COLLECTOR" and ec < cargo and possible & reach(ep, size)}
            if not raiders:
                continue
            home = set(assets[1].values()) & possible
            contested_home = {b for b in home if any(b in cells for cells in raiders.values())}
            actual_enemy_dest = {eid:dest(enemies[eid][0], joint[1-owner].get(eid, "NONE"), size) for eid in raiders}
            actual_dest = dest(pos, joint[owner].get(uid, "NONE"), size)
            threats_by_seat[owner].append({
                "carrier": uid, "cargo": cargo, "raiders": sorted(raiders), "threatened_home": sorted(contested_home),
                "raider_moved_onto_threatened_home": sorted(eid for eid,cell in actual_enemy_dest.items() if cell in contested_home),
                "actual_contesters": sorted(eid for eid,x in actual_enemy_dest.items() if x == actual_dest)})
    return threats_by_seat


def canopy_threats(state, post, config, joint, deaths):
    """Direct seizure windows: a rival planter already on a surviving tree with enough initial cash."""
    result = [[], []]
    dead = {row["worker"]: row for row in deaths}
    size = config["size"]
    for owner, assets in enumerate(state["players"]):
        opposing = state["players"][1-owner]
        if opposing[0] < config["seizeCost"]:
            continue
        for tid,(pos,age) in assets[3].items():
            if age >= config["treeLifespan"]:
                continue
            invaders = [wid for wid,(ep,_,role) in opposing[2].items() if ep==pos and role=="PLANTER"]
            for wid in invaders:
                guards = [uid for uid,(p,_,role) in assets[2].items() if role=="COLLECTOR" and pos in reach(p,size)]
                actual = [uid for uid in guards if dest(assets[2][uid][0],joint[owner].get(uid,"NONE"),size)==pos]
                kill = dead.get(wid, {})
                result[owner].append({"tree":tid,"pos":pos,"planter":wid,"raider_stayed":joint[1-owner].get(wid,"NONE")=="NONE",
                                      "possible_guard":sorted(guards),"actual_guard_at_tree":sorted(actual),
                                      "invader_killed_by_own_collector":kill.get("transfer_to",[])[:2]==["worker",owner] and kill["cell"]==pos,
                                      "seized":tid in post["players"][1-owner][3]})
    return result


def one_game(arm, certificate, package_sha, with_student):
    s, seed = certificate["seat"], certificate["seed"]
    filepath = DEV/arm/f"raw_{seed}_{s}.json.gz"
    tape = read(filepath)
    check((sha(filepath), tape["pack_sha256"], tape["candidate_seat"], tape["seed"]) ==
          (certificate["transcript_sha256"], package_sha, s, seed), "source identity differs")
    check(sha_cells(tape["initial_full_carbon"]) == certificate["map_sha256"], "terrain differs")
    config = tape["game_configuration"]
    check(float(config["moveCost"]) == 0, "geometric opportunity assumes zero movement cost")
    size = config["size"]
    opening = official.initial(tape)
    current = opening
    cash = [Counter(), Counter()]
    counters = [Counter(), Counter()]
    turns = []
    leads = [opening["players"][s][0] - opening["players"][1-s][0]]
    for_replay = []
    with official.ledger() as events, casualties() as deaths:
        for t, joint in enumerate(tape["actions"]):
            threats_now = threats(current, joint, size)
            events.clear()
            deaths.clear()
            post = official.helpers.Board(current, config, joint).next().observation
            sides = [Counter(), Counter()]
            for kind, owner, amount, wid, cell, tid in events:
                sides[owner][kind] += amount
                cash[owner][kind] += amount
            canopy = canopy_threats(current, post, config, joint, deaths)
            by_worker = {e["worker"]: e for e in deaths}
            deposits = {wid: val for kind,p,val,wid,cell,tid in events if kind == "own_delivery" and val > 0}
            for owner in (0,1):
                check(abs(post["players"][owner][0]-current["players"][owner][0]-sum(sides[owner].values())) < 1e-6, "unbalanced turn")
                for row in canopy[owner]:
                    counters[owner]["tree_seizure_windows"] += 1
                    counters[owner]["tree_windows_with_reachable_guard"] += bool(row["possible_guard"])
                    counters[owner]["tree_windows_guard_on_cell"] += bool(row["actual_guard_at_tree"])
                    counters[owner]["tree_windows_guard_kill"] += row["invader_killed_by_own_collector"]
                    counters[owner]["tree_windows_seized"] += row["seized"]
                for risk in threats_now[owner]:
                    uid = risk["carrier"]
                    risk["survived"] = uid in post["players"][owner][2]
                    risk["deposit"] = deposits.get(uid, 0)
                    sink = by_worker.get(uid, {}).get("transfer_to", [None,None,None])
                    risk["captured_by_enemy_collector"] = sink[:2] == ["worker", 1-owner]
                    c = counters[owner]
                    c["carrier_turns_exposed"] += 1
                    c["carrier_turns_survived"] += int(risk["survived"])
                    c["carrier_turns_captured"] += int(risk["captured_by_enemy_collector"])
                    c["carrier_turns_banked"] += int(risk["deposit"] > 0)
                    if risk["threatened_home"]:
                        c["immediate_home_windows"] += 1
                        c["immediate_home_contested"] += bool(risk["raider_moved_onto_threatened_home"])
                        c["immediate_home_banked"] += int(risk["deposit"] > 0)
                        c["immediate_home_deposits"] += risk["deposit"]
                for kill in deaths:
                    if kill["owner"] != owner:
                        continue
                    counters[owner]["lost_"+kill["role"].lower()] += 1
                    if kill["transfer_to"][:2] == ["worker", 1-owner]:
                        counters[owner]["cargo_captured_by_enemy_collector"] += kill["cargo"]
                        counters[owner]["workers_taken_by_enemy_collector"] += 1
            leads.append(post["players"][s][0]-post["players"][1-s][0])
            turns.append({"pre":t, "cash_gain_by_seat":[dict(x) for x in sides], "deaths":list(deaths),
                          "salient_official_payments":[e for e in events if e[0] not in ("tree_absorption","rounding") and abs(e[2])>1e-9],
                          "freight_exposures":threats_now, "tree_exposures":canopy})
            for_replay.append({"pre": t,"before":public_state(current, size),"orders":joint})
            current = post
    final = [p[0] for p in current["players"]]
    scores = certificate["official_reward_by_seat"]
    check(final == tape["last_post_cash"] and scores == tape["official_score_by_seat"] and
          len(turns) == certificate["steps"] and certificate["statuses"] == ["DONE","DONE"], "terminal accounts differ")
    correction = [score-balance for score,balance in zip(scores,final,strict=True)]
    windows = {}
    for width in (1,10):
        width = min(width, len(turns))
        delta = [leads[t+width]-leads[t] for t in range(len(turns)+1-width)]
        for label, order in (("loss",min),("gain",max)):
            start = order(range(len(delta)), key=lambda i: delta[i])
            begin = max(0,start-10)
            windows[f"{label}_{width}"] = {
                "pre": start, "post": start+width, "candidate_minus_enemy_cash_change": delta[start],
                "pre_context_at": {str(k):start-k if start>=k else None for k in (3,5,10)},
                "context":for_replay[begin:start+width]}
    result = {"arm": arm,"seed": seed,"seat": s,"terrain_sha256":certificate["map_sha256"],
              "raw_sha256":sha(filepath),"pack_sha256":package_sha,"steps":len(turns),"scores":scores,
              "official_margin":scores[s]-scores[1-s],"closing_cash":final,"official_minus_closing_cash":correction,
              "physical_cash": [dict(c) for c in cash],"physical_pressure": [dict(c) for c in counters],
              "cash_leads":leads,"turns":turns,"extreme_windows":windows}
    check(abs(sum(cash[s].values())-sum(cash[1-s].values())+leads[0]+correction[s]-correction[1-s]
              -result["official_margin"])<1e-5,"terminal margin not reconciled")
    if with_student:
        result["sapling_contribution"] = ablate(tape)
    return result


def sha_cells(cells):
    check(all(int(v)==v and 0<=v<=255 for v in cells),"unexpected opening terrain encoding")
    return hashlib.sha256(bytes(map(int,cells))).hexdigest()


def ablate(tape):
    pack = DEV/"candidate.py"
    spec = importlib.util.spec_from_file_location("sapling_economic_actual_ablation", pack)
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    decode = package._mixed_actions
    inputs = []
    def capture(x, aux):
        inputs[:] = [x, aux]
        return decode(x, aux)
    package._mixed_actions = capture
    sim = FastCarbon(tape["game_configuration"]).reset(seed=tape["seed"])
    s = tape["candidate_seat"]
    counts = Counter()
    changed = []
    for t, joint in enumerate(tape["actions"]):
        obs = sim.obs(s)
        chosen = package.agent(obs, sim.config)
        check(chosen == joint[s],"candidate failed to reproduce own recorded command")
        check((package._HYBRID["expert_weight"],package._HYBRID["student_weight"]) == (.75,.25),"changed mix")
        try:
            package._HYBRID.update(expert_weight=1.,student_weight=0.)
            no_student = package._FEAT.to_actions(decode(*inputs))
        finally:
            package._HYBRID.update(expert_weight=.75,student_weight=.25)
        edits = []
        for uid in sorted(set(chosen) | set(no_student)):
            if chosen.get(uid) != no_student.get(uid):
                worker = obs["players"][s][2].get(uid)
                role = worker[2].lower() if worker else "base"
                counts[role]+=1
                edits.append([uid, role, no_student.get(uid,"NONE"), chosen.get(uid,"NONE")])
        if edits:
            changed.append([t,edits])
        sim.step(*joint)
    check(list(sim.cash)==tape["last_post_cash"],"policy observation replay differs")
    return {"steps":len(tape["actions"]), "changed_steps":len(changed),"changed_orders_by_role":dict(counts),"events":changed}


def process(arm):
    certificate_path = DEV/arm/"official_replay.json"
    release = read(DEV/"release.json")
    registration = read(DEV/"protocol.json")
    check(sha(DEV/"protocol.json")==PLAN==release["protocol_sha256"],"registration differs")
    certificate = read(certificate_path)
    for rel,h in certificate["engine_source_sha256"].items():
        check(sha(official.ENGINE/rel)==h, "different engine")
    arm_cert = certificate["runs"][arm]
    official_scores = read(DEV/arm/"results.json")
    rows = {(r["seed"],r["candidate_seat"]):r for r in official_scores["games"]}
    check(official_scores["header"]["examiner_md5"]==registration["examiner_md5"],"different examiner")
    pack = release[arm+"_pack_sha256"]
    require_roster = {(seed, s) for seed in registration["maps"] for s in registration["physical_seats"]}
    check(arm_cert["pack_sha256"]==pack==official_scores["header"]["pack_sha256"] and
          len(arm_cert["games"])==len(rows)==4 and
          {(c["seed"],c["seat"]) for c in arm_cert["games"]} == require_roster == rows.keys(),"different cohort")
    evidence = []
    for c in arm_cert["games"]:
        r = rows[c["seed"],c["seat"]]
        check(r["official_score_by_seat"]==c["official_reward_by_seat"] and
              r["raw_sha256"]==c["transcript_sha256"],"different result")
        evidence.append(one_game(arm,c,pack,arm=="candidate"))
    publish(HERE/(arm+".json.gz"),{"official_replay_sha256":sha(certificate_path),"engine_sha256":certificate["engine_source_sha256"],
                                    "implementation_sha256":sha(Path(__file__)),"games":evidence})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=ARMS)
    process(parser.parse_args().arm)

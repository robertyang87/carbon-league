"""Replay every Top 5 appearance from the published initial board and index every score card."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import inspect
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "code"
GAME = CODE / "env_src/carbon_challenge-master"
sys.path[:0] = [str(CODE), str(GAME)]
from fastsim import FastCarbon
sys.path.insert(0, str(HERE.parent))
from map_identity import terrain_sha256, IDENTITY


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")


def store(path, blob):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)


def base(row):
    return {"id": row["id"], "cohort": row["cohort"], "seat_names": row["seat_names"],
            "score_by_seat": row["score_by_seat"], "verdict": "invalid", "terminal_kind": "unknown",
            "frames": 0, "engine_steps_matched": 0, "issues": {}}


def check_one(row, cache, receipt):
    out = base(row)
    issues = out["issues"]

    def check(ok, field, t=None):
        if not ok:
            item = issues.setdefault(field, {"count": 0, "first_frame": t})
            item["count"] += 1
        return ok

    packed = (cache / row["path"]).read_bytes()
    env = json.loads(gzip.decompress(packed))
    cfg, steps = env["configuration"], env["steps"]
    last = len(steps) - 1
    initial = steps[0][0]["observation"]
    seed = cfg["randomSeed"]
    canonical_hash = sha(encoded(env))
    map_hash = sha(encoded(initial["full_carbon"]))
    card = row["card"]
    out.update(frames=len(steps), seed=seed, compressed_sha256=sha(packed),
               canonical_sha256=canonical_hash, initial_map_sha256=map_hash,
               environment_uuid=env["id"], statuses=env["statuses"], final_rewards=env["rewards"])
    check(len(steps) <= cfg["episodeSteps"] and len(steps) >= 2, "episode_length")
    order = card["order"]
    check(sorted(order) == ["a", "b"] and [card["name_" + key] for key in order] == row["seat_names"], "physical_seat_names")
    check(row["id"] == f'{card["id"]}-{card["game"]}', "card_id")
    check([card["scores"][k] for k in order] == row["score_by_seat"], "score_order")
    r = env["rewards"]
    score_ok = len(r) == 2 and all(isinstance(v, (float, int)) and abs(v - ref) <= 0.005001
                                  for v, ref in zip(r, row["score_by_seat"]))
    out["card_scores_match"] = check(score_ok, "terminal_card_score")
    if score_ok:
        winner = order[0 if r[0] > r[1] else 1] if r[0] != r[1] else None
        check(card["winner"] == winner or (winner is None and card["winner"] in ("draw", "tie")), "winner")
    for key, actual in (("id",row["id"]), ("canonical_sha256", canonical_hash), ("sha256",out["compressed_sha256"]),
                        ("environment_uuid",env["id"]), ("transitions",last), ("seed",seed), ("terminal",env["statuses"])):
        check(receipt.get(key) == actual, "receipt_" + key)
    sim = FastCarbon(cfg)
    sim.reset(seed=seed)
    out["seed_alone_recreates_map"] = sim.carbon == initial["full_carbon"]
    check(sim._players_obs() == initial["players"], "initial_seat_assets", 0)
    check(initial["trees"] == {}, "initial_tree_ledger", 0)
    sim.load_observation(initial)
    horizon = cfg["episodeSteps"] - 1
    extinct = []
    checked_actions = 0
    published_cash = None
    for t, pair in enumerate(steps):
        check(isinstance(pair, list) and len(pair) == 2, "pair", t)
        obs = pair[0]["observation"]
        check(obs["step"] == t, "step_sequence", t)
        statuses = [a["status"] for a in pair]
        check(t == last or statuses == ["ACTIVE", "ACTIVE"], "nonfinal_status", t)
        actions = [a.get("action") for a in pair]
        faults = [seat for seat, status in enumerate(statuses) if status not in {"ACTIVE","DONE"}]
        check(all(isinstance(a, dict) or (p in faults and a is None) for p, a in enumerate(actions)), "action_schema", t)
        if t:
            sim.step(actions[0] or {}, actions[1] or {})
            checked_actions += 2
        expected = sim._players_obs()
        extinct = [p for p,(cash,bases,workers,trees) in enumerate(expected)
                   if p not in faults and not workers and not trees and
                   (not bases or cash < min(cfg["recCollectorCost"], cfg["recPlanterCost"]))]
        for seat in faults:
            expected[seat] = [0, expected[seat][1], {}, {}]
        frame_ok = check(expected == obs["players"], "engine_seat_assets", t)
        frame_ok = check(sim.trees_dict == obs["trees"], "engine_tree_creator_and_income", t) and frame_ok
        frame_ok = check(sim.carbon == obs["full_carbon"], "engine_full_carbon", t) and frame_ok
        for seat, entry in enumerate(pair):
            visible = [0] * sim.n_cells
            for wid in sim.player_workers[seat]:
                worker = sim.workers[wid]
                positions = (worker.pos,) if worker.is_collector else sim._nbr4[worker.pos]
                for pos in positions:
                    visible[pos] = sim.carbon[pos]
            view = entry["observation"]
            frame_ok = check(visible == view["carbon"], f"visible_carbon_s{seat}", t) and frame_ok
            frame_ok = check(view.get("player") == seat and view.get("step",t) == t,
                             f"observation_seat_step_s{seat}", t) and frame_ok
            if seat:
                for key in ("players","trees","full_carbon"):
                    if key in view:
                        frame_ok = check(view[key] == obs[key], f"duplicate_{key}", t) and frame_ok
        rewards = [a["reward"] for a in pair]
        if t:
            if faults or extinct:
                expected_rewards = [0 if p in faults else t - cfg["episodeSteps"] - 1 if p in extinct
                                    else published_cash[p] for p in range(2)]
            else:
                expected_rewards = [assets[0] for assets in expected]
            frame_ok = check(expected_rewards == rewards, "official_reward_rule", t) and frame_ok
        else:
            frame_ok = check(rewards == [a[0] for a in expected], "initial_reward", t) and frame_ok
        ended = bool(faults or extinct or t == horizon)
        check((t == last) == ended, "terminal_boundary", t)
        if t and frame_ok:
            out["engine_steps_matched"] += 1
        published_cash = rewards
    check(statuses == env["statuses"] and rewards == env["rewards"], "terminal_header", last)
    kind = ("timeout" if any("TIME" in statuses[s].upper() for s in faults) else "execution_fault") if faults else (
        "extinction" if extinct else "horizon" if last == horizon else "unexplained_early")
    out.update(terminal_kind=kind, extinct_seats=extinct, fault_seats=faults, joint_action_seats=checked_actions,
               final_cash=[p[0] for p in steps[-1][0]["observation"]["players"]],
               verdict="consistent" if not issues else "discrepancy")
    return out


def official(row, cache):
    from zerosum_env.envs.carbon.helpers import Board
    from zerosum_env.envs.carbon import idgen
    env = json.load(gzip.open(cache / row["path"]))
    steps = env["steps"]
    idgen.reset()
    state = copy.deepcopy(steps[0][0]["observation"])
    state["carbon"] = state["full_carbon"]
    for t, pair in enumerate(steps[1:], 1):
        state = Board(state, env["configuration"], [entry["action"] or {} for entry in pair]).next().observation
        actual = pair[0]["observation"]
        players = copy.deepcopy(state["players"])
        for seat,entry in enumerate(pair):
            if entry["status"] not in {"DONE","ACTIVE"}:
                players[seat] = [0, players[seat][1], {}, {}]
        for field, expected, got in (("full_carbon",state["carbon"],actual["full_carbon"]),
                                     ("players",players,actual["players"]),("trees",state["trees"],actual["trees"])):
            if expected != got:
                return {"matched": t-1, "first_divergence": {"frame":t, "field":field}}
    return {"matched": len(steps)-1, "first_divergence": None}


def process(manifest,cache,work,source_hash):
    added = 0
    for row in manifest["games"]:
        key = row["id"]
        dest = work / (key + ".json")
        cert = cache / "receipts" / (key + ".json")
        if not (cache / row["path"]).exists():
            continue
        if cert.exists():
            receipt = json.loads(cert.read_bytes())
        elif row.get("frame_verified"):
            receipt = row  # fetch.py --restore rehydrates payloads; bulk manifest supplies their hashes.
        else:
            continue
        if dest.exists():
            old = json.loads(dest.read_bytes())
            if old.get("source_hash") == source_hash and old.get("compressed_sha256") == receipt.get("sha256"):
                continue
        try:
            audited = check_one(row, cache, receipt)
        except (OSError, ValueError, IndexError, TypeError, KeyError, AssertionError) as exc:
            audited = base(row)
            audited.update(issues={"decode_or_structure": str(exc)}, compressed_sha256=receipt.get("sha256"))
        audited["source_hash"] = source_hash
        store(dest, encoded(audited))
        added += 1
        if added % 100 == 0:
            print(f"validated {added} more",flush=True)
    return added


def report(manifest,cache,work,source_hash):
    rows = []
    evid = {}
    originals = {}
    for row in manifest["games"]:
        dest = work / (row["id"] + ".json")
        if dest.exists():
            result = json.loads(dest.read_bytes())
            assert result.pop("source_hash") == source_hash
        else:
            result = base(row)
            result.update(verdict="score_card_only",terminal_kind="missing_replay",
                          issues={"fetch": row.get("errors", row.get("source_attempt_errors", "no verified receipt and payload"))})
        evid[result["id"]] = row
        rows.append(result)
    assert len(rows) == len(evid)
    map_rows = []
    for result in rows:
        if not result.get("frames"):
            continue
        game = evid[result["id"]]
        with gzip.open(cache / game["path"]) as handle:
            terrain = json.load(handle)["steps"][0][0]["observation"]["full_carbon"]
        if sha(encoded(terrain)) != result["initial_map_sha256"]:
            raise ValueError("initial terrain changed between verification and ledger")
        result["initial_terrain_u8_sha256"] = terrain_sha256(terrain)
        map_rows.append({key: result[key] for key in ("id","seed","initial_terrain_u8_sha256")})
    map_bytes = gzip.compress(b"".join(encoded(r)+b"\n" for r in sorted(map_rows,key=lambda x:x["id"])),mtime=0)
    store(HERE/"validation_maps.jsonl.gz",map_bytes)
    # Deterministic probes by cohort and terminal class. Include every divergence class and every early/fault type.
    samples = {}
    for row in sorted(rows,key=lambda x:x["id"]):
        if row["frames"]:
            cls = (row["cohort"],row["terminal_kind"],",".join(sorted(row["issues"])))
            samples.setdefault(cls, row)
    for row in samples.values():
        result = official(evid[row["id"]],cache)
        originals[row["id"]] = result
        row["original_board"] = result
    duplicates = {}
    for field in ("environment_uuid","canonical_sha256","initial_terrain_u8_sha256","seed"):
        groups = defaultdict(list)
        for r in rows:
            if field in r:
                groups[r[field]].append(r["id"])
        duplicates[field] = {str(key): ids for key,ids in groups.items() if len(ids)>1}
    summary = dict(revision=manifest["revision"],input_manifest_sha256=sha((HERE/"manifest.json").read_bytes()),
                   source_hash=source_hash, validation_script_sha256=sha(Path(__file__).read_bytes()), engine_sha256=sha((CODE/"fastsim.py").read_bytes()),
                   sim_source="code/fastsim.py", official_source="code/env_src/carbon_challenge-master/zerosum_env/envs/carbon",
                   terrain_hash_encoding=IDENTITY, map_ledger_sha256=sha(map_bytes), cards=len(rows), frames=sum(r["frames"] for r in rows),
                   transitions=sum(max(r["frames"]-1,0) for r in rows), matched=sum(r["engine_steps_matched"] for r in rows),
                   count_verdict=dict(Counter(r["verdict"] for r in rows)),
                   count_terminal=dict(Counter(r["terminal_kind"] for r in rows)),
                   count_cohort=dict(Counter(r["cohort"] for r in rows)),
                   seat_scores_match=sum(r.get("card_scores_match",False) for r in rows),
                   seeded_maps_recreated=sum(r.get("seed_alone_recreates_map",False) for r in rows),
                   issue_games=dict(Counter(i for r in rows for i in r["issues"])),
                   original_board=originals,duplicate_groups=duplicates)
    store(HERE/"validation_games.jsonl.gz",gzip.compress(b"".join(encoded(r)+b"\n" for r in rows), mtime=0))
    store(HERE/"validation.json",json.dumps(summary,ensure_ascii=False,indent=2).encode()+b"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    parser.add_argument("--follow",action="store_true",help="Keep reading when fetch has more score cards pending")
    args=parser.parse_args()
    work=args.cache/"validation_work"
    work.mkdir(parents=True,exist_ok=True)
    source_hash = sha(b"".join(inspect.getsource(f).encode() for f in (base, check_one, encoded, sha)) + (CODE/"fastsim.py").read_bytes())
    while True:
        m=json.loads((HERE/"manifest.json").read_bytes())
        added=process(m,args.cache,work,source_hash)
        print("checked",len(list(work.glob("*.json"))),"/",len(m["games"]),"added",added,"manifest",m["state"],flush=True)
        if not args.follow or len(list(work.glob("*.json"))) >= len(m["games"]) or m["state"] in {"complete","partial"}:
            print("summary",report(m,args.cache,work,source_hash)["count_verdict"],flush=True)
            break
        time.sleep(4)


if __name__ == "__main__":
    main()

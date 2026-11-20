"""Expose official ladder actions and rule-reconciled cash flows for the 30 known Top Five–Master3s2 matches."""
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAP = HERE.parents[2]
DIRECT = SAP / "all_top5/master3s2_expert_orders"
LIB = SAP / "master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py"
FROZEN_MANIFEST = "557c34a9194401b03eb7e43876f3b291eda3372055daac450866606731a6d1bb"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = load_module("frozen_expert_orders", DIRECT / "inventory.py")
ledger = load_module("frozen_match_cash", LIB)
require = ledger.require
sha = inventory.sha


@lru_cache
def source_batch(path):
    return {c["id"]: c for c in json.loads(inventory.blob(path))["games"]}


@lru_cache
def score_index(path):
    return {f'{r["id"]}-{r["game"]}': r for r in json.loads(inventory.blob(path))
            if isinstance(r, dict) and r.get("state") == "done" and r.get("game") is not None}


def check_registration(card):
    seat = card["seat"]
    mirror = source_batch(card["source_batch"])[card["id"]]
    names = mirror["seat_names"]
    require(names[seat] == card["expert"] and names[1-seat] == inventory.EXAM, "bad physical seats")
    require(mirror["seat_strategy_ids"][seat] == card["expert_strategy_id"], "different expert registration")
    # Some original match cards only publish the examiner's name. Its registration ID
    # is verified separately against the published source record.
    require(mirror["seat_strategy_ids"][1-seat] in (None, inventory.EXAM_ID), "different examiner registration")
    require(card["examiner_registration"] == inventory.EXAM_ID and mirror["frame_verified"] and
            mirror["clean_terminal"] and mirror["sha256"] == card["replay_sha256"], "invalid official card")
    require(card["score_indexes"], "no direct score reference")
    for path in card["score_indexes"]:
        g = score_index(path)[card["id"]]
        require([g["name_" + x] for x in g["order"]] == names and
                [g["scores"][x] for x in g["order"]] == card["score_by_seat"], "official score index mismatch")


def check_board(sim, obs):
    current = sim.obs(0)
    require(sim.step_count == obs["step"] and current["players"] == obs["players"] and current["trees"] == obs["trees"],
            f'entity state differs from recorded frame at {obs["step"]}')
    require(all(math.isclose(x, y, rel_tol=0., abs_tol=1e-6)
                for x,y in zip(sim.carbon, obs["full_carbon"], strict=True)), "carbon diverged from official frame")


def replay_steps(card, raw):
    frames = raw["steps"]
    first = frames[0][0]["observation"]
    sim = ledger.FastCarbon(raw["configuration"]).load_observation(first)
    sim.cash = ledger.CashEvents(sim.cash)
    check_board(sim, first)
    orders_digest = hashlib.sha256()
    for t, frame in enumerate(frames[1:]):
        before_obs = frames[t][0]["observation"]
        after_obs = frame[0]["observation"]
        require(before_obs["step"] == t and after_obs["step"] == t + 1 and
                all(s["status"] == "ACTIVE" for s in frames[t]), "incomplete official decision")
        orders = [side["action"] or {} for side in frame]
        orders_digest.update(inventory.canonical(orders) + b"\n")
        sim.cash.events.clear()
        sim.step(*orders)
        check_board(sim, after_obs)
        flow = ledger.cash_flow(sim.cash.events)
        for s in (0, 1):
            ledger.check_equal(sum(flow[s].values()), after_obs["players"][s][0] - before_obs["players"][s][0],
                               f'{card["id"]}, turn {t}, seat {s}: component reconciliation')
        yield t, before_obs["players"], orders, flow, after_obs["players"]
    require(orders_digest.hexdigest() == card["ledger"]["joint_orders_sha256"], "order identity differs")
    require(sim.step_count == card["steps"] == raw["configuration"]["episodeSteps"] - 1 and
            raw["statuses"] == [s["status"] for s in frames[-1]] == ["DONE", "DONE"] and
            raw["rewards"] == [s["reward"] for s in frames[-1]] == card["score_by_seat"], "incorrect official terminal")


def games():
    """Yield (identity, checked stream); exhaust steps to check the ordered command digest and official terminal."""
    document = (DIRECT / "manifest.json").read_bytes()
    require(sha(document) == FROZEN_MANIFEST, "direct-game manifest changed")
    manifest = json.loads(document)
    cards = manifest["games"]
    require(manifest["frozen_commit"] == inventory.REV and len(cards) == len({r["id"] for r in cards})
            == len({r["terrain"] for r in cards}) == 30, "incorrect map or game denominator")
    published = inventory.blob("sources/master3s2/README.md")
    require(inventory.EXAM_ID.encode() in published and sha(inventory.blob("sources/master3s2/master3s2.py"))
            == manifest["examiner_reference_sha256"] == inventory.EXAM_SOURCE, "wrong published opponent")
    # Group tar members to avoid repeatedly reading the same large immutable object.
    for card in sorted(cards, key=lambda c: (c["archive"].get("archive", ""), c["id"])):
        check_registration(card)
        contents = inventory.raw_bytes(card["archive"])
        require(sha(contents) == card["replay_sha256"], "wrong replay bytes")
        raw = json.loads(gzip.decompress(contents))
        first = raw["steps"][0][0]["observation"]
        require(raw["id"] == card["uuid"] and raw["configuration"]["randomSeed"] == card["seed"]
                and sha(bytes(first["full_carbon"])) == card["terrain"] and first["step"] == 0
                and len(raw["steps"])-1 == card["steps"], "different replay, map, seed or length")
        meta = dict(id=card["id"], terrain=card["terrain"], seed=card["seed"], actor=card["expert"], seat=card["seat"],
                    official_scores_by_seat=card["score_by_seat"], source="ladder", source_method=card["source_method"],
                    registration_by_name=inventory.EXAM, reference_registration_id=inventory.EXAM_ID,
                    reference_examiner_sha256=inventory.EXAM_SOURCE, in_game_examiner_sha256=None,
                    in_game_search_depth_and_clock=None, configuration=raw["configuration"],
                    recorded_time_control=raw.get("info", {}).get("platform_time_control"),
                    archive=card["archive"], replay_sha256=card["replay_sha256"],
                    joint_orders_sha256=card["ledger"]["joint_orders_sha256"], uuid=card["uuid"],
                    source_game_disposition=card["disposition"])
        yield meta, replay_steps(card, raw)


if __name__ == "__main__":
    games_checked = steps_checked = wins = 0
    for meta, records in games():
        count = sum(1 for _ in records)
        seat = meta["seat"]
        scores = meta["official_scores_by_seat"]
        wins += int(scores[seat] > scores[1-seat])
        games_checked += 1
        steps_checked += count
        print(meta["id"], seat, count, scores, flush=True)
    require((games_checked, steps_checked, wins) == (30, 8970, 29), "wrong completed scope")
    print(json.dumps(dict(games=games_checked, turns=steps_checked, expert_wins=wins), sort_keys=True))

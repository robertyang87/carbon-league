"""Replay a 500-cash deficit and the first recovery in Sapling and ladder games against Master3s2."""
from collections import Counter
from pathlib import Path
from statistics import median
import argparse
import gzip
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from turning_point_ledger import CashEvents, FastCarbon
sys.path.insert(0, str(HERE.parent / "planter_intent_labels"))
from source import CACHE, raw_game


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def games(cache):
    # This analysis compares the original 54 games with the frozen ladder cohort.
    revision = "3d02c08f78e4a2efe954d882c3b8e8648aed2411"
    ledger = (HERE / "latest_games.json").relative_to(ROOT)
    parent = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{ledger}"]))
    for row in parent["games"]:
        path = ROOT / row["sources"][0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["raw_sha256"]:
            raise ValueError(f"Parent tape changed: {path}")
        tape = read(path)
        sim = FastCarbon(tape["game_configuration"]).reset(seed=row["seed"])
        yield dict(cohort="Sapling", entrant="v1", id=path.name, seed=row["seed"], seat=row["seat"],
                   sha256=row["raw_sha256"], board=row["map"]), sim, tape["actions"], tape["official_score_by_seat"], None

    direct = read(BASE / "all_top5/master3s2_direct.json")
    maps = read(HERE.parent / "planter_intent_labels/maps.json.gz")["maps"]
    for appendix in sorted(HERE.parent.glob("planter_intent_labels_holdout_*/maps.json")):
        maps.extend(read(appendix)["maps"])
    by_id = {m["id"]: m for m in maps}
    for row in direct["games"]:
        item = by_id[row["id"]]
        replay = raw_game(item, cache)
        frames = replay["steps"]
        score = [p["reward"] for p in frames[-1]]
        seat = row["entrant_seat"]
        if item["sha256"] != row["source_sha256"] or [score[seat], score[1-seat]] != row["score"]:
            raise ValueError(f'Ladder source or score changed: {row["id"]}')
        sim = FastCarbon(replay["configuration"]).load_observation(frames[0][0]["observation"])
        yield dict(cohort="ladder", entrant=row["entrant"], id=row["id"], seat=seat, seed=row["seed"],
                   examiner_registration_id=item["seat_strategy_ids"][1-seat],
                   sha256=item["sha256"], board=item["terrain"]), sim, [[v["action"] or {} for v in frame]
                   for frame in frames[1:]], score, frames


def trace(info, sim, orders, score, frames):
    seat = info["seat"]
    sim.cash = CashEvents(sim.cash)
    transitions = []
    leads = [round(sim.cash[seat] - sim.cash[1-seat], 2)]
    for turn, joint in enumerate(orders):
        players = sim.obs(seat)["players"]
        before = list(sim.cash)
        sim.cash.events.clear()
        sim.step(*joint)
        totals = [Counter(), Counter()]
        for kind, owner, amount, *_ in sim.cash.events:
            totals[owner][kind] += amount
        if any(abs(before[i] + sum(totals[i].values()) - sim.cash[i]) > 0.00001 for i in (0,1)):
            raise ValueError((info["id"], turn, "cash accounting"))
        after = list(sim.cash)
        if frames and any(abs(after[i] - frames[turn+1][0]["observation"]["players"][i][0]) > .011 for i in (0,1)):
            raise ValueError((info["id"], turn, "platform trajectory"))
        transitions.append(dict(turn=turn, pre_players=players, joint=joint, cash_after=after,
                                cash_events=list(sim.cash.events), channel_by_seat=[dict(t) for t in totals]))
        leads.append(round(after[seat] - after[1-seat], 2))
    crossed = next((i for i, lead in enumerate(leads) if lead <= -500), None)
    recovered = next((i for i in range(crossed + 1, len(leads)) if leads[i] >= 0), None) if crossed is not None else None
    anchors = {}
    for name, index in (("first_deficit", crossed), ("first_recovery", recovered)):
        if index is None:
            continue
        windows = []
        for preceding in (3, 5, 10):
            start = max(0, index - preceding)
            diff = Counter()
            for trans in transitions[start:index]:
                diff.update(trans["channel_by_seat"][seat])
                diff.subtract(trans["channel_by_seat"][1-seat])
            if abs(sum(diff.values()) - (leads[index] - leads[start])) > .02:
                raise ValueError("Lead change does not reconcile")
            windows.append(dict(from_state=start, to_state=index, lead_before=leads[start], lead_after=leads[index],
                                cash_components={k:round(v,4) for k,v in diff.items()}))
        anchors[name] = dict(post_state=index, windows=windows, turns=transitions[max(index-10,0):index])
    return dict(**info, official_scores=score, margin=round(score[seat]-score[1-seat],2),
                transitions=len(orders), cash_margin_end=leads[-1], anchors=anchors)


def summary(rows):
    crossed = [r for r in rows if "first_deficit" in r["anchors"]]
    recovered = [r for r in crossed if "first_recovery" in r["anchors"]]
    def wins(sample):
        return sum(r["margin"] > 0 for r in sample)
    def median_state(sample, key):
        return median(r["anchors"][key]["post_state"] for r in sample) if sample else None
    return dict(games=len(rows), maps=len({r["board"] for r in rows}), wins=wins(rows), ever_down_500=len(crossed),
                down_500_then_won=wins(crossed), recovered_to_even=len(recovered), recovered_then_won=wins(recovered),
                median_first_deficit=median_state(crossed, "first_deficit"),
                median_first_recovery=median_state(recovered, "first_recovery"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    rows = [trace(*record) for record in games(args.cache)]
    groups = sorted({r["entrant"] for r in rows})
    by_group = {g:summary([r for r in rows if r["entrant"]==g]) for g in groups}
    by_cohort = {g:summary([r for r in rows if r["cohort"]==g]) for g in ("Sapling","ladder")}
    encoded = json.dumps(dict(by_cohort=by_cohort, by_entrant=by_group, games=rows),
                         ensure_ascii=False, allow_nan=False, separators=(",",":")).encode()
    (HERE/"deficit_recovery.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    print(json.dumps(by_cohort))

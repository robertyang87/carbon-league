"""Evaluate the frozen survival veto on four archived v1 trajectories. No agents are run."""

import argparse
from collections import Counter, defaultdict
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SURVIVAL = HERE.parent
SOURCE = SURVIVAL.parent / "batches/intake_03"
sys.path.insert(0, str(SURVIVAL))
import extract
from model import score, MOVES

PROTOCOL = "5a0ffa43c06e52e868e6ab8f5d0af43ce25c055ad1e8b6a8b80c1a63f67252cd"
POLICIES = ("v1", "fitted", "geometry", "contact_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def registration():
    require(extract.sha(HERE / "protocol.json") == PROTOCOL, "changed predeclaration")
    plan = extract.load(HERE / "protocol.json")
    for name, h in plan["frozen"].items():
        require(extract.sha(SURVIVAL / name) == h, f"changed frozen input {name}")
    for name, field in [
        ("protocol.json", "protocol_sha256"),
        ("source_receipt.json", "receipt_sha256"),
        ("opportunities.jsonl.gz", "opportunities_sha256"),
        ("source/official_replay.json", "official_replay_sha256"),
    ]:
        require(extract.sha(SOURCE / name) == plan["source"][field], f"changed registered source {name}")
    return plan


def inputs(plan):
    actual = extract.load(SOURCE / "source_receipt.json")
    official = extract.load(SOURCE / "source/official_replay.json")["runs"]["source"]
    cfg = extract.load(SOURCE / "protocol.json")
    native = extract.load(SOURCE / "source/results.json")
    require(
        native["status"] == "complete"
        and native["header"]["pack_sha256"] == cfg["parent_pack_sha256"] == official["pack_sha256"]
        and native["header"]["seeds"] == plan["maps"]
        and actual["protocol_sha256"] == plan["source"]["protocol_sha256"]
        and actual["official_sha256"] == plan["source"]["official_replay_sha256"],
        "unbound parent or previous results",
    )
    games = {(g["seed"], g["seat"]): g for g in actual["games"]}
    verified = {(g["seed"], g["seat"]): g for g in official["games"]}
    expected = {(s, p) for s in plan["maps"] for p in plan["seats"]}
    require(
        set(games) == set(verified) == expected and len(actual["games"]) == len(official["games"]) == 4,
        "wrong map and seat allocation",
    )
    for key, g in games.items():
        o = verified[key]
        require(
            g["map_sha256"] == o["map_sha256"]
            and g["raw_sha256"] == o["transcript_sha256"]
            and g["scores"] == o["official_reward_by_seat"]
            and g["steps"] == o["steps"]
            and o["statuses"] == ["DONE", "DONE"],
            "signed source differs from official terminal",
        )
    return games


def filtered_index(plan):
    counts, index = extract.read_index(SOURCE, {**plan["source"], "maps": plan["maps"]})
    # Prove the chosen movement is reproducible after removing every field except pre-action legality and identity.
    seen = Counter()
    with gzip.open(SOURCE / "opportunities.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            restricted = {k: r[k] for k in ("seed", "seat", "t", "uid", "parent_command")}
            restricted["actions"] = {
                a: {k: r["actions"][a][k] for k in ("entry_legal", "friendly_conflict")} for a in MOVES
            }
            require(
                extract.pair(restricted) == extract.pair(r),
                "choice consumed information outside the public enrollment",
            )
            seen[r["seed"], r["seat"]] += 1
    eligible = {}
    for k, items in index.items():
        current = [item for item in items if item[1] == "NONE" and item[3] and item[4] is not None and item[4] > 0]
        if current:
            eligible[k] = current
    for k, c in counts.items():
        require(seen[k] == c["source_roots"], "missed projected movement")
    return counts, eligible


def name(seed, seat):
    return HERE / f"pair_{seed}_{seat}.json.gz"


def one(plan, game):
    dest = name(game["seed"], game["seat"])
    weights = extract.load(SURVIVAL / "weights.json")
    signature = {
        "protocol": PROTOCOL,
        "source_tape": game["raw_sha256"],
        "model": plan["frozen"]["weights.json"],
        "board": game["map_sha256"],
        "official_helpers": extract.sha(extract.ENGINE / "zerosum_env/envs/carbon/helpers.py"),
        "runner": extract.sha(Path(__file__)),
    }
    if dest.exists():
        require(extract.load(dest)["identity"] == signature, "changed shard or stale runner on resume")
        return
    counts, selected = filtered_index(plan)
    raw = extract.extract_game(game, selected)
    out = []
    for r in raw:
        h, m = r["actions"]
        require(
            r["payable"]
            and r["physical_site"] > 0
            and r["actual"] == h["name"] == "NONE"
            and m["name"] in MOVES
            and m["name"] != "NONE",
            "unintended root or original v1 action",
        )
        require(m["paid"] == 0 and not (h["died"] and h["paid"]), "purchase/died definition mismatch")
        for a in (h, m):
            a["p"] = score(weights, a.pop("x"), a["geometric"])
            a["contact_rate"] = weights["contact_constant"] * float(a["geometric"])
        out.append(r)
    required = sum(len(rs) for (s, p, _), rs in selected.items() if (s, p) == (game["seed"], game["seat"]))
    require(
        len(out) == required and len({(r["t"], r["uid"]) for r in out}) == required, "incomplete root coverage"
    )
    data = {
        "seed": game["seed"],
        "seat": game["seat"],
        "steps": game["steps"],
        "identity": signature,
        "enrollment": dict(counts[game["seed"], game["seat"]]),
        "positive_price_v1_stay_can_pay": len(out),
        "rows": out,
    }
    extract.write(dest, data)
    print(dest.name, len(out), extract.sha(dest), flush=True)


def trigger(policy, h, m, risk):
    if policy == "v1":
        return False
    key = {"fitted": "p", "geometry": "geometric", "contact_rate": "contact_rate"}[policy]
    x, y = map(float, (h[key], m[key]))
    return x >= risk["minimum_stay_probability"] and x >= y + risk["minimum_reduction"]


def effects(rows, policy, risk):
    counts = Counter()
    ordered = sorted(rows, key=lambda r: (r["seed"], r["seat"], r["t"], r["uid"]))
    previous_turn = None
    switched_this_turn = False
    triggered = []
    for r in ordered:
        turn = r["seed"], r["seat"], r["t"]
        if turn != previous_turn:
            switched_this_turn = False
        previous_turn = turn
        h, m = r["actions"]
        flagged = trigger(policy, h, m, risk)
        switched = flagged and not switched_this_turn
        switched_this_turn |= switched
        savable = h["died"] and not m["died"]
        dangerous = not h["died"] and m["died"]
        counts.update(
            dict(
                eligible=1,
                v1_paid=int(h["paid"] > 0),
                v1_deaths=int(h["died"]),
                both_die=int(h["died"] and m["died"]),
                savable=int(savable),
                move_would_kill=int(dangerous),
                all_isolated_triggers=int(flagged),
                edits=int(switched),
                saved=int(switched and savable),
                harmed=int(switched and dangerous),
                unrescued_preventable=int(not switched and savable),
                unrescued_v1_deaths=int(h["died"] and not (switched and savable)),
                unchanged_survival_after_edit=int(switched and h["died"] == m["died"]),
                foregone_purchases=int(switched and h["paid"] > 0),
                chosen_deaths=int((m if switched else h)["died"]),
            )
        )
        counts["foregone_debit"] += h["paid"] if switched else 0
        counts["v1_debit"] += h["paid"]
        if switched:
            triggered.append(
                [*turn, r["uid"], m["name"], h["p"], m["p"], h["died"], m["died"], h["kind"], h["paid"]]
            )
    counts["net_saved"] = counts["saved"] - counts["harmed"]
    require(
        counts["v1_deaths"] - counts["chosen_deaths"] == counts["net_saved"]
        and counts["savable"] == counts["saved"] + counts["unrescued_preventable"],
        "inconsistent survival balance",
    )
    names = "eligible v1_paid v1_debit v1_deaths savable both_die move_would_kill all_isolated_triggers edits saved harmed net_saved unrescued_preventable unrescued_v1_deaths unchanged_survival_after_edit foregone_purchases foregone_debit chosen_deaths".split()
    return {k: counts[k] for k in names}, triggered


def scorecard(plan):
    docs = [extract.load(name(s, p)) for s in plan["maps"] for p in plan["seats"]]
    require(
        all(
            d["identity"]["protocol"] == PROTOCOL and d["identity"]["runner"] == extract.sha(Path(__file__))
            for d in docs
        ),
        "invalid shard",
    )
    risk = plan["risk"]
    groups = {}
    examples = {}

    def summarize(selected):
        rows = [r for d in selected for r in d["rows"]]
        result = {
            "all_source_planter_roots": sum(d["enrollment"]["source_roots"] for d in selected),
            "all_legal_pairs": sum(d["enrollment"].get("eligible_roots", 0) for d in selected),
            "eligible_payment_intents": len(rows),
            "unique_worker_instances": len({(r["seed"], r["seat"], r["uid"]) for r in rows}),
        }
        acts = [a for r in rows for a in r["actions"]]
        result["brier"] = {
            k: (statistics.mean((float(a[f]) - int(a["died"])) ** 2 for a in acts) if acts else None)
            for k, f in [("fitted", "p"), ("geometric", "geometric"), ("contact_rate", "contact_rate")]
        }
        result["policies"] = {k: effects(rows, k, risk)[0] for k in POLICIES}
        return result

    for s in plan["maps"]:
        groups[str(s)] = summarize([d for d in docs if d["seed"] == s])
    by_seat = {f'{d["seed"]}/{d["seat"]}': summarize([d]) for d in docs}
    rows = [r for d in docs for r in d["rows"]]
    for policy in POLICIES:
        examples[policy] = effects(rows, policy, risk)[1]
    pooled = summarize(docs)
    metrics = (
        "edits",
        "saved",
        "harmed",
        "net_saved",
        "unrescued_preventable",
        "foregone_purchases",
        "foregone_debit",
    )
    rates = {
        policy: {
            k: (
                statistics.mean(
                    g["policies"][policy][k] / g["eligible_payment_intents"]
                    for g in groups.values()
                    if g["eligible_payment_intents"]
                )
                if any(g["eligible_payment_intents"] for g in groups.values())
                else None
            )
            for k in metrics
        }
        for policy in POLICIES
    }
    return {
        "protocol_sha256": PROTOCOL,
        "shards": {name(d["seed"], d["seat"]).name: extract.sha(name(d["seed"], d["seat"])) for d in docs},
        "maps": groups,
        "physical_map_seats": by_seat,
        "total": pooled,
        "equal_map_rate_per_eligible_root": rates,
        "equally_weighted_maps_with_eligible_roots": sum(
            g["eligible_payment_intents"] > 0 for g in groups.values()
        ),
        "equal_map_brier": {
            k: (
                statistics.mean(vals)
                if (vals := [g["brier"][k] for g in groups.values() if g["brier"][k] is not None])
                else None
            )
            for k in ("fitted", "geometric", "contact_rate")
        },
        "edit_witness_schema": [
            "seed",
            "seat",
            "turn",
            "worker",
            "fixed_move",
            "fitted_stay",
            "fitted_move",
            "died_stay",
            "died_move",
            "v1_payment_kind",
            "v1_paid",
        ],
        "edits": examples,
        "new_games": 0,
        "formal": False,
    }


def active_priority_work():
    native = 0
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = [os.fsdecode(a) for a in path.read_bytes().split(b"\x00") if a]
        except OSError:
            continue
        if len(args) < 2 or "python" not in Path(args[0]).name:
            continue
        if any(
            a.endswith("/planter_survival/extract.py") or a.endswith("/planter_survival/report.py") for a in args
        ):
            return True
        if ("arm" in args and any(a.endswith("/batch_cli.py") for a in args)) or any(
            a.endswith("tree_pack_arena.py") for a in args
        ):
            native += 2 if any(a.endswith("tree_pack_arena.py") for a in args) else 1
    return native >= 3


def supervise():
    p = None
    stopped = False
    try:
        while active_priority_work():
            time.sleep(4)
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "extract"],
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        while p.poll() is None:
            busy = active_priority_work()
            if busy != stopped:
                p.send_signal(signal.SIGSTOP if busy else signal.SIGCONT)
                stopped = busy
            time.sleep(2)
        require(p.returncode == 0, "extraction interrupted; sealed shards remain reusable")
    finally:
        if p is not None and p.poll() is None:
            if stopped:
                p.send_signal(signal.SIGCONT)
            p.terminate()
            p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("queued", "extract", "report"))
    a = ap.parse_args()
    plan = registration()
    if a.mode == "queued":
        with (HERE / ".queue.lock").open("a") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervise()
    else:
        with (HERE / ".extract.lock").open("a") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.mode == "extract":
                games = inputs(plan)
                for key in sorted(games):
                    one(plan, games[key])
            report = scorecard(plan)
            extract.write(HERE / "results.json", report)
            print(json.dumps(report["total"]), flush=True)


if __name__ == "__main__":
    main()

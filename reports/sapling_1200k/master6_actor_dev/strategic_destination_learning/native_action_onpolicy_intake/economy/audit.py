"""Reconcile both seats' receipts and trace realized cash pivots on eight registered training maps."""

import argparse
import gzip
import importlib.util
import json
import math
import statistics
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
STUDY = INTAKE.parent
POLICY = "87655af5309b6c5468f76441da46ce9f4bd7ff923315b2196b7d636994dd0a9b"
EXAMINER = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconcile = module_at("economic_source_reconciler", STUDY / "policy_window_returns/economy/reconcile.py")
run, ledger = reconcile.run, reconcile.ledger
require = ledger.require
read = reconcile.load


def report_game(card, archive, cfg):
    """Bind every decoded order to its captured transport hash and the official end score."""
    arm = card["arm"]
    raw = archive.read(f"{arm}/receipt.json")
    require(ledger.sha(raw) == card["receipt_sha256"], "Different source receipt")
    receipt = json.loads(raw)
    if card["status"] != "complete":
        game = ledger.replay(card, archive, cfg)
        game["scope"] = (
            "Only confirmed executed prefix; official scores and full-game gains unavailable."
        )
        return game

    tape = archive.read(f"{arm}/turns.jsonl")
    require(
        ledger.sha(tape) == receipt["turns_sha256"] and card["counted"], "Source not a counted match"
    )
    scope = dict(configuration=cfg, seed=card["seed"], seat=card["seat"], terrain=card["terrain"])
    env = run.collect.begin(card["seed"], cfg)
    trajectory = []
    for t, line in enumerate(tape.splitlines()):
        turn = json.loads(line)
        require(
            not env.done
            and t == turn["step"]
            and len(turn["calls"]) == 2
            and all(c["status"] == "ok" for c in turn["calls"]),
            "Broken joint action chronology",
        )
        before = run.board_id(env)
        orders = [c["response"]["action"] for c in turn["calls"]]
        env.step(orders)
        trajectory.append(
            dict(
                step=t,
                orders=orders,
                board_before=before,
                board_after=run.board_id(env),
                cash_after=[p[0] for p in env.states[0].observation.players],
            )
        )
    final = run.terminal(env, card["seat"])
    require(
        len(trajectory) == receipt["official_transitions"] == card["official_transitions"]
        and final["scores"] == receipt["official_scores_by_seat"] == card["official_scores_by_seat"],
        "Fresh official replay scores differ",
    )
    game = reconcile.replay(scope, trajectory, final, arm)
    game["source"] = dict(receipt_sha256=card["receipt_sha256"], turns_sha256=receipt["turns_sha256"])
    game["pivots"] = reconcile.pivots(game)
    game["phases"] = [
        ledger.profile(card["seat"], game["transitions"], t, min(t + 100, len(trajectory)))
        for t in range(0, len(trajectory), 100)
    ]
    # Keep all turns. The references above point into these original joint actions, including their pre-state.
    return game


def analyze(hosted):
    protocol_path = INTAKE / "protocol.json"
    plan = read(protocol_path)
    settlement_path = hosted / "settlement.json"
    settled = read(settlement_path)
    require(
        settled["schema"] == "sapling-physical-policy-training-intake/1"
        and ledger.sha(protocol_path.read_bytes()) == settled["protocol_sha256"],
        "Wrong training collection",
    )
    require(
        plan["files"]["collector"]["sha256"] == POLICY
        and plan["files"]["opponent"]["sha256"] == EXAMINER,
        "Different players",
    )
    roster = [(r["seed"], r["seat"]) for r in plan["schedule"]]
    require(
        len(roster) == len(set(roster)) == 8 and len({s for s, _ in roster}) == 8, "Wrong physical maps"
    )
    cards = {(g["seed"], g["seat"]): g for g in settled["games"]}
    archives = {(b["seed"], b["seat"]): b for b in settled["archives"]}
    require(
        len(cards) == len(settled["games"]) == 8
        and set(cards) == set(roster)
        and len(archives) == len(settled["archives"])
        and set(archives) <= set(cards),
        "Missing or duplicated slots",
    )
    games = []
    cfg = ledger.config()
    for key in roster:
        card = cards[key]
        require(card["arm"] == "collector", "Wrong learner")
        if not card.get("receipt_sha256"):
            games.append(
                dict(
                    seed=key[0],
                    seat=key[1],
                    status=card["status"],
                    reason=card.get("reason"),
                    official_scores_by_seat=None,
                    observed_replay=False,
                )
            )
            continue
        block = archives[key]
        filename = block["blob_filename"]
        require(Path(filename).name == filename, "Invalid archive name")
        path = hosted / filename
        require(
            ledger.sha(path.read_bytes()) == block["sha256"]
            and card["terrain"] == plan["physical_boards"][str(key[0])],
            "Wrong archive or terrain",
        )
        with zipfile.ZipFile(path) as archive:
            require(len(set(archive.namelist())) == len(archive.namelist()), "Duplicated ZIP member")
            game = report_game(card, archive, cfg)
        game.update(observed_replay=True, archive_sha256=block["sha256"], archive_id=block["id"])
        games.append(game)

    complete = [g for g in games if g["status"] == "complete"]

    def components(g, key):
        accounts = g["signed_account_by_seat"]
        return accounts[g["seat"]].get(key, 0.0) - accounts[1 - g["seat"]].get(key, 0.0)

    all_mean = (
        {k: statistics.mean(components(g, k) for g in complete) for k in ledger.KIND}
        if len(complete) == len(games)
        else None
    )
    if all_mean is not None:
        ledger.check_equal(
            sum(all_mean.values()),
            statistics.mean(g["official_margin"] for g in games),
            "Whole collection signed official margin",
        )
    result = dict(
        schema="sapling-onpolicy-training-cash/1",
        run=settled["run_id"],
        cohort="training",
        policy=POLICY,
        examiner=EXAMINER,
        protocol_sha256=settled["protocol_sha256"],
        settlement_sha256=ledger.sha(settlement_path.read_bytes()),
        schema_program=ledger.sha(Path(__file__).read_bytes()),
        retained=len(games),
        complete=len(complete),
        component_means_sapling_minus_examiner=all_mean,
        full_replay_transitions=sum(len(g["transitions"]) for g in complete),
        source_cash_methods=[ledger.sha(Path(m.__file__).read_bytes()) for m in (ledger, reconcile)],
        worker_schema=[
            "cell",
            "load",
            "collector_flag",
            "physical_seat",
            "toroidal_distance_to_nearest_own_base",
        ],
        time="Index t is the exact joint command at state t. [from,to) advances to state to. Lookbacks precede from.",
        games=games,
    )
    return result


def self_test():
    root = STUDY / "elite_direct_policy/development/hosted"
    settled = read(root / "settlement.json")
    key = (621758521, 0)
    card = next(g for g in settled["games"] if (g["seed"], g["seat"], g["arm"]) == (*key, "elite"))
    block = next(a for a in settled["archives"] if (a["seed"], a["seat"]) == key)
    archive_path = root / f"{block['id']}.zip"
    require(ledger.sha(archive_path.read_bytes()) == block["sha256"], "Fixture ZIP changed")
    with zipfile.ZipFile(archive_path) as archive:
        g = report_game(card, archive, ledger.config())
    require(
        g["official_scores_by_seat"] == [8598.34, 9464.66] and len(g["transitions"]) == 299,
        "Fixture scores",
    )
    row = g["transitions"][211]
    require(
        any(
            e[0] == "own_delivery"
            and e[1] == 1
            and math.isclose(e[2], 574.6011, abs_tol=0.001)
            and e[3] == "player-1-worker-22"
            for e in row["cash_events"]
        ),
        "Known cargo payment missing",
    )
    for p in g["pivots"].values():
        for n in (3, 5, 10):
            require(p["lookback"][n]["first_turn"] == max(0, p["from"] - n), "Wrong antecedent index")
    print("One original archive / 299 turns / both official scores / 574.60 proof passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosted", type=Path, default=INTAKE / "hosted")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = analyze(args.hosted)
        print(
            ledger.write(HERE / "results.json.gz", result, True),
            result["complete"],
            result["full_replay_transitions"],
        )

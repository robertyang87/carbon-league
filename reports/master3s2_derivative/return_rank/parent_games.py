"""Reconcile every earlier hosted development game of the frozen 7cf5 parent against original."""
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import runpy
from statistics import mean
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
TOURNAMENTS = ("cold_start/development", "ladder/prediction_feedback/development", "ladder/cargo_defense/development")
PARENT = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
LEDGER = FAMILY.parent / ("sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/"
                          "league_policy/natural_original/economy.py")


def game_record(ledger, cfg, game, archive_path):
    seat = game["seat"]
    with zipfile.ZipFile(archive_path) as archive:
        replay = ledger["replay"](game, archive, cfg)
    scores = game["official_scores_by_seat"]
    assert len(replay["transitions"]) == 299 and scores == replay["official_scores_by_seat"]
    sides = replay["signed_account_by_seat"]
    accounts = [{k: float(v) for k, v in side.items()} for side in sides]
    assert all(abs(sum(a.values()) - score) < 0.001 for a, score in zip(accounts, scores))
    net = lambda s: sum(accounts[s].get(k, 0.) for k in ("ground_rent", "cargo_tax", "seize", "plant"))
    eligible_workers = 0
    distinct_turns = 0
    captures = []
    for row in replay["transitions"]:
        # Count observable opportunities before considering model confidence, occupancy or native HOMEFIX.
        workers = row["pre"]["workers"]
        near = {uid for uid, (_, load, collector, player, distance) in workers.items()
                if collector and player == seat and load >= 100 and 1 <= distance <= 5 and row["t"] <= 293}
        eligible_workers += len(near)
        distinct_turns += bool(near)
        for uid, lost_seat, cargo, paid_to, survivor_seat in row["collisions"]:
            if lost_seat == seat and paid_to == 1-seat and cargo >= 100:
                captures.append(dict(t=row["t"], uid=uid, amount=cargo, distance=workers[uid][4], in_scope=uid in near,
                                     surviving_collector_seat=survivor_seat))
    return dict(seed=game["seed"], seat=seat, terrain=game["terrain"], run=game["workflow_run"], own=scores[seat],
                opponent=scores[1-seat], margin=round(scores[seat]-scores[1-seat],2), win=int(game["winner_seat"]==seat),
                receipt=game["receipt_sha256"], official_steps=299, accounts_by_physical_seat=accounts,
                ds_minus_original_tree=round(net(seat)-net(1-seat),2),
                ds_minus_original_delivery=round(accounts[seat]['own_delivery']-accounts[1-seat]['own_delivery'],2),
                opportunity_worker_steps=eligible_workers, opportunity_steps=distinct_turns,
                heavy_cargo_to_enemy=captures)


def run():
    ledger = runpy.run_path(str(LEDGER))
    cfg = ledger["config"]()
    games = []
    for folder in TOURNAMENTS:
        lane = FAMILY / folder
        raw = (lane / "protocol.json").read_bytes()
        plan = json.loads(raw)
        results = json.loads((lane / "hosted/settlement.json").read_bytes())
        assert results["records_reconciled"] and results["protocol_sha256"] == sha256(raw).hexdigest()
        assert plan["files"]["opponent"]["sha256"] == ORIGINAL
        arms = [arm for arm in plan["arms"] if plan["files"][arm]["sha256"] == PARENT]
        assert len(arms) == 1
        tickets = {(g["seed"],g["seat"]):g for g in results["archives"]}
        for g in results["games"]:
            if g["arm"] not in arms:
                continue
            ticket = tickets[g["seed"], g["seat"]]
            src = lane / "hosted" / (str(ticket["id"]) + ".zip")
            assert sha256(src.read_bytes()).hexdigest() == ticket["sha256"]
            games.append(game_record(ledger, cfg, g, src))
    terrain_counts = Counter(g["terrain"] for g in games)
    assert len(games) == 12 and len(terrain_counts) == 6 and all(c == 2 for c in terrain_counts.values())
    by_map = []
    for terrain in sorted(terrain_counts):
        pair = [g for g in games if g["terrain"] == terrain]
        assert {g["seat"] for g in pair} == {0,1}
        by_map.append(dict(seed=pair[0]["seed"], win=mean(g["win"] for g in pair),
                           margin=mean(g["margin"] for g in pair)))
    draws = np.array([[b["win"],b["margin"]] for b in by_map])[list(itertools.product(range(6),repeat=6)),:].mean(axis=1)
    assert draws.shape == (6**6,2)
    return dict(parent=PARENT, opponent=ORIGINAL, source_tournaments=TOURNAMENTS, games=games, maps=by_map,
                map_bootstrap="all ordered draws, six maps with replacement, empirical percentile 2.5 and 97.5",
                observed=[mean(g['win'] for g in games), mean(g['margin'] for g in games)],
                percentile_interval=np.percentile(draws,[2.5,97.5],axis=0).tolist())


if __name__ == "__main__":
    result=run()
    (HERE/"parent_games.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    print(result['maps'], result['observed'], result['percentile_interval'])
    for g in result['games']:
        print(g['seed'], g['seat'], g['win'], g['margin'],g['ds_minus_original_tree'],g['ds_minus_original_delivery'],
              g['opportunity_worker_steps'], [(r['t'],round(r['amount']),r['in_scope']) for r in g['heavy_cargo_to_enemy']])

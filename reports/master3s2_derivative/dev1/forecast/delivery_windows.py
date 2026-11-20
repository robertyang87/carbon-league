"""Reproduce the two post-match delivery examples from GitHub-signed development receipts."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HOSTED = Path(__file__).resolve().parents[1] / "hosted"
CASES = ((621758581, 1, "player-0-worker-21", 167),
         (621758583, 0, "player-1-worker-36", 249))


def distance(a, b):
    y, x = divmod(a, 15)
    v, u = divmod(b, 15)
    return min((x - u) % 15, (u - x) % 15) + min((y - v) % 15, (v - y) % 15)


def meeting_cells(observation, me, unit):
    players = observation["players"]
    enemy = players[1-me]
    pos, cargo, _ = enemy[2][unit]
    nearest_distance = min(distance(pos, b) for b in enemy[1].values())
    nearest = sorted(b for b in enemy[1].values() if distance(pos, b) == nearest_distance)
    options = []
    for cell in range(225):
        steps = distance(pos, cell)
        if not 1 <= steps <= min(3, nearest_distance - 1):
            continue
        if not any(steps + distance(cell, b) == nearest_distance for b in nearest):
            continue
        for wid, (own_pos, own_cargo, kind) in players[me][2].items():
            if kind == "COLLECTOR" and own_cargo < cargo and distance(own_pos, cell) <= steps:
                options.append(dict(cell=cell, enemy_steps=steps, own_steps=distance(own_pos, cell),
                                    defender=wid, own_pos=own_pos, own_cargo=round(own_cargo, 2)))
    return nearest, nearest_distance, options


def main():
    settlement_raw = (HOSTED/"settlement.json").read_bytes()
    settlement = json.loads(settlement_raw)
    economy = json.loads(gzip.decompress((HOSTED/"economy.json.gz").read_bytes()))
    assert settlement["records_reconciled"] and hashlib.sha256(settlement_raw).hexdigest() == economy["settlement_sha256"]
    for seed, seat, worker, delivered_at in CASES:
        game = next(g for g in settlement["games"] if (g["seed"], g["seat"]) == (seed, seat))
        book = next(g for g in economy["games"] if (g["seed"], g["seat"]) == (seed, seat))
        record = next(a for a in settlement["archives"] if a["id"] == game["archive_id"])
        source = HOSTED / game["archive_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"]
        with zipfile.ZipFile(source) as archive:
            folder = game["arm"] + "/"
            obs = [json.loads(x)["observations"][seat] for x in gzip.decompress(archive.read(folder+"observations.jsonl.gz")).splitlines()]
            commands = [[a["response"]["action"] for a in json.loads(x)["calls"]] for x in archive.read(folder+"turns.jsonl").splitlines()]
        assert len(obs) == len(commands) == game["official_transitions"] == 299
        swing = book["turning_points"]["loss_10"]["swing"]
        event, = [e for e in book["selected_turns"][str(delivered_at)]["cash_events"] if e[0] == "own_delivery" and e[3] == worker]
        print(f"Map {seed}, seat {seat}. States {swing['first_turn']}..{swing['after_last_turn_state']}, cash swing {swing['learner_margin_change']:.2f}")
        print("Deliveries in window (candidate, original):", [round(swing["by_physical_seat"][s].get("own_delivery", 0), 2) for s in (seat, 1-seat)])
        print("Official deposit event:", event)
        print("Physical opponent path (pre-action states):", [obs[t]["players"][1-seat][2][worker][0] for t in range(delivered_at-10, delivered_at+2)])
        for t in (delivered_at-10, delivered_at-5, delivered_at-3):
            point = obs[t]
            assert point["step"] == t and point["player"] == seat
            bases, distance_to_base, options = meeting_cells(point, seat, worker)
            for option in options:
                option["issued_order"] = commands[t][seat].get(option["defender"], "NONE")
            print(json.dumps(dict(step=t, public_enemy_worker=point["players"][1-seat][2][worker],
                                  nearest_bases=bases, distance_to_base=distance_to_base, options=options), ensure_ascii=False))


if __name__ == "__main__":
    main()

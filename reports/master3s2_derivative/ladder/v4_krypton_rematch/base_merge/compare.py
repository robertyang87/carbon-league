"""Test whether handling double arrivals changes the prior eight short return-policy branches."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parents[1] / "v4_fish_plain/route_transfer"
sys.path.insert(0, str(TRANSFER))
from policy import DIRECTIONS, distance, least_tax, move
from register import HELD, sha


def merged_policy(players, seat, uid, orders, step, cfg):
    own, enemy = players[seat], players[1 - seat]
    pos, load, _ = own[2][uid]
    fallback = least_tax(players, seat, uid, orders, step, cfg)
    if load <= 0 or cfg["episodeSteps"] - 2 - step >= 12:
        return fallback
    size = cfg["size"]
    for action in (*DIRECTIONS, "NONE"):
        dest = move(pos, action, size)
        if dest not in own[1].values():
            continue
        if any(orders.get(b, "NONE") != "NONE" and base == dest for b, base in own[1].items()):
            continue
        arriving = [w for other, w in own[2].items()
                    if other != uid and move(w[0], orders.get(other, "NONE"), size) == dest]
        if (len(arriving) != 1 or arriving[0][2] != "COLLECTOR"
                or arriving[0][1] <= 0 or arriving[0][1] == load):
            continue
        if any(w[2] == "COLLECTOR" and distance(w[0], dest, size) <= 1 for w in enemy[2].values()):
            continue
        return action
    return fallback


def run():
    spec = importlib.util.spec_from_file_location("frozen_route_transfer", TRANSFER / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.PLAN = json.loads((TRANSFER / "protocol.json").read_bytes())
    runner.least_tax = merged_policy
    baseline = json.loads(gzip.decompress((TRANSFER / "outcomes.json.gz").read_bytes()))
    settlement = json.loads((HELD / "settlement.json").read_bytes())
    games = []
    for c in runner.PLAN["archive_cards"]:
        key = (c["seed"], c["seat"])
        g = next(x for x in settlement["games"] if x["arm"] == "challenger" and (x["seed"], x["seat"]) == key)
        old = next(x for x in baseline["games"] if (x["seed"], x["seat"]) == key)
        archive = HELD / (str(c["id"]) + ".zip")
        assert sha(archive.read_bytes()) == c["sha256"]
        with zipfile.ZipFile(archive) as z:
            new = runner.one(g, z, runner.rule.econ.config())
        assert new["uid"] == old["uid"] and new["branches"]["recorded"] == old["branches"]["recorded"]
        new["vs_blocked_destination_policy"] = round(new["official_margin_change"] - old["official_margin_change"], 2)
        games.append(new)
    assert len(games) == 8
    output = {"baseline_protocol_sha256": sha((TRANSFER/"protocol.json").read_bytes()), "games": games}
    (HERE/"receipts.json.gz").write_bytes(gzip.compress((json.dumps(output, separators=(",", ":"),allow_nan=False)+"\n").encode(),mtime=0))
    for g in games:
        print(g["seed"],g["seat"],g["vs_blocked_destination_policy"],g["official_margin_change"])


if __name__ == "__main__":
    run()

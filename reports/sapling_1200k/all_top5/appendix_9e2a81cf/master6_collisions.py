"""Reconstruct both sides' actual simultaneous moves at every lost enemy collector."""
from pathlib import Path
import gzip
import json

HERE = Path(__file__).resolve().parent


def destination(position, move):
    x = (position % 15 + {"RIGHT": 1, "LEFT": -1}.get(move, 0)) % 15
    y = (position // 15 + {"DOWN": 1, "UP": -1}.get(move, 0)) % 15
    return y * 15 + x


def main():
    cache = Path("/workspace/sapling-all-top5-20261119")
    rows = []
    for game in json.loads((HERE / "manifest.json").read_text())["games"]:
        if "claude-AlphaCarbon-Master6" not in game["seat_names"]:
            continue
        master = game["seat_names"].index("claude-AlphaCarbon-Master6")
        other = 1 - master
        frames = json.loads(gzip.decompress((cache / game["path"]).read_bytes()))["steps"]
        lost, counts = [], {}
        for t, (pre, post) in enumerate(zip(frames, frames[1:])):
            players = pre[0]["observation"]["players"]
            after = post[0]["observation"]["players"]
            if t in (25, 50, 75, 100, 150, 200, 250):
                counts[t] = [{"collectors": sum(w[2] == "COLLECTOR" for w in p[2].values()),
                              "cash": p[0], "trees": len(p[3])} for p in players]
            ours, theirs = players[master][2], players[other][2]
            our_moves, their_moves = post[master].get("action") or {}, post[other].get("action") or {}
            for uid in sorted(theirs.keys() - after[other][2].keys()):
                pos, cargo, kind = theirs[uid]
                if kind != "COLLECTOR":
                    continue
                cell = destination(pos, their_moves.get(uid))
                meeting = [(k, v[0], v[1], v[2], our_moves.get(k), k in after[master][2]) for k,v in ours.items()
                           if destination(v[0], our_moves.get(k)) == cell]
                lost.append({"t_before": t, "uid": uid, "pos": pos, "cargo": cargo, "order": their_moves.get(uid),
                             "destination": cell, "master_workers_same_cell": meeting, "cash_after": after[other][0],
                             "collectors_after": sum(w[2] == "COLLECTOR" for w in after[other][2].values())})
        rows.append({"id": game["id"], "master_seat": master, "physical_names": game["seat_names"],
                     "score_by_physical_seat": game["score_by_seat"], "physical_seat_counts": counts,
                     "opponent_collector_disappearances": lost})
    if len(rows) != 4:
        raise ValueError("expected four distinct Master6 appearances")
    (HERE / "master6_collisions.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

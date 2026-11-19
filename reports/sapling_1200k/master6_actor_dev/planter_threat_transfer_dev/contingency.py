"""Resolve what actually happened under staying and under the parent's simultaneous move."""
from collections import Counter
import gzip

from pydantic_core import from_json

from registry import HERE, REPLAYS, read, save, sha


def end_cell(cell, command, n):
    row, col = divmod(cell, n)
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1),
              "RIGHT": (0, 1), "NONE": (0, 0)}[command]
    return n * ((row + dr) % n) + (col + dc) % n


def opposition(raw, row, source):
    choice = row["choice"]
    seat = row["seat"]
    step = choice["t"]
    obs = raw["steps"][step][0]["observation"]
    enemy = obs["players"][1 - seat]
    enemy_orders = raw["steps"][step + 1][1 - seat]["action"] or {}
    n = raw["configuration"]["size"]
    start = choice["pos"]
    parent_move = choice["parent_command"] or "NONE"
    dest = end_cell(start, parent_move, n)
    workers = []
    for uid, (cell, load, role) in enemy[2].items():
        cmd = enemy_orders.get(uid) or "NONE"
        stop = end_cell(cell, cmd, n)
        if stop in {start, dest}:
            workers.append({"uid": uid, "role": role, "load": load, "from": cell,
                            "order": cmd, "arrival": stop})
    assert sorted(w["uid"] for w in workers if w["arrival"] == start) == sorted(choice["actual_enemy_arrivals"])
    return {"id": row["id"], "opponent_name": source["opponent_name"], "map": row["map"],
            "seat": seat, "t": step, "uid": choice["uid"], "raw_sha256": source["raw_sha256"],
            "A_cell":start, "parent_order":parent_move, "parent_cell":dest,
            "arriving_enemy_workers":workers, "branches":choice["branches"], "B_veto":choice["veto"],
            "estimate_stay_death":choice["death_prediction"]}


def main():
    rows = read(HERE / "matched.json.gz")["rows"]
    sources = {s["id"]: s for s in read(HERE / "panel.json")["maps"]}
    census = {}
    examples = []
    labels = {(0,0):"both_dead", (0,1):"only_parent_survives", (1,0):"only_A_survives", (1,1):"both_survive"}
    for name in ("new_opponent", "known_new_map", "parent_original"):
        groups = {"all_selected":Counter(), "A_changes_parent":Counter(), "B_vetoes_A":Counter()}
        for row in (r for r in rows if r["split"] == name):
            c = row["choice"]
            if c is None:
                continue
            b = c["branches"]
            outcome = labels[b["A"]["survived"], b["parent"]["survived"]]
            groups["all_selected"][outcome] += 1
            if c["changed"]:
                groups["A_changes_parent"][outcome] += 1
            if c["veto"]:
                groups["B_vetoes_A"][outcome] += 1
            if name == "new_opponent" and not b["parent"]["survived"]:
                source = sources[row["id"]]
                replay = REPLAYS / (row["id"]+".json.gz")
                assert sha(replay) == source["raw_sha256"]
                examples.append(opposition(from_json(gzip.decompress(replay.read_bytes())), row, source))
        census[name] = {g:{l:counts[l] for l in labels.values()} for g,counts in groups.items()}
    save(HERE / "parent_outcomes.json", {
        "matched_sha256":sha(HERE/"matched.json.gz"), "panel_sha256":sha(HERE/"panel.json"),
        "scope":"Historical one-step official rule transitions with matched opponent commands. Deaths belong to the focal planter only.",
        "exclusive_outcomes":census, "every_parent_death_in_new_opponent_small_panel":examples})
    for e in examples:
        print(e["id"], e["opponent_name"], e["t"], e["A_cell"], e["parent_order"], e["parent_cell"], e["branches"]["A"]["expense"], e["arriving_enemy_workers"])


if __name__ == "__main__":
    main()

"""Recover the joint commands and actor-visible responses around the retrospectively selected economic windows."""
import argparse
import gzip
import json
import re
from pathlib import Path

import fetch


def write_windows(directory, replay_cache):
    manifest = fetch.load(directory / "manifest.json")
    cases = json.loads(gzip.decompress((directory / "cases.json.gz").read_bytes()))["rows"]
    cash = json.loads(gzip.decompress((directory / "turn_cash.json.gz").read_bytes()))["games"]
    records = {g["id"]: g for g in cases}
    ledgers = {g["id"]: g for g in cash}
    games = []
    for game in manifest["games"]:
        gid = game["id"]
        economy = records[gid]["economy"]
        turns = ledgers[gid]["turns"]
        raw = (replay_cache / game["path"]).read_bytes()
        if fetch.sha(raw) != game["sha256"]:
            raise ValueError(f"replay changed {gid}")
        frames = json.loads(gzip.decompress(raw))["steps"]
        windows = {}
        for name in ("swing", "decisive"):
            start, end = (economy[name][key] for key in ("a", "b"))
            windows[name] = [max(0, start - 10), start, end]
        actor_names = sorted(set(game["seat_names"]) & set(fetch.NAMES))
        # Capture the lasting loss of the final collector for either player, including the joint commands which removed it.
        for seat in (0, 1):
            has_collector = [any(w[2] == "COLLECTOR" for w in f[0]["observation"]["players"][seat][2].values()) for f in frames]
            if any(has_collector) and not has_collector[-1]:
                lost = max(t for t, exists in enumerate(has_collector) if exists) + 1
                windows[f"final_collector_loss_seat{seat}"] = [max(0, lost-10), lost-1, min(len(turns), lost+1)]
        ticks = sorted({t for start, _, end in windows.values() for t in range(start, end)})
        rows = []
        for t in ticks:
            pre = frames[t][0]["observation"]
            post = frames[t+1][0]["observation"]
            if (pre["step"], post["step"], turns[t][0]) != (t, t+1, t+1):
                raise ValueError(f"wrong action-state alignment {gid} {t}")
            players = pre["players"]
            commands = [agent.get("action") or {} for agent in frames[t+1]]
            if not all(set(commands[i]) <= (players[i][1].keys() | players[i][2].keys()) for i in (0, 1)):
                raise ValueError(f"action belongs to unknown pre-action entity {gid} {t}")
            rows.append({"t_before":t, "public_players_before":players, "commands_as_recorded":commands,
                         "cash_after":turns[t][1],"cash_components":turns[t][2],"delivery_lots":turns[t][3],
                         "captured_in_transit":turns[t][4], "tree_taxes_within_tree_income":turns[t][5]})
        games.append({"id":gid,"physical_seats":game["seat_names"],"registered_names":actor_names,
                      "replay_sha256":game["sha256"],"windows":windows,"transitions":rows})
    if {g["id"] for g in games} != set(records) or set(records) != set(ledgers):
        raise ValueError("window membership differs from the batch")
    payload = {"windows_schema":"[inclusive pre-action start, event-window start, exclusive pre-action end]; each transition t uses commands in replay frame t+1. A missing command means NONE. Final-collector-loss is retrospectively selected. Full state and seat-specific visibility and histories are addressable by ID,t,seat in context.py.",
               "players_schema":"[cash, bases by ID and coordinate, workers by ID as position/cargo/role, trees by ID as position/age]",
               "cash_components_by_physical_seat":["bank","donation","tree","collision_cash","recruit","plant","seize","round"],
               "games":games}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    fetch.atomic(directory / "turn_actions.json.gz", gzip.compress(data, mtime=0))
    for game in games:
        page = directory / (game["id"] + ".md")
        text = page.read_text()
        # The original base-corpus page formatter links a fixed fifth of its first 1,324 games.
        text = re.sub(r"\]\(\.\./economy_data/\d+\.jsonl\.gz\)", "](cases.json.gz)", text)
        if "(turn_actions.json.gz)" not in text:
            text += "\n每个转折附从十步前起的双方逐步实发命令、工号与位置、八项现金和回库工人。按比赛 ID 取本局记录。[双方转折前后的动作与结算](turn_actions.json.gz)\n"
        page.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=fetch.CACHE)
    args = parser.parse_args()
    write_windows(args.directory, args.cache)

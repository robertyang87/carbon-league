"""Reproduce the opening counts and head-to-head tree statistics from the archived replays.

Run inside the carbon-league checkout: python reports/master_family/sapling_replay_checks.py
Git object reads also work in sparse checkouts. The revision pins both match indices and replays.
"""
from collections import Counter
import gzip
import json
import subprocess

REV = "605031c4f3bba28563e7829d3e36e2a0b6c3aaf3"
CORPORA = {"master5s": "claude-AlphaCarbon-Master5s", "master6": "claude-AlphaCarbon-Master6"}


def blob(path):
    return subprocess.check_output(["git", "show", f"{REV}:{path}"])


def tree_events(steps):
    first_tree = [None, None]
    full_roster = [None, None]
    planted = [0, 0]
    captured = [0, 0]
    previous = {}
    for t, row in enumerate(steps):
        players = row[0]["observation"]["players"]
        current = {uid: side for side, p in enumerate(players) for uid in p[3]}
        for side, p in enumerate(players):
            if len(p[2]) == 10 and full_roster[side] is None:
                full_roster[side] = t
        if t:
            for uid, side in current.items():
                if uid not in previous:
                    planted[side] += 1
                    if first_tree[side] is None:
                        first_tree[side] = t
                elif previous[uid] != side:
                    captured[side] += 1
        previous = current
    return dict(first_tree=first_tree, first_full_roster=full_roster, planted=planted,
                captured=captured)


def main():
    games = {}
    tracked = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REV, "data/master5s_corpus", "data/master6_corpus"],
        text=True).splitlines())
    for corpus in CORPORA:
        for row in json.loads(blob(f"data/{corpus}_matches.json")):
            key = (row["id"], row["game"])
            path = f"data/{corpus}_corpus/{key[0]}-{key[1]}.json.gz"
            if path in tracked:
                games[key] = row, path
    openings = {name: Counter() for name in CORPORA.values()}
    head_to_head = []
    for row, path in games.values():
        replay = json.loads(gzip.decompress(blob(path)))
        if replay["rewards"] != [row["scores"][side] for side in row["order"]]:
            raise ValueError(f"replay scores disagree with index: {path}")
        names = [row["name_" + side] for side in row["order"]]
        steps = replay["steps"]
        for seat, name in enumerate(names):
            if name in openings:
                moves = (steps[1][seat]["action"] or {}).values()
                opening = "+".join(sorted(m for m in moves if m in {"RECCOLLECTOR", "RECPLANTER"}))
                openings[name][opening] += 1
        if set(names) == set(openings):
            head_to_head.append(dict(path=path, seats=names, **tree_events(steps)))
    print(json.dumps(dict(revision=REV, replays=len(games), openings=openings,
                          direct_games=head_to_head), indent=2))


if __name__ == "__main__":
    main()

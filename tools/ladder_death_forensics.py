#!/usr/bin/env python3
"""ladder_death_forensics -- classify every worker death in a ladder replay corpus.

For each step transition a worker present at t and absent at t+1 died. Its destination cell is
its position moved by the action recorded for step t+1 (UP = index-15, DOWN = index+15, LEFT/RIGHT
with row wrap; no action = stay). All workers of both players with the same destination form the
collision group (game rule: the unique collector carrying the least carbon survives and takes the
others' carbon; a tie for least, or no collector, deletes everyone). Each death is classified:

  hunted     killed in a group whose survivor is an ENEMY collector (it captures our carbon)
  mutual     group with enemy workers and no survivor (tie for least carbon / no collector)
  friendly   group of own workers only
  own_wins   group with enemy workers whose survivor is OUR collector (we lost a unit, kept the cell)
  enemy_base no collision group; destination is an enemy base (collector with carbon <= smeltCost)
  other      anything else (not reconstructed)

Usage: ladder_death_forensics.py <matches.json> <target name> <replay dir> [--wall name1,name2,...]
Prints JSON lines: per group (wall / rest / all) and per side (ours / theirs): deaths per game by
class, by role, by game phase, mean carbon lost per death, distance of the death cell to the
victim's nearest own base. Descriptive only.
"""
import argparse, collections, glob, gzip, json, os, statistics as st

N = 15


def move(pos, act):
    r, c = divmod(pos, N)
    if act == "UP":
        r = (r - 1) % N
    elif act == "DOWN":
        r = (r + 1) % N
    elif act == "LEFT":
        c = (c - 1) % N
    elif act == "RIGHT":
        c = (c + 1) % N
    return r * N + c


def tdist(a, b):
    ra, ca = divmod(a, N); rb, cb = divmod(b, N)
    dr = abs(ra - rb); dc = abs(ca - cb)
    return min(dr, N - dr) + min(dc, N - dc)


def game_deaths(d):
    """-> list of death dicts (player index as in the replay)."""
    out = []
    steps = d["steps"]
    for t in range(len(steps) - 1):
        o0 = steps[t][0]["observation"]; o1 = steps[t + 1][0]["observation"]
        pl0, pl1 = o0["players"], o1["players"]
        acts = [steps[t + 1][p].get("action") or {} for p in (0, 1)]
        dest = {}
        for p in (0, 1):
            for uid, (pos, carbon, role) in pl0[p][2].items():
                dest[uid] = (move(pos, acts[p].get(uid)), p, float(carbon), role, pos)
        groups = collections.defaultdict(list)
        for uid, v in dest.items():
            groups[v[0]].append(uid)
        bases = [set(pl0[p][1].values()) for p in (0, 1)]
        for p in (0, 1):
            for uid in pl0[p][2]:
                if uid in pl1[p][2]:
                    continue
                cell, _p, carbon, role, pos = dest[uid]
                grp = groups[cell]
                surv = [u for u in grp if u in pl1[dest[u][1]][2]]
                enemies = [u for u in grp if dest[u][1] != p]
                if len(grp) > 1:
                    if not enemies:
                        cls = "friendly"
                    elif surv and dest[surv[0]][1] != p:
                        cls = "hunted"
                    elif surv:
                        cls = "own_wins"
                    else:
                        cls = "mutual"
                elif cell in bases[1 - p]:
                    cls = "enemy_base"
                else:
                    cls = "other"
                killer = dest[surv[0]] if (cls == "hunted" and surv) else None
                out.append({"step": t + 1, "player": p, "role": role, "carbon": carbon, "cls": cls,
                            "base_dist": min((tdist(cell, b) for b in bases[p]), default=None),
                            "enemy_base_dist": min((tdist(cell, b) for b in bases[1 - p]), default=None),
                            "killer_carbon": killer[2] if killer else None,
                            "enemy_roles": sorted(dest[u][3] for u in enemies)})
    return out


def summarize(label, side, deaths, n_games):
    if not n_games:
        return
    by = collections.Counter(x["cls"] for x in deaths)
    role = collections.Counter(x["role"] for x in deaths)
    phase = collections.Counter(min(x["step"] // 50, 5) for x in deaths)
    hunted = [x for x in deaths if x["cls"] == "hunted"]
    print(json.dumps({
        "group": label, "side": side, "games": n_games,
        "deaths_per_game": round(len(deaths) / n_games, 1),
        "by_class_per_game": {k: round(v / n_games, 1) for k, v in by.most_common()},
        "by_role_per_game": {k: round(v / n_games, 1) for k, v in role.most_common()},
        "by_phase_per_game": {f"{50 * k}-{50 * k + 49}": round(phase[k] / n_games, 1) for k in sorted(phase)},
        "carbon_lost_per_game": round(sum(x["carbon"] for x in deaths) / n_games, 1),
        "hunted_carbon_per_death": round(st.mean(x["carbon"] for x in hunted), 1) if hunted else None,
        "hunted_killer_carbon": round(st.mean(x["killer_carbon"] for x in hunted), 1) if hunted else None,
        "hunted_victim_role": dict(collections.Counter(x["role"] for x in hunted)),
        "hunted_own_base_dist": round(st.mean(x["base_dist"] for x in hunted if x["base_dist"] is not None), 2) if hunted else None,
        "hunted_within2_of_own_base": round(sum(1 for x in hunted if x["base_dist"] is not None and x["base_dist"] <= 2) / max(1, len(hunted)), 3),
        "mutual_roles": dict(collections.Counter(x["role"] for x in deaths if x["cls"] == "mutual")),
    }, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches"); ap.add_argument("target"); ap.add_argument("rdir")
    ap.add_argument("--wall", default="")
    a = ap.parse_args()
    wall = {w for w in a.wall.split(",") if w}
    files = {os.path.basename(f).split("-")[0]: f for f in glob.glob(os.path.join(a.rdir, "*.json.gz"))}
    acc = collections.defaultdict(lambda: {"ours": [], "theirs": [], "n": 0})
    skipped = 0
    for m in json.load(open(a.matches)):
        f = files.get(m["id"])
        if not f or a.target not in (m["name_a"], m["name_b"]):
            skipped += 1; continue
        d = json.load(gzip.open(f))
        ra, rb = d["rewards"]; sa, sb = m["scores"]["a"], m["scores"]["b"]
        fwd = abs(ra - sa) < 1.0 and abs(rb - sb) < 1.0; rev = abs(ra - sb) < 1.0 and abs(rb - sa) < 1.0
        if fwd == rev:
            skipped += 1; continue
        slot = "a" if m["name_a"] == a.target else "b"
        me = (0 if slot == "a" else 1) if fwd else (1 if slot == "a" else 0)
        opp = m["name_b"] if slot == "a" else m["name_a"]
        ds = game_deaths(d)
        for key in ("all", "wall" if opp in wall else "rest"):
            acc[key]["n"] += 1
            acc[key]["ours"] += [x for x in ds if x["player"] == me]
            acc[key]["theirs"] += [x for x in ds if x["player"] != me]
    print(json.dumps({"skipped": skipped}))
    for key in ("wall", "rest", "all"):
        for side in ("ours", "theirs"):
            summarize(key, side, acc[key][side], acc[key]["n"])


if __name__ == "__main__":
    main()

"""diag_planter_deaths_v07: planter death-cause attribution over the 8
recorded policy_v0_5 vs reference_juzi games (traces/).

Why: v0.6's snapshot showed our step-250 alive planters 0.75 vs juzi
2.00, matching the late-game (step 200+) planting collapse 4 vs 27.
Before engineering planter survival we need the death ledger: WHEN,
WHERE and BY WHAT our planters die -- and juzi's, same yardstick.

Method (exact, from trace frames): a worker present in frame t-1 and
absent in frame t died during step t. Deaths in this env happen only
in the Phase-2 collision resolution (planters are exempt from the
Phase-3 enemy-base kill, which is collectors-only), so we rebuild the
Phase-2 groups exactly: post-move cell of every frame-(t-1) worker =
prev pos + its frame-t action (missing action = stay; torus wrap),
plus step-t spawns (workers new in frame t, sitting on their base
cell). A planter dies iff its group has >1 member (it can never be
the unique least-cargo collector). Cause = group composition.

Task proxy at death (from positions/trees only, labelled as proxy):
  on_enemy_tree   prev cell held an enemy tree (seizing / contesting)
  enter_enemy_tree post-move cell held an enemy tree (walked in)
  on_own_tree     prev cell held our own tree
  stay_attempt    stayed on a treeless non-base cell (plant attempt)
  at_own_base     post-move cell is an own base cell
  moving          anything else
"""

import gzip
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15
OFF = {"UP": (0, -1), "DOWN": (0, 1), "RIGHT": (1, 0), "LEFT": (-1, 0)}


def move(pos, act):
    if act not in OFF:
        return pos
    dx, dy = OFF[act]
    r, c = divmod(pos, SIZE)
    return ((r + dy) % SIZE) * SIZE + ((c + dx) % SIZE)


def tdist(a, b):
    ra, ca = divmod(a, SIZE)
    rb, cb = divmod(b, SIZE)
    dr = min(abs(ra - rb), SIZE - abs(ra - rb))
    dc = min(abs(ca - cb), SIZE - abs(ca - cb))
    return dr + dc


def bucket(step):
    return f"{(step - 1) // 50 * 50}-{(step - 1) // 50 * 50 + 49}"


def analyse(path):
    tr = json.load(gzip.open(path, "rt"))
    meta = tr["meta"]
    frames = tr["frames"]
    me = meta["policy_side"]
    bases = {int(p): [pos for _bid, pos in v]
             for p, v in meta["bases"].items()}
    first_seen = {}
    deaths = []          # planter death events, both sides
    stats = {}
    for side in (0, 1):
        stats[side] = {
            "planter_deaths": 0, "planter_births": 0,
            "alive250": 0, "late_plants": 0,
            "death_steps": [], "replace_lat": [],
        }
    pending_death = {0: [], 1: []}   # steps of deaths awaiting a birth
    for t in range(1, 301):
        prev, cur = frames[t - 1], frames[t]
        acts = cur["acts"]
        pw, cw = prev["workers"], cur["workers"]
        pt = {v[0]: (k, v[1], v[2]) for k, v in prev["trees"].items()}
        # tree births this step per owner (new tid, not a flip)
        for tid, v in cur["trees"].items():
            if tid not in prev["trees"] and t >= 200:
                stats[v[1]]["late_plants"] += 1
        post = {}   # wid -> post-move cell
        for wid, (pos, cargo, typ) in pw.items():
            player = 0 if wid.startswith("player-0") else 1
            post[wid] = move(pos, acts[player].get(wid))
            if wid not in first_seen:
                first_seen[wid] = t - 1
                if typ == "P":
                    stats[player]["planter_births"] += 1
                    if pending_death[player]:
                        d0 = pending_death[player].pop(0)
                        stats[player]["replace_lat"].append(t - d0)
        groups = defaultdict(list)
        for wid, cell in post.items():
            groups[cell].append(wid)
        for wid, (pos, cargo, typ) in cw.items():   # spawns join groups
            if wid not in pw:
                groups[pos].append(wid)
                if wid not in first_seen:
                    first_seen[wid] = t
                    player = 0 if wid.startswith("player-0") else 1
                    if typ == "P":
                        stats[player]["planter_births"] += 1
                        if pending_death[player]:
                            d0 = pending_death[player].pop(0)
                            stats[player]["replace_lat"].append(t - d0)
        if t == 250:
            for side in (0, 1):
                stats[side]["alive250"] = sum(
                    1 for wid, v in cw.items()
                    if v[2] == "P"
                    and (0 if wid.startswith("player-0") else 1) == side)
        for wid, (pos, cargo, typ) in pw.items():
            if wid in cw:
                continue
            player = 0 if wid.startswith("player-0") else 1
            cell = post[wid]
            grp = groups[cell]
            if typ != "P":
                continue
            stats[player]["planter_deaths"] += 1
            stats[player]["death_steps"].append(t)
            pending_death[player].append(t)
            others = [o for o in grp if o != wid]
            ec = [o for o in others
                  if o in pw and pw[o][2] == "C"
                  and (0 if o.startswith("player-0") else 1) != player]
            ec += [o for o in others
                   if o not in pw and cw[o][2] == "C"
                   and (0 if o.startswith("player-0") else 1) != player]
            ep = [o for o in others
                  if pw.get(o, cw.get(o, [0, 0, "?"]))[2] == "P"
                  and (0 if o.startswith("player-0") else 1) != player]
            fr = [o for o in others if o not in ec and o not in ep]
            tree_here = pt.get(cell)
            prev_tree = pt.get(pos)
            if ec:
                cause = "duel_enemy_collector"
            elif ep:
                cause = "planter_clash_tree" if tree_here else "planter_clash"
            elif fr:
                cause = "friendly_collision"
            elif len(grp) <= 1:
                cause = "unexplained_solo"
            else:
                cause = "other_group"
            # geometry of a duel: who moved?
            duel_geo = None
            killer_cargo = None
            if ec:
                k = ec[0]
                k_prev = pw[k][0] if k in pw else cw[k][0]
                killer_cargo = pw[k][1] if k in pw else 0
                pm = cell != pos
                km = cell != k_prev
                duel_geo = ("both_moved" if pm and km else
                            "planter_moved_in" if pm else
                            "killer_moved_in" if km else "both_stayed")
            if prev_tree and prev_tree[1] != player:
                task = "on_enemy_tree"
            elif tree_here and tree_here[1] != player:
                task = "enter_enemy_tree"
            elif prev_tree and prev_tree[1] == player:
                task = "on_own_tree"
            elif cell in bases[player]:
                task = "at_own_base"
            elif cell == pos:
                task = "stay_attempt"
            else:
                task = "moving"
            # enemy-tree siphon ring: any enemy tree within 8-neigh.
            ring = False
            for tp, (_tid, to, _ta) in pt.items():
                if to != player and tdist(cell, tp) <= 1:
                    ring = True
                    break
            deaths.append({
                "game": os.path.basename(path), "side": player,
                "is_us": player == me, "step": t, "bucket": bucket(t),
                "wid": wid, "age": t - first_seen[wid],
                "cell": cell, "prev_cell": pos, "cause": cause,
                "duel_geo": duel_geo, "killer_cargo": killer_cargo,
                "task": task,
                "dist_base": min(tdist(cell, b) for b in bases[player]),
                "enemy_tree_ring": ring,
                "group_size": len(grp),
            })
    return deaths, stats, me


def main():
    files = [os.path.join(HERE, "traces",
                          f"policy_v0_5_vs_reference_juzi_seed{s}_side{d}.json.gz")
             for s in (1000, 1001, 1002, 1003) for d in (0, 1)]
    all_deaths, agg = [], {True: [], False: []}
    per_game = []
    for f in files:
        deaths, stats, me = analyse(f)
        all_deaths += deaths
        juzi = 1 - me
        per_game.append((os.path.basename(f), stats[me], stats[juzi]))
        agg[True].append(stats[me])
        agg[False].append(stats[juzi])
    us = [d for d in all_deaths if d["is_us"]]
    jz = [d for d in all_deaths if not d["is_us"]]
    out = {"events_us": us, "events_juzi": jz}

    def show(name, evts):
        print(f"\n== {name}: {len(evts)} planter deaths "
              f"({len(evts) / 8:.2f}/game)")
        print("  by cause:", dict(Counter(d['cause'] for d in evts)))
        print("  by task :", dict(Counter(d['task'] for d in evts)))
        print("  by step bucket:")
        for b in ("0-49", "50-99", "100-149", "150-199", "200-249",
                  "250-299"):
            sub = [d for d in evts if d["bucket"] == b]
            print(f"    {b}: {len(sub)}",
                  dict(Counter(d['cause'] for d in sub)) if sub else "")
        print("  duel geometry:", dict(Counter(
            d['duel_geo'] for d in evts if d['duel_geo'])))
        print("  killer cargo buckets:", dict(Counter(
            "0" if d['killer_cargo'] == 0 else
            "1-99" if d['killer_cargo'] < 100 else ">=100"
            for d in evts if d['killer_cargo'] is not None)))
        print("  dist-to-base:", dict(Counter(
            "<=2" if d['dist_base'] <= 2 else
            "3-4" if d['dist_base'] <= 4 else ">=5"
            for d in evts)))
        print("  in enemy-tree ring:", sum(d['enemy_tree_ring']
                                           for d in evts))
        print("  planter age at death: median",
              sorted(d['age'] for d in evts)[len(evts) // 2] if evts else None)
    show("US (v0.5)", us)
    show("JUZI", jz)
    for lbl, key in (("US", True), ("JUZI", False)):
        ss = agg[key]
        print(f"\n{lbl}: alive@250/game "
              f"{sum(s['alive250'] for s in ss) / 8:.2f}, "
              f"late plants (step>=200)/game "
              f"{sum(s['late_plants'] for s in ss) / 8:.2f}, "
              f"planter births/game "
              f"{sum(s['planter_births'] for s in ss) / 8:.2f}, "
              f"replace latency (deaths followed by a birth): "
              f"{[x for s in ss for x in s['replace_lat']]}")
    with open(os.path.join(HERE, "diag_planter_deaths_v07.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("\nwrote diag_planter_deaths_v07.json")


if __name__ == "__main__":
    main()

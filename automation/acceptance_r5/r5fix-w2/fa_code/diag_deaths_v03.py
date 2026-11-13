"""diag_deaths_v03: offline death-cargo forensics on the v0.2 vs juzi
traces (traces/policy_v0_2_vs_reference_juzi_seed*_side*.json.gz).

v2 classifier (after frame probes of the costliest 'other' deaths):
a victim's death cell is its DESTINATION cell, not necessarily its
frame-t cell.  Endgame deaths 1 step from base with a friendly
collector sitting on the base in frame t+1 and our cash jumping are
base-mouth consolidation: the winner took the combined cargo and
deposited it the same step -- the cargo is RECOVERED, and the eval
ledger's death_cargo overstates the true loss by that amount.

Classes (priority order):
  base_consol   death cell is / reaches an OWN base with a friendly
                worker on it in frame t+1 (cargo recovered)
  enemy_base    died on an ENEMY base cell (cargo donated)
  hunt_kill     an enemy collector wins the death cell (sits there in
                frame t+1, or is the unique neighbour whose cargo grew
                by ~the victim's cargo: a converging collision)
  planter_clash an enemy planter sits on the death cell in frame t+1
  friendly      a friendly collector wins the victim's frame-t cell
                (own collision, cargo consolidated -- recovered)
  mutual        enemy worker died on the victim's cell the same step
                and no collector survives there
  other         unresolved

recovered := base_consol + friendly.  True loss := the rest.
Per-game lost-cargo totals are cross-checked against the eval ledger
(eval_v0_1_results.jsonl, tag eval_v02): trace totals must equal
ledger death_cargo up to trace rounding (<=0.3/game observed).
"""

import gzip
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15


def torus_dist(a, b):
    ax, ay = a % SIZE, a // SIZE
    bx, by = b % SIZE, b // SIZE
    return min(abs(ax - bx), SIZE - abs(ax - bx)) + \
        min(abs(ay - by), SIZE - abs(ay - by))


def nbr4(cell):
    x, y = cell % SIZE, cell // SIZE
    return [((y + 1) % SIZE) * SIZE + x, ((y - 1) % SIZE) * SIZE + x,
            y * SIZE + (x + 1) % SIZE, y * SIZE + (x - 1) % SIZE]


def nbr8(cell):
    x, y = cell % SIZE, cell // SIZE
    return [((y + dy) % SIZE) * SIZE + (x + dx) % SIZE
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def analyse(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    meta = tr["meta"]
    me = meta["policy_side"]
    foe = 1 - me
    my_bases = {pos for _, pos in meta["bases"][str(me)]}
    foe_bases = {pos for _, pos in meta["bases"][str(foe)]}
    frames = tr["frames"]
    events = []
    for t in range(300):
        cur, nxt = frames[t], frames[t + 1]
        cw, nw = cur["workers"], nxt["workers"]
        for wid, (pos, cargo, typ) in cw.items():
            if not wid.startswith(f"player-{me}-") or wid in nw:
                continue
            cand_cells = [pos] + nbr4(pos)
            occ = {c: [w for w, (p2, c2, t2) in nw.items() if p2 == c]
                   for c in cand_cells}
            cause, cell = "other", pos
            # 1) base consolidation: a candidate cell is an own base
            #    and a friendly worker occupies it in frame t+1
            for c in cand_cells:
                if c in my_bases and any(
                        w.startswith(f"player-{me}-") for w in occ[c]):
                    cause, cell = "base_consol", c
                    break
            if cause == "other" and pos in my_bases:
                cause, cell = "base_consol", pos
            # 2) enemy base
            if cause == "other":
                for c in cand_cells:
                    if c in foe_bases and occ[c]:
                        cause, cell = "enemy_base", c
                        break
            # 3) enemy winner on the frame-t cell
            if cause == "other" and occ[pos]:
                w0 = occ[pos][0]
                if w0.startswith(f"player-{foe}-"):
                    cause = "hunt_kill" if nw[w0][2] == "C" else "planter_clash"
                    cell = pos
            # 4) converging enemy collector: unique neighbour whose
            #    cargo grew by ~the victim's cargo
            if cause == "other" and typ == "C" and cargo > 0:
                grown = []
                for c in nbr4(pos):
                    for w in occ[c]:
                        if w.startswith(f"player-{foe}-") \
                                and nw[w][2] == "C" \
                                and nw[w][1] >= cargo - 1.0:
                            prev = cw.get(w, [None, 0])[1]
                            if nw[w][1] - prev >= cargo - 1.0:
                                grown.append((w, c))
                if len(grown) == 1:
                    cause, cell = "hunt_kill", grown[0][1]
            # 5) friendly winner on the frame-t cell
            if cause == "other" and occ[pos]:
                cause, cell = "friendly", pos
            # 6) mutual: enemy died on the frame-t cell, no survivor
            if cause == "other":
                foe_died = [w for w, (p2, c2, t2) in cw.items()
                            if p2 == pos and w.startswith(f"player-{foe}-")
                            and w not in nw]
                if foe_died and not occ[pos]:
                    cause, cell = "mutual", pos
            foe_tree_cells = {v[0] for v in cur["trees"].values()
                              if v[1] == foe}
            siphon = cell in foe_tree_cells or \
                any(n in foe_tree_cells for n in nbr8(cell))
            d_base = min(torus_dist(pos, b) for b in my_bases)
            foe_coll = [(p2, c2) for w, (p2, c2, t2) in cw.items()
                        if w.startswith(f"player-{foe}-") and t2 == "C"]
            d_foe = min((torus_dist(pos, p2) for p2, _ in foe_coll),
                        default=99)
            events.append(dict(
                step=t + 1, wid=wid, typ=typ, cargo=round(float(cargo), 1),
                cell=cell, cause=cause, siphon=bool(siphon),
                d_base=d_base, d_foe=d_foe,
                recovered=cause in ("base_consol", "friendly")))
    return meta, events


def band(cargo):
    if cargo < 60:
        return "<60"
    if cargo < 150:
        return "60-150"
    if cargo < 300:
        return "150-300"
    return ">=300"


def bucket(step):
    lo = (step - 1) // 50 * 50 + 1
    return f"{lo}-{min(300, lo + 49)}"


def main():
    files = sorted(f for f in os.listdir(os.path.join(HERE, "traces"))
                   if f.startswith("policy_v0_2_vs_reference_juzi"))
    led = {}
    with open(os.path.join(HERE, "eval_v0_1_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["tag"] == "eval_v02":
                led[(r["seed"], r["policy_side"])] = \
                    r["ledger_pol"]["death_cargo"]
    all_events = []
    print(f"{'game':22s} {'deaths':>6s} {'cargo':>8s} {'ledger':>8s} {'diff':>6s}")
    for fn in files:
        meta, ev = analyse(os.path.join(HERE, "traces", fn))
        for e in ev:
            e["seed"] = meta["seed"]
            e["side"] = meta["policy_side"]
        all_events += ev
        tot = sum(e["cargo"] for e in ev)
        lg = led.get((meta["seed"], meta["policy_side"]), float("nan"))
        print(f"seed{meta['seed']} side{meta['policy_side']:<8d} "
              f"{len(ev):6d} {tot:8.1f} {lg:8.1f} {tot - lg:+6.1f}")
    n = len(all_events)
    tot = sum(e["cargo"] for e in all_events)
    ng = len(files)
    print(f"\nTOTAL {n} deaths ({n / ng:.1f}/game), ledger cargo {tot:.0f} "
          f"({tot / ng:.0f}/game)")
    rec = sum(e["cargo"] for e in all_events if e["recovered"])
    print(f"recovered (base_consol+friendly): {rec:.0f} "
          f"({rec / ng:.0f}/game) -> TRUE loss {(tot - rec) / ng:.0f}/game")

    def agg(events, keyfn, title, base_tot):
        d = defaultdict(lambda: [0, 0.0])
        for e in events:
            k = keyfn(e)
            d[k][0] += 1
            d[k][1] += e["cargo"]
        print(f"\n--- {title} ---")
        for k in sorted(d, key=lambda k: -d[k][1]):
            c, s = d[k]
            print(f"  {k:16s} n={c:3d} ({c / max(1, len(events)):4.0%})  "
                  f"cargo={s:7.0f} ({s / base_tot:4.0%})")

    agg(all_events, lambda e: e["cause"], "by cause (all deaths)", tot)
    true_loss = [e for e in all_events if not e["recovered"]]
    tl = sum(e["cargo"] for e in true_loss)
    agg(true_loss, lambda e: e["cause"], "TRUE loss by cause", tl)
    agg(true_loss, lambda e: band(e["cargo"]), "TRUE loss by cargo band", tl)
    agg(true_loss, lambda e: bucket(e["step"]), "TRUE loss by step bucket", tl)
    agg(true_loss, lambda e: "siphon zone" if e["siphon"] else "open",
        "TRUE loss by enemy-tree siphon zone", tl)
    agg(true_loss, lambda e: f"d_foe<={min(e['d_foe'], 3)}"
        if e["d_foe"] <= 3 else "d_foe>=4",
        "TRUE loss by nearest enemy collector dist (frame t)", tl)
    agg(all_events, lambda e: e["typ"], "all deaths by victim type", tot)
    agg([e for e in all_events if e["typ"] == "P"],
        lambda e: e["cause"], "PLANTER deaths by cause", 1.0)
    top = sorted(true_loss, key=lambda e: -e["cargo"])[:12]
    print("\ncostliest TRUE-loss deaths:")
    for e in top:
        print(f"  seed{e['seed']} side{e['side']} step{e['step']:3d} "
              f"{e['typ']} cargo={e['cargo']:6.0f} cause={e['cause']:13s} "
              f"siphon={e['siphon']} d_base={e['d_base']} d_foe={e['d_foe']}")
    with open(os.path.join(HERE, "diag_deaths_v03.json"), "w") as f:
        json.dump(all_events, f)


if __name__ == "__main__":
    main()

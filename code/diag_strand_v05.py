"""diag_strand_v05: forensic on the v0.4 vs example seed 2001 P1 game --
final stranded cargo 427 (the other three example games were <= 48).

Reads ONLY the recorded trace. Questions:
  - which of our workers (player 1) hold cargo at step 300, how much;
  - for each holder: position/cargo/action trace over the last 60
    steps -- stationary (frozen), oscillating, or still travelling;
  - who occupied the BFS-descent cells toward the nearest own base
    (planter-block / collector-chain / enemy -- example has 0 workers,
    so any blocker is ours);
  - when did each holder last stand on an own base cell, and what did
    it do afterwards.
"""

import gzip
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15

FN = os.path.join(HERE, "traces",
                  "policy_v0_4_vs_example_seed2001_side1.json.gz")


def dist(a, b):
    ax, ay = a % SIZE, a // SIZE
    bx, by = b % SIZE, b // SIZE
    return (min(abs(ax - bx), SIZE - abs(ax - bx))
            + min(abs(ay - by), SIZE - abs(ay - by)))


def nbr4(pos):
    x, y = pos % SIZE, pos // SIZE
    return [((y + dy) % SIZE) * SIZE + (x + dx) % SIZE
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]


def main():
    tr = json.load(gzip.open(FN, "rt"))
    meta = tr["meta"]
    frames = tr["frames"]
    me = meta["policy_side"]
    my_bases = [pos for _bid, pos in meta["bases"][str(me)]]
    print(f"policy_side={me} my_bases={my_bases} "
          f"final_cash={meta['final_cash']}")

    last = frames[-1]
    holders = {wid: v for wid, v in last["workers"].items()
               if wid.startswith(f"player-{me}-") and v[1] > 0}
    print(f"\nfinal stranded: total "
          f"{sum(v[1] for v in holders.values()):.2f} over "
          f"{len(holders)} workers")
    for wid, (pos, cargo, typ) in sorted(
            holders.items(), key=lambda kv: -kv[1][1]):
        print(f"  {wid} type={typ} pos={pos} cargo={cargo} "
              f"dist_to_base={min(dist(pos, b) for b in my_bases)}")

    for wid, (pos, cargo, typ) in sorted(
            holders.items(), key=lambda kv: -kv[1][1]):
        if cargo < 20:
            continue
        print(f"\n===== {wid} (final cargo {cargo}, final pos {pos}) =====")
        # last visit to an own base cell
        last_base = None
        for fr in frames:
            w = fr["workers"].get(wid)
            if w and w[0] in my_bases:
                last_base = fr["step"]
        print(f"last step standing on own base: {last_base}")
        # position run analysis over the last 60 frames
        seq = []
        for fr in frames[-61:]:
            w = fr["workers"].get(wid)
            if w is None:
                seq.append(None)
            else:
                act = None
                if "acts" in fr and fr["step"] >= 1:
                    act = fr["acts"][me].get(wid)
                seq.append((fr["step"], w[0], w[1], act))
        moves = sum(1 for a, b in zip(seq, seq[1:])
                    if a and b and a[1] != b[1])
        stays = sum(1 for s in seq if s and s[3] is None)
        print(f"last 60 frames: position changes={moves}, "
              f"stay-actions={stays}")
        # longest stationary run (same pos) anywhere in the game
        best_run, run, prev = 0, 0, None
        best_span = None
        for fr in frames:
            w = fr["workers"].get(wid)
            if w is None:
                run, prev = 0, None
                continue
            if w[0] == prev:
                run += 1
            else:
                run, prev = 1, w[0]
            if run > best_run:
                best_run, best_span = run, (fr["step"] - run + 1,
                                            fr["step"], w[0], w[1])
        print(f"longest stationary run: {best_run} steps "
              f"(span steps {best_span[0]}-{best_span[1]}, pos "
              f"{best_span[2]}, cargo then {best_span[3]})")
        # step-by-step for the last 35 steps
        print(" step | pos | cargo | act | descent-cell occupants")
        for fr in frames[-35:]:
            w = fr["workers"].get(wid)
            if w is None:
                print(f" {fr['step']:4d} | GONE")
                continue
            p, cg = w[0], w[1]
            act = fr["acts"][me].get(wid) if "acts" in fr else None
            nb = min(my_bases, key=lambda b: dist(p, b))
            d0 = dist(p, nb)
            desc = [c for c in nbr4(p) if dist(c, nb) == d0 - 1] \
                if d0 > 0 else []
            occ = []
            at = {v[0]: k for k, v in fr["workers"].items()}
            for c in desc:
                occ.append(f"{c}:{'free' if c not in at else at[c]}")
            print(f" {fr['step']:4d} | {p:3d} | {cg:7.2f} | "
                  f"{str(act):5s} | {occ}")
        # who are the neighbours' blockers in aggregate over last 35?
        block = Counter()
        for fr in frames[-35:]:
            w = fr["workers"].get(wid)
            if not w:
                continue
            p = w[0]
            nb = min(my_bases, key=lambda b: dist(p, b))
            d0 = dist(p, nb)
            if d0 == 0:
                continue
            at = {v[0]: (k, v[2]) for k, v in fr["workers"].items()}
            for c in nbr4(p):
                if dist(c, nb) == d0 - 1 and c in at:
                    block[at[c]] += 1
        if block:
            print(" descent blockers (last 35 steps):", dict(block))


if __name__ == "__main__":
    main()

"""analyze_s4_traces: trace-level camp accounting for the S4 games.

For every traces/fork_s4_*_vs_*.json.gz, from the point of view of
each player:
  posts        = that player's collectors standing on a diagonal cell
                 of an ENEMY base (the camp posts chosen by base_v53)
  camper_steps = sum over frames of #posted collectors
  intercepts   = transitions where an enemy collector with cargo >= 20
                 disappears and a posted collector within distance <= 1
                 of the dead collector's last cell gains >= 80% of the
                 dead collector's cargo in the same transition
                 (collision steal resolved on/near the post cell)
  post_gain    = total positive cargo deltas of posted collectors that
                 did not change cell (interception + on-post collecting)
  camper_deaths= posted collectors that disappear next frame
Aggregates printed per file stem pair; raw per-game rows to stdout.
"""

import glob
import gzip
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 15


def dist(a, b):
    ra, ca = divmod(a, SIZE)
    rb, cb = divmod(b, SIZE)
    dr = min(abs(ra - rb), SIZE - abs(ra - rb))
    dc = min(abs(ca - cb), SIZE - abs(ca - cb))
    return dr + dc


def diagonals(cell):
    r, c = divmod(cell, SIZE)
    return {((r + dr) % SIZE) * SIZE + (c + dc) % SIZE
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1))}


def analyze(path):
    d = json.load(gzip.open(path))
    meta = d["meta"]
    bases = {int(p): [pos for _, pos in v]
             for p, v in meta["bases"].items()}
    frames = d["frames"]
    out = {}
    for p in (0, 1):
        foe = 1 - p
        post_cells = set()
        for b in bases[foe]:
            post_cells |= diagonals(b)
        camper_steps = 0
        intercepts = 0
        intercept_cargo = 0.0
        post_gain = 0.0
        camper_deaths = 0
        for t in range(1, len(frames)):
            prev = frames[t - 1]["workers"]
            cur = frames[t]["workers"]
            posted_prev = {wid: w for wid, w in prev.items()
                           if wid.startswith(f"player-{p}-")
                           and w[2] == "C" and w[0] in post_cells}
            camper_steps += len(posted_prev)
            dead_foe = {wid: w for wid, w in prev.items()
                        if wid.startswith(f"player-{foe}-")
                        and wid not in cur and w[2] == "C" and w[1] >= 20}
            for wid, w in posted_prev.items():
                if wid not in cur:
                    camper_deaths += 1
                    continue
                cw = cur[wid]
                gain = cw[1] - w[1]
                if gain > 0 and cw[0] == w[0]:
                    post_gain += gain
                if gain > 0:
                    for ewid, ew in dead_foe.items():
                        if dist(ew[0], cw[0]) <= 1 and gain >= 0.8 * ew[1]:
                            intercepts += 1
                            intercept_cargo += ew[1]
                            break
        out[p] = dict(camper_steps=camper_steps, intercepts=intercepts,
                      intercept_cargo=round(intercept_cargo, 1),
                      post_gain=round(post_gain, 1),
                      camper_deaths=camper_deaths)
    return meta, out


def main():
    agg = defaultdict(lambda: defaultdict(float))
    files = sorted(glob.glob(os.path.join(
        HERE, "traces", "fork_s4_*_vs_*.json.gz")))
    print(f"{len(files)} trace files")
    for path in files:
        meta, out = analyze(path)
        pol = meta["policy"]
        side = meta["policy_side"]
        key = f"{pol} (side{side})"
        for p in (0, 1):
            who = pol if p == side else meta["opp"]
            a = agg[who]
            a["games"] += 0.5  # each file contributes both players; count per player-role below
            a[f"camper_steps_{'pol' if p == side else 'opp'}"] += out[p]["camper_steps"]
            a[f"intercepts_{'pol' if p == side else 'opp'}"] += out[p]["intercepts"]
            a[f"intercept_cargo_{'pol' if p == side else 'opp'}"] += out[p]["intercept_cargo"]
            a[f"post_gain_{'pol' if p == side else 'opp'}"] += out[p]["post_gain"]
            a[f"camper_deaths_{'pol' if p == side else 'opp'}"] += out[p]["camper_deaths"]
        print(os.path.basename(path),
              "| pol:", out[side], "| opp:", out[1 - side])
    print("\nTOTALS by strategy (pol = as the recorded policy side):")
    for who, a in sorted(agg.items()):
        print(who, dict(a))


if __name__ == "__main__":
    main()

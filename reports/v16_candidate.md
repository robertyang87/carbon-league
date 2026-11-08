# v16 candidate (2026-11-08, not submitted)

Goal: rank 1 on the heavy ladder. v15 (online) is ~2610-2630, 5th-6th; Master2s leads at ~2710.

## Components
- **Policy net `cnet2_b128v7`** (not in git; `data/weights/`): b128 cnet2, warm start from b128v6, 4 epochs on
  dataset v7 (`data/v7_train.sh`, builder `code/build_v6.py`): 1375 top-bot games / 482k decisions,
  slot 3 = Godslayer-P2C, WB-* family held out (exam), per-unit decision weights (`cnet2_build.unit_weights`),
  phase-aware teacher weights (Master2 x1.5 for steps <= 240, x0.5 after; 1C slots x1.5 after step 200),
  base recruit decisions at steps 40-100 x3.
- **Phase style switch** (`cnet2_search` TEACHER / TEACHER_LATE / LATE_STEP): slot 5 (Master2 economy) until
  step 240, then slot 7 (1C-Zero-v11 endgame seizing/defence). Wrapper: `code/exp/cnet2_v7s57_240.py`.
- **Test-time search** (`code/exp/cnet2_v7s_mfxn.py`, uncapped; `code/exp/cnet2_v7s_mfxc.py`, time-capped):
  cnet2_search MCTS on 1 key unit, 12 root sims, horizon 3, online mixture opponent model, s64 rollouts,
  narrowed anti-bankruptcy guard. Capped version: wall-clock BUDGET 0.5 s / HARD 0.75 s per step (env SB/SH).

## Results (paired, same maps; exams never used in training)
| candidate | vs v13 | vs WB-532 | vs Master2s stand-in |
|---|---|---|---|
| v16 net, no search, vs v15 | +129 +/- 38 (800 games) | +104 +/- 101 | +127 +/- 84 |
| v16 + search (uncapped), vs v16 net | +302 +/- 68 (200) | pending | pending |
| v16 + search (uncapped), vs v15 | **+445 +/- 80** (200); 192/200 wins | pending | pending |
| v16 + search capped at platform-equivalent budget | running | running | running |

Timing: uncapped search 186 ms/step mean, 266 ms worst on an idle Mac (platform ~3x slower);
one 3.0 s spike seen on the cloud box -> the shipped version must be time-capped.

## Open questions for review
1. Is the capped search robust to platform slowness (1 s/step, 5 s overage, max 3 slow steps, 3 s hard)?
2. Tail: worst game -1972 with search vs -1342 without (v15: -2001). Bankruptcy check pending.
3. Opponent model for search uses our own style heads; v13 is same-family, so +302 may be optimistic
   vs foreign bots (earlier: search gave +100..+200 vs foreign opponents).

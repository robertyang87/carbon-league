# v16 (submitted 2026-11-08 19:3x) and v17 prep — review notes

## What is online
`claude-opus55-cloud-v16` = `code/exp/submission_cnet2_v16s.py` (md5 645c17dc…, not in git: it embeds weights).
Built with `cnet2_search_pack.py` from:
- policy `cnet2_b128v7` (dataset v7), phase switch: slot 5 (Master2) until step 240, then slot 7 (v11-preview)
- rollout net `cnet2_s64v7` (same 8-slot table as b128v7, retrained on dataset v7; val 0.831 vs 0.780 for old s64;
  NumPy/torch parity max logit diff 1.1e-5)
- search CONF (platform clock): MODE mcts, ROOT_S 10, MAX_KEY 1, MAX_BR 5, H 3, MIX over STYLES 0..7, ID_PRIOR 0,
  BUDGET 0.5 / HARD 0.75 / ABORT_DL 0.9 s, root fallback (exception -> `{}`), forced single-thread BLAS.
  Float equivalent for local exams: `code/exp/cnet2_v7s_mfxn8_r10.py` (fixed count, no clock).

## Local exams (float, fixed-count search; paired on the same maps)
| exam | vs v16 net (no search) | vs v15 | tail (<-1000 games, worst) |
|---|---|---|---|
| v13, 200 | +466 ± 71 | **+608 ± 78** (198/200 wins) | 0, -467 (v15: 2, -2001) |
| WB-532, 200 | +388 ± 65 | +492 ± 101 | 1, -1483 (v15: 5, -1661) |
| Master2s stand-in (old), 200 | +337 ± 94 | +464 ± 93 | 0, -914 |

## Ladder after 35 games: 25-10, 2603, 7th — the local gain did not show up
- Platform timing: 370-393 ms/step per game, last step 435-473 ms, no game near the limits.
- Wins come from delivery volume (+1200/game); seizures are a net loss (we seize 224, lose 446 per game;
  local exam had us seizing ~400).
- Master line is the big hole: v16 0-5 vs Master2/Master2s (v15: 3-10).
- The old Master2s stand-in was far too weak (v15 beat it 75% but went 1-6 vs the real one; it lost 225 cargo/game
  vs ~550 for the real Master2s against v15). Its exam result is void; stand-in v2 is being built (below).

## Open question 1: is the shipped search truncated by its clock caps?
| packed file, vs v13 | conditions | paired vs float search |
|---|---|---|
| **with caps (as shipped)** | **platform, 35 ladder games**: 370-393 ms/step per game (spread 6%), last step 435-473 ms, no game near any limit | (no paired number; direct timing evidence only) |
| with caps (as shipped) | cloud, 60 procs (heavily contended) | -407 ± 75 (200) |
| with caps (as shipped) | Mac, 4 procs, 363 ms/step | -286 ± 149 (40) |
| caps removed (`submission_cnet2_v16s_nocap.py`) | cloud, fixed count | -149 ± 134 (first 60) |
| caps removed | Mac, 4 procs, same 40 maps as row 2 | running (decisive) |
- Platform argument against heavy truncation (Fable): uniform 380 ms/step and a 450 ms last step; a full
  10-round step is ~450 ms on the platform, so with BUDGET 0.5 and the 1.3x projection a few steps may lose the
  10th round, not most.
- Agreed regardless of the outcome: caps go to BUDGET 0.7 / HARD 0.85 / ABORT_DL 0.95 so they only catch spikes.
- If removing the caps does not restore the gain, the remaining cause is the opponent model on foreign bots
  (our search predicts its own family well; on the ladder it seizes less than v15 did).
- Second suspect: int8 quantization of the new rollout net (first time it is quantized); rows 3 vs float.

## Open question 2: endgame interception vs the Master line
Master2 game (#27): +978 at step 250, overturned by 1549 in the last 49 steps while its 10 collectors ran home.
Search looks at 1 key unit per step (MAX_KEY=1). Plan for v17: MAX_KEY 2-3 in the last 50 steps only
(those steps often have no searched unit, so the time is there).

## v17 in progress
- Dataset v8 (`code/build_v8.py`, `data/v8_train.sh`): v7 + our ladder losses with the winner as teacher (only
  winners that map to a slot) + winners of games against tree-rush families (无始 / Godslayer) x2.
  无始 is NOT a teacher: it is a held-out exam. Training b128v8 from b128v7.
- Stand-ins v2 (`build_m2s2.py`, `build_m2v2.py`, `build_ws.py`): Master2s from all 137 games (held out),
  Master2 from ~300 games (practice target only — Master2 is a teacher), 无始6/7 (held out); each also with our
  search on top (`*_s.py`). Acceptance by ledger (`data/stand_check.py --vs v15`): collectors@75, collector-steps
  and cargo lost within ±15% of the real bot against the same opponent (Master2s vs v15: lost ~550).
- Comparisons on search-enabled stand-ins only like-for-like (net vs net or search vs search): the stand-in shares
  our b128v7 base, so our opponent model predicts it unrealistically well.

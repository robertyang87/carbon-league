# K4TacSearch-v1 clone+search probe - preregistration (2026-11-12 14:20)

Owner decision (2026-11-12 14:18, DGX side chat): build a K4TacSearch-v1
stand-in as CLONE + SEARCH, in two stages. Probe first; formal stand-in only
if the probe passes. Do not displace running work.

## Baseline (on-box record, h24 bigeval.jsonl)
- pk_k4r1 (K4TacSearch-v1 clone, packaged) vs E0 (echo_e0): mean -824, se 60,
  wins 31/200, seed0 994000, worst_ms 183.
- Same pack vs Master2 real: mean -143, se 56, wins 90/200.
- NOTE: the -786 figure in older summaries is superseded by this on-box
  record (-824). The probe re-runs its own SEARCH=False arm anyway.

## Probe (stage 1)
- One variable only: the pk_k4r1 wrapper with SEARCH flipped False -> True
  (fleet cnet2_search DEFAULTS otherwise: MAX_BR=16, BUDGET=0.38,
  MIN_DEPTH=3, MODE=sweep). Same weights cnet2_k4r1L.npz.
- A/B in one session on h24: SEARCH=False arm and SEARCH=True arm,
  both vs E0, 200 pairs, SAME fresh seed range.
- PASS line (preregistered): SEARCH=True mean >= -330 (recovers >= 60% of
  the 824 deficit) AND worst_ms <= 200 (arena step budget; baseline arm's
  worst_ms 183 on the same harness).
- STRONG PASS: mean >= -100 (v1's edge is mostly play-time search).
- FAIL: mean < -330 -> STOP. No formal stand-in. Report numbers only.

## Stage 2 (only on PASS)
- Formal stand-in = k4r1 clone base (ds_k4 / k4_corpus already on DGX) +
  the winning search config, packaged; full calibration battery
  (Master2 real, opp_w3b, opp_m3echo, E0, 200+ each) and a style comparison
  against the two real losses K4TacSearch-v1 dealt our family
  (margins -1888 / -2168 in the ladder records).

## Queue discipline
- h24: probe runs after the m3s2r1 final calibration finishes.
- h23: m3s2g1 control calibration in parallel (weights via DGX relay).
- M7Bs (DGX) and Studio distillation are not displaced.

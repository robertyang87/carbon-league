# Search-budget probe (b) — final report

Date: 2026-11-18. Design: DESIGN_BUDGET_PROBE_20261117.md (approved
2026-11-17 09:48). Driver: 搜索组. Question: does deepening the
annotation-time search budget (T1 = K16/M8/H2 → TD = K32/M16/H3) improve
teacher labels? TR = K64/M32/H4 is the convergence reference on a
subsample (not ground truth).

## Execution record

Gen-2 pin throughout (annotate e9c7a0fb + teacher d446e5b6, five identity
files unchanged), weights 06a03d9c, pi0=pi1=1.0, no isolation file.
Rebuilt on spot-64 after the spot-44 loss (see root causes). Config gates
recomputed on-seat before launch, both verbatim: cost_T1 (994000,
[200,203), 6 roots) rc 0, 95.95 s, config 8787930e…; cost_TD rc 0,
418.39 s, config 1348fd16…. Clean unit prices on spot-64: **T1 16.0
s/root, TD 69.7 s/root** (spot-44 anchors were 25.8 / 113.5).
Legs (all rc 0): T1_g0 1,549.4 s (96 roots); TD_g0 9,721.4 s (96);
TR_g0 4,976.6 s (24, mixed contention); TD_g1 8,252.9 s (96). From
09:08 the seat dual-ran the owner's production annotation (62 workers)
by 作战室 arbitration — walls stretched, labels unaffected (per-root
deterministic; verified below).

## Readings (game selfplay-994000 unless noted)

**(a) Policy-top flip T1→TD: 21/96 = 21.9%** (pre-registered positive
line ≥10% — met). Arms prefix-aligned 96/96.
**(a2) Value-top flip: 74/96 = 77.1%**; in 27 of those, TD's winner is
outside T1's candidate set entirely.
**(b) Convergence error vs TR** (24 shared roots in [200,212), candidates
aligned by arm identity): T1 err 0.0174 (384 candidate pairs), TD err
0.0145 (600 pairs; TD covers 100% of TR's recorded candidates, T1 covers
64%). **Reduction +16.2%** — below the +20% positive line, above the
<5% negative line. Paired per-root diff +0.0028, se 0.0032, t = 0.87
(TD better on 16/24 roots): the point estimate is inside noise at n=24.
**(c) Realized-U error of the chosen candidate's Q** (replay result.u):
policy choice |Q−u| T1 0.2667 vs TD 0.2667 (paired diff −0.0001, TD
better 44/96); value choice T1 0.2767 vs TD 0.2791 (diff −0.0025, TD
better 39/96). **No improvement in either variant.**

**Determinism cross-check**: the rebuilt shards reproduce the lost
spot-44 run's (a)/(c) readings to the printed digits (21/96, 74/96,
0.2667/0.2667, preview |ΔQ| mean 0.0160) — the 44 loss cost wall time,
not information.

## Verdict mapping (pre-registered)

Positive required (a) ≥10% AND (b) ≥20% AND (c) same-direction. (a)
met; (b) +16.2% misses its line and is not significant; (c) flat/worse.
Negative required flips <3% or reduction <5% — neither holds.
**Middle zone: raw numbers to 策略组 for the ruling**, per the design.
Substance in one line: doubling-and-a-bit the budget re-ranks labels
heavily and moves Q ~16% closer to the deep reference, but on this
sample it does not make the chosen candidate's Q predict the realized
outcome any better.

## Sample reduction (as ordered to state)

Original batch: (a)/(c) on two games (994000+994001), (b) on 24 roots.
Delivered: (a)/(c) on **one game** (994000, 96 roots); (b) as designed
(24 roots). T1_g1 was downgraded to spare capacity in the rebuild
ruling and not re-run; TD_g1's shard exists (gen-aaa43305e8f4cdd6) and
can pair with a future T1_g1 if the second game is ever wanted.

## Root causes (complete list)

1. spot-34 released 2026-11-17 ~11:08 mid-probe; label legs there were
   skeletons only (cost_TD survived via DGX's pull).
2. spot-44 released 2026-11-17 16:28 with disk destroyed, zero recovery:
   T1_g0/TD_g0/T1_g1 shards (~3.9 h compute) and the in-flight TR leg
   lost. DGX's ordered wholesale pull never landed (tailnet offline,
   public IP reassigned). Mitigation since: shards pulled to the driver
   the moment legs land (done this run) + the artifact-registry /
   release-gate rules the owner set on 2026-11-17.
3. Original chain design misread k0_annotate's pool granularity (it
   maps over games, not roots) — per-game walls were ~50x the per-root
   estimate; rebuilt chain is per-game with legs concurrent.
4. Supply tar aa96d8ca lacks eval_v53.py (+env_src): two rc-1 starts on
   spot-64 before the gap was found and patched from the seat's
   production tree (eval_v53 45792dfb verified). DGX has logged the
   packaging fix for future kits.
5. Seat contention from 09:08 (owner-ordered production annotation on
   the same seat, arbitrated dual-run): TD_g0 wall 9,721 s vs ~6,690 s
   unloaded estimate. No reading impact.

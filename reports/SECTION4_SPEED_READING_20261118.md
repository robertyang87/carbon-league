# §4 formal speed reading — small-teacher tiers B and C vs teacher (spot-64)

Date: 2026-11-18 07:26–07:39 CST, spot-64 (unloaded, load 0.00–0.13 verified),
single thread (OMP/MKL/OPENBLAS/NUMEXPR = 1), batch = 1, production cnet2_np
path, pol.act only (Feat.observe excluded, per design §5.1).
Fixed manifest: 2,048 positions (train split f7b50267; step band × seat
stratified 341×4 + 342×2; seeded walk 20261118; extraction via dual-Feat
Route-A tracking; per-position replay sha256 recorded in
manifest_positions.json). Students are random-init (latency is shape-only).

## Readings (n = 2,048 each)

| net | params | mean | p95 | band 0–99 | band 100–199 | band 200–299 |
|---|---|---|---|---|---|---|
| teacher it50 | 3.481M | 28.281 ms | 28.640 | 28.134 | 28.371 | 28.337 |
| student B (48/4/2/96/2/E24) | 0.528M | 7.341 ms | 7.480 | 7.204 | 7.420 | 7.398 |
| student C (32/3/1/64/1/E16) | 0.188M | 2.923 ms | 3.018 | 2.819 | 2.985 | 2.967 |

Ratios vs teacher: **B 3.85x**, **C 9.67x**. B reproduced across two runs
(7.348 / 7.341 mean) — readings are stable, distribution tight (act cost is
structure-dominated; band split is flat for every net).

## §4 gates (registration: mean ≤ 5 ms AND ≥ 8x)

- **Tier B: FAIL both** (7.341 ms > 5; 3.85x < 8).
- **Tier C: PASS both** (2.923 ms, 42% under the cap; 9.67x, 21% over the floor).

## Why B missed (measured, not conjectured)

The design estimated B at ≈4–5 ms and the teacher at ~35–45 ms on this seat
class. Both were off: the teacher is faster here (28.3 ms), and B's latency
has a high fixed floor — at 64M MACs the per-call time is dominated by
interpreter/small-matrix overhead, not math, so shrinking MACs 11.2x bought
only 3.85x. C's 3.2x fewer MACs than B plus one fewer attention layer cut the
floor to 2.92 ms. (My earlier VM prototype ratio of 9.37x for B was an
artifact of the VM's slow cores, where compute dominates; the formal seat
reading supersedes it.)

## Disposition (for 策略组's ruling)

Per the design's decision-10 order (B fails → drop to C before considering
inference-side optimization; never relax the registered numbers): Tier C is
the §4-passing architecture. C's known cost is precision risk (design §1.3:
Tier 1 unlikely, Tier 2 the planned landing) — whether to train C as the
main line is the ruling question; the measurement question is closed.

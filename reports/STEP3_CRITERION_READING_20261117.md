# §4 Step-3 criterion reading — vnet leaf vs heuristic leaf (comparator instrument)

Date: 2026-11-17 (arms ran 22:46–22:54 CST, spot-62). Caliber: 策略组 ruling 2026-11-17 15:58
(questions submission b6911118; ack da9789b9). Registration: reports/REGISTRATION_SECTION4_OWN_SEARCH_20261117.md
(commit 48d89a03). Fixed seed set 20291101–20291400 (300 seeds), same vehicle, same seeds,
leaf-only swap; leaf = it50 vnet direct read (VNetLeafAdapter, U = (tanh z + 1)/2, Δz = 0.0067 A0 bonus).

## Identity (all read back on-seat, spot-62)

- Harness bigeval.py 354bcd1b, eval_v53.py 45792dfb; Echo4 base echo4_int8.py 819250bf.
- Baseline arm pack echo4_search_int8_resync.py cba5923c (leaf-① heuristic).
- Vnet arm pack echo4_search_int8_cmp3.py d39e0627; vnet_it50.npz 1c9b6273 (source
  code/search_proto/vnet_it50.npz, md5 1c9b6273; commit 0fc00403).
- Env: venv python 3.12.3 / numpy 2.5.4; thread caps OMP/MKL/OPENBLAS/NUMEXPR = 1 (both arms);
  EVO_WORKERS = 32; run script run_step3_48.sh c458d46e.

## Readings (paired, 300/300 seeds, 0 error rows)

| arm | mean margin | se | mean U |
|---|---|---|---|
| baseline (leaf-①) | −631.71 | 41.0 | 0.2662 |
| vnet leaf (it50) | +124.45 | 62.7 | 0.5787 |
| **paired ΔU (vnet − base)** | **+0.3124** | 0.0226 | **95% CI [+0.2680, +0.3569]** |

- Arm summaries (bigeval): baseline wins 50/300, mean −632, se 41; vnet wins 184/300,
  mean +124, se 63; both FAIL=false, search_games 297/300, 0 search errors, 0 root errors.
- Changed totals (comparator overrides): baseline 28,546 vs vnet 5,499 — the vnet leaf
  overrides far less often, yet flips the margin by +756 points.

## Criterion verdict: PASS

Pinned criterion: baseline margin < 0 AND vnet-leaf margin ≥ 0.
- Baseline −631.71 < 0 ✓. Vnet +124.45 ≥ 0 ✓.
- Paired ΔU +0.3124 with the entire 95% CI above zero — the leaf swap is a decisive
  positive, not a noise-band effect.

## Notes / deviations (all disclosed)

- Seats: the run landed on spot-62 after spot-48 (reclaimed) and spot-54 (reclaimed);
  all supplies rebuilt from repo/DGX, identities re-verified on-seat before launch.
- Two failed starts on 54 (system python3 had no numpy; then uncapped BLAS threads
  spun the workers to load 534 with zero completions in 11 min) — diagnosed with a
  capped control game (seconds), fixed by venv PATH + thread caps; zero games were
  produced by either failed start, no seed contamination.
- On 62 the arms ran concurrently with reading block 3 (62 workers, readout_supply)
  rather than after it; both sides thread-capped, pace unaffected
  (arm 1 in ~3 min, arm 2 in ~5 min).
- The vnet pack's runtime npz load at agent construction is a diagnostic-only
  deviation (packs normally make no file access at play time); recorded here per
  the registration's verification notes.
- One baseline row (seed 20291115): margin 0.0, srch 0 — a degenerate game; retained
  in the paired analysis (n = 300).

## Next

§2 critical path in order: ① forward-net distillation (stand-in speed measurement
first: random-init Tier-B weights through cnet2_np), ② critic distillation,
③ batched inference, ④ engine. Stage check 2026-11-20 20:00.

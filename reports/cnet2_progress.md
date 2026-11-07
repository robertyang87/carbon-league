# cnet2 progress

## 2026-11-07 ~09:30 — first end-to-end (small model)
- Pipeline: cnet2_feat (canonical full board, 33 planes, belief w/ dynamics + mirror fill, 8-step unit history), cnet2_build (pool of 10 procs, ~1 min for 451 games), cnet2_model (torch), cnet2_np (NumPy; parity vs torch max |dlogit| 4.8e-7, |dvalue| 1.1e-7), cnet2_agent, cnet2_pack.
- Dataset v1: 451 games (dedup ladder_top + top_corpus), 157k teacher decisions (144k train / 13k val, 38 val games).
- Small model cnet2_s64 (C=64, NB=4, D=128, 2 attn layers, GRU decoder; 0.84M params), 4 epochs (~7.5 min MPS):
  val acc collector 0.760, planter 0.708, base 0.930, all 0.780 (workers 0.747).
- vs opp_1cz_v5, seeds 960000-015 (32 games): 0/32 wins, mean -2116 ± 151, worst -4396 (seed 960014);
  ledger pol/opp: deposits 6680/7681, tree income 3373/2905, death cargo 235/335. ms/step 62 mean / 284 worst (while MPS trains and 10 eval workers run).
- Diagnosis seed 960003: ~100 deaths per side, 81 of them planter-vs-planter mutual kills (P0P1); cnet2 overspends on planting (47 trees, 2769 plant spend vs 13 / 603).
- Next: 8x128 model (2.95M params) training on dataset v2 (516 games, 178k samples).

## 2026-11-07 ~10:45 — small-model variants, baselines, pack check
All vs opp_1cz_v5, seeds 960000-015 x both sides (32 games; note: the harness makes side 0 and side 1 of a seed exact mirrors, so effectively 16 independent games).
| agent | wins | mean ± se | worst |
|---|---|---|---|
| v10 (cand_c1z_v1) baseline | 11/32 | -201 ± 153 | -1851 |
| cnet2_s64 teacher=1 (v9 style), pure | 0/32 | -2116 ± 151 | -4396 |
| cnet2_s64 teacher=0 (Master style), pure | 0/32 | -1610 ± 215 | -5330 |
| cnet2_s64 t=1 + shield/nocollide/endgame | 0/32 | -1707 ± 155 | -2993 |
| cnet2_s64 t=0 + shield/nocollide/endgame | 0/32 | -1468 ± 205 | -4522 |
| cnet2_s64 t=0 + rules + planter-shield | 0/32 | -10226 ± 278 | -14165 (planter shield is catastrophic: drop) |
| cnet2_b128 epoch-4 checkpoint (val 0.751), t=0 pure | 0/32 | -3490 ± 543 | -14534 (bankrupt: 3 early recruits died, cash 6 < 30) |
| cnet2_b128 epoch-4, t=0 + rules | 0/32 | -2989 ± 138 | -5181 |
- Feature sanity: s64 agreement with v5's own moves in fastsim self-play (diagnostic only, not training): collector 0.715, planter 0.695, base 0.846 — close to replay val, so no train/serve feature mismatch.
- Training speed: replaced padding_mode='circular' with a single index_select wrap + bf16 autocast (MPS): conv stack ~2x faster. NumPy parity after the change 1.0e-5.
- Pack: cnet2_pack.py (int8 per-row, zlib+base64) s64 file 1.18 MB; packed file == unpacked q8 wrapper game-for-game (2/2 games identical margins).
- b128 (C128 x 8 blocks, 2.95M params) val all: ep4 0.751, ep5 0.784, ep6 0.800 (collector 0.777, planter 0.743, base 0.944); still training (14 epochs).

## 2026-11-07 10:40-11:00 — cnet2_b128 (final, 14 epochs)
- Model C=128 x 8 GN res blocks, D=128 tokens, 2 attention layers (4 heads, torus rel-pos bias), GRU decoder; 2,946,610 params. Training 14 epochs x 159 steps (bs 1024, OneCycle lr 1.5e-3, bf16 autocast), ~5 min/epoch on MPS.
- Val (44 held-out games, 14.7k decisions): all 0.826, collector 0.804, planter 0.775, base 0.955 (workers 0.797). Curve: ep4 .751, ep7 .806, ep10 .824, ep13 .826.
- **vs opp_1cz_v5, seeds 960000-031 x 2 sides (64 games), pure net, teacher style 0 (Master): 54/64 wins, +744 ± 116, worst -2207 (960001)**; ledger pol/opp deposits 8828/8177, tree 2675/2554, death cargo 104/447. ms/step 42 mean, 260 worst (under 10 parallel workers).
- **vs v10 (cand_c1z_v1), same seeds: 61/64, +1811 ± 151, worst -2128 (960022)**; ledger pol/opp deposits 8381/7085, tree 3072/2576, death cargo 459/166.
- ep7 checkpoint (val .806) was already 20/32 +391 ± 151 vs v5; ep4 (val .751) 0/32 -3490 — game strength is very sensitive to BC accuracy in this range.
- Unloaded-ish timing (2 games, PPO at nice 19 running): ep7 weights mean 53 ms, p99 124 ms, max 254 ms.
- Packed: exp/cnet2_b128_t0_packed.py (int8, 3.56 MB) == unpacked q8 wrapper (exp/cnet2_b128_t0_q8.py) in 3/3 games (identical final cash). f16 pack 7.3 MB.
- Next: cnet2_b160 (C=160 x 10 blocks, D=160, 5.50M params) training, 18 epochs.

## 2026-11-07 11:10 — teacher style, rules, TTA, submission candidate
- b128 teacher style 1 (1C-Zero-v9 embedding), pure, vs opp_1cz_v5 seeds 960000-031 x2: **60/64, +1142 ± 93, worst -659 (960027)**; ledger pol/opp deposits 8750/8300, tree 3280/1808, death cargo 254/426. (style 0: 54/64 +744 ± 116.)
- b128 style 0 + rule layer (shield/nocollide/endgame) vs v5: 52/64, +813 ± 115 — rules ≈ neutral for the big net.
- Symmetry TTA, offline teacher-forced val (3000 decisions): k=1 0.8259, k=2 0.8298, k=4 0.8321 (collector .802/.806/.809, planter .781/.789/.789).
- Submission candidate exp/cnet2_b128_t1_packed.py (int8, 3,567,277 B, md5 94fcf38a…): identical to unpacked q8 wrapper in 3/3 games (+1422, +2428, +2203 vs v5); standalone run (python -I, empty dir): numpy only, 84 MB peak RSS, first call 0.10 s.
- cnet2_b160 (10x160, 5.50M params) training: ep0 0.565 (13 min/epoch under load).

## 2026-11-07 ~11:50 — more b128 t1 results
- b128 t1 int8 (q8 weights == packed v12 file) vs opp_1cz_v5, 960000-031 x2: 58/64, +1314 ± 119, worst -400; ms mean 123 / worst 808 (heavy contention).
- b128 t1 (fp32) vs v10 cand_c1z_v1: 62/64, +1634 ± 115, worst -175; ledger pol/opp deposits 8339/7345, tree 3160/1844, death cargo 581/292.
- b128 t1 + 2-view symmetry TTA vs v5: 62/64, +1020 ± 83, worst -369; ms mean 172 / worst 969 → no gain, not worth the latency.
- cnet2_pack.py fix: no top-level __file__ (platform execs source without it); every pack now checked with data/sim_platform_load.py (t1 pack: load ok 0.21 s, one game vs v9 won 7965-5555).
- Dataset v3 built (540 games, 185k decisions); b128v3 warm-start retrain queued until b160 finishes (GPU reserved for b160).

## 2026-11-07 ~12:30 — teacher-style sweep (b128 fp32 pure vs v5, seeds 960000-015 x2, 32 games)
| style | wins | mean ± se |
|---|---|---|
| t1 1C-Zero-v9 | 32/32 | +1275 ± 113 |
| t3 1C-Zero-v7 | 30/32 | +1154 ± 147 |
| t2 1C-Zero-v8-preview | 32/32 | +1088 ± 135 |
| t0 Master | 26/32 | +465 ± 160 |
| t4 RyanX-V11 | 20/32 | +363 ± 137 |
Quiet-machine inference (b128): 12-13 ms/step mean, ~100 ms worst.

## 2026-11-07 12:45 — b160 run 1 stopped, b128v3, plant ledger
- b160 run 1 (lr 1.5e-3, ds v2): val ep2 .652, ep3 .679, ep4 .645 (regression), ep5 .675, ep6 .694 vs b128 .800 at ep6 → stopped (lr too high for 10x160). Log: data/cnet2_train_b160_run1.log.
- b128v3: warm start from b128, ds v3, lr 5e-4, 6 epochs: val (v3 val set) .8247 → .8324 (collector .812, planter .782, base .959).
- vs opp_1cz_v5, 960000-031 x2, int8 weights, 3 workers:
  - v12 = b128 t1: 58/64, +1314 ± 119, worst -400; ms 22/181; plants/game 32.8 (spend 1430) vs v5 11.1 (429); tree income 3278 vs 1636; seizes 26 vs 32.5; recruits 90 vs 90.
  - b128v3 t1: 56/64, +1092 ± 88, worst -801; ms 32/321; plants 32.6 (1517) vs 12.4 (491); tree income 3318 vs 1716 → not better than v12.
- b128v3 packed (exp/cnet2_b128v3_t1_packed.py): sim_platform_load ok, packed == q8 wrapper 3/3 games.
- b160 run 2 started 12:35: lr 6e-4, ds v3, 18 epochs.

## 2026-11-07 ~14:30 — b160 run 2, dataset v4, b128v4
- b160 run 2 (lr 6e-4, ds v3): val .551, .625, .701, then .529 at ep3 (confirmed fp32 CPU .537), train loss spike .687→1.056 at ep4 → diverged, stopped. Log data/cnet2_train_b160_run2.log.
- Dataset v4 (data/cnet2_ds_v4; CNET2_WEIGHTED=1): 556 games, 189,865 decisions, 49 val games; per-decision weight = 2 (teacher side won) x 1.5 (both rated >= 2400); hist w1.0 17,342 / 1.5 26,312 / 2.0 119,899 / 3.0 26,312 (mean 1.98). 17 games our bots lost are included with the winner as teacher (6 v12 losses, w=3.0); ours_corpus rule added 2 extra.
- cnet2_train.py: weighted CE when w.npy present (BC_USE_W=0 to disable); BC_SUBSET, BC_PCT, BC_CLIP; cnet2_model ZI option (zero-init residual branch).
- b128v4: warm start b128, ds v4, lr 5e-4, 6 epochs: val (v4 val) .826 → .832 (collector .811, planter .782, base .959).
- Gate 1, b128v4 int8 vs v12 (submission_cnet2_v12), seeds 973000-023 x2: 24/48, +104 ± 100, worst -1193 → not significant. ms 9.2 mean / 31 worst.
- Big-model recipe ready (code/cnet2_big_recipe.sh: C160 NB10 D160 ZI=1, fp32, lr 5e-4, 15% warmup, clip 1.0); CPU sanity passes; numpy parity 3.6e-7.
- Gate 2 (style stand-ins, seeds 973000-007 x2, 16 games each; b128v4 vs v12 on same seeds): t0 +119±344 (8/16) vs +97±201 (10/16); t3 +643±195 (12/16) vs +612±203 (14/16); t2 +70±226 (8/16) vs -111±158 (8/16); t4 +707±309 (12/16) vs +380±186 (10/16).
- Gate 3, b128v4 int8 vs opp_1cz_v5 960000-031 x2: **62/64, +1540 ± 113**, worst -577 (960005); ms 46/305 (2 workers + RL load); plants 34.6 (1637) vs 11.1 (453); tree income 3589 vs 1584; deposits 8592 vs 8105; death cargo 282 vs 474. (v12 same seeds: 58/64 +1314 ± 119.)
- Packed: exp/cnet2_b128v4_t1_packed.py (md5 66375f15…), sim_platform_load ok.

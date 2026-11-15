# Probe producer binding (C1 / spot-29)

## Historical result (already run, do NOT re-run)
- SPROBE-S29-1, 2026-11-13 11:26 CST, spot-29, rc=0
- Producer: code/producer/selfplay_gen.py **md5 b275e952da14d644e6fab41e4ce941c8** (16,464 B) — **OLD version, pre-F11**
- Script: code/producer/sp_probe_s29.sh md5 629b4d1160f6eddf25a1f5940ccc3404
- Result: SPGEN-RESULT games 56, games_per_min 82.6, mean_steps 299.0, workers 56 — **labelled OLD-VERSION**
- Note: the old script file sp-probe-s29.sh (md5 47e98870…, base64-embedded) is VOID, never run.

## Superseded binding (2026-11-13 12:33, war room order)
- Producer: code/producer/selfplay_gen.py md5 8aed7f48245cf8b4393e5a9f69782566 (29,153 B), Codex REVIEW_FULL F11, HEAD 4f12f901
- Superseded by the a7ed8f95 binding below on 2026-11-13 13:23 (engineering group task, owner order).

## Current binding (from 2026-11-13 13:23, engineering group task, owner order)
- Repo HEAD: main **a7ed8f9552ffc7217c8ea4fe432d575713b4295f**
- Producer: code/producer/selfplay_gen.py **md5 95d2e29fbec8cfcebee57bf9bf0d25ed** (37,499 B)
- Verified by C1 from carbon-league main at a7ed8f95 on 2026-11-13 (blob 9a7938b1); cross-checked against the engineering group's own read-back (same md5).
- All future probes and production game generation MUST use this version and verify by this md5 before running.
- New producer carries a recovery identity block (weights/config/code hash, written to <out>/identity.json). The old probe output dir /root/carbon/data/selfplay_probe/s29 on spot-29 was produced by the OLD producer and has NO identity.json — do NOT resume/adopt it under the new producer; it would be judged identity_mismatch. Before any first resume with the new producer, check identity.json.
- Discrepancy RESOLVED (2026-11-13, engineering group): at a7ed8f95 the script was still the old issue (md5 629b4d1160…) with GEN_MD5 naming the old producer; engineering fixed it in commit **015002e6922b2e2e6555685baa894d0d7acbd188** (main HEAD now 015002e6, only this pin changed). Re-issued script: md5 **406fd25e8be0413a9f062eeef285658a** (4,061 B), GEN_MD5=95d2e29fbec8cfcebee57bf9bf0d25ed, no old value residual — verified by C1 from main. Deployed to spot-29 (/root/carbon/sp_probe_s29.sh + code/producer/sp_probe_s29.sh); the on-machine 629b4d11… script is VOID. Future s29 probes fetch the script fresh from main anyway.

## 991400 probe binding (from 2026-11-13 14:30, war room order)
- Repo HEAD: main **45d57c1cf04d4d81e8d81e2a9afa35340c36000c** (P1; a7ed8f95 pin superseded)
- Producer: code/producer/selfplay_gen.py **95d2e29fbec8cfcebee57bf9bf0d25ed** (37,499 B) — unchanged from a7ed
- Model: code/cnet2_model.py **3bb5857f833a94797ac6cdc2a5a28963** — unchanged
- Bigeval: code/bigeval.py **327ff8073932cc8bd59c3e59090d2647** — unchanged
- Eval: code/eval_v53.py **aa4c5bb63c879caffeb33b0b25c498d7** — unchanged
- Build: code/cnet2_build.py **d6e867187ceec289c0e28a53399eb01a** (66,847 B) — P1
- Train: code/cnet2_train.py **39e749cb177e3d72511e8d9fc986da80** (49,861 B) — P1
- All six verified by C1 from main at 45d57c1c on 2026-11-13 14:30, and read back on spot-29 (all six on-disk md5s match; weights cnet2_m3s2r1L.npz b6db08c9… in place; machine idle).
- Builder re-deploy baseline "main ee6164be" is VOID (per war room).
- BLOCKER resolved (2026-11-13 14:35, war room): engineering issued code/producer/sp_probe_s29_991400.sh, commit bf190bcf31e94f4585b3fe08850a729a491c4d68, md5 **a42c99dd9f34ff04fbe8bc99e0565757** (4,359 B), marker SPROBE-S29-2, seed range 991400–991600, out /root/carbon/data/selfplay_probe/s29_991400, log /tmp/sprobe_s29_991400.log. C1 verified from main (md5/size/marker/seeds/GEN_MD5=95d2e29f…/weights b6db08c9…, plain text, no encoded payload) and deployed to spot-29 (/root/carbon/sp_probe_s29_991400.sh + code/producer/sp_probe_s29_991400.sh).

## SPROBE-S29-2 result (991400, run 2026-11-13 14:36:40–14:38, rc=0, P1 pin)
- SPGEN-RESULT: games 56, wall_s 41.7, total_wall_s 82.5, **games_per_min 80.52**, mean_steps 299.0, mean_margin 0.0, mean_u 0.5, workers 56, cpu_seconds 4473.2, eff_cores 107.2, errors [], budget_hit=games, stub=false, adopted 0, identity_mismatches 0. Identity block present (weights b6db08c9…, config_hash f872ee1c…, code_md5 eval_v53 aa4c5bb6/fastsim aefe6c83/search 10a0674e/selfplay_gen 95d2e29f).
- Three-piece evidence: script pid + 56 workers ran (LOAD-BEFORE 0.08 → CPU-BUSY 84% → LOAD-AFTER 40.24); output /root/carbon/data/selfplay_probe/s29_991400 has identity.json (14:38) + index.json (56 entries) + replays/ 56 files (selfplay-991400.json.gz …); no lingering processes after.
- Notes: requested 20 games, produced 56 (one full 56-worker wave, same as SPROBE-S29-1); all 56 games draws (mean_margin 0.0, mirror self-play), same as the 991200 probe. "adopted 0" on a fresh output dir (0 pre-existing games to adopt); 56 replays + index entries landed, identity_mismatches 0.
- Do NOT re-run SPROBE-S29-2; the next step is the gen-1 production taskbook from the war room.
- No new games with any older producer before the gen-1 production taskbook arrives.

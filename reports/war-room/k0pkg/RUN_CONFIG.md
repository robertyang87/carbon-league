# K0 prerequisite-4 annotation — distribution package / run config (as-run on DGX)

Source run: DGX spark-29db, launched 2026-11-14 18:07 CST, PGID 1394500,
stopped by owner order 2026-11-14 19:05 CST with 0 shards landed (nothing to
deduct; all 200 games in this package go to the C1/C2 partition).
Stopped progress archived at DGX:/home/arm/junyang/k0_prod_stage/progress_stopped_20261114.json

## Verbatim DGX command line
python k0_annotate.py --weights /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_heads_v2.npz --replays /home/arm/junyang/carbon/data/gen1r_shard1/replays --games /home/arm/junyang/k0_prod_stage/games_shard1.json --out /home/arm/junyang/carbon/data/k0_annot_gen1r_shard1 --isolation-file /home/arm/junyang/k0_prod_stage/iso_shard0.json --isolation-file /home/arm/junyang/k0_prod_stage/iso_test323.json --workers 16

## Identity (direct read from the stopped run's progress.json)
- config_sha256 = 9a2b2440e9267680c9b27a1d51a50db9cacdbd04582ae101ef54abaa86d8d10c
- weights_md5    = 06a03d9cfeec239af3065a49fefec382
- games_sha256   = eee281718f5744adb2f6fa78f3df9e4d90bc2d88e18d6ba8b11528a698795861
- selected_sha256= 756c9541f938b9766f74838ef46799dc437ef683d4bfbafae5a8e3592ea01a7f
- window [200, 299], pi0 = pi1 = 1.0, seed 12345
- code_md5s: k0_annotate.py 7f567a2e8544b78e51d16aa7660bc47b,
  stage1_teacher_gate.py e922d9d890c3e1eff6643e947289acb0,
  stage0_teacher.py c6ade5ced42e2c54a77c1499163aa151,
  cnet2_feat.py 2a45b08b6f97508541e62867e408748e,
  cnet2_search.py 94f7fb6438bb7577b141e388f068e122,
  cnet2_np.py d8c15ad4fdd8ee079363cf215d52daf3,
  fastsim.py aefe6c83826efae4de83bef8c643c80b
  (eval_v53.py 45792dfbea3101812836fe77e6752706 is imported by the gate for
  CFG but is NOT part of the code_md5s identity set; it ships in code/ and
  reads code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json
  relative to its own directory at import — keep code/ and code/env_src/
  together, do not relocate eval_v53.py alone.)

## Parameters: DGX run value vs code default (k0_annotate.py argparse)
- --window-lo 200 = default 200; --window-hi 299 = default 299
- --belief sample = default "sample"; --teacher-style 2 = default 2
- --K/--M/--H = CANON defaults K=16, M=8, H=2 (not passed on the DGX command line)
- --seed-teacher 7000 = default; --train-temp 1.0 = default 1.0
- --behavior-temp 1.0 = default 1.0; --seed 12345 = default 12345
- --pi0 1.0 / --pi1 1.0 = defaults
Non-default items on the DGX run were ONLY: --weights/--replays/--games/--out
(paths, machine-local), --isolation-file x2 (the two files ship in this
package unchanged), and --workers 16 (machine-local; each machine uses its
own worker count). No compute-semantics parameter deviated from default.
Target on C1/C2: recomputed config_sha256 must equal 9a2b2440... (above).

## torch
The annotate import closure (the 7 identity files + eval_v53) contains no
torch import (only a docstring mention in cnet2_np.py). A live worker's
/proc maps on DGX contained 0 torch mappings. torch is NOT required;
a venv without torch runs this unchanged.

## Timing shards (designated by DGX coordinator)
- C1 (spot-29): selfplay-994400, selfplay-994401, selfplay-994402, selfplay-994403
  (C1's partition will start at the head of the game order.)
- C2 (spot-31): selfplay-994596, selfplay-994597, selfplay-994598, selfplay-994599
  (C2's partition will end at the tail of the game order.)
Timing shards double as each machine's first-shard verification: send them
back to DGX first; DGX runs --verify + identity read-back, and only then
releases the full run and fixes the partition boundary from the two
measured per-root rates. Timing shards are production shards unless DGX
verification fails (status stated explicitly in DGX's reply).

## Shard return
- C1 -> DGX /home/arm/junyang/k0_dist/returns/c1/
- C2 -> DGX /home/arm/junyang/k0_dist/returns/c2/
Sync landed shards about every 15 minutes, together with a copy of the
machine's current progress.json. DGX owns union completeness (200 games)
and the production join.

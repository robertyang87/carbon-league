# leg-1 post-chain record — rebuild238 (train-241 → 238)

Filed by DGX line 2026-11-18 ~01:55 CST, per CHAIN_MANIFEST.json step 6
(all generation ids / md5s below verified by DGX-line direct read on L20, not by receipt alone).

## Outcome
- **Training started L20 2026-11-18 01:45**, PID 19645, from joined238.
  Recipe: parent weights 06a03d9cfeec239af3065a49fefec382 (L20 md5-verified) /
  bs 128 / lr 5e-4 / β=1.0 / seed 7 / max-updates **5,749** (=round(5.17×142,324/128)) /
  out `/root/carbon/katago/gen1_leg1_student.npz`.
  Liveness at filing: process in place, GPU 100%, log fresh (update 98/5,749 at 01:48,
  lr 0.0005, ~1.23 s/update → finish ~03:43 估计）.
  Training completion receipt: war room's (this record covers delivery only).

## Products (registered in artifact registry v4.3, 47 artifacts)
| product | generation | manifest / key facts | location |
|---|---|---|---|
| merged238 | gen-181a281d2311f41b | manifest_sha256 ab98fd7b139ca7a2253fc7a6cc715d8fb1d4c80af1d67b39b506f734a315516c; 142,324 roots / 238 games | L20 /root/carbon/leg1_chain/rebuild238/merged238/ |
| build238 | gen-4e2a6937158479d5 | 142,324 samples, val_games 22, cfg_fp f982640561fc4fe4, ladder_frame true | L20 /root/carbon/leg1_chain/rebuild238/build238/ |
| joined238 | gen-1bb5e738c653c7fd | manifest_sha256 91d510cf2fcd6d3443bf6042dc7b44ac92143605970936989bdd4ab8316fb131; 142,324 roots, candidates 2,232,423; join two-step + `--verify` PASS | L20 /root/carbon/leg1_chain/joined238/ (DGX holds NO joined238 copy) |
| prescan238 | — (report md5 93726dd317aa0e2716c73905d06529b7) | roots 142,324; inconsistent 0 / straddle 0 / tripping 0; PRESCAN238_CLEAN 01:35:45 | L20 /root/carbon/leg1_chain/prescan238.json |

## Chain actually run (deviations from CHAIN_MANIFEST.json, all owner/war-room ordered)
1. **Collect — re-seal, not whole-root tar.** Manifest step 0 (tar whole annot_train
   roots from seats) was impossible: spot-61 released 22:42, spot-62 offline ~23:19,
   spot-60 offline 23:32. Recovery = re-seal on spot-63 from the DGX-pulled shards
   (annotator's own deterministic seal; weights-path string binds the gen id).
   Proof: three independent gen-id matches (pkg1 gen-455748998d7ec317,
   pkg3 gen-d5f9b2cc1a00696b, pkg4 gen-9b18eacafa23c7a5); pkg2 gen-a14a0c5de30bf61e
   by the same proven procedure. DGX-path trial re-seals are VOID (deleted).
2. **Merge + build on DGX** (merged241 gen-511694d8afa67b96, 144,118 roots;
   build241 gen-642eb8709a8e2519, RC=0), then **transfer to spot-63** (301 files,
   md5 301/301) for join/prescan per amendment eb64d2e6 / CHAIN_AMENDMENT_JOIN63.
3. **Join on 63 failed (war-room command error):** `k0_join.py --annot … --build …
   --out joined241 --verify` in one command — `--verify` is a standalone re-check
   mode, so the join never executed (RC=1). Correct form is two steps
   (join, then `--out O --verify`); SOP §5 fixed to v1.2.
4. **Seat change (owner ruling 2026-11-18 00:33, SOP v1.1):** build+join on L20,
   prescan with the joined product on L20. War room took join+prescan on 63 first
   (00:14), then the owner re-seated the chain to L20. DGX delivered merged241 +
   build241 + tools (7 files) to L20 /root/carbon/leg1_chain/ (inputs 301/301 OK,
   tools 7/7 OK, 00:56). **joined241 was built on L20: gen-58d88b621e239c92.**
5. **Prescan gate stopped the chain (prescan_leg1.json, md5 6eb5a92e):**
   joined241, 144,118 roots — 3 tripping roots (root 25161/game 42/step 22,
   root 105896/game 177/step 25, root 128638/game 215/step 34; cash 59.99–60.01,
   recruits_at_10 true). Per owner prescan hard gate (2026-11-17 19:10) +
   one-shot exclusion (2026-11-17 18:38), the 3 games were excluded whole:
   264e444df51624a0ec8747b1240e92e4-1, ac669558cc94d933c8ec7f715b3633e2-1,
   d9616829957abd5a4cd42b06e5f6893a-1 (598 roots each).
6. **Rebuild238 on L20 (war room, driver with step gates):** DGX supplied
   (supply-only) the four re-sealed annot_train roots (315/315 md5 OK),
   k0_annotate.py 7f567a2e, order_train241.json affa20a3, three work scripts,
   and the remaining six pinned code files (code/ 7/7 OK).
   merge_excl → build_src238 (238 entries, 0 short games) → build238 →
   join (two-step) → prescan238 clean. Free byte audit AUDIT_PASS
   (142,324 rows, 0 mismatches, tail segments byte-equal excl. gid).
   MERGE_RECORD.json in merged238/ lists the 3 excluded games with source
   generations (exclusion_ruling 6c98cc58).
7. **Delivery deviation from manifest step 6:** joined238 was NOT copied to
   `/root/carbon/katago/top4_leg1_joined/`; it stays at
   `/root/carbon/leg1_chain/joined238/` and training reads it there directly
   (owner 00:33 seat ruling: product consumed in place on L20).

## Caliber
- Annotation: train-241 only, full window [0,299), config 3cc2e67e…,
  pinned pair k0_annotate 7f567a2e + teacher c6ade5ce / weights 06a03d9c.
- Order file: order_train241.json, md5 affa20a3, 241 games.
- Sources: pkg1 spot-63 gen-455748998d7ec317 (36,478) / pkg2 spot-61
  gen-a14a0c5de30bf61e (35,880) / pkg3 spot-60 gen-d5f9b2cc1a00696b (35,880) /
  pkg4 spot-62 gen-9b18eacafa23c7a5 (35,880). Total annotated 144,118 roots.
- Annot registry: all 241 leg1_241 records flipped running→done with this
  filing (3 excluded games annotated + sealed, excluded from merged238 only).

## Receipts
- DGX→L20 delivery (merged241/build241/tools): war room 00:57 (submission 9bd8e5e0).
- Rebuild238 supply (315 files + 3 work scripts + 6 code files): war room
  01:11 / 01:13 / 01:19 (submissions 2b007db5, 1bf5b774, e8662628).
- This filing: war room + Robert, 2026-11-18 ~01:55.

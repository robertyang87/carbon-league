# Fast-line 138 rebuild — evidence pack (Strategy verification)

Filed by the war room 2026-11-17 ~19:12 CST, for Strategy's registry verification of the
138-game rebuild (rulings 757d17db + 77623940, co-cited with 6c98cc58 / 83a5db39).
Companion archive: `reports/fastline138_evidence.tgz` (md5 `4d029ed27f9b5db09cb2514ea7fb5412`,
48K) — verbatim originals pulled from DGX / spot-58 / L20, no re-derived hashes.

## Chain (all direct-read verified)

| Step | Generation | Identity |
|---|---|---|
| Merge (DGX, tool `merge_annot_gens_excl.py` 57cc1310) | gen-adebe932da1eb6f5 | 138 games / 82,524 roots; manifest_sha256 `01f40a8bf14c403d589571a23d8f84ee00ac3b19ce784f9e1cad9506b22f98ea` |
| Build (spot-58, `cnet2_build.py` 5c2dd58e) | gen-75ed865a73fd02b1 | 138 games / 82,524 samples; cfg_fp `f982640561fc4fe4`; 138/138 shard ckeys ≡ build146 |
| Join (spot-58, `k0_join_fallback.py` fac55acb) | gen-56e91737a08c5fb6 | 82,524 roots / 1,293,795 candidates; manifest_sha256 `b91e6f2f04de34c978cb1968f32d7019b39cb7d7e163c5aaffd36726b1b81016`; --verify PASS |

Excluded (9 games, whole-game, zero entry in joined meta): voided
`8759f8217d439cc5127f10aa2139935a-1` (208 roots) + the 8 threshold-crossing games
`1c659232…`, `1f170a1f…`, `22c26fb4…`, `31568ffd…`, `4c086ee0…` (trigger, game 55),
`699fc9e2…`, `b4e45fac…`, `c0308ebb…` (598 roots each). Arithmetic: 87,308 − 208 − 8×598 = 82,524.

## Independent checks in this pack

- **Delivery 21/21**: `joined138_md5_58.txt` and `joined138_md5_l20.txt` are byte-identical
  (file md5 `12dc5bf156a7ac72e36768490ecfbe2a`, 21 files each) — spot-58 source vs
  L20 `/root/carbon/data/master5s_fastline_joined138/`.
- **L20 re-verification** (`l20_verify_joined138.txt`, re-run 19:10, rc=0): meta total 82,524,
  games 138, unique ids 138, all 598 roots/game; `excluded present: []`; teacher config
  `3cc2e67e…a1ff3` (full window), weights `06a03d9c…`; sources = annotation gen-adebe932 /
  build gen-75ed865a (manifest `7f2e8d46…`).
- **Join log** (`s58/join138.log`): `k0_joined` gen-56e91737, total 82,524, reused false,
  candidates 1,293,795, sentinel translation 319,068 entries.
- **Retrain**: L20 from scratch 19:01:14 CST, PID 554380, 3,333 steps (bs 128 / lr 5e-4 /
  β 1.0 / seed 7), ~1.21 s/step, ETA ~20:08 (estimate). Stop-loss: death on a root outside
  the 8 excluded games = stop + report.

## Pack contents (md5, as filed)

| File | md5 |
|---|---|
| REBUILD_RECORD.json | 2fc83221577fc6b645ea1459b6540f12 |
| MERGE_RECORD.json | a898040795b1f99903ddc825b64748c0 |
| merged138_current.json | b6f12af23e5c99537fc9c7129d31b9a5 |
| merged138_generations/gen-adebe932da1eb6f5/manifest.json | 00e734e75461c764d6a9aaa3eedb50e2 |
| merged138_generations/gen-adebe932da1eb6f5/meta.json | 787a11d95f4951e8b428274318a59037 |
| s58/build138/current.json | d0b77fcdd9ce75efcdec05ba2b5aaafb |
| s58/build138/meta.json | da60b6e473ba98218da77a07d790cc3f |
| s58/build138/generations/gen-75ed865a73fd02b1/manifest.json | e017c26b072a41320bb56a66962feca2 |
| s58/build138/generations/gen-75ed865a73fd02b1/meta.json | 08ab02ac029ca501231c17d0e8833e73 |
| s58/build138.log | 8537c34b683ea9e27adbc1d74d00416d |
| s58/joined138/current.json | 1bbebb0506b44bd610c5544f2a95ebdb |
| s58/joined138/generations/gen-56e91737a08c5fb6/manifest.json | 3720c6d072c5b3bda39cfbf8e125a23e |
| s58/joined138/generations/gen-56e91737a08c5fb6/meta.json | 1e1276e0ca172f66d8417ea3da51f759 |
| s58/join138.log | 8b3d99ae17b6dce8a90be086e8bcb5fa |
| joined138_md5_58.txt | 12dc5bf156a7ac72e36768490ecfbe2a |
| joined138_md5_l20.txt | 12dc5bf156a7ac72e36768490ecfbe2a |
| l20_verify_joined138.txt | 170b7cc33cbc3498f4491205b677cab3 |

Provenance: `REBUILD_RECORD.json`, `MERGE_RECORD.json`, `merged138_*` from DGX
`/home/arm/junyang/k0_dist/master5s_fastline/`; `s58/*` from spot-58
`/root/carbon/fastline_rebuild/`; `l20_verify_joined138.txt` and the L20 md5 list from
L20 direct read. Diagnosis files referenced in REBUILD_RECORD (md5 9dbaf286 / 6b040870)
are warehoused on DGX at the same fastline path.

Superseded: the 145-game rebuild (gen-4f80c4dc2f956ef6 / gen-50aa24668b6d0ea1) is VOID —
never join. The 146-game joined gen-82a26d661360b330 remains on L20 as evidence only.

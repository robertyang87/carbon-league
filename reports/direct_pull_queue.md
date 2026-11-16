# 语料直拉队 (DGX line) — corpus direct-pull queue

Status: **QUEUE EMPTY — all items pulled and closed by the war room**
(see Current status below; the BLOCKED-era text is kept as history).

## Current status (2026-11-16 12:25 CST, DGX line)

- Queue EMPTY; every item below is DONE + FULLY CLOSED by the war room.
  Awaiting the next war-room order.
- 2026-11-16 09:11 echo4 full re-pull (owner order): list refreshed to
  **243** games (matches.json md5 a1905f0b), corpus 243/243 OK BAD 0,
  manifest 243 lines md5 79678f11. Warehoused to main (commit 6e67aa9f);
  the old 3-game top-up debt is written off.
- 2026-11-16 09:13 Echo5 placement corpus pull: **29** games
  (matches.json md5 dfafd7b9), corpus 29/29 OK BAD 0, manifest 29 lines
  md5 b5bcecf8. In main (list, manifest and all 29 replay files verified
  blob-identical against DGX on 2026-11-16 by tree compare).
- Echo4 in main likewise: list, manifest and all 243 replay files
  verified blob-identical against DGX on 2026-11-16.
- Outstanding top-up debts (post-pull live drift; ride the next patrol
  cycle, war room/patrol handles): k4v6 2, fablebc2 4, fablebc3 4,
  m3s3z 5, M9XJ-630 2, hakimi （哈基米基准） 1, Master4s 1,
  k4v3 1 (af1b8009625ec55b0d7130612b0e5e07),
  krypton-m128 1 (cec8be4ca575dc7fb24f0a3bf172a890),
  fablesym8v4 3 (IDs as recorded in its closure entry).
- 2026-11-16 warehousing (owner order): this file is committed to the
  repo as reports/direct_pull_queue.md; calibration-chain cells 1–9
  results under data/k0_calib/; opponent-corpus replay files beyond
  list+manifest are NOT in the repo for most labels (2,377 files,
  ~578 MB on DGX only) — surfaced to the war room 2026-11-16.

Execute only in genuine idle windows, or on war-room re-prioritization.
Corpus only, no training, no triggers. Never touch it570 frozen files.
Counts re-read live from the 117 endpoint at execution time.
Route: 117-endpoint DGX direct pull (`code/ladder.py matches <hex>` then `replays`,
LADDER_BASE=https://117.72.210.134, ≤30/min, skips existing); ladder-replay-pull
route is blocked — do not use it. Per-file verify (gzip-load + "steps");
manifest = one line per corpus file `{md5} data/<label>_corpus/<fn>`;
manifest md5s to the war room for warehousing. Hex IDs resolved from the
117 board at execution time. Label = directory label below.

## Formation (war room, 2026-11-15 ~21:38)

1. **m3s2/m3s combined backfill + pull0045 remainder — AT TOP.**
   pull0045 scope: CloudSearch-v1 gap 107, ProSearch-v1 gap 40,
   Search256-v1 gap 58, Search256-v2 gap 20; plus fable6; **m3s3z gap 25**
   (its pull also feeds the M3s3z bridge-check 4a receipts);
   krypton-search256 gap 24, krypton-prosearch gap 23, k4v2 gap 21.
2. 小肥鱼v5 gap 100 (triggered #4, 2827±31, 100 games 77W; list warehoused +
   read-back verified).
   - Hex 2856c37287900ee3cba8eca53b827da9 (war room 05:12, board
     direct read 05:10).
   - **DONE 05:22**: 小肥鱼v5 — list refreshed to **155**
     (matches.json md5 5d33db0a809fd8a18488f00eeaa5d6f3), corpus
     0→155 (new 155; first pass cut by the 280 s timeout at 103,
     second pass +52), OK 155 BAD 0, manifest md5
     b47d7d2c4687cb81b5b994958078c157
     (data/xiaofeiyu5_corpus.manifest, 155 lines). Reported 05:22
     for 合账. **FULLY CLOSED 05:25**: war room warehoused the list
     (155, md5 5d33db0a, commit 050f104a, read-back verified) and
     the manifest (155 lines, commit bbc66de3, read-back
     byte-identical); all 155 corpus files match list game IDs
     1:1, zero drift. Label xiaofeiyu5 confirmed — no rename
     needed. Hex 2856c37287900ee3cba8eca53b827da9.
3. M9XJ-630 gap 100.
   - Hex b3fa633052e17329f34054bab0090128 (war room 05:12, board
     direct read 05:10).
   - **DONE 05:29**: M9XJ-630 — list refreshed to **145**
     (matches.json md5 269bc906d53e540c6d598b94856472ea), corpus
     0→145 (new 145; first pass cut by the 280 s timeout at 103,
     second pass +42), OK 145 BAD 0, manifest md5
     fd8ede16dbac989c9be6632f521be2fd
     (data/m9xj630_corpus.manifest, 145 lines). Reported 05:29 for
     合账. **FULLY CLOSED 05:34**: war room warehoused the list
     (now 147, md5 c4270d13, byte-identical to the 117 endpoint,
     no refresh needed) and the manifest (145 lines, commit
     13e00878, read-back byte-identical); all 145 corpus files
     match list game IDs 1:1; the 2-game drift is post-pull new
     games — top-up rides the next patrol cycle. Hex
     b3fa633052e17329f34054bab0090128.
4. 哈基米基准 gap 104 (triggered 2026-11-15 19:26 patrol, 104 games on 117).
   - Hex 66ab03a1462c6fc6ce366a83dca18827 (war room 05:12, board
     direct read 05:10).
   - **DONE 05:40**: 哈基米基准 — list refreshed to **179**
     (matches.json md5 2b6f9ce84ffe7c4e4814fe8d21d01392), corpus
     0→179 (new 179; first pass cut by the 280 s timeout at 107,
     second pass +72), OK 179 BAD 0, manifest md5
     9bff408b00eb3275a3a496a4317dd09a
     (data/hakimi_corpus.manifest, 179 lines). Reported 05:40 for
     合账. **FULLY CLOSED 05:42**: war room warehoused the list
     (180, new file, md5 d4272044, commit a4841f55, read-back
     verified) and the manifest (179 lines, commit 60307f36,
     read-back byte-identical); all 179 corpus files match list
     game IDs 1:1; the 1-game drift is post-pull new games —
     top-up rides the next patrol cycle. Label hakimi confirmed —
     no conflict with hakimi3pro. Hex
     66ab03a1462c6fc6ce366a83dca18827.
5. Master4s gap 101 (triggered 2026-11-15 21:38 sync, list warehoused
   commit b321f3960d33, read-back verified).
   - Hex 6389c346461bce295badcf52a1dd74c6 (war room 05:12, board
     direct read 05:10).
   - **DONE 05:47**: Master4s — list refreshed to **142**
     (matches.json md5 f8eac204eba85599a1f27a89abd4a6a6), corpus
     0→142 (new 142; first pass cut by the 280 s timeout at 102,
     second pass +33 with 1 file 429'd — paused 60 s, single-file
     retry landed; third pass +7), OK 142 BAD 0, manifest md5
     39b008e5bf5d1c3ac0fd970e5a953a3b
     (data/master4s_corpus.manifest, 142 lines). Reported 05:47
     for 合账. **FULLY CLOSED 06:04**: war room — list refreshed
     to 143 in warehouse (commit 96498698, md5 9b5e9a9c,
     read-back consistent); manifest 142 lines warehoused (commit
     7809d5a7, md5 39b008e5, read-back consistent); all 142 corpus
     files are within the 143-game list; 117 endpoint now 143
     games; the 1-game difference is live drift, logged as a
     top-up debt for the next patrol cycle (master4s 1 game).
     **Items 2–5 all pulled + warehoused.**
   - **⑥ tail dispatched 06:04** (war room, board direct read
     05:50, board 30/60): pull in registration order, four-cell
     report each: 1. m9xj600 (hex TO FIND — delisted, 117 frozen
     107 games, last game 11-15 17:17; my record hex
     b1cde1fda59f9bab81e56f40de92302d from the 2026-11-15 08:18
     dispatch — verifying; gap ~54) → 2. hakimi3pro
     aff6e57c5eba8857d5546783c7d39a46 → 3. echo4
     65da7084ba2adf5c1ae555ad3f3931bc → 4. k4v3
     9d596068f39e0b8cca317c3bb27f8ae5 → 5. fable6
     5b453cb4f7c49439673deda697d46a9b → 6. lite-v8
     3a4d267d9fcea50794f48d38d3e1360d → 7. krypton-m128
     96db8cfd61a73a534807cc404d046e31 → 8. krypton-csplant
     adaf275d51a177b7a9cf8224ab66c467 → 9. krypton
     0f2c23daf04aaf650576575d2482b940.
   - **DONE 06:10**: m9xj600 — list refreshed to **107** (frozen;
     matches.json md5 5aa3ca6002b640c2ed14bd4f16a22f18), corpus
     52→107 (new 55; first pass +51 with 1 file 429'd — paused
     60 s, single-file retry landed; second pass +4), OK 107
     BAD 0, manifest md5 0f024b67b7a893d23a3da31dfd7747bc
     (data/m9xj600_corpus.manifest, 107 lines). Reported 06:10
     for 合账. **FULLY CLOSED 06:15**: war room — warehouse list
     107 games, md5 5aa3ca60 byte-identical to my report, no
     refresh needed, zero drift (frozen 107); corpus manifest 107
     lines warehoused (commit 650a221f, read-back md5 consistent);
     all 107 corpus files within the list, gap 0. Hex
     b1cde1fda59f9bab81e56f40de92302d confirmed four ways (live
     endpoint + refresh_1056.py + refresh_1256.py + war-room
     2026-11-15 dispatch records in ~/memory/2026-11-15.md).
   - **IN PROGRESS 06:11**: hakimi3pro — list refreshed to **306**
     (live count governs; queue's gap 83 was stale), corpus 181,
     gap 125 (pull running). Hex aff6e57c5eba8857d5546783c7d39a46.
   - **DONE 06:18**: hakimi3pro — list refreshed to **306**
     (matches.json md5 dd347e67cad0d57190f81ead6fd4d87f), corpus
     181→306 (new 125; first pass cut by the 280 s timeout at
     282, second pass +24), OK 306 BAD 0, manifest md5
     971fba4775bc6c65da38f384c90f0588
     (data/hakimi3pro_corpus.manifest, 306 lines). Reported 06:18
     for 合账. **FULLY CLOSED 06:21**: war room — 117 endpoint
     306 games; warehouse list refreshed from 305 to 306 (commit
     aacc3a43, md5 dd347e67 byte-identical to my report); corpus
     manifest 306 lines warehoused (commit 594fd5d8, md5 971fba47,
     read-back consistent); all 306 corpus files within the
     306-game list, gap 0. Hex aff6e57c5eba8857d5546783c7d39a46.
   - **IN PROGRESS 06:18**: echo4 — list refreshed to **229**,
     corpus 0 in this clone (the repo's 51 warehoused files predate
     the local HEAD f82191eb — war room reconciles on warehousing),
     gap 229 (pull running). Hex 65da7084ba2adf5c1ae555ad3f3931bc.
   - **DONE 06:30**: echo4 — list refreshed to **229**
     (matches.json md5 eb6db66aabcfe60d3c5787a0b9492030), corpus
     0→229 (new 229; first pass cut by the 280 s timeout at 110,
     second pass cut at 218, third pass +10 with 1 file 429'd —
     paused 60 s, single-file retry landed; fourth pass +1),
     OK 229 BAD 0, manifest md5 fe2a886a306cd642f6e1741a88fa6001
     (data/echo4_corpus.manifest, 229 lines). Reported 06:30 for
     合账. **FULLY CLOSED 06:39**: war room — 117 endpoint 232
     games (md5 dfab4dc0); warehouse list already 232 games,
     byte-identical; corpus manifest 229 lines warehoused (commit
     3fb7072a, md5 fe2a886a, read-back consistent); all 229 corpus
     files within the 232-game list. Reconciliation: ① the
     warehouse's original 51 files individually md5-verified,
     byte-identical to the new list, no conflict; ② gap of 3
     games (044cecf401771cba78c84a5bf476285a,
     1ea0f9e035849f17311a95c636ac5c45,
     681309d330f45fdfa5ae7c91a0683825) — new games played after my
     pull, live drift, top-up rides the next patrol cycle.
     Hex 65da7084ba2adf5c1ae555ad3f3931bc.
   - **IN PROGRESS 06:30**: k4v3 — list refreshed to **331**
     (live count governs; queue's gap 52 was stale), corpus 251,
     gap 80 (pull running). Hex 9d596068f39e0b8cca317c3bb27f8ae5.
   - **DONE 06:37**: k4v3 — list refreshed to **331**
     (matches.json md5 5e69f96d67b9ff265a92a4427b289eea), corpus
     251→331 (new 80, single pass, no timeout cut), OK 331 BAD 0,
     manifest md5 934e87a111498c99597c1b6551bdd4ae
     (data/k4v3_corpus.manifest, 331 lines). Reported 06:37 for
     合账. **FULLY CLOSED 06:40**: war room — 117 endpoint 332
     games (md5 9e21758f); warehouse list refreshed from 331 to
     332 (commit a01c7081, md5 9e21758f, byte-identical to my
     reported list); corpus manifest 331 lines warehoused (commit
     0410cefe, md5 934e87a1, read-back consistent); all 331 corpus
     files within the 332-game list, gap of 1 game
     (af1b8009625ec55b0d7130612b0e5e07) — a new game played after
     my pull, live drift, top-up rides the next patrol cycle.
     Hex 9d596068f39e0b8cca317c3bb27f8ae5.
   - **IN PROGRESS 06:37**: fable6 — list refreshed to **333**
     (live count governs; queue's gap 52 was stale), corpus 256,
     gap 77 (pull running). Hex 5b453cb4f7c49439673deda697d46a9b.
   - **DONE 06:42**: fable6 — list refreshed to **333**
     (matches.json md5 abd902d3843d493af3360ca858b3bf5f), corpus
     256→333 (new 77, single pass, no timeout cut), OK 333 BAD 0,
     manifest md5 ff645a1a24f85fc2cb54d4f8b215afc9
     (data/fable6_corpus.manifest, 333 lines). Reported 06:42 for
     合账. **FULLY CLOSED 06:45**: war room — 117 endpoint 333
     games; warehouse list already 333 games, md5 abd902d3
     byte-identical to my report, no refresh needed, zero drift;
     corpus manifest 333 lines warehoused (commit 2e8bcf05, md5
     ff645a1a, read-back consistent); all 333 corpus files within
     the list, gap 0. Hex 5b453cb4f7c49439673deda697d46a9b.
   - **IN PROGRESS 06:42**: lite-v8 — list refreshed to **339**
     (live count governs; queue's gap 47 was stale), corpus 268,
     gap 71 (pull running). Hex 3a4d267d9fcea50794f48d38d3e1360d.
   - **DONE 06:47**: lite-v8 — list refreshed to **339**
     (matches.json md5 c7e709f2e7e3b6c3d1eb5baff868e316), corpus
     268→339 (new 71, single pass, no timeout cut), OK 339 BAD 0,
     manifest md5 919a390378a97d71abd5d066f6ad7463
     (data/lite-v8_corpus.manifest, 339 lines). Reported 06:47
     for 合账. **FULLY CLOSED 06:50**: war room — 117 endpoint
     339 games; warehouse list refreshed from 338 to 339 (commit
     a593693c, md5 c7e709f2 byte-identical to my report); corpus
     manifest 339 lines warehoused (commit 6d79053a, md5 919a3903,
     read-back consistent); all 339 corpus files within the
     339-game list, gap 0. Hex 3a4d267d9fcea50794f48d38d3e1360d.
   - **IN PROGRESS 06:47**: krypton-m128 — list refreshed to
     **237** (live count governs), corpus 0 (full pull),
     gap 237 (pull running). Hex 96db8cfd61a73a534807cc404d046e31.
   - **DONE 06:58**: krypton-m128 — list refreshed to **237**
     (matches.json md5 7512e766c96d2518faaaae5707094a99), corpus
     0→237 (new 237; first pass cut by the 280 s timeout at 100,
     second pass cut at 205, third pass +32), OK 237 BAD 0,
     manifest md5 b2205108402cf3310063a9bbddc1ef31
     (data/krypton-m128_corpus.manifest, 237 lines). Reported
     06:58 for 合账. **FULLY CLOSED 07:11**: war room — 117
     endpoint 238 games (md5 c69411b0); warehouse list refreshed
     from 236 to 238 (commit 8126fbc7, md5 c69411b0,
     byte-identical to my reported list); corpus manifest 237
     lines warehoused (commit 4579adc0, md5 b2205108, read-back
     consistent); all 237 corpus files within the 238-game list,
     gap of 1 game (cec8be4ca575dc7fb24f0a3bf172a890) — a new game
     played after my pull, live drift, top-up rides the next
     patrol cycle. Hex 96db8cfd61a73a534807cc404d046e31.
   - **IN PROGRESS 06:58**: krypton-csplant — list refreshed to
     **265** (live count governs; queue's gap 60 was stale),
     corpus 192, gap 73 (pull running).
     Hex adaf275d51a177b7a9cf8224ab66c467.
   - **DONE 07:04**: krypton-csplant — list refreshed to **265**
     (matches.json md5 ac0e101495ac9730682dceb1434e6e3a), corpus
     192→265 (new 73, single pass, no timeout cut), OK 265 BAD 0,
     manifest md5 7dd3e8c427b413c34f64ea5975fde713
     (data/krypton-csplant_corpus.manifest, 265 lines). Reported
     07:04 for 合账. **FULLY CLOSED 07:12**: war room — 117
     endpoint 265 games; warehouse list already 265 games, md5
     ac0e1014 byte-identical to my report, no refresh needed,
     zero drift; corpus manifest 265 lines warehoused (commit
     a7c9faae, md5 7dd3e8c4, read-back consistent); all 265 corpus
     files within the list, gap 0.
     Hex adaf275d51a177b7a9cf8224ab66c467.
   - **DONE 07:06**: krypton (⑥ tail last item) — list refreshed
     to **244** (matches.json md5 0dd548dfaf8b360594e005122c4277dc),
     corpus 206→244 (new 38, single pass, no timeout cut),
     OK 244 BAD 0, manifest md5 56d2a0f3228958d812e455967d1f8fc8
     (data/krypton_corpus.manifest, 244 lines). Reported 07:06
     for 合账. **FULLY CLOSED 07:12**: war room — 117 endpoint
     244 games; warehouse list already 244 games, md5 0dd548df
     byte-identical to my report, no refresh needed, zero drift;
     corpus manifest 244 lines warehoused (commit 570c72d4, md5
     56d2a0f3, read-back consistent); all 244 corpus files within
     the list, gap 0. **⑥ tail all nine items closed**: m9xj600
     107, hakimi3pro 306, echo4 229, k4v3 331, fable6 333,
     lite-v8 339, krypton-m128 237, krypton-csplant 265,
     krypton 244. Hex 0f2c23daf04aaf650576575d2482b940.
   - **DONE 07:10**: fablesym8v4 FULL RE-PULL (war-room 06:14
     order, trigger 102 games) — list refreshed to **127** (the
     list had grown from 102 to 127 by pull time; live count
     governs; matches.json md5 6dcbe1e467766ce50352406946781f62),
     corpus 76→127 (new 51, single pass, no timeout cut),
     OK 127 BAD 0, manifest md5 19e8555c9c3d84ff05564ffb143ee9d4
     (data/fablesym8v4_corpus.manifest, 127 lines). Reported
     07:10 for 合账. **FULLY CLOSED 07:13**: war room — 117
     endpoint 130 games (md5 c12b121e; 127 at my pull time, grew
     by 3 after); warehouse list refreshed to 130 (commit
     715f25ce); corpus manifest 127 lines warehoused, old 76-line
     manifest replaced (commit 1f054c16, md5 19e8555c, read-back
     consistent); all 127 corpus files within the 130-game list,
     gap of 3 games (0ea2f47e…, 271779eb…, c6f0778d…) — new games
     played after my pull, live drift, top-up rides the next
     patrol cycle; the old 2-game top-up debt is void with the
     new manifest. Hex 278bb13924c5e9993dc339f86b3de6f1.
   - **QUEUE EMPTY 07:10**: ⑥ tail (echo4, k4v3, fable6, lite-v8,
     krypton-m128, krypton-csplant, krypton) + hakimi3pro +
     fablesym8v4 re-pull all done and reported. Awaiting the next
     order.
   - **QUEUED 06:14** (war room, new trigger — AFTER the ⑥ tail):
     fablesym8v4 (ClaudeFable5.2-Sym8-v4) — **TRIGGERED at 100
     games** (117 endpoint 102 games, #6; list warehoused 102
     games, commit 662a3bfac8b9). DGX corpus already 76 files
     (from the 05:11 pull); pull the FULL set against the current
     102-game list and regenerate the manifest (replays skips
     existing — top-up is 26 files). The old 76-line manifest
     (commit b7b196d8) and the 2-game top-up debt are VOID with
     the new manifest — no separate top-up. Hex
     278bb13924c5e9993dc339f86b3de6f1, label fablesym8v4. Same
     caliber: ≤30/min, per-file gzip verify, four-cell report.
6. Tail, in order: m9xj600 47 ＞ m3s3z 79 ＞ hakimi3pro 83 ＞ echo4 82 ＞
   k4v3 52 ＞ fable6 52 ＞ lite-v8 47 ＞ krypton-m128 211 ＞
   krypton-cloudsearch 142 ＞ krypton-csplant 54 ＞ krypton 27.
   - Superseded by the ⑥ tail dispatch above (m9xj600 DONE 107,
     hakimi3pro in progress at 306).

## Added 2026-11-15 23:59 (war room order) — NEW TAIL

7. **k4v6** (stephen-K4TacSearch-v6) — BC 收口补丁 4b 口径：只做清单＋语料保险，
   不训练、不触发；100 局触发仍以侦查兵后续报为准。
   - 23:11 新面孔巡检：#7、57 局 35 胜，语料 0 件（war-room message 23:59  quoted 57）。
   - Fresher verified numbers: match list data/k4v6_matches.json 57→65
     refreshed + warehoused 2026-11-15 23:33, commit 292a4b82 (read-back
     verified); live games 67 @23:58 war-room sync. Corpus
     data/k4v6_corpus does NOT exist = 0 files. Gap = live game count at
     execution (65 @23:33, 67 @23:58 — re-read, do not freeze).
   - Note (do not re-trip): the "k4v6 0->16 / 16->28" lines in old worker
     exec-logs are MATCH-LIST growth (data/k4v6_matches.json), NOT corpus
     files. Corpus has always been 0; ladder-replay-pull last 5 runs all
     failure (latest 2026-11-13 00:10).
   - At execution: refresh list from the 117 board, pull the replay gap,
     verify per file, manifest md5 to the war room.
   - **TRIGGERED 2026-11-16 ~01:26** (war-room notice, this chat 01:40):
     #7, 2653±28, 117 endpoint **102 games, 65 wins (63.7%)**,
     avg +358.1, median +413.5; first-move 33/52, second-move 32/50,
     both hands verified; vs Echo2 direct 2/2, avg +1825.4. Match list
     data/k4v6_matches.json refreshed to 102, warehoused + read-back
     verified, commit c7179faba221. Corpus gap now **102** (live at
     execution governs). Position unchanged: tail; pull in a genuine
     idle window only, corpus insurance only, no training, no trigger
     chain (BC closure patch 4b). War room acked 2026-11-16 ~01:41;
     report back on completion.
   - **DONE 04:47**: k4v6 — list refreshed to 154 (matches.json
     md5 e7c3d10ec588d577b4d7dd2a66f6aae2), corpus 0→154 (new 154;
     first pass cut by the 280 s timeout at 103, second pass +51),
     OK 154 BAD 0, manifest md5 af34913bc8b1c9bd3ecb4789ba052920
     (data/k4v6_corpus.manifest, 154 lines). Reported 04:47 for
     合账. **FULLY CLOSED 04:50**: war room warehoused the list
     (refreshed to 156, md5 101b195c, commit 3d316f09) and the
     manifest (154 lines, commit 39dca5c5, read-back byte-identical);
     the 2-game list-vs-corpus drift is post-pull new games — top-up
     rides the next patrol cycle, no action now. Hex:
     5422677057dbcca6cdf11cb60b65ba71 (war room, board direct read
     04:39).
   - Next: fablebc2 (111, triggered) — hex requested from war room
     04:48 (batch ask incl. fablebc3 + fablesym8v4).

## Added 2026-11-16 02:54 (war-room order) — NEW TAIL, after k4v6

8. **fablebc2** — gap **137** (war room 04:50: board 127 at 04:50;
   live count at execution governs).
   - **DONE 04:57**: fablebc2 — list refreshed to **137**
     (matches.json md5 aa2cac8c318a4efc4fc4c54dbd86fc4f), corpus
     0→137 (new 137; first pass cut by the 280 s timeout at 109,
     second pass +28), OK 137 BAD 0, manifest md5
     d73bf785916bd437835df05ecdbe49a2
     (data/fablebc2_corpus.manifest, 137 lines). Reported 04:57 for
     合账. **FULLY CLOSED 05:01**: war room warehoused the list
     (refreshed to 141, md5 248398aa, commit 4a276a86) and the
     manifest (137 lines, commit 0ea4810f, read-back byte-identical);
     all 137 corpus files match list game IDs 1:1; the 4-game drift
     is post-pull new games — top-up rides the next patrol cycle.
     Hex: 7fe8858658a7155d72f66a49607c36b0 (war room, board
     direct read 04:50).
   - **TRIGGER CONFIRMED** (war room 04:31): scout 04:09 patrol:
     111 games, #7. Disposition unchanged: sync + corpus insurance
     only, NO training.
   - Next: fablebc3 (116, in progress) — hex
     7d4a08d68ec0020abc7568d03e8e8384 (war room, board direct read
     04:50).
9. **fablebc3** — gap **116** (war room 04:50: board 100 at 04:50;
   live count at execution governs).
   - **DONE 05:05**: fablebc3 — list refreshed to **116**
     (matches.json md5 d89c5fda8b77bd76c6a305a95a5b5070), corpus
     0→116 (new 116; first pass cut by the 280 s timeout at 112,
     second pass +3, 1 file 429'd — paused 60 s, single-file retry
     landed), OK 116 BAD 0, manifest md5
     fa3c3545152aa49ddeb78475647ee177
     (data/fablebc3_corpus.manifest, 116 lines). Reported 05:05 for
     合账. **FULLY CLOSED 05:09**: war room warehoused the list
     (refreshed to 120, md5 52d8dd35, commit ffd2c008) and the
     manifest (116 lines, commit 3e235d09, read-back byte-identical);
     all 116 corpus files match list game IDs 1:1; the 4-game drift
     is post-pull new games — top-up rides the next patrol cycle.
     The 429 handling (60 s pause + single-file retry) is recorded.
     Hex: 7d4a08d68ec0020abc7568d03e8e8384 (war room, board
     direct read 04:50).
   - **TRIGGERED** (war room 04:38): 100 games. Same caliber:
     sync + corpus insurance only, no training.
   - Next: fablesym8v4 (76, in progress) — hex
     278bb13924c5e9993dc339f86b3de6f1 (war room, board direct read
     04:50).
10. **fablesym8v4** (ClaudeFable5.2-Sym8-v4) — list
    data/fablesym8v4_matches.json (scout 03:38 snapshot 42 games; war
    room read-back verified 52 games, sha 1b524812, crossed the
    50-game mark); corpus 0 files; gap **76** (war room 04:50:
    board 61; list refreshed at execution to 76 — live count
    governs). Same caliber as items 7–9: list + corpus insurance
    only, no training; pull in queue-head order, small batches,
    report each completed item to the war room (four-cell md5 table).
    At the 100-game trigger: sync + fact pack only, no BC training.
    Hex: 278bb13924c5e9993dc339f86b3de6f1 (war room, board direct
    read 04:50).
   - **DONE 05:11**: fablesym8v4 — list refreshed to **76**
     (matches.json md5 c247a67ccef247f875547fb230c9abdc), corpus
     0→76 (new 76, single pass, no timeout cut), OK 76 BAD 0,
     manifest md5 92edfc3c9e6dc93f12b5cce42c10d4bb
     (data/fablesym8v4_corpus.manifest, 76 lines). Reported 05:11
     for 合账. **FULLY CLOSED 05:12**: war room warehoused the list
     (refreshed to 78, md5 ae1d8fd8, commit b4b6c7cc) and the
     manifest (76 lines, commit b7b196d8, read-back byte-identical);
     all 76 corpus files match list game IDs 1:1; the 2-game drift
     is post-pull new games — top-up rides the next patrol cycle.
     Hex: 278bb13924c5e9993dc339f86b3de6f1 (war room, board direct
     read 04:50). **Tail four (k4v6, fablebc2, fablebc3, fablesym8v4)
     all pulled + closed.**
   - **Hexes received 05:12** (war room, board direct read 05:10),
     in order: 小肥鱼v5 (#3) 2856c37287900ee3cba8eca53b827da9;
     M9XJ-630 (#2) b3fa633052e17329f34054bab0090128;
     哈基米基准 (#9) 66ab03a1462c6fc6ce366a83dca18827;
     Master4s (#1) 6389c346461bce295badcf52a1dd74c6.
   - **IN PROGRESS 05:13**: 小肥鱼v5 — list refreshed to **155**
     (ladder name 小肥鱼v5 confirmed in the entries); corpus 0,
     gap 155 (pull running). Label xiaofeiyu5 is my provisional
     romanization (no feiyu label existed in the repo; ladder name
     is Chinese) — flagged to the war room in the report; rename
     on request.
   - Current tail order (war room 04:38, updated): k4v6 DONE →
     fablebc2 DONE → fablebc3 DONE → fablesym8v4 (76, in progress,
     last tail item).

Robert (this chat, 03:02:56):「可以现在一点点拉取过来么」— afraid
the route will block again later. The owner's order opens the window:
start pulling now, bit by bit, in queue-head order, throttled
(≤30/min, background), instead of waiting for a "genuine idle
window". The queue's BLOCKED status is superseded for execution by
this order; the caliber (list + corpus insurance only, no training)
is unchanged.
   - **DONE 03:04**: krypton-search256-v2 gap 20 — corpus 105→125,
     OK 125 BAD 0, manifest md5 b8e8a1e3bdbad23ee4eb3af861b9f1ec
     (data/krypton-search256-v2_corpus.manifest). **FULLY CLOSED
     03:29**: war room warehoused the manifest (main commit bd7108af,
     blob 3d2c200b, read-back md5 byte-identical); matches.json md5
     d9e99d0d2132b3f1cf951803c1868a5f also verified.
   - **DONE 03:09**: krypton-prosearch gap 58 (live list 216,
     was 184 at pull0043) — corpus 158→216, OK 216 BAD 0,
     manifest md5 5c98c1fb71255187b3bb20047ca9fcd0
     (data/krypton-prosearch_corpus.manifest). **FULLY CLOSED
     03:29**: war room warehoused the manifest (main commit 2030c5f9,
     blob e73f9310, read-back md5 byte-identical); matches.json md5
     f00eb50ad602f9865a066df6a7b40e11 also verified.
   - **DONE 03:22–03:29**: CloudSearch-v1 — list refreshed to 280
     (matches.json md5 a03a18a641445a1e3a5866fc0618efa6), corpus
     125→280 (new 155), OK 280 BAD 0, manifest md5
     4e24e60908ffdcb0229ec91c7d4ba79c
     (data/krypton-cloudsearch_corpus.manifest, 280 lines). Reported
     to war room 03:29 for 合账. Note: 117 board now shows 281
     games for CloudSearch-v1 (rank 29) — one game landed during/after
     the pull; re-refresh on the next cycle.
   - **TOP-UP 03:47**: CloudSearch-v1 +1 game (fbda4b33…, newface
     watch commit e3160aebc1 "280->281") — list 281, matches.json
     md5 76a8abf90781bf2e946032c697d3a0fa, corpus 281, OK 281 BAD 0,
     manifest md5 52ae0557622288c8e7a917cbe244e630 (281 lines).
     Re-reported 03:47. **FULLY CLOSED 04:12**: war room warehoused
     the manifest (main commit e61a35ee, read-back md5 52ae0557
     byte-identical).
   - **DONE 04:12**: Search256-v1 — list refreshed to 176
     (matches.json md5 e7c0472305b2129b66105e9ccac3d10b), corpus
     136→176 (new 40), OK 176 BAD 0, manifest md5
     5d853d204c5c226089662b4712b4b9d5
     (data/krypton-search256_corpus.manifest, 176 lines). Reported
     04:12 for 合账. **FULLY CLOSED 04:28**: war room warehoused
     the manifest (main commit a521bba0, read-back md5 5d853d20
     byte-identical). Hex from scout (04:12, war room):
     2c57e7cd01dee46bef98643398ab4ca3.
   - **DONE 04:20**: m3s3z — list refreshed to 249
     (matches.json md5 9ee6f72bef87b75354b8bbd9edb34079), corpus
     133→249 (new 116; first pass landed 242/249, 7 missing
     retried and all 7 landed), OK 249 BAD 0, manifest md5
     3c1052dcaf042e731eef2a894b3dbd48
     (data/m3s3z_corpus.manifest, 249 lines). Reported 04:20 for
     合账. **Reconciliation 04:28**: war room — my list 249 is
     AHEAD of the repo list 247 (md5 7a0fe8dd) by 2 new games
     (c22888df vs BC-v3, 34ae18d1 vs Sym8-v4, both m3s3z wins);
     repo has 0 missing. Manifest pulled + verified by war room,
     held pending newface-watch refreshing the repo list to 249,
     then warehoused — no re-pull needed. Hex from scout:
     f9fdbce38def1a78eb9c77f8711a1103. **FULLY CLOSED 05:17**:
     war room warehoused the manifest (249 lines, commit 9bf5460a,
     read-back md5 3c1052dc byte-identical); repo list now 254,
     all 249 corpus files match list game IDs 1:1; the 5-game drift
     is post-pull new games — top-up rides the next patrol cycle.
     The 05:15 trip's "m3s3z writes in the last 60 min" flag was my
     own 04:12–04:20 pull (confirmed to the war room 05:17).
   - **DONE 04:29**: k4v2 — list refreshed to **202** (matches.json
     md5 0037eb63879ffbfc82435770ee08db2a; committed list was 351
     — the ladder dropped ~149 older games from the endpoint;
     hex verified right, strategy stephen-K4TacSearch-v2), corpus
     326→402 (new 76 — all 202 live-list games now in corpus, plus
     200 legacy files from the old 351-list retained as insurance),
     OK 402 BAD 0, manifest md5 e667926ce324c787056d9d14f638b459
     (data/k4v2_corpus.manifest, 402 lines — full-corpus per the
     "one line per corpus file" rule; list 202 ≠ corpus 402 due to
     the ladder-side shrink). Reported 04:29 for 合账. **FULLY
     CLOSED 04:36**: war room warehoused the manifest (main commit
     077b88ff, read-back md5 e667926c byte-identical); list 202
     all in corpus (0 missing), 200 legacy files individually
     identified; shrink confirmed real (patrol lists 220→213→210→
     205→200; my 202 was a between-patrol snapshot, the extra 2
     were 11-13 old games dropped after my pull — timing, not a
     data error). Hex from scout: aaa02959b15d57e4e28f80d665ae3482.
   - Next: k4v6 (gap 124) — **hex requested from war room 04:36**
     (queue file has no k4v6 hex; matches.json carries names only).

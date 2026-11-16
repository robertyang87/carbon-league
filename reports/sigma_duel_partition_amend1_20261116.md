# sigma seed-8 duel partition — AMEND1 registration (2026-11-16)

Basis: split ruling 10eca5d7 (md5 5fe93282) + AMEND1 ruling commit 45417749
(reports/RULING_SIGMA_DUEL_SPLIT_AMEND1_20261116.md, md5 94d74f95).
Registered by DGX line before L20 starts, per AMEND1 §2.2.

## Boundary (live read at ruling time)

- DGX direct read 2026-11-16 23:41:33 CST: games 991000-991018 complete,
  game 991019 IN PROGRESS on DGX (k0_duel PID 2833598).
- Per AMEND1 §2.1: the in-progress game stays with DGX and runs to
  completion; completed games are not recomputed; only unstarted games
  transfer.

## Three-way partition (union exactly 50, zero overlap)

| executor | games | count | state at registration |
|---|---|---|---|
| DGX | selfplay-991000 .. selfplay-991019 | 20 | 19 done, 991019 in progress |
| L20 (gpu-l20-6) | selfplay-991020 .. selfplay-991025 | 6 | unstarted |
| spot-38 | selfplay-991026 .. selfplay-991033 | 8 | done, synced to DGX |
| spot-37 | selfplay-991034 .. selfplay-991041 | 8 | done, synced to DGX |
| spot-36 | selfplay-991042 .. selfplay-991049 | 8 | done, synced to DGX |

Total: 20 + 6 + 8 + 8 + 8 = 50. No game appears twice.

## Anti-double-run

- DGX driver is stopped by a boundary watcher the moment
  duel-selfplay-991019.json lands, so DGX cannot start 991020.
  Merge happens by relaunching the sigma8 driver after 50/50; it skips
  landed shards and merges only.
- Sync guard source list is expanded to include L20's out dir
  (/root/carbon/sigma_duel/out on gpu-l20-6) in the same round as this
  registration; guard pulls L20 shards into the DGX shard dir
  (temp + rename, DGX provenance wins, same-game dual source = stop
  and report per original ruling condition 2).

## L20 start conditions (AMEND1 §2.3 / §2.4)

- On-machine md5 read-back before start: student npz 7e563a1f,
  duel eight-piece code set, parent weights 06a03d9c, roots manifest
  25f5bb7f. Staged by DGX line at /root/carbon/sigma_duel/ on L20
  (student.npz, manifests/, runner.sh) plus calibration replays for
  991020-991025 at the absolute path
  /home/arm/junyang/carbon/data/k0_calib_holdout/replays/.
- Pilot priority (§2.4): L20 duel shards may only use CPU released
  after its gen-1 annotation segment completes; on resource collision
  L20's unstarted duel games revert to DGX. L20's start announcement
  must state the annotation segment's completion status.
- Per-game provenance for the sigma report: DGX games = DGX (ARM),
  L20 games = L20 (x86), spot games as registered in the original
  partition file.

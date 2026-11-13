# snap_70 calibration, Echo2 field (C2, spot-31, 2026-11-13)

Tested: snap_70 (snipe_n1 it70 snapshot, attention arm, it70)
Pack: code/exp/snap_70.py, md5 ece3ff5294aaca0be0563be66571d9d7, 4,598,882 B, commit 54bf2c13
Source weight: snap_70.npz, md5 22c5075b0a16e90c72f1c63fa2e940fb, 13,972,532 B
Config: bare-net, SEARCH off; EVO_WORKERS=62. Chain 2026-11-13 19:32:46 -> 19:34:47 CST, rc=0.
Smoke n=2 vs Echo2 seed 999000: rc=0, err 0, worst_ms 93.0.
Master2 field (seed 996000) run in parallel by C1 on spot-29; calibration verdict
needs both fields (Master2 mean - 785 >= +150 is the go threshold).

## vs Echo2 (claude-opus55-cloud-Echo2), seed 999000, 200 games
- wins 121, mean +237.0, se 52.0, lo2se 133.0, worst -2938, worst_ms 118.0, err 0
- launcher JSON + independent recompute from 200-line snapshot agree exactly.

## Archive
- Raw lines: synced/h31/bigeval_live/snap_70_echo2.jsonl (md5 bbe8415e1ba5d4ba5a9f279f4d362915, 835,297 B)

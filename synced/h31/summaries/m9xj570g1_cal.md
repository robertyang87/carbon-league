# it570 g1 calibration (C2, spot-31, 2026-11-13)

Tested: m9xj570g1 (it570 replica, g1 control arm, best ep19, val 0.8448)
Pack: code/exp/m9xj570g1.py, md5 e6e190a8f16250b1ffa45d5913d42d97, 3,730,147 B, commit 9ef9c031
Source weight: cnet2_m9xj570g1.npz, md5 6f15554daf0642c00e0e9d9d37649bba, 11,825,320 B
Config: bare-net, SEARCH off, guard as it320/Lite-V7 calibration; EVO_WORKERS=62.
Chain: /root/carbon/data/it570g1_chain.sh, 2026-11-13 19:12:38 -> 19:15:35 CST, both fields rc=0.
Smoke n=2 vs Master2 seed 996000: rc=0, err 0, worst_ms 67.0.

## Field 1 vs Master2 (true body), seed 996000, 200 games
- wins 173, mean +880.9, se 61.0, lo2se 758.8, worst -3620, worst_ms 83.0, err 0
- launcher JSON + independent recompute from 200-line snapshot agree exactly.
- Calibration score = 880.9 - 785 = +95.9 (< +150 gate) -> FAIL

## Field 2 vs Echo2 (claude-opus55-cloud-Echo2), seed 999000, 200 games
- wins 158, mean +678.9, se 58.8, lo2se 561.2, worst -1856, worst_ms 86.0, err 0
- launcher JSON + independent recompute from 200-line snapshot agree exactly.

## Bridge (true body it570 vs Echo2 on the ladder, frozen 89-game set)
- True body direct record: 2 games, 2 wins, margins +3133.66 / +2820.94, mean +2977.3, se 156.4
- Replica sits (2977.3 - 678.9) / sqrt(156.4^2 + 58.8^2) = 13.76 SE below -> bridge FAIL
- Note: true-body bridge sample is only n=2; SE is noisy, but the gap is ~13.8 SE.

## Verdict
- g1 calibration FAIL (+95.9 < +150). No pool, no re-exam (war room to confirm).
- Raw lines: synced/h31/bigeval_live/m9xj570g1_m2.jsonl (md5 7ea857772fd1ad587a618aaaa02565c0, 835,305 B)
- Raw lines: synced/h31/bigeval_live/m9xj570g1_echo2.jsonl (md5 876f91c326a6324c6763cb28dd834a5b, 835,297 B)

## Ops note
- Archive push first attempt mistakenly targeted main (commit 2ee845b4, no workflow runs
  triggered); file removed from main by delete commit a63598d1, then both files pushed
  to sync/h31 correctly (commits 2384f9ba / cae21d51, read-back md5 verified).

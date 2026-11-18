# Budget Race B — Subject Identity Pin (Addendum)

Date: 2026-11-19 (CST). Filed by the war room (registry keeper).

- Prereg: reports/PREREG_BUDGET_RACE_20261118.md (5fa5c5eb); pairing seal: reports/SEAL_BUDGET_RACE_PAIRING_FIX_20261118.md (ee6290b1).
- B subject: ring-2 student (250-game short ring; 249 games after prescan exclusion of selfplay-960003).
  - Student NPZ: loop2_student.npz, md5 4c49f28146bd3912282db8c80d77faab (sha256 prefix 69e45626, per L20 receipt).
  - Student package: loop2_student_int8.py, md5 162d6142af95cffd7f71f31100fbf3f2.
  - Training input: joined gen-f7b7c986b869c2de; 6,014 updates; parent loop1_student (e28a5dd1).
- Opponents, pre-fire verbatim re-verification on Studio (passed): Master3s2 wrapper md5 b4cde262695d38270e16bde33875a443; Echo6 md5 25f144f3d9ecc87be9073d135d3ab8c8.
- Driver: bigeval 354bcd1b; eval_v53 45792dfb; run_budget_b.sh md5 6c4667c7.
- Fields: vs Master3s2 seeds 962000-962199 (tag budgetB_3s2_962000); vs Echo6 seeds 962200-962399 (tag budgetB_echo6_962200); 200 games each, side=seed%2, identical (seed,side) set to the A field.
- Fired: 2026-11-19 01:30:18 CST on Studio (driver PID 23131).
- Record note: the identity values above were fixed and recorded before firing — in L20's plan-section-2 fill-in table (01:28 CST) and the war-room dispatch to Studio — and Studio's start-of-work receipt verified them on-machine verbatim before the field started. This addendum is the registry filing of those values.

- Companion filing note: Studio's parallel addendum (filed dbbce880) recorded the same pinned values verbatim; after crossed messages it was finally withdrawn by Studio (deletion commit c14e18a5) and will not be restored. This file is the sole pin record.
